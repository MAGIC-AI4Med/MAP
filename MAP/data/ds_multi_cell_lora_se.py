# data/ds_multi_cell_lora_se.py
# Multi-cell-line dataset for LoRA fine-tuning SE model.
# Key change: load SE inputs (gene_ids / expr) from per-cell-line memmap files,
# mirroring the HVG memmap usage in your new framework.
#
# Expected per cell line under:  <se_inputs_base_dir>/<cell_line>/
#   - se_gene_ids.npy  (int32)  shape [N, pad_length]
#   - se_expr.npy      (float16 or float32) shape [N, pad_length]
#   - se_shape.json    {"N": int, "pad_length": int}
#
import os
import json
import ast
from collections import Counter
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem


def extract_concentration(drug_conc_str: str) -> float:
    try:
        conc_list = ast.literal_eval(drug_conc_str)
        return float(conc_list[0][1])
    except Exception:
        return -1.0


class TahoePerturbDatasetSE(Dataset):
    """Dataset that returns SE-model inputs (gene_ids/expressions) for control and perturb sets.

    After DataLoader collation:
      control_gene_ids:      [B, set_size, pad_length] int32
      control_expressions:   [B, set_size, pad_length] float16/float32
      perturb_gene_ids:      [B, set_size, pad_length] int32
      perturb_expressions:   [B, set_size, pad_length] float16/float32
      *_hvg_vectors:         [B, set_size, n_hvg] float32 (or log1p normalized float32)
    """

    def __init__(
        self,
        cell_lines: List[str],
        split: str,
        base_dir: str,
        se_inputs_base_dir: str,
        hvg_not_yet_normed: bool = True,
        set_size: int = 64,
        is_train: bool = True,
        sequential: bool = False,
        return_control_hvg: bool = False,
        UC: bool = False,
        se_expr_dtype: str = "float16",
    ):
        super().__init__()
        self.cell_lines = list(cell_lines)
        self.split = split
        self.base_dir = base_dir
        self.se_inputs_base_dir = se_inputs_base_dir
        self.hvg_not_yet_normed = bool(hvg_not_yet_normed)
        self.set_size = int(set_size)
        self.is_train = bool(is_train)
        self.sequential = bool(sequential)
        self.return_control_hvg = bool(return_control_hvg)
        self.UC = bool(UC)
        self.se_expr_dtype = se_expr_dtype

        self.num_cell_lines = len(self.cell_lines)
        self.cell_line_to_idx = {cl: i for i, cl in enumerate(self.cell_lines)}

        self.cell_line_data: Dict[str, Dict[str, Any]] = {}
        self.hvg_mmaps: Dict[str, np.memmap] = {}
        self.se_gene_mmaps: Dict[str, np.memmap] = {}
        self.se_expr_mmaps: Dict[str, np.memmap] = {}
        self.se_shapes: Dict[str, Tuple[int, int]] = {}  # cl -> (N, L)
        self.emb_mmaps: Dict[str, np.memmap] = {}


        self.total_data_samples = 0
        self.total_control_samples = 0

        import pickle

        for cell_line in self.cell_lines:
            cell_line_dir = os.path.join(self.base_dir, cell_line)
            meta_path = os.path.join(cell_line_dir, f"{cell_line}_meta.pkl")
            if not os.path.isfile(meta_path):
                raise FileNotFoundError(f"Missing meta: {meta_path}")

            with open(meta_path, "rb") as f:
                meta = pickle.load(f)

            all_ds_indices = meta["ds_level_index"]
            all_drug_conc = meta["drug_conc"]

            suffix = "_UC" if self.UC else ""
            split_json = os.path.join(cell_line_dir, f"{split}_indices{suffix}.json")
            control_json = os.path.join(cell_line_dir, f"control_indices{suffix}.json")

            data_indices = self._filter_indices(split_json, all_ds_indices, all_drug_conc)
            control_indices = self._filter_indices(control_json, all_ds_indices, all_drug_conc)

            drug_conc_to_rows = {}
            for row_idx in data_indices:
                dc = all_drug_conc[row_idx]
                drug_conc_to_rows.setdefault(dc, []).append(row_idx)

            self.cell_line_data[cell_line] = {
                "cell_line_dir": cell_line_dir,
                "meta": meta,
                "data_indices": data_indices,
                "control_indices": control_indices,
                "drug_conc_to_matching_rows": drug_conc_to_rows,
            }

            self.total_data_samples += len(data_indices)
            self.total_control_samples += len(control_indices)

            # Validate SE memmap metadata exists and matches N
            se_dir = os.path.join(self.se_inputs_base_dir, cell_line)
            shape_path = os.path.join(se_dir, "se_shape.json")
            if not os.path.isfile(shape_path):
                raise FileNotFoundError(f"Missing se_shape.json: {shape_path}")
            with open(shape_path, "r") as f:
                shape = json.load(f)
            N = int(shape["N"])
            L = int(shape["pad_length"])
            if N != len(all_ds_indices):
                raise ValueError(
                    f"SE memmap N mismatch for {cell_line}: se_shape N={N}, meta ds_level_index N={len(all_ds_indices)}"
                )
            self.se_shapes[cell_line] = (N, L)

        # Sampling weights across cell lines
        self.cell_line_weights = self._calculate_cell_line_weights()

        # Sequential mode: evaluation only
        self._seq_mode_active = (self.sequential and (not self.is_train))
        if self._seq_mode_active:
            self._setup_sequential_mode()

        print(
            f"[OK] TahoePerturbDatasetSE(memmap) | split={split} | cell_lines={len(self.cell_lines)} | "
            f"data={self.total_data_samples} | control={self.total_control_samples}"
        )

    def _cell_line_one_hot(self, cell_line: str) -> torch.Tensor:
        idx = self.cell_line_to_idx[cell_line]
        ohe = torch.zeros(self.num_cell_lines, dtype=torch.float32)
        ohe[idx] = 1.0
        return ohe

    def _filter_indices(self, json_path, all_ds_indices, all_drug_conc):
        with open(json_path, "r") as f:
            target = set(json.load(f))
        valid = []
        for row_idx, ds_idx in enumerate(all_ds_indices):
            if ds_idx in target:
                conc_str = all_drug_conc[row_idx]
                if "Sacubitril/Valsartan" in conc_str or "Verteporfin" in conc_str:
                    continue
                valid.append(row_idx)
        return valid

    def _calculate_cell_line_weights(self):
        weights = np.array([len(self.cell_line_data[cl]["data_indices"]) for cl in self.cell_lines], dtype=float)
        weights = weights / weights.sum()
        return weights

    def _get_hvg_mmap(self, cell_line: str) -> np.memmap:
        if cell_line in self.hvg_mmaps:
            return self.hvg_mmaps[cell_line]
        cell_line_dir = self.cell_line_data[cell_line]["cell_line_dir"]
        meta = self.cell_line_data[cell_line]["meta"]
        hvg_path = os.path.join(cell_line_dir, f"{cell_line}_hvg.npy")
        self.hvg_mmaps[cell_line] = np.memmap(hvg_path, dtype="float32", mode="r", shape=meta["shape_hvg"])
        return self.hvg_mmaps[cell_line]

    def _get_se_mmaps(self, cell_line: str) -> Tuple[np.memmap, np.memmap]:
        if cell_line in self.se_gene_mmaps and cell_line in self.se_expr_mmaps:
            return self.se_gene_mmaps[cell_line], self.se_expr_mmaps[cell_line]

        se_dir = os.path.join(self.se_inputs_base_dir, cell_line)
        gene_path = os.path.join(se_dir, "se_gene_ids.npy")
        expr_path = os.path.join(se_dir, "se_expr.npy")
        if not (os.path.isfile(gene_path) and os.path.isfile(expr_path)):
            raise FileNotFoundError(f"Missing SE memmaps for {cell_line} under {se_dir}")

        N, L = self.se_shapes[cell_line]
        self.se_gene_mmaps[cell_line] = np.memmap(gene_path, dtype="int32", mode="r", shape=(N, L))
        self.se_expr_mmaps[cell_line] = np.memmap(expr_path, dtype=self.se_expr_dtype, mode="r", shape=(N, L))
        return self.se_gene_mmaps[cell_line], self.se_expr_mmaps[cell_line]
    
    def _get_emb_mmap(self, cell_line: str) -> np.memmap:
        """懒加载获取指定细胞系的 embedding memmap 对象"""
        if cell_line in self.emb_mmaps:
            return self.emb_mmaps[cell_line]

        cell_line_dir = self.cell_line_data[cell_line]["cell_line_dir"]
        meta = self.cell_line_data[cell_line]["meta"]

        emb_path = os.path.join(cell_line_dir, f"{cell_line}_embeddings.npy")
        
        if not os.path.isfile(emb_path):
            raise FileNotFoundError(f"Missing embedding file: {emb_path}")

        self.emb_mmaps[cell_line] = np.memmap(
            emb_path,
            dtype=meta["dtype"],
            mode="r",
            shape=meta["shape_emb"]
        )
        
        return self.emb_mmaps[cell_line]


    def _process_hvg(self, hvg_batch_np):
        hvg_tensor = torch.from_numpy(hvg_batch_np)
        if self.hvg_not_yet_normed:
            total_counts = hvg_tensor.sum(dim=1, keepdim=True)
            total_counts[total_counts == 0] = 1.0
            normalized = 1000 * hvg_tensor / total_counts
            return torch.log1p(normalized)
        return hvg_tensor

    def _sample_control_rows(self, cell_line):
        control_indices = self.cell_line_data[cell_line]["control_indices"]
        n_control = len(control_indices)
        idx = np.random.choice(n_control, self.set_size, replace=(n_control < self.set_size))
        return [control_indices[i] for i in idx]

    def _calculate_sampling_weights(self, cell_line):
        cell_data = self.cell_line_data[cell_line]
        data_indices = cell_data["data_indices"]
        meta = cell_data["meta"]

        conc_counts = Counter()
        for row_idx in data_indices:
            conc_counts[meta["drug_conc"][row_idx]] += 1

        perturb_counts = {k: v for k, v in conc_counts.items() if "0.0," not in k}
        if not perturb_counts:
            raise ValueError(f"No perturbation conditions for {cell_line}")

        keys = list(perturb_counts.keys())
        counts = np.array([perturb_counts[k] for k in keys], dtype=float)
        weights = counts / counts.sum()
        return weights, keys

    def _sample_perturb_rows(self, cell_line, target_conc_str):
        matching = self.cell_line_data[cell_line]["drug_conc_to_matching_rows"][target_conc_str]
        if len(matching) >= self.set_size:
            return np.random.choice(matching, self.set_size, replace=False).tolist()
        return np.random.choice(matching, self.set_size, replace=True).tolist()

    def _setup_sequential_mode(self):
        self._seq_cell_smiles_to_rows = {}
        for cell_line in self.cell_lines:
            data_indices = self.cell_line_data[cell_line]["data_indices"]
            meta = self.cell_line_data[cell_line]["meta"]
            for row_idx in data_indices:
                smi = meta["drug_smiles"][row_idx]
                key = (cell_line, smi)
                self._seq_cell_smiles_to_rows.setdefault(key, []).append(row_idx)
        self.sequential_cell_smiles = sorted(self._seq_cell_smiles_to_rows.keys())
        print(f"[SEQ] unique (cell_line, smiles) = {len(self.sequential_cell_smiles)}")

    def __len__(self):
        if self._seq_mode_active:
            return len(self.sequential_cell_smiles)
        else:
            base_length = 2 * self.total_data_samples // self.set_size
            return max(base_length, 1000)

    def _fetch_se_by_rows(self, cell_line: str, rows: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        gene_mm, expr_mm = self._get_se_mmaps(cell_line)
        gene = torch.from_numpy(gene_mm[rows])  # int32
        expr = torch.from_numpy(expr_mm[rows]).float()  # float16/float32
        return gene, expr

    def __getitem__(self, index):
        # Sequential mode (eval)
        if self._seq_mode_active:
            cell_line, target_smiles = self.sequential_cell_smiles[index]
            matching_rows = self._seq_cell_smiles_to_rows[(cell_line, target_smiles)]
            canon_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(target_smiles))

            sampled_rows = np.random.choice(matching_rows, self.set_size, replace=(len(matching_rows) < self.set_size))
            sampled_rows = list(map(int, sampled_rows))
            control_rows = self._sample_control_rows(cell_line)

            # Extract concentration (same pattern as train mode)
            meta = self.cell_line_data[cell_line]["meta"]
            conc = extract_concentration(meta["drug_conc"][sampled_rows[0]])

            # ===== NEW: load perturb embeddings for supervision (sequential mode) =====
            emb_mmap = self._get_emb_mmap(cell_line)
            perturb_embeddings = torch.from_numpy(emb_mmap[sampled_rows])

            perturb_gene_ids, perturb_expr = self._fetch_se_by_rows(cell_line, sampled_rows)
            control_gene_ids, control_expr = self._fetch_se_by_rows(cell_line, control_rows)

            hvg_mmap = self._get_hvg_mmap(cell_line)
            perturb_hvg_vectors = self._process_hvg(hvg_mmap[sampled_rows])
            control_hvg_vectors = self._process_hvg(hvg_mmap[control_rows])

            cell_line_idx = self.cell_line_to_idx[cell_line]
            cell_line_ohe = self._cell_line_one_hot(cell_line)

            return {
                "control_gene_ids": control_gene_ids,
                "control_expressions": control_expr,
                "perturb_gene_ids": perturb_gene_ids,
                "perturb_expressions": perturb_expr,
                'perturb_embeddings': perturb_embeddings,
                "perturb_hvg_vectors": perturb_hvg_vectors,
                "control_hvg_vectors": control_hvg_vectors,
                "drug_smiles": canon_smiles,
                "drug_conc": conc,  # ← Changed from -1.0 to actual concentration
                "cell_line": cell_line,
                "cell_line_idx": cell_line_idx,
                "cell_line_ohe": cell_line_ohe,
            }

        # Train/random mode
        cell_line = np.random.choice(self.cell_lines, p=self.cell_line_weights)
        cell_data = self.cell_line_data[cell_line]
        meta = cell_data["meta"]

        # lazily cache per-cell-line perturbation weights
        cache_key = f"_perturbation_weights_{cell_line}"
        if not hasattr(self, cache_key):
            w, keys = self._calculate_sampling_weights(cell_line)
            setattr(self, cache_key, w)
            setattr(self, f"_unique_perturbations_{cell_line}", keys)

        weights = getattr(self, cache_key)
        unique = getattr(self, f"_unique_perturbations_{cell_line}")

        target_drug_conc_str = np.random.choice(unique, p=weights)
        conc = extract_concentration(target_drug_conc_str)

        control_rows = self._sample_control_rows(cell_line)
        perturb_rows = self._sample_perturb_rows(cell_line, target_drug_conc_str)

        # ===== NEW: load perturb embeddings for supervision =====
        emb_mmap = self._get_emb_mmap(cell_line)
        perturb_embeddings = torch.from_numpy(emb_mmap[perturb_rows])


        control_gene_ids, control_expr = self._fetch_se_by_rows(cell_line, control_rows)
        perturb_gene_ids, perturb_expr = self._fetch_se_by_rows(cell_line, perturb_rows)

        hvg_mmap = self._get_hvg_mmap(cell_line)
        control_hvg_vectors = self._process_hvg(hvg_mmap[control_rows])
        perturb_hvg_vectors = self._process_hvg(hvg_mmap[perturb_rows])

        raw_smi = meta["drug_smiles"][int(perturb_rows[0])]
        canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(raw_smi))

        cell_line_idx = self.cell_line_to_idx[cell_line]
        cell_line_ohe = self._cell_line_one_hot(cell_line)

        return {
            "control_gene_ids": control_gene_ids,
            "control_expressions": control_expr,
            "perturb_gene_ids": perturb_gene_ids,
            "perturb_expressions": perturb_expr,
            "perturb_hvg_vectors": perturb_hvg_vectors,
            'perturb_embeddings': perturb_embeddings,
            "control_hvg_vectors": control_hvg_vectors,
            "drug_smiles": canon_smi,
            "drug_conc": conc,
            "cell_line": cell_line,
            "cell_line_idx": cell_line_idx,
            "cell_line_ohe": cell_line_ohe,
        }
