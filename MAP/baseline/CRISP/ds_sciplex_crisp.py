# ds_sciplex_crisp.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
from typing import Dict, List, Tuple, Optional, Any

try:
    from rdkit import Chem
except Exception:
    Chem = None

import scanpy as sc


# =========================================================
# (1) Sciplex dataset 类：从 ds_sciplex.py 适配而来
# =========================================================

def remove_non_alphanumeric(s: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9]", "", str(s))


def ensure_condition_and_control(adata):
    """
    Sciplex 专用逻辑（对齐 D2_split_UD.py）:
      - condition：优先 obs['condition']，否则由 obs['sm_name'] 清洗得到
      - is_control：优先 obs['control']，否则 obs['vehicle']，最后才用 condition == 'DMSO'
    """
    obs = adata.obs

    # 1) condition 列
    if "condition" in obs.columns:
        condition = obs["condition"].astype(str).values
    elif "sm_name" in obs.columns:
        cond = obs["sm_name"].astype(str).map(remove_non_alphanumeric)
        cond = cond.replace({"DimethylSulfoxide": "DMSO"})
        condition = cond.values.astype(str)
    else:
        raise KeyError("Need obs['condition'] or obs['sm_name'] to build condition.")

    # 统一 DMSO 拼写
    condition = np.array([("DMSO" if c == "DimethylSulfoxide" else c) for c in condition], dtype=object).astype(str)

    # 2) control mask - Sciplex 优先级顺序
    if "control" in obs.columns:
        is_control = obs["control"].astype(int).values.astype(bool)
    elif "vehicle" in obs.columns:
        is_control = obs["vehicle"].astype(int).values.astype(bool)
    else:
        # fallback
        is_control = (condition == "DMSO")

    return condition, is_control


class SciplexPerturbDataset(Dataset):
    """
    Sciplex 多 cell-type 数据集（对齐 CRISP 训练需求）
    """

    def __init__(
        self,
        cell_types,
        split="train",
        h5ad_path="/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/external_dataset_sciplex/raw/sciplex3_pp_hvgenes_scFM_resplit_filtered_ESM2.h5ad",
        shards_dir="/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/ours_3/shards/sciplex",
        hvg_root="/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/external_dataset_sciplex/preprocessed/sciplex3_hvg_by_celltype_seurat_top2000",
        split_dir="/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/external_dataset_sciplex/split",
        set_size=32,
        UC = False,
        is_train=True,
        sequential=False,
        hvg_not_yet_normed=False,
        seed=42,
    ):
        self.cell_types = list(cell_types)
        self.split = split
        self.h5ad_path = h5ad_path
        self.shards_dir = shards_dir
        self.hvg_root = hvg_root
        self.split_dir = split_dir
        self.set_size = int(set_size)
        self.UC = UC
        self.is_train = bool(is_train)
        self.sequential = bool(sequential)
        self.hvg_not_yet_normed = bool(hvg_not_yet_normed)
        self.rng = np.random.RandomState(seed)

        # ============ 1) 读取 split ============
        postfix = '_UC.json' if self.UC else '.json'
        control_json = os.path.join(split_dir, "control" + postfix)
        train_json = os.path.join(split_dir, "train" + postfix)
        test_json = os.path.join(split_dir, "test" + postfix)

        with open(control_json, "r") as f:
            self.control_ds = set(json.load(f))
        with open(train_json, "r") as f:
            self.train_ds = set(json.load(f))
        with open(test_json, "r") as f:
            self.test_ds = set(json.load(f))

        if split == "train":
            self.data_ds = self.train_ds
        elif split == "test":
            self.data_ds = self.test_ds
        else:
            raise ValueError("split must be 'train' or 'test' for Sciplex.")

        # ============ 2) 读取 h5ad 元信息（backed） ============
        adata = sc.read_h5ad(h5ad_path, backed="r")

        if "cell_type" not in adata.obs.columns:
            raise KeyError("Missing obs['cell_type'] in h5ad.")
        if "SMILES" not in adata.obs.columns:
            raise KeyError("Missing obs['SMILES'] in h5ad.")
        if "dose_val" not in adata.obs.columns:
            raise KeyError("Missing obs['dose_val'] in h5ad (Sciplex specific).")

        self.obs_names = np.asarray(adata.obs_names).astype(str)
        self.cell_type_arr = np.asarray(adata.obs["cell_type"]).astype(str)
        self.smiles_arr = np.asarray(adata.obs["SMILES"]).astype(str)
        self.dose_arr = np.asarray(adata.obs["dose_val"]).astype(float)  # Sciplex 特有

        # condition + control
        self.condition_arr, self.is_control_arr = ensure_condition_and_control(adata)

        # ============ 3) 载入 embedding shards ============
        self.embeddings, self.emb_ds_index, self.emb_cell_type, self.emb_smiles = self._load_all_shards(shards_dir)
        self.emb_dim = int(self.embeddings.shape[1])

        self.ds_to_emb_row = {}
        for r, ds_idx in enumerate(self.emb_ds_index):
            ds_idx = int(ds_idx)
            if ds_idx not in self.ds_to_emb_row:
                self.ds_to_emb_row[ds_idx] = r

        # ============ 4) cell_type 数据池构建 ============
        self.cell_type_data = {}
        total_data = 0

        for ct in self.cell_types:
            ct_mask = (self.cell_type_arr == ct)
            ct_all = np.where(ct_mask)[0].astype(int)

            ct_control = [i for i in ct_all.tolist() if i in self.control_ds]
            ct_data = [i for i in ct_all.tolist() if i in self.data_ds]

            ct_control = [i for i in ct_control if i in self.ds_to_emb_row]
            ct_data = [i for i in ct_data if i in self.ds_to_emb_row]

            drug_to_rows = {}
            for ds_idx in ct_data:
                drug_id = self.condition_arr[ds_idx]
                drug_to_rows.setdefault(drug_id, []).append(ds_idx)

            self.cell_type_data[ct] = {
                "data_ds": ct_data,
                "control_ds": ct_control,
                "drug_to_ds": drug_to_rows,
            }

            total_data += len(ct_data)

        self.total_data_samples = total_data

        # ============ 5) cell_type 采样权重 ============
        weights = []
        for ct in self.cell_types:
            weights.append(len(self.cell_type_data[ct]["data_ds"]))
        weights = np.asarray(weights, dtype=float)
        if weights.sum() == 0:
            raise RuntimeError("No data samples found for given cell_types and split.")
        self.cell_type_weights = weights / weights.sum()

        # ============ 6) HVG 缓存 ============
        self._hvg_cache = {}

        # ============ 7) Sequential 模式 ============
        self._seq_mode_active = (self.sequential and (not self.is_train))
        if self._seq_mode_active:
            self._setup_sequential_mode()

    def _load_all_shards(self, shards_dir):
        summary_path = os.path.join(shards_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary = json.load(f)
            files = summary.get("files", [])
            if files:
                part_paths = [os.path.join(shards_dir, fn) for fn in files]
            else:
                part_paths = sorted(glob.glob(os.path.join(shards_dir, "embeddings_part_*.pt")))
        else:
            part_paths = sorted(glob.glob(os.path.join(shards_dir, "embeddings_part_*.pt")))

        if not part_paths:
            raise FileNotFoundError(f"No shard files found in {shards_dir}")

        embs = []
        ds_idx = []
        cell_types = []
        smiles = []

        for p in part_paths:
            d = torch.load(p, map_location="cpu")
            embs.append(d["embeddings"])
            ds_idx.extend(d["ds_level_index"])
            cell_types.extend(d["cell_line_id"])
            smiles.extend(d["drug_smiles"])

        embeddings = torch.cat(embs, dim=0).contiguous()
        embeddings = embeddings.numpy()
        ds_idx = np.asarray(ds_idx, dtype=np.int64)
        cell_types = np.asarray(cell_types, dtype=object).astype(str)
        smiles = np.asarray(smiles, dtype=object).astype(str)

        return embeddings, ds_idx, cell_types, smiles

    def _load_hvg_for_cell_type(self, ct: str):
        if ct in self._hvg_cache:
            return self._hvg_cache[ct]

        ct_safe = ct.replace("/", "_")
        ct_dir = os.path.join(self.hvg_root, ct_safe)

        x_path = os.path.join(ct_dir, "X_hvg_top2000_seurat.npz")
        if not os.path.exists(x_path):
            cand = glob.glob(os.path.join(ct_dir, "X_hvg_top2000_seurat*.npz"))
            if not cand:
                raise FileNotFoundError(f"HVG npz not found under {ct_dir}")
            x_path = cand[0]

        cell_ids_path = os.path.join(ct_dir, "cell_ids.npy")
        if not os.path.exists(cell_ids_path):
            raise FileNotFoundError(f"cell_ids.npy not found under {ct_dir}")

        X_hvg = sparse.load_npz(x_path).tocsr()
        cell_ids = np.load(cell_ids_path, allow_pickle=True).astype(str)

        obsname2row = {name: i for i, name in enumerate(cell_ids.tolist())}

        self._hvg_cache[ct] = {
            "X_hvg": X_hvg,
            "obsname2row": obsname2row,
            "n_hvg": int(X_hvg.shape[1]),
        }
        return self._hvg_cache[ct]

    def _fetch_hvg_rows(self, ct: str, ds_indices):
        info = self._load_hvg_for_cell_type(ct)
        X_hvg = info["X_hvg"]
        obsname2row = info["obsname2row"]

        rows = []
        for ds_idx in ds_indices:
            obs_name = self.obs_names[int(ds_idx)]
            r = obsname2row.get(obs_name, None)
            if r is None:
                raise KeyError(f"obs_name {obs_name} (ds={ds_idx}) not found in HVG cell_ids for ct={ct}")
            rows.append(r)

        mat = X_hvg[rows].toarray().astype(np.float32)
        return mat

    def _sample_control_ds(self, ct: str):
        pool = self.cell_type_data[ct]["control_ds"]
        if len(pool) == 0:
            raise RuntimeError(f"No control cells found for cell_type={ct}")
        if len(pool) >= self.set_size:
            return self.rng.choice(pool, self.set_size, replace=False).tolist()
        return self.rng.choice(pool, self.set_size, replace=True).tolist()

    def _sample_perturb_ds(self, ct: str, drug_id: str):
        pool = self.cell_type_data[ct]["drug_to_ds"].get(drug_id, [])
        if len(pool) == 0:
            raise RuntimeError(f"No perturb cells for (ct={ct}, drug_id={drug_id}) in split={self.split}")
        if len(pool) >= self.set_size:
            return self.rng.choice(pool, self.set_size, replace=False).tolist()
        return self.rng.choice(pool, self.set_size, replace=True).tolist()

    def _choose_drug_weighted(self, ct: str):
        drug_to_ds = self.cell_type_data[ct]["drug_to_ds"]
        drugs = list(drug_to_ds.keys())
        counts = np.asarray([len(drug_to_ds[d]) for d in drugs], dtype=float)
        p = counts / counts.sum()
        return self.rng.choice(drugs, p=p)

    def _process_hvg(self, hvg_np: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(hvg_np)
        if self.hvg_not_yet_normed:
            total = x.sum(dim=1, keepdim=True)
            total[total == 0] = 1.0
            x = 1000.0 * x / total
            x = torch.log1p(x)
        return x

    def _setup_sequential_mode(self):
        self._seq_key_to_ds = {}
        for ct in self.cell_types:
            for ds_idx in self.cell_type_data[ct]["data_ds"]:
                smi = self.smiles_arr[int(ds_idx)]
                key = (ct, smi)
                self._seq_key_to_ds.setdefault(key, []).append(int(ds_idx))
        self.sequential_keys = sorted(self._seq_key_to_ds.keys())

    def __len__(self):
        if self._seq_mode_active:
            return len(self.sequential_keys)
        if self.is_train:
            base_len = 2 * max(self.total_data_samples, 1) // max(self.set_size, 1)
            return max(int(base_len), 1000)
        return -1

    def __getitem__(self, index):
        # -------- Sequential mode --------
        if self._seq_mode_active:
            ct, raw_smiles = self.sequential_keys[index]
            candidate_ds = self._seq_key_to_ds[(ct, raw_smiles)]

            if len(candidate_ds) >= self.set_size:
                pert_ds = self.rng.choice(candidate_ds, self.set_size, replace=False).tolist()
            else:
                pert_ds = self.rng.choice(candidate_ds, self.set_size, replace=True).tolist()

            ctrl_ds = self._sample_control_ds(ct)

            pert_rows = [self.ds_to_emb_row[int(i)] for i in pert_ds]
            ctrl_rows = [self.ds_to_emb_row[int(i)] for i in ctrl_ds]
            perturb_embeddings = torch.from_numpy(self.embeddings[pert_rows]).float()
            control_embeddings = torch.from_numpy(self.embeddings[ctrl_rows]).float()

            perturb_hvg = self._process_hvg(self._fetch_hvg_rows(ct, pert_ds))
            control_hvg = self._process_hvg(self._fetch_hvg_rows(ct, ctrl_ds))

            canon = raw_smiles
            if Chem is not None:
                m = Chem.MolFromSmiles(raw_smiles)
                if m is not None:
                    canon = Chem.MolToSmiles(m)

            drug_name = self.condition_arr[int(pert_ds[0])]
            drug_conc = float(self.dose_arr[int(pert_ds[0])])

            return {
                "control_embeddings": control_embeddings,
                "perturb_embeddings": perturb_embeddings,
                "perturb_hvg_vectors": perturb_hvg,
                "control_hvg_vectors": control_hvg,
                "drug_smiles": canon,
                "drug_conc": drug_conc,
                "cell_line": ct,
                "target_drug_conc_str": drug_name,
                "final_ds_indices": pert_ds,
                "control_ds_indices": ctrl_ds,
            }

        # -------- Random sampling mode --------
        ct = self.rng.choice(self.cell_types, p=self.cell_type_weights)
        drug_id = self._choose_drug_weighted(ct)

        ctrl_ds = self._sample_control_ds(ct)
        pert_ds = self._sample_perturb_ds(ct, drug_id)

        pert_rows = [self.ds_to_emb_row[int(i)] for i in pert_ds]
        ctrl_rows = [self.ds_to_emb_row[int(i)] for i in ctrl_ds]

        perturb_embeddings = torch.from_numpy(self.embeddings[pert_rows]).float()
        control_embeddings = torch.from_numpy(self.embeddings[ctrl_rows]).float()

        perturb_hvg = self._process_hvg(self._fetch_hvg_rows(ct, pert_ds))
        control_hvg = self._process_hvg(self._fetch_hvg_rows(ct, ctrl_ds))

        raw_smiles = self.smiles_arr[int(pert_ds[0])]
        canon = raw_smiles
        if Chem is not None:
            m = Chem.MolFromSmiles(raw_smiles)
            if m is not None:
                canon = Chem.MolToSmiles(m)
        
        drug_conc = float(self.dose_arr[int(pert_ds[0])])

        return {
            "control_embeddings": control_embeddings,
            "perturb_embeddings": perturb_embeddings,
            "perturb_hvg_vectors": perturb_hvg,
            "control_hvg_vectors": control_hvg,
            "drug_smiles": canon,
            "drug_conc": drug_conc,
            "cell_line": ct,
        }


# =========================================================
# (2) SmilesEmbeddingStore：与 Tahoe 版本对齐（不变）
# =========================================================

class SmilesEmbeddingStore:
    """
    加载预计算的 SMILES embeddings
    """
    def __init__(
        self,
        emb_pt_path: str,
        device: str = "cpu",
        canonicalize: bool = True,
        isomeric_smiles: bool = True,
    ):
        self.emb_pt_path = emb_pt_path
        self.device = torch.device(device)
        self.canonicalize = bool(canonicalize)
        self.isomeric_smiles = bool(isomeric_smiles)

        obj = torch.load(emb_pt_path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "embeddings" in obj and isinstance(obj["embeddings"], dict):
            emb_dict = obj["embeddings"]
            self.meta = obj.get("meta", None)
        elif isinstance(obj, dict):
            emb_dict = obj
            self.meta = None
        else:
            raise ValueError("Unrecognized smiles embedding file format")

        norm_dict: Dict[str, torch.Tensor] = {}
        for k, v in emb_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(np.asarray(v), dtype=torch.float32)
            norm_dict[str(k)] = v.detach().cpu().float()

        self.raw_emb_dict = norm_dict
        self.emb_dict: Dict[str, torch.Tensor] = {}

        if self.canonicalize:
            try:
                from rdkit import Chem as _Chem
                self.Chem = _Chem
            except Exception:
                raise ImportError("RDKit is required for canonicalize=True")

            for s, emb in self.raw_emb_dict.items():
                cs = self._canonicalize_smiles(s)
                if cs is None:
                    continue
                self.emb_dict[cs] = emb
            if len(self.emb_dict) == 0:
                raise ValueError("No valid SMILES keys after canonicalization")
        else:
            self.emb_dict = dict(self.raw_emb_dict)

        any_emb = next(iter(self.emb_dict.values()))
        self._dim = int(any_emb.numel())

    @property
    def dim(self) -> int:
        return self._dim

    def _canonicalize_smiles(self, smiles: str) -> Optional[str]:
        s = str(smiles).strip()
        if s == "":
            return None
        m = self.Chem.MolFromSmiles(s)
        if m is None:
            return None
        return self.Chem.MolToSmiles(m, canonical=True, isomericSmiles=self.isomeric_smiles)

    def get(self, smiles: str) -> torch.Tensor:
        s0 = str(smiles)
        if self.canonicalize:
            cs = self._canonicalize_smiles(s0)
            if cs is not None and cs in self.emb_dict:
                return self.emb_dict[cs]
            if s0 in self.raw_emb_dict:
                return self.raw_emb_dict[s0]
            raise KeyError(f"SMILES not found: {s0}, canonical: {cs}")
        else:
            if s0 not in self.emb_dict:
                raise KeyError(f"SMILES not found: {s0}")
            return self.emb_dict[s0]


# =========================================================
# (3) CRISP Sciplex Collator
# =========================================================

class CRISPSciplexCollator:
    """
    将 SciplexPerturbDataset 的输出转换为 CRISP 原版训练所需的格式
    
    与 NIPS collator 的唯一区别：
      - dosages 字段使用真实浓度（从 batch[i]["drug_conc"] 读取，不是固定 -1.0）
    """

    def __init__(
        self,
        smiles_emb_store: SmilesEmbeddingStore,
        cell_line_to_idx: Dict[str, int],
        device: str = "cpu",
        neg_strategy: str = "same_drug_diff_celltype",
    ):
        self.smiles_emb_store = smiles_emb_store
        self.cell_line_to_idx = cell_line_to_idx
        self.device = device
        self.neg_strategy = neg_strategy

        self.num_genes = None

    def __call__(self, batch: List[Dict[str, Any]]) -> Tuple:
        B = len(batch)
        S = batch[0]["perturb_embeddings"].shape[0]

        if self.num_genes is None:
            self.num_genes = batch[0]["perturb_hvg_vectors"].shape[1]

        # ========= 正样本 =========
        genes_list = []
        paired_cell_emb_list = []
        drugs_pre_list = []
        dosages_list = []
        celltype_idx_list = []

        for sample in batch:
            genes_list.append(sample["perturb_hvg_vectors"])
            paired_cell_emb_list.append(sample["control_embeddings"])

            drug_emb = self.smiles_emb_store.get(sample["drug_smiles"])
            drugs_pre_list.append(drug_emb)

            dosages_list.append(float(sample["drug_conc"]))  # ✅ Sciplex 真实浓度
            celltype_idx_list.append(self.cell_line_to_idx[sample["cell_line"]])

        genes = torch.cat(genes_list, dim=0)
        paired_cell_embeddings = torch.cat(paired_cell_emb_list, dim=0)
        drugs_pre = torch.stack(drugs_pre_list, dim=0).repeat_interleave(S, dim=0)
        dosages = torch.tensor(np.repeat(dosages_list, S), dtype=torch.float32)
        celltype_idx = torch.tensor(np.repeat(celltype_idx_list, S), dtype=torch.long)
        degs = torch.zeros_like(genes, dtype=torch.bool)

        # ========= 负样本 =========
        neg_indices = self._sample_negative_indices(batch)

        neg_genes_list = []
        neg_paired_cell_emb_list = []
        neg_drugs_pre_list = []
        neg_dosages_list = []
        neg_celltype_idx_list = []

        for i, neg_idx in enumerate(neg_indices):
            neg_sample = batch[neg_idx]

            neg_genes_list.append(neg_sample["perturb_hvg_vectors"])
            neg_paired_cell_emb_list.append(neg_sample["control_embeddings"])

            # 负样本使用"与正样本相同药物"的 embedding
            neg_drug_emb = self.smiles_emb_store.get(batch[i]["drug_smiles"])
            neg_drugs_pre_list.append(neg_drug_emb)

            neg_dosages_list.append(float(batch[i]["drug_conc"]))  # ✅ 使用正样本浓度
            neg_celltype_idx_list.append(self.cell_line_to_idx[neg_sample["cell_line"]])

        neg_genes = torch.cat(neg_genes_list, dim=0)
        neg_paired_cell_embeddings = torch.cat(neg_paired_cell_emb_list, dim=0)
        neg_drugs_pre = torch.stack(neg_drugs_pre_list, dim=0).repeat_interleave(S, dim=0)
        neg_dosages = torch.tensor(np.repeat(neg_dosages_list, S), dtype=torch.float32)
        neg_celltype_idx = torch.tensor(np.repeat(neg_celltype_idx_list, S), dtype=torch.long)
        neg_degs = torch.zeros_like(neg_genes, dtype=torch.bool)

        # ========= 元信息 =========
        metadata = {
            "smiles": [sample["drug_smiles"] for sample in batch],
            "cell_lines": [sample["cell_line"] for sample in batch],
            "drug_names": [sample.get("target_drug_conc_str", "Unknown") for sample in batch],
            "concentrations": [sample["drug_conc"] for sample in batch],  # ✅ 真实浓度
            "set_size": S,
            "control_hvg": torch.cat([sample["control_hvg_vectors"] for sample in batch], dim=0),
        }

        return (
            genes,
            paired_cell_embeddings,
            None,  # drugs_idx
            dosages,
            degs,
            celltype_idx,
            neg_genes,
            neg_paired_cell_embeddings,
            None,  # neg_drugs_idx
            neg_dosages,
            neg_degs,
            neg_celltype_idx,
            None,  # covariates
            None,  # neg_covariates
            drugs_pre,
            neg_drugs_pre,
            metadata,
        )

    def _sample_negative_indices(self, batch: List[Dict[str, Any]]) -> List[int]:
        """
        same_drug_diff_celltype 策略（与 NIPS 一致）
        """
        B = len(batch)
        neg_indices: List[int] = []

        for i in range(B):
            pos_smiles = batch[i]["drug_smiles"]
            pos_cell = batch[i]["cell_line"]

            candidates = []
            for j in range(B):
                if j != i and batch[j]["drug_smiles"] == pos_smiles and batch[j]["cell_line"] != pos_cell:
                    candidates.append(j)

            if candidates:
                neg_idx = np.random.choice(candidates)
            else:
                diff_cell_candidates = [j for j in range(B) if batch[j]["cell_line"] != pos_cell]
                if diff_cell_candidates:
                    neg_idx = np.random.choice(diff_cell_candidates)
                else:
                    neg_idx = np.random.choice([j for j in range(B) if j != i]) if B > 1 else i

            neg_indices.append(int(neg_idx))

        return neg_indices


# =========================================================
# (4) build_crisp_dataloader：Sciplex 版本
# =========================================================

def build_crisp_dataloader(
    cell_types: List[str],
    split: str,
    drug_emb_pt_path: str,
    batch_size: int,
    set_size: int,
    UC: bool = False,
    num_workers: int = 4,
    device: str = "cpu",
    hvg_not_yet_normed: bool = False,
    is_train: bool = True,
    sequential: bool = False,
    seed: int = 42,
    # --- Sciplex dataset paths ---
    h5ad_path: str = "/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/external_dataset_sciplex/raw/sciplex3_pp_hvgenes_scFM_resplit_filtered_ESM2.h5ad",
    shards_dir: str = "/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/ours_3/shards/sciplex",
    hvg_root: str = "/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/external_dataset_sciplex/preprocessed/sciplex3_hvg_by_celltype_seurat_top2000",
    split_dir: str = "/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/external_dataset_sciplex/split",
) -> Tuple[DataLoader, int, int, int, Dict[str, int]]:
    """
    构建 Sciplex -> CRISP 训练/评估 DataLoader

    Returns:
        loader: DataLoader
        num_genes: genes 维度（HVG 数）
        fm_dim: paired_cell_embeddings 维度
        drug_emb_dim: 药物 embedding dim
        cell_line_to_idx: cell_type -> idx
    """
    # cell_type -> idx
    cell_line_to_idx = {ct: i for i, ct in enumerate(sorted(cell_types))}

    # smiles embedding store
    smiles_emb_store = SmilesEmbeddingStore(
        drug_emb_pt_path,
        device="cpu",
        canonicalize=True,
    )

    # dataset
    dataset = SciplexPerturbDataset(
        cell_types=cell_types,
        split=split,
        h5ad_path=h5ad_path,
        shards_dir=shards_dir,
        hvg_root=hvg_root,
        split_dir=split_dir,
        set_size=set_size,
        UC=UC,
        is_train=is_train,
        sequential=sequential,
        hvg_not_yet_normed=hvg_not_yet_normed,
        seed=seed,
    )

    # collator
    collator = CRISPSciplexCollator(
        smiles_emb_store=smiles_emb_store,
        cell_line_to_idx=cell_line_to_idx,
        device=device,
        neg_strategy="same_drug_diff_celltype",
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        collate_fn=collator,
        drop_last=is_train,
        pin_memory=True,
    )

    # dim inference
    sample_batch = next(iter(loader))
    num_genes = int(sample_batch[0].shape[1])
    fm_dim = int(sample_batch[1].shape[1])
    drug_emb_dim = int(smiles_emb_store.dim)

    return loader, num_genes, fm_dim, drug_emb_dim, cell_line_to_idx


# =========================================================
# (5) 可选：简单自检
# =========================================================

def _quick_sanity_check():
    # Sciplex 的三个细胞类型
    cell_types = ["A549", "K562", "MCF7"]

    loader, num_genes, fm_dim, drug_dim, ct2i = build_crisp_dataloader(
        cell_types=cell_types,
        split="train",
        drug_emb_pt_path="/path/to/smiles_embedding.pt",  # 替换成你的药物 embedding 路径
        batch_size=2,
        set_size=32,
        num_workers=0,
        is_train=True,
        sequential=False,
        hvg_not_yet_normed=False,
    )

    batch = next(iter(loader))
    genes, paired_cell_embeddings = batch[0], batch[1]
    dosages = batch[3]  # ✅ Sciplex 真实浓度
    drugs_pre, neg_drugs_pre = batch[14], batch[15]
    metadata = batch[16]

    print("=" * 80)
    print("Sciplex CRISP Collator Sanity Check")
    print("=" * 80)
    print(f"genes: {genes.shape}")  # [B*S, G]
    print(f"paired_cell_embeddings: {paired_cell_embeddings.shape}")  # [B*S, FM]
    print(f"dosages (first 5): {dosages[:5].tolist()}")  # ✅ 应该是真实浓度值
    print(f"drugs_pre: {drugs_pre.shape}")  # [B*S, D]
    print(f"neg_drugs_pre: {neg_drugs_pre.shape}")  # [B*S, D]
    print(f"metadata keys: {list(metadata.keys())}")
    print(f"metadata concentrations: {metadata['concentrations']}")  # ✅ 真实浓度
    print("=" * 80)


if __name__ == "__main__":
    # 默认不跑；需要时手动打开
    # _quick_sanity_check()
    pass