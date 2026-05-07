# nips_crisp_multicell.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from scipy import sparse
from typing import Dict, List, Tuple, Optional, Any

try:
    from rdkit import Chem
except Exception:
    Chem = None

import scanpy as sc


# =========================================================
# (1) 原封不动的 NIPS dataset 类：从 ds_nips.py 复制粘贴
# =========================================================

def remove_non_alphanumeric(s: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9]", "", str(s))


def ensure_condition_and_control(adata):
    """
    与 D_split.py 保持一致的控制组判定语义：
      - condition：优先 obs['condition']，否则由 obs['sm_name'] 清洗得到，并把 DimethylSulfoxide -> DMSO
      - is_control：优先 obs['neg_control']，否则 condition == 'DMSO'
    """
    obs = adata.obs

    if "condition" in obs.columns:
        condition = obs["condition"].astype(str).values
    elif "sm_name" in obs.columns:
        cond = obs["sm_name"].astype(str).map(remove_non_alphanumeric)
        cond = cond.replace({"DimethylSulfoxide": "DMSO"})
        # 仅为了可追溯性（backed 模式下不能写 adata.obs；这里不强写回）
        condition = cond.values.astype(str)
    else:
        raise KeyError("Need obs['condition'] or obs['sm_name'] to build condition.")

    condition = np.array([("DMSO" if c == "DimethylSulfoxide" else c) for c in condition], dtype=object).astype(str)

    if "neg_control" in obs.columns:
        is_control = obs["neg_control"].astype(int).values.astype(bool)
    else:
        is_control = (condition == "DMSO")

    return condition, is_control


class NIPSPerturbDataset(Dataset):
    """
    NIPS 多 cell-type 数据集（对齐 ds_multi_cell.TahoePerturbDataset 的采样/输出语义）

    依赖的预处理产物：
      1) B_ SE embedding shards:
         shards_dir/summary.json
         shards_dir/embeddings_part_*.pt
         每个 pt 必须含 keys: embeddings, ds_level_index, cell_line_id(cell_type), drug_smiles, drug_conc
      2) C_ HVG per cell-type:
         hvg_root/<cell_type>/X_hvg_top2000_seurat.npz
         hvg_root/<cell_type>/cell_ids.npy (obs_names)
         hvg_root/<cell_type>/hvg_genes_top2000_seurat.npy
      3) D_ split:
         split_dir/control.json, train.json, test.json

    输出（与 Tahoe 对齐）：
      {
        'control_embeddings': [set_size, emb_dim]
        'perturb_embeddings': [set_size, emb_dim]
        'perturb_hvg_vectors': [set_size, n_hvg]   (log1p after libnorm if hvg_not_yet_normed)
        'control_hvg_vectors': [set_size, n_hvg]
        'drug_smiles': str (canonical if rdkit available)
        'drug_conc': float  (NIPS 无浓度，返回 -1.0)
        'cell_line': str (这里等于 cell_type)
        # sequential 模式额外字段与 Tahoe 类似
      }
    """

    def __init__(
        self,
        cell_types,
        split="train",  # 'train' or 'test'
        h5ad_path="external_dataset_nips/raw/nips_pp_scFM_resplit.filtered_ESM2.h5ad",
        shards_dir="external_dataset_nips/SE_emb/nips",
        hvg_root="external_dataset_nips/preprocessed/nips_hvg_by_celltype_seurat_top2000",
        split_dir="external_dataset_nips/split",
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
            raise ValueError("split must be 'train' or 'test' for NIPS.")

        # ============ 2) 读取 h5ad 元信息（backed） ============
        # 只用 obs / obs_names；避免载入大 X
        adata = sc.read_h5ad(h5ad_path, backed="r")

        if "cell_type" not in adata.obs.columns:
            raise KeyError("Missing obs['cell_type'] in h5ad.")
        if "SMILES" not in adata.obs.columns:
            raise KeyError("Missing obs['SMILES'] in h5ad.")

        self.obs_names = np.asarray(adata.obs_names).astype(str)
        self.cell_type_arr = np.asarray(adata.obs["cell_type"]).astype(str)
        self.smiles_arr = np.asarray(adata.obs["SMILES"]).astype(str)

        # condition + control（语义与 D_split 对齐）
        self.condition_arr, self.is_control_arr = ensure_condition_and_control(adata)

        # ============ 3) 载入 embedding shards，并建立 global_idx -> row 映射 ============
        self.embeddings, self.emb_ds_index, self.emb_cell_type, self.emb_smiles = self._load_all_shards(shards_dir)
        self.emb_dim = int(self.embeddings.shape[1])

        # global idx -> row in embeddings
        self.ds_to_emb_row = {}
        for r, ds_idx in enumerate(self.emb_ds_index):
            ds_idx = int(ds_idx)
            # 若重复，以第一次为准（理论上不应重复）
            if ds_idx not in self.ds_to_emb_row:
                self.ds_to_emb_row[ds_idx] = r

        # ============ 4) 为每个 cell_type 构建 data/control 的“可采样索引池” ============
        # 重要：control 必须来自同一 cell_type
        self.cell_type_data = {}
        total_data = 0

        for ct in self.cell_types:
            ct_mask = (self.cell_type_arr == ct)

            # 全局 ds index（0..n_obs-1）中属于该 ct 的
            ct_all = np.where(ct_mask)[0].astype(int)

            # control / perturb split（使用 D_split 已经产出的 control.json 更一致）
            ct_control = [i for i in ct_all.tolist() if i in self.control_ds]
            ct_data = [i for i in ct_all.tolist() if i in self.data_ds]

            # 过滤掉那些在 embeddings shards 里找不到的（理论上不应该缺）
            ct_control = [i for i in ct_control if i in self.ds_to_emb_row]
            ct_data = [i for i in ct_data if i in self.ds_to_emb_row]

            # 在该 cell_type 下，建立 “drug(condition) -> ds_indices” 映射（用于按频率采样）
            drug_to_rows = {}
            for ds_idx in ct_data:
                drug_id = self.condition_arr[ds_idx]  # 避免用 SMILES 做 split id
                drug_to_rows.setdefault(drug_id, []).append(ds_idx)

            self.cell_type_data[ct] = {
                "data_ds": ct_data,
                "control_ds": ct_control,
                "drug_to_ds": drug_to_rows,  # key: drug_id(str), value: list[ds_idx]
            }

            total_data += len(ct_data)

        self.total_data_samples = total_data

        # ============ 5) cell_type 采样权重（按 data 规模比例） ============
        weights = []
        for ct in self.cell_types:
            weights.append(len(self.cell_type_data[ct]["data_ds"]))
        weights = np.asarray(weights, dtype=float)
        if weights.sum() == 0:
            raise RuntimeError("No data samples found for given cell_types and split.")
        self.cell_type_weights = weights / weights.sum()

        # ============ 6) 懒加载 HVG：每个 cell_type 读一次，并缓存 ============
        self._hvg_cache = {}  # ct -> dict(matrix, obsname2row)

        # ============ 7) Sequential 模式 ============
        self._seq_mode_active = (self.sequential and (not self.is_train))
        if self._seq_mode_active:
            self._setup_sequential_mode()

    # ---------------- Embedding shards ----------------
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

        embeddings = torch.cat(embs, dim=0).contiguous()  # [N, 2048]
        embeddings = embeddings.numpy()  # numpy for fast slicing
        ds_idx = np.asarray(ds_idx, dtype=np.int64)
        cell_types = np.asarray(cell_types, dtype=object).astype(str)
        smiles = np.asarray(smiles, dtype=object).astype(str)

        return embeddings, ds_idx, cell_types, smiles

    # ---------------- HVG loading & mapping ----------------
    def _load_hvg_for_cell_type(self, ct: str):
        if ct in self._hvg_cache:
            return self._hvg_cache[ct]

        ct_safe = ct.replace("/", "_")
        ct_dir = os.path.join(self.hvg_root, ct_safe)

        # 依据 C_ 脚本的命名
        x_path = os.path.join(ct_dir, "X_hvg_top2000_seurat.npz")
        if not os.path.exists(x_path):
            # 兜底：匹配任意 top2000 seurat_v3
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
        """
        ds_indices: list[int] global idx
        return: np.ndarray [len(ds_indices), n_hvg] float32
        """
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

    # ---------------- Sampling helpers ----------------
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

    # ---------------- HVG post-processing (same as Tahoe) ----------------
    def _process_hvg(self, hvg_np: np.ndarray) -> torch.Tensor:
        """
        hvg_np: [B, n_hvg] raw counts
        return: torch.FloatTensor [B, n_hvg]
        """
        x = torch.from_numpy(hvg_np)  # float32
        if self.hvg_not_yet_normed:
            total = x.sum(dim=1, keepdim=True)
            total[total == 0] = 1.0
            x = 1000.0 * x / total
            x = torch.log1p(x)
        return x

    # ---------------- Sequential mode ----------------
    def _setup_sequential_mode(self):
        """
        构建 unique (cell_type, SMILES) -> list[ds_idx] 的映射（仅 data split）
        """
        self._seq_key_to_ds = {}
        for ct in self.cell_types:
            for ds_idx in self.cell_type_data[ct]["data_ds"]:
                smi = self.smiles_arr[int(ds_idx)]
                key = (ct, smi)
                self._seq_key_to_ds.setdefault(key, []).append(int(ds_idx))
        self.sequential_keys = sorted(self._seq_key_to_ds.keys())

    # ---------------- Dataset API ----------------
    def __len__(self):
        if self._seq_mode_active:
            return len(self.sequential_keys)
        if self.is_train:
            base_len = 2 * max(self.total_data_samples, 1) // max(self.set_size, 1)
            return max(int(base_len), 1000)
        return -1

    def __getitem__(self, index):
        # -------- Sequential mode (val/test) --------
        if self._seq_mode_active:
            ct, raw_smiles = self.sequential_keys[index]
            candidate_ds = self._seq_key_to_ds[(ct, raw_smiles)]

            # 采样 perturb
            if len(candidate_ds) >= self.set_size:
                pert_ds = self.rng.choice(candidate_ds, self.set_size, replace=False).tolist()
            else:
                pert_ds = self.rng.choice(candidate_ds, self.set_size, replace=True).tolist()

            # 采样 control（同 ct）
            ctrl_ds = self._sample_control_ds(ct)

            # embeddings
            pert_rows = [self.ds_to_emb_row[int(i)] for i in pert_ds]
            ctrl_rows = [self.ds_to_emb_row[int(i)] for i in ctrl_ds]
            perturb_embeddings = torch.from_numpy(self.embeddings[pert_rows]).float()
            control_embeddings = torch.from_numpy(self.embeddings[ctrl_rows]).float()

            # hvg
            perturb_hvg = self._process_hvg(self._fetch_hvg_rows(ct, pert_ds))
            control_hvg = self._process_hvg(self._fetch_hvg_rows(ct, ctrl_ds))

            # smiles canonical
            canon = raw_smiles
            if Chem is not None:
                m = Chem.MolFromSmiles(raw_smiles)
                if m is not None:
                    canon = Chem.MolToSmiles(m)

            # drug id（避免 smiles 做 id）
            drug_name = self.condition_arr[int(pert_ds[0])]

            return {
                "control_embeddings": control_embeddings,
                "perturb_embeddings": perturb_embeddings,
                "perturb_hvg_vectors": perturb_hvg,
                "control_hvg_vectors": control_hvg,
                "drug_smiles": canon,
                "drug_conc": -1.0,
                "cell_line": ct,
                "target_drug_conc_str": drug_name,  # 对齐 Tahoe 的“展示字段”
                "final_ds_indices": pert_ds,
                "control_ds_indices": ctrl_ds,
            }

        # -------- Train/random sampling mode --------
        # 1) 采样 cell_type（按 data 规模比例）
        ct = self.rng.choice(self.cell_types, p=self.cell_type_weights)

        # 2) 在该 cell_type 内按频率采样 drug_id(condition)
        drug_id = self._choose_drug_weighted(ct)

        # 3) 采样 control / perturb ds indices（同 cell_type）
        ctrl_ds = self._sample_control_ds(ct)
        pert_ds = self._sample_perturb_ds(ct, drug_id)

        # 4) embeddings
        pert_rows = [self.ds_to_emb_row[int(i)] for i in pert_ds]
        ctrl_rows = [self.ds_to_emb_row[int(i)] for i in ctrl_ds]

        perturb_embeddings = torch.from_numpy(self.embeddings[pert_rows]).float()
        control_embeddings = torch.from_numpy(self.embeddings[ctrl_rows]).float()

        # 5) HVG + 预处理（libnorm + log1p）
        perturb_hvg = self._process_hvg(self._fetch_hvg_rows(ct, pert_ds))
        control_hvg = self._process_hvg(self._fetch_hvg_rows(ct, ctrl_ds))

        # 6) drug smiles（模型可能仍用 smiles 做条件输入；split 用的是 condition）
        raw_smiles = self.smiles_arr[int(pert_ds[0])]
        canon = raw_smiles
        if Chem is not None:
            m = Chem.MolFromSmiles(raw_smiles)
            if m is not None:
                canon = Chem.MolToSmiles(m)

        return {
            "control_embeddings": control_embeddings,
            "perturb_embeddings": perturb_embeddings,
            "perturb_hvg_vectors": perturb_hvg,
            "control_hvg_vectors": control_hvg,
            "drug_smiles": canon,
            "drug_conc": -1.0,
            "cell_line": ct,
        }


# =========================================================
# (2) SmilesEmbeddingStore：与 Tahoe 版本对齐
# =========================================================

class SmilesEmbeddingStore:
    """
    加载预计算的 SMILES embeddings（与 Tahoe 版本一致）
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
# (3) NIPS -> CRISP collator：逻辑尽量与 Tahoe collator 一致
#     仅对齐 NIPS 的字段差异
# =========================================================

class CRISPNIPSCollator:
    """
    将 NIPSPerturbDataset 的输出转换为 CRISP 原版训练所需的格式

    输出 tuple 对齐 Tahoe 版本的 CRISP collator：
        (
            genes,
            paired_cell_embeddings,
            drugs_idx(None),
            dosages,
            degs,
            celltype_idx,
            neg_genes,
            neg_paired_cell_embeddings,
            neg_drugs_idx(None),
            neg_dosages,
            neg_degs,
            neg_celltype_idx,
            covariates(None),
            neg_covariates(None),
            drugs_pre,
            neg_drugs_pre,
            metadata
        )

    说明（与 Tahoe 保持一致）：
      - genes := perturb_hvg_vectors  拼接成 [B*S, G]
      - paired_cell_embeddings := control_embeddings 拼接成 [B*S, FM_dim]
      - dosages：NIPS 无浓度，dataset 已返回 -1.0，这里沿用
      - degs：没有预计算 DEG mask，置零 bool
      - 负样本策略默认 same_drug_diff_celltype：
            在 batch 内找 “同 SMILES、不同 cell_line(cell_type)” 的样本作为负样本来源；
            找不到则随机选不同 cell_line；再找不到则退化为随机（与 Tahoe 实现一致）
      - neg_drugs_pre：负样本使用“与正样本相同的药物 embedding”（同 Tahoe）
    """

    def __init__(
        self,
        smiles_emb_store: SmilesEmbeddingStore,
        cell_line_to_idx: Dict[str, int],  # NIPS 这里是 cell_type -> idx
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
        S = batch[0]["perturb_embeddings"].shape[0]  # set_size

        if self.num_genes is None:
            self.num_genes = batch[0]["perturb_hvg_vectors"].shape[1]

        # ========= 正样本 =========
        genes_list = []
        paired_cell_emb_list = []
        drugs_pre_list = []
        dosages_list = []
        celltype_idx_list = []

        for sample in batch:
            genes_list.append(sample["perturb_hvg_vectors"])          # [S, G]
            paired_cell_emb_list.append(sample["control_embeddings"])  # [S, FM_dim]

            drug_emb = self.smiles_emb_store.get(sample["drug_smiles"])  # [D]
            drugs_pre_list.append(drug_emb)

            dosages_list.append(float(sample["drug_conc"]))  # NIPS 为 -1.0
            celltype_idx_list.append(self.cell_line_to_idx[sample["cell_line"]])

        genes = torch.cat(genes_list, dim=0)  # [B*S, G]
        paired_cell_embeddings = torch.cat(paired_cell_emb_list, dim=0)  # [B*S, FM_dim]
        drugs_pre = torch.stack(drugs_pre_list, dim=0).repeat_interleave(S, dim=0)  # [B*S, D]
        dosages = torch.tensor(np.repeat(dosages_list, S), dtype=torch.float32)  # [B*S]
        celltype_idx = torch.tensor(np.repeat(celltype_idx_list, S), dtype=torch.long)  # [B*S]
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

            # 负样本使用“与正样本相同药物”的 embedding（同 Tahoe）
            neg_drug_emb = self.smiles_emb_store.get(batch[i]["drug_smiles"])
            neg_drugs_pre_list.append(neg_drug_emb)

            neg_dosages_list.append(float(batch[i]["drug_conc"]))  # 同剂量（NIPS 仍为 -1.0）
            neg_celltype_idx_list.append(self.cell_line_to_idx[neg_sample["cell_line"]])

        neg_genes = torch.cat(neg_genes_list, dim=0)
        neg_paired_cell_embeddings = torch.cat(neg_paired_cell_emb_list, dim=0)
        neg_drugs_pre = torch.stack(neg_drugs_pre_list, dim=0).repeat_interleave(S, dim=0)
        neg_dosages = torch.tensor(np.repeat(neg_dosages_list, S), dtype=torch.float32)
        neg_celltype_idx = torch.tensor(np.repeat(neg_celltype_idx_list, S), dtype=torch.long)
        neg_degs = torch.zeros_like(neg_genes, dtype=torch.bool)

        # ========= 元信息（评测用）=========
        metadata = {
            "smiles": [sample["drug_smiles"] for sample in batch],
            "cell_lines": [sample["cell_line"] for sample in batch],
            "drug_names": [sample.get("target_drug_conc_str", "Unknown") for sample in batch],
            "concentrations": [sample["drug_conc"] for sample in batch],
            "set_size": S,
            "control_hvg": torch.cat([sample["control_hvg_vectors"] for sample in batch], dim=0),
        }

        return (
            genes,
            paired_cell_embeddings,
            None,  # drugs_idx (不使用)
            dosages,
            degs,
            celltype_idx,
            neg_genes,
            neg_paired_cell_embeddings,
            None,  # neg_drugs_idx (不使用)
            neg_dosages,
            neg_degs,
            neg_celltype_idx,
            None,  # covariates
            None,  # neg_covariates
            drugs_pre,       # ✅ 预计算药物 embedding
            neg_drugs_pre,   # ✅ 负样本药物 embedding
            metadata,        # ✅ 元信息
        )

    def _sample_negative_indices(self, batch: List[Dict[str, Any]]) -> List[int]:
        """
        same_drug_diff_celltype:
          - 在 batch 内找相同 SMILES 但不同 cell_line 的样本
          - 找不到就找不同 cell_line 的样本
          - 再找不到就随机（与 Tahoe 一致）
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
# (4) build_crisp_dataloader：对齐 Tahoe 形式，但参数适配 NIPS
# =========================================================

def build_crisp_dataloader(
    cell_types: List[str],
    split: str,
    drug_emb_pt_path: str,
    batch_size: int,
    set_size: int,
    UC: False,
    num_workers: int = 4,
    device: str = "cpu",
    hvg_not_yet_normed: bool = False,
    is_train: bool = True,
    sequential: bool = False,
    seed: int = 42,
    # --- NIPS dataset paths ---
    h5ad_path: str = "external_dataset_nips/raw/nips_pp_scFM_resplit.filtered_ESM2.h5ad",
    shards_dir: str = "external_dataset_nips/SE_emb/nips",
    hvg_root: str = "external_dataset_nips/preprocessed/nips_hvg_by_celltype_seurat_top2000",
    split_dir: str = "external_dataset_nips/split",
) -> Tuple[DataLoader, int, int, int, Dict[str, int]]:
    """
    构建 NIPS -> CRISP 训练/评估 DataLoader（接口对齐 Tahoe 版本）

    Returns:
        loader: DataLoader
        num_genes: genes 维度（HVG 数）
        fm_dim: paired_cell_embeddings 维度（scFM embedding dim）
        drug_emb_dim: 药物 embedding dim
        cell_line_to_idx: cell_type -> idx
    """
    # cell_type -> idx（对齐 Tahoe 习惯：sorted）
    cell_line_to_idx = {ct: i for i, ct in enumerate(sorted(cell_types))}

    # smiles embedding store
    smiles_emb_store = SmilesEmbeddingStore(
        drug_emb_pt_path,
        device="cpu",
        canonicalize=True,
    )

    # dataset
    dataset = NIPSPerturbDataset(
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
    collator = CRISPNIPSCollator(
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

    # dim inference (与 Tahoe build 一致：取一个 batch)
    sample_batch = next(iter(loader))
    num_genes = int(sample_batch[0].shape[1])     # genes.shape[1]
    fm_dim = int(sample_batch[1].shape[1])        # paired_cell_embeddings.shape[1]
    drug_emb_dim = int(smiles_emb_store.dim)

    return loader, num_genes, fm_dim, drug_emb_dim, cell_line_to_idx


# =========================================================
# (5) 可选：简单自检（不会自动运行）
# =========================================================

def _quick_sanity_check():
    cell_types = ["T cells CD4+", "NK cells", "Myeloid cells", "B cells", "T cells CD8+", "T regulatory cells"]

    loader, num_genes, fm_dim, drug_dim, ct2i = build_crisp_dataloader(
        cell_types=cell_types,
        split="train",
        drug_emb_pt_path="/path/to/gene_symbol_to_embedding_ESM2.pt",  # 这里替换成你的 smiles emb pt
        batch_size=2,
        set_size=32,
        num_workers=0,
        is_train=True,
        sequential=False,
        hvg_not_yet_normed=False,
    )

    batch = next(iter(loader))
    genes, paired_cell_embeddings = batch[0], batch[1]
    drugs_pre, neg_drugs_pre = batch[14], batch[15]
    metadata = batch[16]

    print("genes:", genes.shape)  # [B*S, G]
    print("paired_cell_embeddings:", paired_cell_embeddings.shape)  # [B*S, FM]
    print("drugs_pre:", drugs_pre.shape)  # [B*S, D]
    print("neg_drugs_pre:", neg_drugs_pre.shape)  # [B*S, D]
    print("metadata keys:", list(metadata.keys()))


if __name__ == "__main__":
    # 默认不跑；需要时手动打开
    # _quick_sanity_check()
    pass
