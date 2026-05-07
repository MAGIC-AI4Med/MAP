# tahoe_crisp_multicell_dataset.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import ast
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from rdkit import Chem
from typing import Dict, List, Tuple, Optional, Any


def extract_concentration(conc_str: str) -> float:
    """从drug_conc字符串中提取浓度数值"""
    conc_list = ast.literal_eval(conc_str)
    assert len(conc_list) == 1 and len(conc_list[0]) == 3
    concentration = conc_list[0][1]
    return float(concentration)


class TahoePerturbDataset(Dataset):
    """多细胞系扰动数据集 - 保持原有实现不变"""
    def __init__(
        self,
        cell_lines: List[str],
        split: str = 'train',
        base_dir: str = "/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/ours_3/preprocessed",
        hvg_setting: str = "seurat_v3",
        hvg_not_yet_normed: bool = True,
        set_size: int = 128,
        is_train: bool = True,
        sequential: bool = False,
        return_control_hvg: bool = False,
        UC = False,
    ):
        self.cell_lines = cell_lines
        self.split = split
        self.base_dir = base_dir
        self.hvg_not_yet_normed = hvg_not_yet_normed
        self.is_train = is_train
        self.set_size = set_size
        self.sequential = sequential
        self.return_control_hvg = return_control_hvg
        self.UC = UC

        # 存储每个细胞系的数据
        self.cell_line_data = {}
        self.emb_mmaps = {}
        self.hvg_mmaps = {}
        
        total_data_samples = 0
        total_control_samples = 0

        # 1. 遍历每个细胞系，加载元数据
        for cell_line in cell_lines:
            print(f"\n{'='*60}")
            print(f"📚 Loading cell line: {cell_line}")
            print(f"{'='*60}")
            
            cell_line_dir = os.path.join(base_dir, cell_line)
            meta_path = os.path.join(cell_line_dir, f"{cell_line}_meta.pkl")

            postfix = '_UC.json' if self.UC else '.json'
            
            # # 根据 split 拼接 JSON 文件路径
            # if split == 'train':
            #     data_json_path = os.path.join(cell_line_dir, "train_indices.json")
            # elif split == 'internal_test':
            #     data_json_path = os.path.join(cell_line_dir, "internal_test_indices.json")
            # elif split == 'external_test':
            #     data_json_path = os.path.join(cell_line_dir, "external_test_indices.json")
            # else:
            #     raise ValueError(f"Unknown split: {split}")
            
            # control_json_path = os.path.join(cell_line_dir, "control_indices.json")


            # 根据 split 拼接 JSON 文件路径
            if split == 'train':
                data_json_path = os.path.join(cell_line_dir, "train_indices"+postfix)
            elif split == 'internal_test':
                data_json_path = os.path.join(cell_line_dir, "internal_test_indices"+postfix)
            elif split == 'external_test':
                data_json_path = os.path.join(cell_line_dir, "external_test_indices"+postfix)
            else:
                raise ValueError(f"Unknown split: {split}. Must be 'train', 'internal_test', or 'external_test'.")
            
            # Control 始终加载
            control_json_path = os.path.join(cell_line_dir, "control_indices"+postfix)
            
            # 加载元数据
            print(f"📥 Loading metadata from: {meta_path}")
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            
            # 过滤数据和控制组索引
            data_indices = self._filter_indices(
                data_json_path, 
                meta["ds_level_index"],
                meta["drug_conc"]
            )
            control_indices = self._filter_indices(
                control_json_path,
                meta["ds_level_index"],
                meta["drug_conc"]
            )
            
            # 构建该细胞系的采样映射
            drug_conc_to_matching_rows = {}
            for row_idx in data_indices:
                conc_str = meta["drug_conc"][row_idx]
                if conc_str not in drug_conc_to_matching_rows:
                    drug_conc_to_matching_rows[conc_str] = []
                drug_conc_to_matching_rows[conc_str].append(row_idx)
            
            # 存储该细胞系的所有信息
            self.cell_line_data[cell_line] = {
                "meta": meta,
                "data_indices": data_indices,
                "control_indices": control_indices,
                "drug_conc_to_matching_rows": drug_conc_to_matching_rows,
                "cell_line_dir": cell_line_dir,
            }
            
            total_data_samples += len(data_indices)
            total_control_samples += len(control_indices)
            
            print(f"✅ {cell_line}: {len(data_indices)} data samples, {len(control_indices)} control samples")
        
        self.total_data_samples = total_data_samples
        self.total_control_samples = total_control_samples
        
        # 2. 计算细胞系采样权重
        self.cell_line_weights = self._calculate_cell_line_weights()
        
        # 3. Sequential模式设置
        self._seq_mode_active = (self.sequential and not self.is_train)
        if self._seq_mode_active:
            self._setup_sequential_mode()
        
        print(f"\n{'='*60}")
        print(f"✅ Multi-cell-line dataset initialized")
        print(f"   Split: {split}")
        print(f"   Total cell lines: {len(self.cell_lines)}")
        print(f"   Total data samples: {self.total_data_samples}")
        print(f"   Total control samples: {self.total_control_samples}")
        print(f"{'='*60}\n")

    def _calculate_cell_line_weights(self):
        """计算每个细胞系的采样权重"""
        weights = []
        for cell_line in self.cell_lines:
            weights.append(len(self.cell_line_data[cell_line]["data_indices"]))
        
        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()
        
        print(f"\n📊 Cell line sampling weights:")
        for cell_line, weight in zip(self.cell_lines, weights):
            print(f"   {cell_line}: {weight:.4f}")
        
        return weights

    def _get_mmap(self, cell_line: str):
        """懒加载获取指定细胞系的 memmap 对象"""
        if cell_line not in self.emb_mmaps:
            cell_line_dir = self.cell_line_data[cell_line]["cell_line_dir"]
            meta = self.cell_line_data[cell_line]["meta"]
            
            emb_path = os.path.join(cell_line_dir, f"{cell_line}_embeddings.npy")
            hvg_path = os.path.join(cell_line_dir, f"{cell_line}_hvg.npy")
            
            self.emb_mmaps[cell_line] = np.memmap(
                emb_path, 
                dtype=meta["dtype"], 
                mode='r', 
                shape=meta["shape_emb"]
            )
            self.hvg_mmaps[cell_line] = np.memmap(
                hvg_path, 
                dtype='float32', 
                mode='r', 
                shape=meta["shape_hvg"]
            )
            
        return self.emb_mmaps[cell_line], self.hvg_mmaps[cell_line]

    def _filter_indices(self, json_path: str, all_ds_indices: List, all_drug_conc: List):
        """读取 split json 并返回对应的 memmap 行号列表"""
        with open(json_path, 'r') as f:
            target_ds_indices = set(json.load(f))
        
        print(f"📋 Filtering indices for {os.path.basename(json_path)}...")
        valid_row_indices = []
        
        for row_idx, ds_idx in enumerate(all_ds_indices):
            if ds_idx in target_ds_indices:
                conc_str = all_drug_conc[row_idx]
                if 'Sacubitril/Valsartan' in conc_str or 'Verteporfin' in conc_str:
                    continue
                valid_row_indices.append(row_idx)
                
        print(f"   Matched {len(valid_row_indices)} samples.")
        return valid_row_indices

    def _setup_sequential_mode(self):
        """为顺序模式构建索引"""
        self._seq_cell_smiles_to_rows = {}
        
        for cell_line in self.cell_lines:
            data_indices = self.cell_line_data[cell_line]["data_indices"]
            meta = self.cell_line_data[cell_line]["meta"]
            
            for row_idx in data_indices:
                smi = meta["drug_smiles"][row_idx]
                key = (cell_line, smi)
                if key not in self._seq_cell_smiles_to_rows:
                    self._seq_cell_smiles_to_rows[key] = []
                self._seq_cell_smiles_to_rows[key].append(row_idx)
        
        self.sequential_cell_smiles = sorted(self._seq_cell_smiles_to_rows.keys())
        
        print(f"\n🔄 Sequential mode: {len(self.sequential_cell_smiles)} unique (cell_line, SMILES) pairs")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Sequential模式
        if self._seq_mode_active:
            cell_line, target_smiles = self.sequential_cell_smiles[index]
            matching_rows = self._seq_cell_smiles_to_rows[(cell_line, target_smiles)]
            
            canon_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(target_smiles))
            
            emb_mmap, hvg_mmap = self._get_mmap(cell_line)
            meta = self.cell_line_data[cell_line]["meta"]
            
            if len(matching_rows) >= self.set_size:
                sampled_rows = np.random.choice(matching_rows, self.set_size, replace=False)
            else:
                sampled_rows = np.random.choice(matching_rows, self.set_size, replace=True)

            first_row = sampled_rows[0]
            drug_conc_str = meta["drug_conc"][first_row]
            
            try:
                conc_list = ast.literal_eval(drug_conc_str)
                drug_name = conc_list[0][0] if len(conc_list) > 0 and len(conc_list[0]) > 0 else "Unknown"
                concentration = conc_list[0][1] if len(conc_list[0]) > 1 else -1.0
            except:
                drug_name = "Unknown"
                concentration = -1.0
            
            perturb_embeddings = torch.from_numpy(emb_mmap[sampled_rows])
            perturb_hvg_raw = hvg_mmap[sampled_rows]
            perturb_hvg_vectors = self._process_hvg(perturb_hvg_raw)
            
            control_rows = self._sample_control_rows(cell_line)
            control_embeddings = torch.from_numpy(emb_mmap[control_rows])
            control_hvg_vectors = self._process_hvg(hvg_mmap[control_rows])
            
            return {
                'control_embeddings': control_embeddings,
                'perturb_embeddings': perturb_embeddings,
                'perturb_hvg_vectors': perturb_hvg_vectors,
                'control_hvg_vectors': control_hvg_vectors,
                'drug_smiles': canon_smiles,
                'drug_conc': concentration,
                'cell_line': cell_line,
                'target_drug_conc_str': drug_name,
                "final_ds_indices": [meta["ds_level_index"][i] for i in sampled_rows],
                "control_ds_indices": [meta["ds_level_index"][i] for i in control_rows],
            }

        # 训练/随机模式
        cell_line = np.random.choice(self.cell_lines, p=self.cell_line_weights)
        
        cell_data = self.cell_line_data[cell_line]
        meta = cell_data["meta"]
        emb_mmap, hvg_mmap = self._get_mmap(cell_line)
        
        cache_key = f"_perturbation_weights_{cell_line}"
        if not hasattr(self, cache_key):
            weights, unique_perturbations = self._calculate_sampling_weights(cell_line)
            setattr(self, cache_key, weights)
            setattr(self, f"_unique_perturbations_{cell_line}", unique_perturbations)
        
        perturbation_weights = getattr(self, cache_key)
        unique_perturbations = getattr(self, f"_unique_perturbations_{cell_line}")
        
        target_drug_conc_str = np.random.choice(unique_perturbations, p=perturbation_weights)
        conc = extract_concentration(target_drug_conc_str)
        
        control_rows = self._sample_control_rows(cell_line)
        control_embeddings = torch.from_numpy(emb_mmap[control_rows])
        control_hvg_vectors = self._process_hvg(hvg_mmap[control_rows])
        
        perturb_rows = self._sample_perturb_rows(cell_line, target_drug_conc_str)
        perturb_embeddings = torch.from_numpy(emb_mmap[perturb_rows])
        perturb_hvg_vectors = self._process_hvg(hvg_mmap[perturb_rows])
        
        raw_smi = meta["drug_smiles"][perturb_rows[0]]
        canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(raw_smi))
        
        return {
            'control_embeddings': control_embeddings,
            'perturb_embeddings': perturb_embeddings,
            'perturb_hvg_vectors': perturb_hvg_vectors,
            'control_hvg_vectors': control_hvg_vectors,
            'drug_smiles': canon_smi,
            'drug_conc': conc,
            'cell_line': cell_line,
        }

    def _sample_control_rows(self, cell_line: str):
        """返回采样的 Control 组行号列表"""
        control_indices = self.cell_line_data[cell_line]["control_indices"]
        n_control = len(control_indices)
        
        if n_control >= self.set_size:
            indices = np.random.choice(n_control, self.set_size, replace=False)
        else:
            indices = np.random.choice(n_control, self.set_size, replace=True)
        
        return [control_indices[i] for i in indices]

    def _sample_perturb_rows(self, cell_line: str, target_conc_str: str):
        """返回采样的 Perturb 组行号列表"""
        drug_conc_to_matching_rows = self.cell_line_data[cell_line]["drug_conc_to_matching_rows"]
        matching_rows = drug_conc_to_matching_rows[target_conc_str]
        
        if len(matching_rows) >= self.set_size:
            sampled_rows = np.random.choice(matching_rows, self.set_size, replace=False)
        else:
            sampled_rows = np.random.choice(matching_rows, self.set_size, replace=True)
            
        return sampled_rows

    def _process_hvg(self, hvg_batch_np):
        """处理 HVG 向量"""
        hvg_tensor = torch.from_numpy(hvg_batch_np)
        
        if self.hvg_not_yet_normed:
            total_counts = hvg_tensor.sum(dim=1, keepdim=True)
            total_counts[total_counts == 0] = 1.0 
            normalized_counts = 1000 * hvg_tensor / total_counts
            return torch.log1p(normalized_counts)
        else:
            return hvg_tensor

    def _calculate_sampling_weights(self, cell_line: str):
        """计算指定细胞系的采样权重"""
        cell_data = self.cell_line_data[cell_line]
        data_indices = cell_data["data_indices"]
        meta = cell_data["meta"]
        
        conc_counts = Counter()
        for row_idx in data_indices:
            conc_counts[meta["drug_conc"][row_idx]] += 1
            
        perturbation_counts = {}
        for conc_str, count in conc_counts.items():
            if '0.0,' not in conc_str:
                perturbation_counts[conc_str] = count
                
        if not perturbation_counts:
            raise ValueError(f"No perturbation conditions found for cell line {cell_line}")
            
        unique_perturbations = list(perturbation_counts.keys())
        counts = np.array(list(perturbation_counts.values()))
        weights = counts / counts.sum()
        
        return weights, unique_perturbations

    def __len__(self):
        if self._seq_mode_active:
            return len(self.sequential_cell_smiles)
        if self.is_train:
            base_length = 2 * self.total_data_samples // self.set_size
            return max(base_length, 1000)
        else:
            return -1


class CRISPTahoeCollator:
    """
    将 TahoePerturbDataset 的输出转换为 CRISP 原版训练所需的格式
    
    CRISP 原版需要:
        - genes (正样本)
        - paired_cell_embeddings (正样本的配对控制)
        - drugs_idx / drugs_pre (正样本药物)
        - dosages (正样本剂量)
        - degs (正样本DEG mask)
        - celltype_idx (正样本细胞类型)
        - neg_genes (负样本)
        - neg_paired_cell_embeddings (负样本的配对控制)
        - neg_drugs_idx / neg_pre (负样本药物)
        - neg_dosages (负样本剂量)
        - neg_degs (负样本DEG mask)
        - neg_celltype_idx (负样本细胞类型)
        - covariates (可选)
        - neg_covariates (可选)
    """
    
    def __init__(
        self,
        smiles_emb_store,  # SmilesEmbeddingStore 实例
        cell_line_to_idx: Dict[str, int],  # {'CVCL_0023': 0, ...}
        device: str = "cpu",
        neg_strategy: str = "same_drug_diff_celltype",
    ):
        self.smiles_emb_store = smiles_emb_store
        self.cell_line_to_idx = cell_line_to_idx
        self.device = device
        self.neg_strategy = neg_strategy
        
        self.num_genes = None  # 会在第一个batch时自动设置
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Tuple:
        """
        Args:
            batch: List of samples from TahoePerturbDataset
        
        Returns:
            Tuple of (genes, paired_cell_embeddings, drugs_idx, dosages, degs, celltype_idx,
                     neg_genes, neg_paired_cell_embeddings, neg_drugs_idx, neg_dosages, 
                     neg_degs, neg_celltype_idx, covariates, neg_covariates)
        """
        B = len(batch)
        S = batch[0]['perturb_embeddings'].shape[0]  # set_size
        
        if self.num_genes is None:
            self.num_genes = batch[0]['perturb_hvg_vectors'].shape[1]
        
        # === 正样本数据 ===
        genes_list = []
        paired_cell_emb_list = []
        drugs_pre_list = []
        dosages_list = []
        celltype_idx_list = []
        
        for sample in batch:
            # genes: [S, G]
            genes_list.append(sample['perturb_hvg_vectors'])
            
            # paired control embeddings: [S, FM_dim]
            paired_cell_emb_list.append(sample['control_embeddings'])
            
            # drug embedding (precomputed): [D]
            drug_emb = self.smiles_emb_store.get(sample['drug_smiles'])
            drugs_pre_list.append(drug_emb)
            
            # dosage: float
            dosages_list.append(float(sample['drug_conc']))
            
            # cell type index
            celltype_idx_list.append(self.cell_line_to_idx[sample['cell_line']])
        
        # 拼接为 [B*S, ...] 格式
        genes = torch.cat(genes_list, dim=0)  # [B*S, G]
        paired_cell_embeddings = torch.cat(paired_cell_emb_list, dim=0)  # [B*S, FM_dim]
        
        # drugs_pre: [B*S, D]
        drugs_pre = torch.stack(drugs_pre_list, dim=0).repeat_interleave(S, dim=0)
        
        # dosages: [B*S]
        dosages = torch.tensor(np.repeat(dosages_list, S), dtype=torch.float32)
        
        # celltype_idx: [B*S]
        celltype_idx = torch.tensor(np.repeat(celltype_idx_list, S), dtype=torch.long)
        
        # DEGs mask (全0，因为我们没有预计算DEG)
        degs = torch.zeros_like(genes, dtype=torch.bool)
        
        # === 负样本数据 (same_drug_diff_celltype) ===
        # 策略: 在同一batch内，找到相同药物但不同细胞系的样本作为负样本
        neg_indices = self._sample_negative_indices(batch)
        
        neg_genes_list = []
        neg_paired_cell_emb_list = []
        neg_drugs_pre_list = []
        neg_dosages_list = []
        neg_celltype_idx_list = []
        
        for i, neg_idx in enumerate(neg_indices):
            neg_sample = batch[neg_idx]
            
            neg_genes_list.append(neg_sample['perturb_hvg_vectors'])
            neg_paired_cell_emb_list.append(neg_sample['control_embeddings'])
            
            # 负样本使用相同的药物 (同一个SMILES)
            neg_drug_emb = self.smiles_emb_store.get(batch[i]['drug_smiles'])
            neg_drugs_pre_list.append(neg_drug_emb)
            
            neg_dosages_list.append(float(batch[i]['drug_conc']))  # 同剂量
            neg_celltype_idx_list.append(self.cell_line_to_idx[neg_sample['cell_line']])
        
        neg_genes = torch.cat(neg_genes_list, dim=0)
        neg_paired_cell_embeddings = torch.cat(neg_paired_cell_emb_list, dim=0)
        neg_drugs_pre = torch.stack(neg_drugs_pre_list, dim=0).repeat_interleave(S, dim=0)
        neg_dosages = torch.tensor(np.repeat(neg_dosages_list, S), dtype=torch.float32)
        neg_celltype_idx = torch.tensor(np.repeat(neg_celltype_idx_list, S), dtype=torch.long)
        neg_degs = torch.zeros_like(neg_genes, dtype=torch.bool)


        # ✅ 收集元信息（用于评测）
        metadata = {
            'smiles': [sample['drug_smiles'] for sample in batch],
            'cell_lines': [sample['cell_line'] for sample in batch],
            'drug_names': [sample.get('target_drug_conc_str', 'Unknown') for sample in batch],
            'concentrations': [sample['drug_conc'] for sample in batch],
            'set_size': batch[0]['perturb_embeddings'].shape[0],
            # 保存control的HVG用于计算delta
            'control_hvg': torch.cat([sample['control_hvg_vectors'] for sample in batch], dim=0),
        }
        
        # 返回格式与 CRISP 原版一致
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
            drugs_pre,  # ✅ 新增: 正样本的预计算药物embedding
            neg_drugs_pre,  # ✅ 新增: 负样本的预计算药物embedding
            metadata,  # ✅ 新增
        )
    
    def _sample_negative_indices(self, batch: List[Dict]) -> List[int]:
        """
        为每个样本采样一个负样本索引
        
        策略: same_drug_diff_celltype
            - 在batch内找相同SMILES但不同cell_line的样本
            - 如果找不到，随机选一个不同cell_line的样本
        """
        B = len(batch)
        neg_indices = []
        
        for i in range(B):
            pos_smiles = batch[i]['drug_smiles']
            pos_cell = batch[i]['cell_line']
            
            # 找到相同药物但不同细胞系的候选
            candidates = []
            for j in range(B):
                if j != i and batch[j]['drug_smiles'] == pos_smiles and batch[j]['cell_line'] != pos_cell:
                    candidates.append(j)
            
            if candidates:
                neg_idx = np.random.choice(candidates)
            else:
                # 如果找不到相同药物的，就找不同细胞系的
                diff_cell_candidates = [j for j in range(B) if batch[j]['cell_line'] != pos_cell]
                if diff_cell_candidates:
                    neg_idx = np.random.choice(diff_cell_candidates)
                else:
                    # 实在找不到就随机选一个
                    neg_idx = np.random.choice([j for j in range(B) if j != i]) if B > 1 else i
            
            neg_indices.append(neg_idx)
        
        return neg_indices


class SmilesEmbeddingStore:
    """
    加载预计算的 SMILES embeddings
    (与您单细胞系版本相同的实现)
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
            raise ValueError(f"Unrecognized smiles embedding file format")

        norm_dict: Dict[str, torch.Tensor] = {}
        for k, v in emb_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(np.asarray(v), dtype=torch.float32)
            norm_dict[str(k)] = v.detach().cpu().float()

        self.raw_emb_dict = norm_dict
        self.emb_dict: Dict[str, torch.Tensor] = {}

        self._rdkit_ok = False
        if self.canonicalize:
            try:
                from rdkit import Chem
                self.Chem = Chem
                self._rdkit_ok = True
            except Exception as e:
                raise ImportError("RDKit is required for canonicalize=True")

        if self.canonicalize:
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


def build_crisp_dataloader(
    cell_lines: List[str],
    split: str,
    drug_emb_pt_path: str,
    base_dir: str,
    batch_size: int,
    set_size: int,
    num_workers: int = 4,
    device: str = "cpu",
    hvg_not_yet_normed: bool = True,
    is_train: bool = True,
    sequential: bool = False,
    seed: int = 0,
    UC: bool=False,
) -> Tuple[DataLoader, int, int, int, Dict[str, int]]:
    """
    构建 CRISP 训练/评估的 DataLoader
    
    Returns:
        loader: DataLoader
        num_genes: 基因数
        fm_dim: FM embedding 维度
        drug_emb_dim: 药物 embedding 维度
        cell_line_to_idx: 细胞系到索引的映射
    """
    # 构建细胞系索引
    cell_line_to_idx = {cl: i for i, cl in enumerate(sorted(cell_lines))}
    
    # 加载药物embedding
    smiles_emb_store = SmilesEmbeddingStore(
        drug_emb_pt_path, 
        device="cpu", 
        canonicalize=True
    )
    
    # 创建数据集
    dataset = TahoePerturbDataset(
        cell_lines=cell_lines,
        split=split,
        base_dir=base_dir,
        set_size=set_size,
        is_train=is_train,
        sequential=sequential,
        hvg_not_yet_normed=hvg_not_yet_normed,
        UC=UC,
    )
    
    # 创建collator
    collator = CRISPTahoeCollator(
        smiles_emb_store=smiles_emb_store,
        cell_line_to_idx=cell_line_to_idx,
        device=device,
        neg_strategy="same_drug_diff_celltype",
    )
    
    # 创建DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        collate_fn=collator,
        drop_last=is_train,
        pin_memory=True,
    )
    
    # 获取维度信息 (从第一个batch)
    sample_batch = next(iter(loader))
    num_genes = sample_batch[0].shape[1]  # genes.shape[1]
    fm_dim = sample_batch[1].shape[1]  # paired_cell_embeddings.shape[1]
    drug_emb_dim = smiles_emb_store.dim
    
    return loader, num_genes, fm_dim, drug_emb_dim, cell_line_to_idx