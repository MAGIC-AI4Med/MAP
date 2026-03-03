import os
import json
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from datasets import load_dataset

import data.utils as utils


EXPONENTIATED_UMIS_LIMIT = 5_000_000
RAW_COUNT_HEURISTIC_THRESHOLD = 35



class Tahoe100mDataset(Dataset):
    def __init__(self, data_hf_revision="affe86a848ac896240aa75fe1a2b568051f3b850", data_json=None):
        self.data_json = data_json
        self.d_lst_full, self.ds_metadata = self._load_hf_ds(data_hf_revision)
        if self.data_json is not None:
            self.d_lst = self.filter_data()
        else:
            self.d_lst = self.d_lst_full

    def _load_hf_ds(self, revision="affe86a848ac896240aa75fe1a2b568051f3b850"):
        t0 = time.time()
        print('Loading cached data, this might take a rather long while...')
        ds = load_dataset("tahoebio/Tahoe-100M", revision=revision)
        print(f"Loaded Tahoe-100M. took {np.round(time.time() - t0)} seconds.")

        metadata = load_dataset("tahoebio/Tahoe-100M", "sample_metadata", revision=revision, split="train")
        ds_metadata = {d['sample']: d for d in metadata}

        return ds["train"], ds_metadata
    
    def filter_data(self):
        assert os.path.exists(self.data_json), f"Path not found: {self.data_json}."

        with open(self.data_json, "r", encoding="utf-8") as f:
            data_index = json.load(f)

        print('filtering data...')
        d_lst_new = []
        for idx in tqdm(data_index):
            d_lst_new.append(self.d_lst_full[idx])
        
        return d_lst_new

    def save_filter_index(self, cell_line_id, save_dir='other_data', total=None):
        index_list = []
        json_path = os.path.join(save_dir, f"output_{cell_line_id}.json")
        os.makedirs(save_dir, exist_ok=True)

        with tqdm(enumerate(self.d_lst), total=len(self.d_lst), desc="Filtering index") as pbar:
            for i, data in pbar:
                if data['cell_line_id'] == cell_line_id:
                    index_list.append(i)
                if total is not None and len(index_list) == total:
                    break
                pbar.set_postfix({'Filtered': len(index_list)})

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(index_list, f, ensure_ascii=False, indent=2)

        print(f"saved index of cell line {cell_line_id}, containing {len(index_list)} samples.")
    
    def __len__(self):
        return len(self.d_lst)
    
    def __getitem__(self, index):
        sample = self.d_lst[index]

        gene_ids = torch.tensor(sample['genes']).reshape(1, -1)
        expressions = torch.tensor(sample['expressions']).reshape(1, -1)

        drug_conc = self.ds_metadata[sample['sample']]['drugname_drugconc']
        drug_smiles = sample['canonical_smiles']

        cell_line_id = sample['cell_line_id']        
        return index, gene_ids, expressions, drug_smiles, drug_conc, cell_line_id
    

class scTahoe100mDataset(Dataset):
    def __init__(self, 
        cell_line,
        json_name = 'train_indices.json',
        data_split_base="preprocessed/",
        data_hf_revision="affe86a848ac896240aa75fe1a2b568051f3b850", 
    ):
        self.data_split_base = os.path.join(data_split_base, cell_line)
        self.data_json = os.path.join(data_split_base, json_name)
        self.d_lst_full, self.ds_metadata = self._load_hf_ds(data_hf_revision)
        self.index_map = self.filter_data_multi_json()

    def _load_hf_ds(self, revision="affe86a848ac896240aa75fe1a2b568051f3b850"):
        t0 = time.time()
        print('Loading cached data, this might take a rather long while...')
        ds = load_dataset("tahoebio/Tahoe-100M", revision=revision)
        print(f"Loaded Tahoe-100M. took {np.round(time.time() - t0)} seconds.")

        metadata = load_dataset("tahoebio/Tahoe-100M", "sample_metadata", revision=revision, split="train")
        ds_metadata = {d['sample']: d for d in metadata}

        return ds["train"], ds_metadata
    
    def filter_data_multi_json(self):
        assert os.path.exists(self.data_split_base), f"Path not found: {self.data_split_base}."

        json_files = [
            os.path.join(self.data_split_base, fn)
            for fn in os.listdir(self.data_split_base)
            if fn.lower().endswith(".json")
        ]
        assert len(json_files) > 0, f"No json files found in: {self.data_split_base}"

        merged_indices = []
        for jf in sorted(json_files):
            with open(jf, "r", encoding="utf-8") as f:
                lst = json.load(f)
            if not isinstance(lst, list):
                print(f"File {jf} does not contain a list, got {type(lst)}.")
                continue
            merged_indices.extend(lst)

        return merged_indices

    def filter_data(self):
        assert os.path.exists(self.data_json), f"Path not found: {self.data_json}."

        with open(self.data_json, "r", encoding="utf-8") as f:
            data_index = json.load(f)

        print('filtering data...')
        d_lst_new = []
        for idx in tqdm(data_index):
            d_lst_new.append(self.d_lst_full[idx])
        
        return d_lst_new

    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, index):
        dataset_level_index = self.index_map[index]
        sample = self.d_lst_full[self.index_map[index]]

        gene_ids = torch.tensor(sample['genes']).reshape(1, -1)
        expressions = torch.tensor(sample['expressions']).reshape(1, -1)

        drug_conc = self.ds_metadata[sample['sample']]['drugname_drugconc']
        drug_smiles = sample['canonical_smiles']

        cell_line_id = sample['cell_line_id']        
        return index, gene_ids, expressions, drug_smiles, drug_conc, cell_line_id, dataset_level_index


class CellDatasetCollator(object):
    def __init__(self, cfg, valid_gene_mask=None, ds_emb_mapping_inference=None, is_train=False, precision=None):
        self.pad_length = cfg.dataset.pad_length
        self.P = cfg.dataset.P
        self.N = cfg.dataset.N
        self.S = cfg.dataset.S
        self.cfg = cfg
        self.training = is_train
        self.precision = precision
        self.valid_gene_mask = valid_gene_mask

        self.ds_emb_mapping = self._load_gene_mapping()

    def _load_gene_mapping(self):
        all_embeddings_path = utils.get_embedding_cfg(self.cfg).all_embeddings
        gene_token_data = torch.load(all_embeddings_path, weights_only=False)

        with open('gene_vocab_hgnc.json', 'r') as f:
            cell_index_to_hgnc = json.load(f)
        
        valid_genes_list = list(gene_token_data.keys())
        global_pos = {g: i for i, g in enumerate(valid_genes_list)}

        cell_index_to_global = {}
        for cell_gene_idx_str, hgnc_symbol in cell_index_to_hgnc.items():
            global_idx = global_pos.get(hgnc_symbol, -1)
            cell_index_to_global[int(cell_gene_idx_str)] = global_idx
        
        index_ls = [-1, -1, -1]
        for i in range(3, len(cell_index_to_global)+3):
            index_ls.append(cell_index_to_global[i])
        
        assert len(index_ls) == len(cell_index_to_global)+3, (f"{len(index_ls)},{len(cell_index_to_global)}")
        return index_ls
    
    def __call__(self, batch):
        num_aug = getattr(self.cfg.model, "num_downsample", 1)
        if num_aug > 1 and self.training:
            batch = [item for item in batch for _ in range(num_aug)]

        batch_size = len(batch)
        batch_sentences = torch.zeros((batch_size, self.pad_length), dtype=torch.int32)
        batch_sentences_counts = torch.zeros((batch_size, self.pad_length))

        drug_smiles_list = []
        drug_conc_list = []
        cell_line_id_list = []
        dataset_level_index_list = []

        i = 0
        max_len = self.cfg.dataset.pad_length
        for idx, gene_ids, expressions, drug_smiles, drug_conc, cell_line_id, dataset_level_index in batch:
            if self.valid_gene_mask is not None:
                valid_mask = self.valid_gene_mask
            else:
                valid_mask = None
            
            downsample_fraction = 1.0 if (num_aug > 1 and i % num_aug == 0 and self.training) else None

            (bs, cell_sentence_counts) = self.sample_cell_sentences(expressions, gene_ids, valid_mask, downsample_fraction)

            batch_sentences[i, :] = bs

            if self.cfg.model.counts and cell_sentence_counts is not None:
                batch_sentences_counts[i, :] = cell_sentence_counts
            i += 1

            drug_smiles_list.append(drug_smiles)
            drug_conc_list.append(drug_conc)
            cell_line_id_list.append(cell_line_id)
            dataset_level_index_list.append(dataset_level_index)

        if self.precision is not None and batch_sentences_counts is not None:
            batch_sentences_counts = batch_sentences_counts.to(dtype=self.precision)

        return (
            batch_sentences[:, :max_len],
            batch_sentences_counts if self.cfg.model.counts else None,
            drug_smiles_list,
            drug_conc_list,
            cell_line_id_list,
            dataset_level_index_list
        )
        
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def is_raw_integer_counts(self, counts: torch.Tensor) -> bool:
        max_val = torch.max(counts).item()
        if max_val > RAW_COUNT_HEURISTIC_THRESHOLD:
            return True

        total_umis = int(torch.expm1(counts).sum().item())
        if total_umis > EXPONENTIATED_UMIS_LIMIT:
            return True

        return False
        
    def sample_cell_sentences(self, counts_raw, gene_ids, valid_gene_mask=None, downsample_frac=None):
        if torch.isnan(counts_raw).any():
            raise ValueError(f"NaN values in counts for dataset. counts_raw:{counts_raw}")

        if torch.any(counts_raw < 0):
            counts_raw = F.relu(counts_raw)
        
        counts_raw = counts_raw[:, 1:]
        gene_ids = gene_ids[:, 1:]

        ds_emb_mapping = torch.tensor(self.ds_emb_mapping, dtype=torch.long)

        counts_filtered_list = []
        gene_ids_filtered_list = []

        for c in range(counts_raw.shape[0]):
            current_gene_ids = gene_ids[c]
            current_counts = counts_raw[c]
            
            valid_genes_mask = ds_emb_mapping[current_gene_ids] != -1
            
            valid_counts = current_counts[valid_genes_mask]
            valid_gene_ids = current_gene_ids[valid_genes_mask]
            
            counts_filtered_list.append(valid_counts)
            gene_ids_filtered_list.append(valid_gene_ids)
        
        max_valid_genes = max(len(counts) for counts in counts_filtered_list)
        counts_raw_filtered = torch.zeros((counts_raw.shape[0], max_valid_genes))
        gene_ids_filtered = torch.zeros((gene_ids.shape[0], max_valid_genes), dtype=gene_ids.dtype)

        valid_gene_counts = []
        for c in range(counts_raw.shape[0]):
            num_valid = len(counts_filtered_list[c])
            valid_gene_counts.append(num_valid)
            
            if num_valid > 0:
                counts_raw_filtered[c, :num_valid] = counts_filtered_list[c]
                gene_ids_filtered[c, :num_valid] = gene_ids_filtered_list[c]
        
        counts_raw = counts_raw_filtered
        gene_ids = gene_ids_filtered

        if self.is_raw_integer_counts(counts_raw):
            total_umis = int(counts_raw.sum(axis=1).item())
            count_expr_dist = counts_raw / counts_raw.sum(axis=1, keepdim=True)
            counts_raw = torch.log1p(counts_raw)
        else:
            exp_log_counts = torch.expm1(counts_raw)
            total_umis = int(exp_log_counts.sum(axis=1).item())
            count_expr_dist = exp_log_counts / exp_log_counts.sum(axis=1, keepdim=True)

        original_counts_raw = counts_raw.clone()

        counts = counts_raw

        if counts.sum() == 0:
            expression_weights = F.softmax(counts, dim=1)
        else:
            expression_weights = counts / torch.sum(counts, dim=1, keepdim=True)

        cell_sentences = torch.zeros((counts.shape[0], self.cfg.dataset.pad_length))
        cell_sentence_counts = torch.zeros((counts.shape[0], self.cfg.dataset.pad_length))
        mask = torch.zeros((counts.shape[0], self.cfg.dataset.pad_length), dtype=torch.bool)


        for c, cell in enumerate(counts):
            current_cell_gene_ids = gene_ids[c]
            num_valid_genes = valid_gene_counts[c]

            if num_valid_genes == 0: continue

            valid_cell = cell[:num_valid_genes]
            valid_gene_ids = current_cell_gene_ids[:num_valid_genes]

            indices = torch.randperm(num_valid_genes)
            shuffled_cell = valid_cell[indices]
            shuffled_genes_ranked_exp = torch.argsort(shuffled_cell, descending=True)
            genes_ranked_exp = indices[shuffled_genes_ranked_exp]

            cell_sentences[c, 0] = self.cfg.dataset.cls_token_idx

            if num_valid_genes >= self.cfg.dataset.pad_length - 1:
                selected_indices = genes_ranked_exp[:self.cfg.dataset.pad_length - 1]
                selected_gene_ids = valid_gene_ids[selected_indices]
                
                cell_sentences[c, 1:] = ds_emb_mapping[selected_gene_ids]
            else:
                selected_gene_ids = valid_gene_ids[genes_ranked_exp]
                cell_sentences[c, 1:num_valid_genes + 1] = ds_emb_mapping[selected_gene_ids]

                remaining_slots = self.cfg.dataset.pad_length - 1 - num_valid_genes
                if remaining_slots > 0:
                    all_valid_global_genes = set()
                    for local_idx in range(len(ds_emb_mapping)):
                        global_idx = ds_emb_mapping[local_idx].item()
                        if global_idx != -1:
                            if valid_gene_mask is None or valid_gene_mask[local_idx]:
                                all_valid_global_genes.add(global_idx)
                    
                    expressed_global_genes = set(cell_sentences[c, 1:num_valid_genes + 1].cpu().numpy())
                    unexpressed_global_genes = list(all_valid_global_genes - expressed_global_genes)

                    if len(unexpressed_global_genes) >= remaining_slots:
                        selected_unexpressed = np.random.choice(unexpressed_global_genes, remaining_slots, replace=False)
                    else:
                        selected_unexpressed = np.random.choice(unexpressed_global_genes, remaining_slots, replace=True) if unexpressed_global_genes else [0] * remaining_slots
                    
                    cell_sentences[c, num_valid_genes + 1:] = torch.tensor(selected_unexpressed)

            global_to_local_pos = {}
            for local_pos in range(num_valid_genes):
                global_id = ds_emb_mapping[valid_gene_ids[local_pos]].item()
                global_to_local_pos[global_id] = local_pos

            for pos in range(1, min(num_valid_genes + 1, self.cfg.dataset.pad_length)):
                global_gene_id = cell_sentences[c, pos].item()
                if global_gene_id in global_to_local_pos:
                    local_pos = global_to_local_pos[global_gene_id]
                    cell_sentence_counts[c, pos] = 100 * expression_weights[c, local_pos]
        
        return(
            cell_sentences,
            cell_sentence_counts,
        )





class scTahoe100mDatasetSEOnly(Dataset):
    def __init__(self, 
        cell_line,
        json_name = 'train_indices.json',
        data_split_base="preprocessed/",
        data_hf_revision="affe86a848ac896240aa75fe1a2b568051f3b850", 
    ):
        self.data_split_base = os.path.join(data_split_base, cell_line)
        self.data_json = os.path.join(data_split_base, json_name)
        self.d_lst_full, _ = self._load_hf_ds(data_hf_revision)
        self.index_map = self.filter_data_multi_json()

    def _load_hf_ds(self, revision="affe86a848ac896240aa75fe1a2b568051f3b850"):
        t0 = time.time()
        print('Loading cached data, this might take a rather long while...')
        ds = load_dataset("tahoebio/Tahoe-100M", revision=revision)
        print(f"Loaded Tahoe-100M. took {np.round(time.time() - t0)} seconds.")
        return ds["train"], None

    
    def filter_data_multi_json(self):
        assert os.path.exists(self.data_split_base), f"Path not found: {self.data_split_base}."
        
        json_files = [
            os.path.join(self.data_split_base, fn)
            for fn in os.listdir(self.data_split_base)
            if fn.lower().endswith(".json") and not fn.endswith("_UC.json")
        ]
        assert len(json_files) > 0, f"No json files found in: {self.data_split_base}"
        
        print(f"Found {len(json_files)} non-UC JSON files:")
        for jf in sorted(json_files):
            print(f"  - {os.path.basename(jf)}")
        
        merged_indices = []
        for jf in sorted(json_files):
            with open(jf, "r", encoding="utf-8") as f:
                lst = json.load(f)
            if isinstance(lst, list):
                print(f"  Loaded {len(lst)} indices from {os.path.basename(jf)}")
                merged_indices.extend(lst)
        
        print(f"Total loaded: {len(merged_indices)} indices")
        return merged_indices

    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, index):
        sample = self.d_lst_full[self.index_map[index]]
        
        gene_ids = torch.tensor(sample['genes'], dtype=torch.int32)
        expressions = torch.tensor(sample['expressions'], dtype=torch.float32)
        
        return gene_ids, expressions


class CellDatasetCollatorSEOnly(object):
    def __init__(self, cfg, valid_gene_mask=None):
        self.pad_length = cfg.dataset.pad_length
        self.cfg = cfg
        self.valid_gene_mask = valid_gene_mask
        self.ds_emb_mapping = self._load_gene_mapping()

    def _load_gene_mapping(self):
        all_embeddings_path = utils.get_embedding_cfg(self.cfg).all_embeddings
        gene_token_data = torch.load(all_embeddings_path, weights_only=False)

        with open('gene_vocab_hgnc.json', 'r') as f:
            cell_index_to_hgnc = json.load(f)
        
        valid_genes_list = list(gene_token_data.keys())
        global_pos = {g: i for i, g in enumerate(valid_genes_list)}

        cell_index_to_global = {}
        for cell_gene_idx_str, hgnc_symbol in cell_index_to_hgnc.items():
            global_idx = global_pos.get(hgnc_symbol, -1)
            cell_index_to_global[int(cell_gene_idx_str)] = global_idx
        
        index_ls = [-1, -1, -1]
        for i in range(3, len(cell_index_to_global)+3):
            index_ls.append(cell_index_to_global[i])
        
        return torch.tensor(index_ls, dtype=torch.long)
    
    def __call__(self, batch):
        batch_size = len(batch)
        batch_sentences = torch.zeros((batch_size, self.pad_length), dtype=torch.int32)
        batch_sentences_counts = torch.zeros((batch_size, self.pad_length), dtype=torch.float32)

        for i, (gene_ids, expressions) in enumerate(batch):
            bs, counts = self.sample_cell_sentences_fast(expressions, gene_ids)
            batch_sentences[i] = bs
            batch_sentences_counts[i] = counts

        return batch_sentences, batch_sentences_counts
        
    def is_raw_integer_counts(self, counts: torch.Tensor) -> bool:
        max_val = torch.max(counts).item()
        if max_val > RAW_COUNT_HEURISTIC_THRESHOLD:
            return True
        total_umis = int(torch.expm1(counts).sum().item())
        return total_umis > EXPONENTIATED_UMIS_LIMIT
        
    def sample_cell_sentences_fast(self, counts_raw, gene_ids):
        counts_raw = counts_raw[1:]
        gene_ids = gene_ids[1:]

        valid_mask = self.ds_emb_mapping[gene_ids] != -1
        counts_raw = counts_raw[valid_mask]
        gene_ids = gene_ids[valid_mask]
        
        num_valid_genes = len(counts_raw)
        if num_valid_genes == 0:
            cell_sentences = torch.zeros(self.pad_length, dtype=torch.int32)
            cell_sentences[0] = self.cfg.dataset.cls_token_idx
            return cell_sentences, torch.zeros(self.pad_length, dtype=torch.float32)

        if self.is_raw_integer_counts(counts_raw):
            counts_raw = torch.log1p(counts_raw)

        if counts_raw.sum() > 0:
            expression_weights = counts_raw / counts_raw.sum()
        else:
            expression_weights = F.softmax(counts_raw, dim=0)

        cell_sentences = torch.zeros(self.pad_length, dtype=torch.int32)
        cell_sentence_counts = torch.zeros(self.pad_length, dtype=torch.float32)
        
        cell_sentences[0] = self.cfg.dataset.cls_token_idx

        indices = torch.randperm(num_valid_genes)
        shuffled_counts = counts_raw[indices]
        ranked_indices = torch.argsort(shuffled_counts, descending=True)
        genes_ranked = indices[ranked_indices]

        if num_valid_genes >= self.pad_length - 1:
            selected = genes_ranked[:self.pad_length - 1]
            cell_sentences[1:] = self.ds_emb_mapping[gene_ids[selected]]
            cell_sentence_counts[1:] = 100 * expression_weights[selected]
        else:
            cell_sentences[1:num_valid_genes + 1] = self.ds_emb_mapping[gene_ids[genes_ranked]]
            cell_sentence_counts[1:num_valid_genes + 1] = 100 * expression_weights[genes_ranked]
            
            remaining = self.pad_length - 1 - num_valid_genes
            if remaining > 0:
                all_valid_global = set()
                for local_idx in range(len(self.ds_emb_mapping)):
                    global_idx = self.ds_emb_mapping[local_idx].item()
                    if global_idx != -1:
                        all_valid_global.add(global_idx)
                
                expressed_global = set(cell_sentences[1:num_valid_genes + 1].tolist())
                unexpressed = list(all_valid_global - expressed_global)
                
                if len(unexpressed) >= remaining:
                    selected = np.random.choice(unexpressed, remaining, replace=False)
                else:
                    selected = np.random.choice(unexpressed, remaining, replace=True) if unexpressed else [0] * remaining
                
                cell_sentences[num_valid_genes + 1:] = torch.tensor(selected, dtype=torch.int32)

        return cell_sentences, cell_sentence_counts