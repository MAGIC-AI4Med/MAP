import os
import logging
import torch
from datasets import load_dataset
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import time
import json
import requests
import copy

import data.utils as utils

import warnings
warnings.filterwarnings('ignore')

log = logging.getLogger(__file__)

EXPONENTIATED_UMIS_LIMIT = 5_000_000
RAW_COUNT_HEURISTIC_THRESHOLD = 35

def create_dataloader(
    cfg,
    workers=1,
    data_dir=None,
    datasets=None,
    shape_dict=None,
    adata=None,
    adata_name=None,
    shuffle=False,
    sentence_collator=None,
    protein_embeds=None,
    precision=None,
):
    pass

class Tahoe100mDataset(Dataset):
    def __init__(self, data_hf_revision="affe86a848ac896240aa75fe1a2b568051f3b850", data_json=None):
        self.data_json = data_json
        self.d_lst_full = self._load_hf_ds(data_hf_revision)
        if self.data_json is not None:
            self.d_lst = self.filter_data()
        else:
            self.d_lst = self.d_lst_full
        
    def _load_hf_ds(self, revision="affe86a848ac896240aa75fe1a2b568051f3b850"):
        t0 = time.time()
        print('HuggingFace Connection Status:', requests.get("https://huggingface.co").status_code)
        print('Loading cached data, this might take a rather long while...')
        ds = load_dataset("tahoebio/Tahoe-100M", revision=revision)
        print(f"Loaded Tahoe-100M. took {np.round(time.time() - t0)} seconds.")
        
        return ds["train"]
    
    def filter_data(self):
        assert os.path.exists(self.data_json), f"Path not found: {self.data_json}."

        with open(self.data_json, "r", encoding="utf-8") as f:
            data_index = json.load(f)

        print('filtering data...')
        d_lst_new = []
        for idx in tqdm(data_index):
            d_lst_new.append(self.d_lst_full[idx])
        
        return d_lst_new

    def save_filter_index(self, cell_line_id, save_dir='/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/base/other_data', total=None):
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
        dataset_num = 0
        return index, gene_ids, expressions, dataset_num
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        for k, v in self.__dict__.items():
            if k in ['d_lst_full']:  
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result


class VCIDatasetSentenceCollator(object):
    def __init__(self, cfg, valid_gene_mask=None, ds_emb_mapping_inference=None, is_train=True, precision=None):
        self.pad_length = cfg.dataset.pad_length
        self.P = cfg.dataset.P
        self.N = cfg.dataset.N
        self.S = cfg.dataset.S
        self.cfg = cfg
        self.training = is_train
        self.precision = precision

        if 'LOCAL_RANK' in os.environ:
            self.local_rank = int(os.environ['LOCAL_RANK'])
        else:
            self.local_rank = 0

        self.use_dataset_info = getattr(cfg.model, "dataset_correction", False)
        self.batch_tabular_loss = getattr(cfg.model, "batch_tabular_loss", False)

        if valid_gene_mask is not None:
            self.valid_gene_mask = valid_gene_mask
            self.dataset_to_protein_embeddings = ds_emb_mapping_inference

        else:
            gene_mask_file = utils.get_embedding_cfg(self.cfg).valid_genes_masks
            if gene_mask_file is not None:
                self.valid_gene_mask = torch.load(gene_mask_file, weights_only=False)
            else:
                self.valid_gene_mask = None

        self.ds_emb_mapping = self._load_gene_mapping()

    def _load_gene_mapping(self):
        all_embeddings_path = utils.get_embedding_cfg(self.cfg).all_embeddings
        gene_token_data = torch.load(all_embeddings_path, weights_only=False)

        with open('/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/base/other_data/gene_vocab_hgnc.json', 'r') as f:
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
        masks = torch.zeros((batch_size, self.pad_length), dtype=torch.bool)

        idxs = torch.zeros(batch_size, dtype=torch.int32)
        if self.cfg.loss.name == "tabular":
            task_num = self.P + self.N + self.S
        else:
            task_num = self.P + self.N
        Xs = torch.zeros((batch_size, (task_num)), dtype=torch.int32)
        Ys = torch.zeros((batch_size, (task_num)))

        largest_cnt = max([x[1].shape[1] for x in batch])
        batch_weights = torch.zeros((batch_size, largest_cnt))

        total_counts_all = None
        if self.cfg.model.rda:
            total_counts_all = torch.zeros(batch_size)

        if self.cfg.loss.name == "tabular":
            if "global_size" not in self.__dict__:
                self.global_size = utils.get_embedding_cfg(self.cfg).num
            shared_genes = torch.randint(
                    low=0, high=self.global_size, size=(self.S,), device=masks.device, dtype=torch.long
                )
        else:
            shared_genes = None
        

        dataset_nums = torch.zeros(batch_size, dtype=torch.int32)

        i = 0
        max_len = 0
        for idx, gene_ids, expressions, dataset_num in batch:
            if self.valid_gene_mask is not None:
                valid_mask = self.valid_gene_mask
            else:
                valid_mask = None
            
            downsample_fraction = 1.0 if (num_aug > 1 and i % num_aug == 0 and self.training) else None
            
            (bs, xx, yy, batch_weight, mask, cell_total_counts, cell_sentence_counts) = self.sample_cell_sentences(
                expressions, gene_ids, shared_genes, valid_mask, downsample_fraction
            )

            batch_sentences[i, :] = bs
            masks[i, :] = mask
            batch_weight = batch_weight.squeeze()
            batch_weights[i, : len(batch_weight)] = batch_weight

            max_len = max(max_len, self.cfg.dataset.pad_length)
            idxs[i] = idx

            Xs[i] = xx
            Ys[i] = yy.squeeze()
            dataset_nums[i] = dataset_num

            if self.cfg.model.rda and cell_total_counts is not None:
                total_counts_all[i] = cell_total_counts[0]
            if self.cfg.model.counts and cell_sentence_counts is not None:
                batch_sentences_counts[i, :] = cell_sentence_counts
            i += 1

        if self.precision is not None:
            Ys = Ys.to(dtype=self.precision)
            batch_weights = batch_weights.to(dtype=self.precision)
            if total_counts_all is not None:
                total_counts_all = total_counts_all.to(dtype=self.precision)
            if batch_sentences_counts is not None:
                batch_sentences_counts = batch_sentences_counts.to(dtype=self.precision)
        
        return (
            batch_sentences[:, :max_len],
            Xs,
            Ys,
            idxs,
            batch_weights,
            masks,
            total_counts_all if self.cfg.model.rda else None,
            batch_sentences_counts if self.cfg.model.counts else None,
            dataset_nums if self.use_dataset_info else None,
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

    import time
    def sample_cell_sentences(self, counts_raw, gene_ids, shared_genes=None, valid_gene_mask=None, downsample_frac=None):
        if torch.isnan(counts_raw).any():
            log.error(f"NaN values in counts for dataset")
        
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
        original_gene_ids = gene_ids.clone()

        num_aug = getattr(self.cfg.model, "num_downsample", 1)
        if num_aug > 1:
            if downsample_frac is None:
                downsample_frac = torch.empty(1).uniform_(0.3, 1.0).item()

            down_umis = int(total_umis * downsample_frac)
            if down_umis > 0 and downsample_frac < 1.0:
                genes_sampled = torch.multinomial(count_expr_dist.squeeze(), down_umis, replacement=True)
                counts_aug_flat = torch.zeros_like(counts_raw.view(-1))
                counts_aug_flat.scatter_add_(0, genes_sampled, torch.ones(down_umis, dtype=counts_aug_flat.dtype))
                counts_aug = counts_aug_flat.view_as(counts_raw)
                counts_aug = torch.log1p(counts_aug)
            else:
                counts_aug = counts_raw
            
            counts_raw = counts_aug

        if valid_gene_mask is not None:
            counts_filtered_list2 = []
            gene_ids_filtered_list2 = []
            
            for c in range(counts_raw.shape[0]):
                current_gene_ids = gene_ids[c]
                current_counts = counts_raw[c]
                
                nonzero_mask = current_gene_ids != 0
                
                if nonzero_mask.any():
                    nonzero_gene_ids = current_gene_ids[nonzero_mask]
                    nonzero_counts = current_counts[nonzero_mask]
                    
                    valid_mask = valid_gene_mask[nonzero_gene_ids]
                    
                    final_gene_ids = nonzero_gene_ids[valid_mask]
                    final_counts = nonzero_counts[valid_mask]
                else:
                    final_gene_ids = torch.tensor([], dtype=gene_ids.dtype)
                    final_counts = torch.tensor([], dtype=counts_raw.dtype)
                
                counts_filtered_list2.append(final_counts)
                gene_ids_filtered_list2.append(final_gene_ids)
            
            max_valid_genes2 = max(len(counts) for counts in counts_filtered_list2) if counts_filtered_list2 else 1
            
            counts_raw = torch.zeros((counts_raw.shape[0], max_valid_genes2))
            gene_ids = torch.zeros((gene_ids.shape[0], max_valid_genes2), dtype=gene_ids.dtype)
            original_counts_raw = torch.zeros((counts_raw.shape[0], max_valid_genes2))
            original_gene_ids = torch.zeros((gene_ids.shape[0], max_valid_genes2), dtype=gene_ids.dtype)
            
            valid_gene_counts = []
            
            for c in range(len(counts_filtered_list2)):
                num_valid = len(counts_filtered_list2[c])
                valid_gene_counts.append(num_valid)
                
                if num_valid > 0:
                    counts_raw[c, :num_valid] = counts_filtered_list2[c]
                    gene_ids[c, :num_valid] = gene_ids_filtered_list2[c]
                    original_counts_raw[c, :num_valid] = counts_filtered_list2[c]
                    original_gene_ids[c, :num_valid] = gene_ids_filtered_list2[c]

        counts = counts_raw
        original_counts = original_counts_raw
        
        if counts.sum() == 0:
            expression_weights = F.softmax(counts, dim=1)
        else:
            expression_weights = counts / torch.sum(counts, dim=1, keepdim=True)

        cell_sentences = torch.zeros((counts.shape[0], self.cfg.dataset.pad_length))
        cell_sentence_counts = torch.zeros((counts.shape[0], self.cfg.dataset.pad_length))
        mask = torch.zeros((counts.shape[0], self.cfg.dataset.pad_length), dtype=torch.bool)

        if self.cfg.loss.name == "tabular":
            task_num = self.cfg.dataset.P + self.cfg.dataset.N + self.cfg.dataset.S
        else:
            task_num = self.cfg.dataset.P + self.cfg.dataset.N

        task_counts = torch.zeros((counts.shape[0], task_num))
        task_sentence = torch.zeros((counts.shape[0], task_num))

        if self.cfg.model.rda:
            cell_total_counts = torch.zeros((counts.shape[0],))
        else:
            cell_total_counts = None


        for c, cell in enumerate(counts):
            t0 = time.time()
            current_cell_gene_ids = gene_ids[c]
            num_valid_genes = valid_gene_counts[c]
            
            if num_valid_genes == 0:
                continue
                
            valid_cell = cell[:num_valid_genes]
            valid_gene_ids = current_cell_gene_ids[:num_valid_genes]
            
            num_pos_genes = torch.sum(valid_cell > 0)
            
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

            t3 = time.time()

            if num_pos_genes > self.cfg.dataset.P:
                expressed_indices = genes_ranked_exp[:num_pos_genes]
                selected_exp_indices = expressed_indices[torch.randperm(num_pos_genes)[:self.cfg.dataset.P]]
            else:
                expressed_indices = genes_ranked_exp[:max(1, num_pos_genes)]
                selected_exp_indices = expressed_indices[torch.randint(0, len(expressed_indices), (self.cfg.dataset.P,))]
            
            selected_exp_gene_ids = valid_gene_ids[selected_exp_indices]
            task_sentence[c, :self.cfg.dataset.P] = ds_emb_mapping[selected_exp_gene_ids]
            task_counts[c, :self.cfg.dataset.P] = original_counts[c, selected_exp_indices]

            all_valid_global_genes = set()
            for local_idx in range(len(ds_emb_mapping)):
                global_idx = ds_emb_mapping[local_idx].item()
                if global_idx != -1:
                    if valid_gene_mask is None or valid_gene_mask[local_idx]:
                        all_valid_global_genes.add(global_idx)

            current_expressed_global = set(ds_emb_mapping[valid_gene_ids].cpu().numpy())
            unexpressed_global_genes = list(all_valid_global_genes - current_expressed_global)
            
            if len(unexpressed_global_genes) >= self.cfg.dataset.N:
                selected_unexp_global = np.random.choice(unexpressed_global_genes, self.cfg.dataset.N, replace=False)
            else:
                selected_unexp_global = np.random.choice(unexpressed_global_genes, self.cfg.dataset.N, replace=True) if unexpressed_global_genes else [0] * self.cfg.dataset.N
            
            task_sentence[c, self.cfg.dataset.P:self.cfg.dataset.P + self.cfg.dataset.N] = torch.tensor(selected_unexp_global)
            task_counts[c, self.cfg.dataset.P:self.cfg.dataset.P + self.cfg.dataset.N] = 0.0

            if shared_genes is not None and self.cfg.loss.name == "tabular":
                task_sentence[c, self.cfg.dataset.P + self.cfg.dataset.N:] = shared_genes
                
                if 'global_to_local_pos' not in locals():
                    global_to_local_pos = {}
                    for local_pos in range(num_valid_genes):
                        global_id = ds_emb_mapping[valid_gene_ids[local_pos]].item()
                        global_to_local_pos[global_id] = local_pos
                
                shared_counts = torch.zeros(len(shared_genes))
                for i, global_gene_id in enumerate(shared_genes):
                    global_id = global_gene_id.item()
                    if global_id in global_to_local_pos:
                        local_pos = global_to_local_pos[global_id]
                        shared_counts[i] = original_counts[c, local_pos]
                
                task_counts[c, self.cfg.dataset.P + self.cfg.dataset.N:] = shared_counts

            if self.cfg.model.rda:
                cell_total_counts[c] = torch.sum(task_counts[c])

            if self.cfg.loss.name == "cross_entropy":
                task_counts[c] = (task_counts[c] > 0).float()

            task_gene_set = set(task_sentence[c].cpu().numpy())
            potential_mask = torch.tensor([
                gene_id.item() in task_gene_set for gene_id in cell_sentences[c]
            ], dtype=torch.bool)

            target_mask_count = int(self.cfg.task.mask * self.cfg.dataset.pad_length)
            current_mask_count = potential_mask.sum().item()

            if current_mask_count > target_mask_count:
                mask_indices = torch.where(potential_mask[1:])[0] + 1
                keep_indices = torch.randperm(len(mask_indices))[:target_mask_count]
                selected_indices = mask_indices[keep_indices]
                
                final_mask = torch.zeros_like(potential_mask)
                final_mask[selected_indices] = True
                mask[c] = final_mask
            elif current_mask_count < target_mask_count:
                non_masked = ~potential_mask
                non_masked_indices = torch.where(non_masked[1:])[0] + 1
                
                additional_needed = min(target_mask_count - current_mask_count, len(non_masked_indices))
                if additional_needed > 0:
                    additional_indices = non_masked_indices[torch.randperm(len(non_masked_indices))[:additional_needed]]
                    potential_mask[additional_indices] = True
                
                mask[c] = potential_mask
            else:
                mask[c] = potential_mask

            mask[c, 0] = False

        return (
            cell_sentences,
            task_sentence,
            task_counts,
            counts,
            mask,
            cell_total_counts if self.cfg.model.rda else None,
            cell_sentence_counts if self.cfg.model.counts else None,
        )