import os
import time
import requests
import json
import pickle
from collections import defaultdict
from tqdm import tqdm
import random

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from datasets import load_dataset
import scanpy as sc
from anndata import AnnData
import gc

CELL_LINES = ['CVCL_0023', 'CVCL_0069', 'CVCL_0480', 'CVCL_0131', 'CVCL_0218', 'CVCL_0504', 'CVCL_1056', 'CVCL_1098']

def load_hf_ds(revision="affe86a848ac896240aa75fe1a2b568051f3b850"):
    t0 = time.time()
    print('HuggingFace Connection Status:', requests.get("https://huggingface.co").status_code)
    print('Loading cached data, this might take a rather long while...')
    ds = load_dataset("tahoebio/Tahoe-100M", revision=revision)
    print(f"Loaded Tahoe-100M. took {np.round(time.time() - t0)} seconds.")
    
    return ds["train"]

def load_split_indices_for_all_cell_lines(cell_lines):
    split_files = [
        "control_indices.json",
        "external_test_indices.json", 
        "internal_test_indices.json",
        "train_indices.json"
    ]
    
    cell_line_indices = {}
    all_indices = []
    
    for cell_line in cell_lines:
        base_path = f"path/to/preprocessed/{cell_line}/"
        cell_line_all_indices = []
        
        print(f"\nLoading indices for {cell_line}...")
        for split_file in split_files:
            file_path = os.path.join(base_path, split_file)
            try:
                with open(file_path, 'r') as f:
                    indices = json.load(f)
                    cell_line_all_indices.extend(indices)
                    all_indices.extend(indices)
                    print(f"  - Added {len(indices)} indices from {split_file}")
            except FileNotFoundError:
                print(f"  Warning: {split_file} not found for {cell_line}")
                continue
        
        cell_line_indices[cell_line] = cell_line_all_indices
        print(f"  Total for {cell_line}: {len(cell_line_all_indices)} indices")
    
    print(f"\nGrand total across all cell lines: {len(all_indices)} indices")
    return all_indices, cell_line_indices


def sample_balanced_cells_per_batch(cell_line_indices, cells_per_batch):
    print(f"\nSampling {cells_per_batch} cells per cell line for balanced HVG calculation...")
    
    sampled_indices_dict = {}
    
    for cell_line, indices in cell_line_indices.items():
        actual_sample_size = min(cells_per_batch, len(indices))
        
        if actual_sample_size < len(indices):
            sampled = random.sample(indices, actual_sample_size)
            print(f"  {cell_line}: sampled {actual_sample_size} from {len(indices)} cells")
        else:
            sampled = indices
            print(f"  {cell_line}: only {len(indices)} cells available, using all")
        
        sampled_indices_dict[cell_line] = sampled
    
    total_sampled = sum(len(v) for v in sampled_indices_dict.values())
    print(f"\nTotal sampled cells across all batches: {total_sampled}")
    
    return sampled_indices_dict


def collect_all_genes(ds, all_indices, sample_ratio=0.02):
    print(f"Collecting all unique genes...")
    
    sample_size = max(1000, int(len(all_indices) * sample_ratio))
    sampled_indices = random.sample(all_indices, min(sample_size, len(all_indices)))
    print(f"   Scanning {len(sampled_indices)} cells ({len(sampled_indices)/len(all_indices)*100:.1f}%)")
    
    all_genes = set()
    for idx in tqdm(sampled_indices, desc="Scanning genes"):
        sample_dict = ds[idx]
        genes = sample_dict['genes']
        all_genes.update(genes)
    
    all_genes = sorted(list(all_genes))
    print(f"Found {len(all_genes)} unique genes")
    
    return all_genes


def build_adata_with_batches(ds, sampled_indices_dict, all_genes):
    print("Building multi-batch AnnData object...")
    
    n_genes = len(all_genes)
    gene_to_idx = {gene_id: idx for idx, gene_id in enumerate(all_genes)}
    
    all_cell_indices = []
    all_batch_labels = []
    
    for cell_line in sorted(sampled_indices_dict.keys()):
        indices = sampled_indices_dict[cell_line]
        all_cell_indices.extend(indices)
        all_batch_labels.extend([cell_line] * len(indices))
    
    n_cells = len(all_cell_indices)
    print(f"Matrix shape: {n_cells} cells × {n_genes} genes")
    print(f"Batch distribution:")
    batch_counts = pd.Series(all_batch_labels).value_counts().sort_index()
    for batch, count in batch_counts.items():
        print(f"   {batch}: {count} cells")
    
    print("Constructing sparse matrix...")
    row_indices = []
    col_indices = []
    data_values = []
    
    for cell_idx, dataset_idx in enumerate(tqdm(all_cell_indices, desc="Processing cells")):
        sample_dict = ds[dataset_idx]
        genes = sample_dict['genes']
        expressions = sample_dict['expressions']
        
        for gene_id, exp_val in zip(genes, expressions):
            if gene_id in gene_to_idx:
                row_indices.append(cell_idx)
                col_indices.append(gene_to_idx[gene_id])
                data_values.append(exp_val)
    
    X = coo_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(n_cells, n_genes),
        dtype=np.float32
    )
    X = X.tocsr()
    
    print(f"Sparse matrix created: {X.shape}, {X.nnz} non-zero elements ({X.nnz/(n_cells*n_genes)*100:.2f}% density)")
    
    var_df = pd.DataFrame(index=[f"gene_{gene_id}" for gene_id in all_genes])
    var_df['gene_id'] = all_genes
    
    obs_df = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    obs_df['dataset_idx'] = all_cell_indices
    obs_df['batch'] = all_batch_labels
    
    adata = AnnData(X=X, obs=obs_df, var=var_df)
    
    print(f"Multi-batch AnnData object created!")
    print(f"   - adata.X: {adata.X.shape} (sparse: {type(adata.X).__name__})")
    print(f"   - adata.obs: {adata.obs.shape} (with 'batch' column)")
    print(f"   - adata.var: {adata.var.shape}")
    
    return adata


def calculate_hvg_with_batches(
    adata,
    n_top_genes=2000,
    flavor='seurat_v3',
    batch_key='batch',
    span=0.3
):
    print(f"Calculating batch-aware highly variable genes...")
    print(f"   Flavor: {flavor}")
    print(f"   Batch key: {batch_key}")
    print(f"   Target HVG count: {n_top_genes}")
    
    if flavor in ['seurat_v3', 'seurat_v3_paper']:
        print("   Using raw counts (as expected for seurat_v3)")
    else:
        print("   Warning: flavor != seurat_v3, will normalize data first")
        adata.layers['counts'] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    
    print("   Running scanpy.pp.highly_variable_genes with batch_key...")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=flavor,
        batch_key=batch_key,
        span=span,
        subset=False,
        inplace=True
    )
    
    n_hvg = adata.var['highly_variable'].sum()
    print(f"\nSelected {n_hvg} highly variable genes")
    
    if 'highly_variable_nbatches' in adata.var.columns:
        nbatches_dist = adata.var.loc[adata.var['highly_variable'], 'highly_variable_nbatches'].value_counts().sort_index()
        print(f"\nHVG distribution across batches:")
        for n_batches, count in nbatches_dist.items():
            print(f"   Present in {n_batches} batch(es): {count} genes")
        
        if 'highly_variable_intersection' in adata.var.columns:
            n_intersection = adata.var['highly_variable_intersection'].sum()
            print(f"   HVG in ALL batches: {n_intersection} genes ({n_intersection/n_hvg*100:.1f}%)")
    
    if flavor in ['seurat_v3', 'seurat_v3_paper']:
        score_col = 'variances_norm'
    else:
        score_col = 'dispersions_norm'
    
    if score_col in adata.var.columns:
        hvg_scores = adata.var.loc[adata.var['highly_variable'], score_col]
        print(f"\nHVG score ({score_col}) range: {hvg_scores.min():.3f} - {hvg_scores.max():.3f}")
    
    hvg_genes = adata.var[adata.var['highly_variable']].copy()
    if 'highly_variable_rank' in hvg_genes.columns:
        hvg_genes = hvg_genes.sort_values('highly_variable_rank')
    
    print(f"\nTop 10 HVG genes:")
    for i, (gene_name, row) in enumerate(list(hvg_genes.head(10).iterrows())):
        gene_id = row['gene_id']
        score = row.get(score_col, 0)
        n_batches = row.get('highly_variable_nbatches', 'N/A')
        print(f"  {i+1:2d}. {gene_name} (ID: {gene_id}, score: {score:.3f}, in {n_batches} batches)")
    
    return adata


def extract_hvg_vectors_directly(ds, indices, hvg_gene_ids):
    print(f"    Extracting HVG vectors for {len(indices)} cells...")
    
    hvg_gene_set = set(hvg_gene_ids)
    gene_to_hvg_idx = {gene_id: idx for idx, gene_id in enumerate(hvg_gene_ids)}
    
    cell_hvg_vectors = {}
    n_hvg = len(hvg_gene_ids)
    
    for dataset_idx in tqdm(indices, desc="    Processing cells"):
        sample_dict = ds[dataset_idx]
        genes = sample_dict['genes']
        expressions = sample_dict['expressions']
        
        hvg_vector = np.zeros(n_hvg, dtype=np.float32)
        
        for gene_id, exp_val in zip(genes, expressions):
            if gene_id in hvg_gene_set:
                hvg_idx = gene_to_hvg_idx[gene_id]
                hvg_vector[hvg_idx] = exp_val
        
        cell_hvg_vectors[dataset_idx] = hvg_vector
    
    sample_vectors = list(cell_hvg_vectors.values())[:min(100, len(cell_hvg_vectors))]
    mean_nonzero = np.mean([np.count_nonzero(v) for v in sample_vectors])
    print(f"    Average non-zero HVG per cell (sampled): {mean_nonzero:.1f}/{n_hvg}")
    print(f"    Extracted HVG vectors for {len(cell_hvg_vectors)} cells")
    
    return cell_hvg_vectors


def main():
    N_TOP_GENES = 2000
    FLAVOR = 'seurat_v3'
    CELLS_PER_BATCH = 10000
    GENE_COLLECTION_SAMPLE_RATIO = 0.02
    
    print(f"Configuration:")
    print(f"   Cell lines: {len(CELL_LINES)} ({', '.join(CELL_LINES)})")
    print(f"   Cells per batch for HVG: {CELLS_PER_BATCH}")
    print(f"   HVG flavor: {FLAVOR}")
    print(f"   Target HVG count: {N_TOP_GENES}")
    
    print("\n" + "="*80)
    print("STEP 1: Loading dataset and indices")
    print("="*80)
    ds = load_hf_ds()
    all_indices, cell_line_indices = load_split_indices_for_all_cell_lines(CELL_LINES)
    print(f"\nTotal cells across all cell lines: {len(all_indices)}")
    
    print("\n" + "="*80)
    print("STEP 2: Balanced sampling from each cell line")
    print("="*80)
    sampled_indices_dict = sample_balanced_cells_per_batch(
        cell_line_indices, 
        CELLS_PER_BATCH
    )
    
    total_sampled = sum(len(v) for v in sampled_indices_dict.values())
    print(f"\nSampled {total_sampled} cells for HVG calculation")
    
    print("\n" + "="*80)
    print("STEP 3: Collecting all genes")
    print("="*80)
    all_sampled_indices = []
    for indices in sampled_indices_dict.values():
        all_sampled_indices.extend(indices)
    
    all_genes = collect_all_genes(ds, all_sampled_indices, GENE_COLLECTION_SAMPLE_RATIO)
    
    print("\n" + "="*80)
    print("STEP 4: Building multi-batch AnnData")
    print("="*80)
    adata_multi_batch = build_adata_with_batches(
        ds,
        sampled_indices_dict,
        all_genes
    )
    
    print("\n" + "="*80)
    print("STEP 5: Computing batch-aware HVG")
    print("="*80)
    adata_multi_batch = calculate_hvg_with_batches(
        adata_multi_batch,
        n_top_genes=N_TOP_GENES,
        flavor=FLAVOR,
        batch_key='batch',
        span=0.3
    )
    
    print("\n" + "="*80)
    print("STEP 6: Extracting HVG gene list and metadata")
    print("="*80)
    
    hvg_mask = adata_multi_batch.var['highly_variable'].values
    hvg_gene_ids = adata_multi_batch.var.loc[hvg_mask, 'gene_id'].tolist()
    print(f"Selected {len(hvg_gene_ids)} HVG genes")
    
    hvg_var = adata_multi_batch.var[adata_multi_batch.var['highly_variable']].copy()
    hvg_info = {
        'hvg_gene_ids': hvg_gene_ids,
        'n_hvg': len(hvg_gene_ids),
        'flavor': FLAVOR,
        'n_top_genes': N_TOP_GENES,
        'cell_lines': CELL_LINES,
        'cells_per_batch': CELLS_PER_BATCH,
        'total_cells_for_hvg': total_sampled,
        'gene_collection_sample_ratio': GENE_COLLECTION_SAMPLE_RATIO,
        'batch_aware': True,
    }
    
    if 'highly_variable_nbatches' in hvg_var.columns:
        hvg_info['hvg_nbatches'] = hvg_var['highly_variable_nbatches'].tolist()
    if 'highly_variable_intersection' in hvg_var.columns:
        hvg_info['n_hvg_in_all_batches'] = hvg_var['highly_variable_intersection'].sum()
    
    if 'variances_norm' in hvg_var.columns:
        hvg_info['hvg_scores'] = hvg_var['variances_norm'].tolist()
    elif 'dispersions_norm' in hvg_var.columns:
        hvg_info['hvg_scores'] = hvg_var['dispersions_norm'].tolist()
    
    if 'highly_variable_rank' in hvg_var.columns:
        hvg_info['hvg_ranks'] = hvg_var['highly_variable_rank'].tolist()
    
    print("\nReleasing multi-batch AnnData to free memory...")
    del adata_multi_batch
    gc.collect()
    print("Memory freed")
    
    print("\n" + "="*80)
    print("STEP 7: Extracting HVG vectors for each cell line")
    print("="*80)
    
    for i, cell_line in enumerate(CELL_LINES, 1):
        print(f"\n  [{i}/{len(CELL_LINES)}] Processing {cell_line}...")
        indices_for_this_cell_line = cell_line_indices[cell_line]
        print(f"      Total cells: {len(indices_for_this_cell_line)}")
        
        cell_line_hvg_vectors = extract_hvg_vectors_directly(
            ds, 
            indices_for_this_cell_line, 
            hvg_gene_ids
        )
        
        base_path = f"path/to/preprocessed/{cell_line}/"
        os.makedirs(base_path, exist_ok=True)
        
        vectors_pkl_path = os.path.join(base_path, f"{cell_line}_cell_hvg_vectors_{FLAVOR}_batch_aware.pkl")
        print(f"      Saving to {vectors_pkl_path}...")
        with open(vectors_pkl_path, 'wb') as f:
            pickle.dump(cell_line_hvg_vectors, f)
        
        file_size_mb = os.path.getsize(vectors_pkl_path) / (1024 * 1024)
        print(f"      Saved {len(cell_line_hvg_vectors)} cells ({file_size_mb:.2f} MB)")
        
        del cell_line_hvg_vectors
        gc.collect()
    
    print("\n" + "="*80)
    print("STEP 8: Saving shared HVG information")
    print("="*80)
    
    shared_path = f"path/to/preprocessed/shared_hvg/"
    os.makedirs(shared_path, exist_ok=True)
    
    hvg_pkl_path = os.path.join(shared_path, f"top{N_TOP_GENES}_hvg_info_{FLAVOR}_batch_aware.pkl")
    print(f"Saving HVG info to {hvg_pkl_path}...")
    with open(hvg_pkl_path, 'wb') as f:
        pickle.dump(hvg_info, f)
    print(f"Saved HVG info pickle")
    
    hvg_json = {k: v for k, v in hvg_info.items() if k not in ['hvg_scores', 'hvg_ranks', 'hvg_nbatches']}
    hvg_json['hvg_gene_ids_preview'] = hvg_gene_ids[:100]
    hvg_json['total_hvg_genes'] = len(hvg_gene_ids)
    
    hvg_json_path = os.path.join(shared_path, f"top{N_TOP_GENES}_hvg_{FLAVOR}_batch_aware.json")
    print(f"Saving HVG info to {hvg_json_path}...")
    with open(hvg_json_path, 'w') as f:
        json.dump(hvg_json, f, indent=2)
    print(f"Saved HVG info JSON (preview)")
    
    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)
    print(f"  HVG genes selected: {len(hvg_gene_ids)}")
    print(f"  Flavor used: {FLAVOR} (batch-aware)")
    print(f"  HVG calculation based on:")
    for cell_line in sorted(sampled_indices_dict.keys()):
        n_sampled = len(sampled_indices_dict[cell_line])
        print(f"      {cell_line}: {n_sampled} cells")
    print(f"      Total: {total_sampled} cells")
    
    print(f"\n  Cell lines processed: {len(CELL_LINES)}")
    total_cells_processed = 0
    for cell_line in CELL_LINES:
        n_cells = len(cell_line_indices[cell_line])
        total_cells_processed += n_cells
        print(f"      {cell_line}: {n_cells:,} cells -> HVG vectors saved")
    
    print(f"  Total cells processed: {total_cells_processed:,}")
    
    if 'n_hvg_in_all_batches' in hvg_info:
        print(f"\n  HVG stability across batches:")
        print(f"      Present in ALL {len(CELL_LINES)} batches: {hvg_info['n_hvg_in_all_batches']} genes")
    
    print(f"\n  Output locations:")
    print(f"      Cell line HVG vectors: /preprocessed/<cell_line>/<cell_line>_cell_hvg_vectors_{FLAVOR}_batch_aware.pkl")
    print(f"      Shared HVG info: {shared_path}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Starting Batch-Aware HVG Calculation Pipeline")
    print("="*80)
    
    main()  
    
    print("\n" + "="*80)
    print("All tasks completed successfully!")
    print("="*80)