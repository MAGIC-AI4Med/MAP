import os
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm

def convert_dataset_to_mmap(cell_line, data_base, hvg_base, output_dir, hvg_setting='seurat_v3'):
    print(f"Converting {cell_line}...")

    shards_dir = os.path.join(data_base, cell_line)
    summary_path = os.path.join(shards_dir, "summary.json")
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    meta_ds_indices = []
    meta_drug_conc = []
    meta_drug_smiles = []
    meta_cell_line_ids = []
    
    total_samples = 0
    embedding_dim = None
    
    print("   Scanning shards for metadata...")
    for i in range(summary['file_count']):
        file_path = os.path.join(shards_dir, f"embeddings_part_{i:04d}.pt")
        data = torch.load(file_path, map_location="cpu")
        
        if embedding_dim is None:
            embedding_dim = data["embeddings"].shape[1]
            
        n = data["embeddings"].shape[0]
        total_samples += n
        
        meta_ds_indices.extend(data["ds_level_index"])
        meta_drug_conc.extend(data["drug_conc"])
        meta_drug_smiles.extend(data["drug_smiles"])
        meta_cell_line_ids.extend(data["cell_line_id"])

    print(f"   Total samples: {total_samples}, Dim: {embedding_dim}")

    emb_path = os.path.join(output_dir, cell_line, f"{cell_line}_embeddings.npy")
    emb_fp = np.memmap(emb_path, dtype='float32', mode='w+', shape=(total_samples, embedding_dim))
    
    print("   Writing embeddings to disk...")
    start_idx = 0
    for i in tqdm(range(summary['file_count']), desc="Writing Embeddings"):
        file_path = os.path.join(shards_dir, f"embeddings_part_{i:04d}.pt")
        data = torch.load(file_path, map_location="cpu")
        batch_len = data["embeddings"].shape[0]
        
        emb_fp[start_idx : start_idx + batch_len] = data["embeddings"].numpy()
        start_idx += batch_len
    
    emb_fp.flush()
    del emb_fp

    hvg_path_data = os.path.join(hvg_base, f"{cell_line}/{cell_line}_cell_hvg_vectors_{hvg_setting}.pkl")
    print(f"   Loading original HVG dict from {hvg_path_data}...")
    with open(hvg_path_data, 'rb') as f:
        full_hvg_dict = pickle.load(f)
    
    hvg_dim = 2000
    hvg_path = os.path.join(output_dir, cell_line, f"{cell_line}_hvg.npy")
    hvg_fp = np.memmap(hvg_path, dtype='float32', mode='w+', shape=(total_samples, hvg_dim))
    
    print("   Writing HVG vectors to disk...")
    batch_size = 5000
    buffer = np.zeros((batch_size, hvg_dim), dtype='float32')
    
    for i, ds_idx in enumerate(tqdm(meta_ds_indices, desc="Writing HVG")):
        local_idx = i % batch_size
        
        if ds_idx in full_hvg_dict:
            vec = full_hvg_dict[ds_idx]
            if isinstance(vec, torch.Tensor):
                vec = vec.cpu().numpy()
            buffer[local_idx] = vec
        else:
            buffer[local_idx] = 0
            
        if local_idx == batch_size - 1 or i == total_samples - 1:
            write_len = local_idx + 1
            start_pos = i - local_idx
            hvg_fp[start_pos : start_pos + write_len] = buffer[:write_len]
            buffer.fill(0)

    hvg_fp.flush()
    del hvg_fp
    del full_hvg_dict

    ds_idx_to_row = {ds_idx: i for i, ds_idx in enumerate(meta_ds_indices)}
    
    meta_data = {
        "ds_level_index": meta_ds_indices,
        "drug_conc": meta_drug_conc,
        "drug_smiles": meta_drug_smiles,
        "cell_line_id": meta_cell_line_ids,
        "ds_idx_to_row": ds_idx_to_row,
        "shape_emb": (total_samples, embedding_dim),
        "shape_hvg": (total_samples, hvg_dim),
        "dtype": "float32"
    }
    
    meta_out_path = os.path.join(output_dir, cell_line, f"{cell_line}_meta.pkl")
    with open(meta_out_path, 'wb') as f:
        pickle.dump(meta_data, f)
        
    print(f"Conversion complete for {cell_line}!")

convert_dataset_to_mmap(
    "CVCL_0218",
    "path/to/shards",
    "path/to/preprocessed",
    "path/to/preprocessed"
)