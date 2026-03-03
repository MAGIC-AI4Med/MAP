import os
import time
import glob
import json
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader

from model.cell_emb import CellEmbeddingExtractor
from data.ds_tahoe_se import scTahoe100mDataset, CellDatasetCollator
from data.utils import get_embedding_cfg


def get_embeddings(cfg):
    all_pe = torch.load(get_embedding_cfg(cfg).all_embeddings, weights_only=False)
    if isinstance(all_pe, dict):
        all_pe = torch.vstack(list(all_pe.values()))

    all_pe = all_pe.cuda()
    return all_pe


def find_resume_point(save_dir):
    existing_files = glob.glob(os.path.join(save_dir, "embeddings_part_*.pt"))
    
    if not existing_files:
        print("No existing files found. Starting from beginning.")
        return 0, 0, 0
    
    file_numbers = []
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        try:
            file_num = int(filename.split('_')[-1].split('.')[0])
            file_numbers.append(file_num)
        except:
            continue
    
    if not file_numbers:
        print("No valid part files found. Starting from beginning.")
        return 0, 0, 0
    
    max_file_num = max(file_numbers)
    file_counter = max_file_num + 1
    
    total_samples = 0
    for i in range(max_file_num + 1):
        file_path = os.path.join(save_dir, f"embeddings_part_{i:04d}.pt")
        if os.path.exists(file_path):
            try:
                data = torch.load(file_path, map_location="cpu")
                total_samples += data["count"]
            except:
                print(f"Warning: Failed to load {file_path}")
                continue
    
    print(f"Resume detected: {max_file_num + 1} files exist, {total_samples} samples processed")
    return file_counter, total_samples, total_samples


def main():
    cell_line = 'CVCL_0504'
    print('current cell line: ', cell_line)
    cfg = OmegaConf.load("path/to/configs/se600m.yaml")

    generator = torch.Generator()
    dataset = scTahoe100mDataset(cell_line = cell_line)
    collator = CellDatasetCollator(cfg, is_train=True)

    extractor = CellEmbeddingExtractor()

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.model.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.dataset.num_train_workers,
        persistent_workers=True,
        pin_memory=True,
        generator=generator,
    )

    torch.backends.cudnn.benchmark = True

    extractor.se.eval()
    model = extractor.se
    pe = model.pe_embedding
    assert pe is not None, "pe_embedding not initialized, please check CellEmbeddingExtractor initialization."

    COLLECT_EVERY = 200

    save_dir = f"path/to/shards/{cell_line}"
    os.makedirs(save_dir, exist_ok=True)
    
    file_counter, total_samples, processed_samples = find_resume_point(save_dir)
    skip_batches = processed_samples // cfg.model.batch_size
    
    temp_embs_gpu = []
    temp_cell_line_ids = []
    temp_drug_smiles = []
    temp_drug_concs = []
    temp_ds_level_index = []

    print("Extracting SE embeddings over the whole dataset...")
    print(f"Batch collection strategy: every {COLLECT_EVERY} batches")
    if skip_batches > 0:
        print(f"Resuming from batch {skip_batches} (skipping {processed_samples} samples)")
    print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    
    with torch.no_grad():
        batch_idx = 0
        for batch in tqdm(dataloader, desc="Batches"):
            if batch_idx < skip_batches:
                batch_idx += 1
                continue
                
            start_time = time.time()

            transfer_start = time.time()
            _, _, drug_smiles_list, drug_conc_list, cell_line_id_list, dataset_level_index_list = batch
            transfer_time = time.time() - transfer_start

            inference_start = time.time()
            emb = model.infer(batch)
            inference_time = time.time() - inference_start

            temp_store_start = time.time()
            temp_embs_gpu.append(emb)
            temp_cell_line_ids.extend(cell_line_id_list)
            temp_drug_smiles.extend(drug_smiles_list)
            temp_drug_concs.extend(drug_conc_list)
            temp_ds_level_index.extend(dataset_level_index_list)
            temp_store_time = time.time() - temp_store_start

            total_time = time.time() - start_time

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Total={total_time:.3f}s, "
                    f"Transfer={transfer_time:.3f}s, "
                    f"Inference={inference_time:.3f}s, "
                    f"TempStore={temp_store_time:.3f}s")

            batch_idx += 1
            
            relative_batch = batch_idx - skip_batches
            should_collect = (relative_batch % COLLECT_EVERY == 0) or (batch_idx == len(dataloader))
            
            if should_collect:
                collect_start = time.time()
                
                if temp_embs_gpu:
                    print(f"  -> Collecting {len(temp_embs_gpu)} batches to file...")
                    
                    gpu_concat_start = time.time()
                    batch_embs_gpu = torch.cat(temp_embs_gpu, dim=0)
                    gpu_concat_time = time.time() - gpu_concat_start
                    
                    cpu_transfer_start = time.time()
                    batch_embs_cpu = batch_embs_gpu.detach().cpu()
                    cpu_transfer_time = time.time() - cpu_transfer_start
                    
                    save_start = time.time()
                    part_file = f"embeddings_part_{file_counter:04d}.pt"
                    part_path = os.path.join(save_dir, part_file)
                    
                    torch.save({
                        "embeddings": batch_embs_cpu,
                        "ds_level_index": temp_ds_level_index,
                        "cell_line_id": temp_cell_line_ids,
                        "drug_smiles": temp_drug_smiles,
                        "drug_conc": temp_drug_concs,
                        "count": batch_embs_cpu.shape[0]
                    }, part_path)
                    
                    save_time = time.time() - save_start
                    collect_time = time.time() - collect_start
                    
                    print(f"  -> Saved {batch_embs_cpu.shape[0]} samples to {part_file} "
                          f"({batch_embs_cpu.numel() * batch_embs_cpu.element_size() / 1024**2:.1f}MB)")
                    
                    total_samples += batch_embs_cpu.shape[0]
                    file_counter += 1
                    
                    temp_embs_gpu = []
                    temp_cell_line_ids = []
                    temp_drug_smiles = []
                    temp_drug_concs = []
                    temp_ds_level_index = []
                    
                    del batch_embs_gpu, batch_embs_cpu
                    torch.cuda.empty_cache()
                    
                    current_gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    print(f"  -> GPU memory after collection: {current_gpu_memory:.1f}GB")
            
            if batch_idx % 50 == 0:
                current_gpu_memory = torch.cuda.memory_allocated() / 1024**3
                pending_batches = len(temp_embs_gpu)
                print(f"\n--- Progress Update ---")
                print(f"Processed batches: {batch_idx}")
                print(f"Collected samples: {total_samples}")
                print(f"Saved files: {file_counter}")
                print(f"Pending batches in GPU: {pending_batches}")
                print(f"Current GPU memory: {current_gpu_memory:.1f}GB")
                print(f"----------------------\n")

    if total_samples == 0:
        print("No embeddings collected!")
        return

    print(f"Total samples collected: {total_samples}")
    print(f"Total files saved: {file_counter}")

    final_gpu_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"Final GPU memory: {final_gpu_memory:.1f}GB")

    summary_path = os.path.join(save_dir, "summary.json")
    summary = {
        "total_samples": total_samples,
        "file_count": file_counter,
        "embedding_dim": 2048,
        "files": [f"embeddings_part_{i:04d}.pt" for i in range(file_counter)]
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary to {summary_path}")

    print("\n=== Performance Summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Embedding files: {file_counter}")
    print(f"Collection frequency: every {COLLECT_EVERY} batches")
    print(f"Final memory usage: {final_gpu_memory:.1f}GB")


if __name__ == '__main__':
    main()