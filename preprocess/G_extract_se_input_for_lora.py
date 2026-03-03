import os
import json
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader


def extract_se_inputs_memmap(cell_line, cfg_path, out_base, batch_size=256):
    from data.ds_tahoe_se import scTahoe100mDatasetSEOnly, CellDatasetCollatorSEOnly

    print(f"\n{'='*60}")
    print(f"Processing {cell_line}")
    print(f"{'='*60}")
    
    print("1. Loading config...")
    cfg = OmegaConf.load(cfg_path)
    pad_length = cfg.dataset.pad_length
    
    print("2. Creating output directory...")
    out_dir = os.path.join(out_base, cell_line)
    os.makedirs(out_dir, exist_ok=True)
    
    print("3. Initializing dataset and loader...")
    dataset = scTahoe100mDatasetSEOnly(cell_line=cell_line)
    collator = CellDatasetCollatorSEOnly(cfg)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    
    N = len(dataset)
    print(f"Dataset size: {N}, Pad length: {pad_length}")
    
    print("4. Creating memmap files...")
    gene_ids_mm = np.memmap(
        os.path.join(out_dir, "se_gene_ids.npy"),
        dtype="int32",
        mode="w+",
        shape=(N, pad_length),
    )
    expr_mm = np.memmap(
        os.path.join(out_dir, "se_expr.npy"),
        dtype="float16",
        mode="w+",
        shape=(N, pad_length),
    )
    
    print("5. Extracting and writing data...")
    write_ptr = 0
    
    for batch_gene_ids, batch_expr in tqdm(loader, desc=f"Extracting SE inputs for {cell_line}"):
        bsz = batch_gene_ids.shape[0]
        
        gene_ids_mm[write_ptr:write_ptr + bsz] = batch_gene_ids.numpy()
        expr_mm[write_ptr:write_ptr + bsz] = batch_expr.numpy().astype(np.float16)
        
        write_ptr += bsz
        
        if write_ptr % (1000 * batch_size) == 0:
            gene_ids_mm.flush()
            expr_mm.flush()
    
    print("6. Flushing and saving metadata...")
    gene_ids_mm.flush()
    expr_mm.flush()
    
    with open(os.path.join(out_dir, "se_shape.json"), "w") as f:
        json.dump({"N": N, "pad_length": pad_length}, f)
    
    print(f"Done with {cell_line}! N={N}, L={pad_length}")
    print(f"Files saved to: {out_dir}")
    print(f"{'='*60}\n")
    
    return cell_line


def process_cell_line_wrapper(args):
    cell_line, cfg_path, out_base, batch_size = args
    
    try:
        result = extract_se_inputs_memmap(
            cell_line=cell_line,
            cfg_path=cfg_path,
            out_base=out_base,
            batch_size=batch_size,
        )
        return (True, result)
    except Exception as e:
        print(f"Error processing {cell_line}: {str(e)}")
        import traceback
        traceback.print_exc()
        return (False, cell_line)


def run_sequential(cell_lines, cfg_path, out_base, batch_size):
    results = []
    for cell_line in cell_lines:
        print(f"\nStarting processing for {cell_line}...")
        result = process_cell_line_wrapper((cell_line, cfg_path, out_base, batch_size))
        results.append(result)
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--single', action='store_true', help='Process single cell line')
    parser.add_argument('--cell_line', type=str, help='Cell line name')
    parser.add_argument('--cfg_path', type=str, help='Config file path')
    parser.add_argument('--out_base', type=str, help='Output base directory')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--cl', type=int)
    
    args = parser.parse_args()
    
    if args.single:
        result = process_cell_line_wrapper((
            args.cell_line,
            args.cfg_path,
            args.out_base,
            args.batch_size
        ))
        exit(0 if result[0] else 1)
    
    CELL_LINES_LST = ['CVCL_0023', 'CVCL_0069', 'CVCL_0480', 'CVCL_0131', 'CVCL_1056', 'CVCL_1098']
    CELL_LINES = [CELL_LINES_LST[args.cl]]
    CFG_PATH = "path/to/configs/se600m.yaml"
    OUT_BASE = "path/to/preprocessed_se_inputs_memmap"
    BATCH_SIZE = 512
    
    print(f"Starting processing for {len(CELL_LINES)} cell lines")
    print(f"Cell lines: {CELL_LINES}")
    print(f"Config file: {CFG_PATH}")
    print(f"Output directory: {OUT_BASE}")
    print(f"Batch size: {BATCH_SIZE}\n")
    
    results = run_sequential(CELL_LINES, CFG_PATH, OUT_BASE, BATCH_SIZE)
    
    successful = [cell_line for success, cell_line in results if success]
    failed = [cell_line for success, cell_line in results if not success]
    
    print("\n" + "="*60)
    print("Processing complete!")
    print(f"Successful: {len(successful)}/{len(CELL_LINES)}")
    if successful:
        print(f"Successful cell lines: {successful}")
    if failed:
        print(f"Failed cell lines: {failed}")
    print("="*60)