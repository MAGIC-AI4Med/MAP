# G2_extract_se_inputs_sharded_indexed.py
# Efficient sharded storage for SE inputs, with an index for O(1) lookup.
#
# Output (per cell line):
#   se_inputs_index.json           # ds_level_index -> [shard_id, offset]
#   se_inputs_shard_0000.pt        # dict with tensors: ds_idx, gene_ids, expr
#
# Storage choices:
#   - gene_ids: int32
#   - expr: float16 (cast to float32 at runtime)
#
# NOTE: This script preserves ds_level_index exactly as provided by scTahoe100mDataset.
#
import os
import sys
import json
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def preprocess_one_cell_line(cell_line: str, split_base: str, out_base: str, shard_size: int = 50_000):
    # Heavy import inside worker
    from data.ds_tahoe_se import scTahoe100mDataset

    out_dir = os.path.join(out_base, cell_line)
    os.makedirs(out_dir, exist_ok=True)

    ds = scTahoe100mDataset(cell_line=cell_line, data_split_base=split_base)
    n = len(ds)
    print(f"[INFO] {cell_line}: N={n:,} | shard_size={shard_size:,}")

    index = {}  # ds_idx (str) -> [shard_id, offset]
    shard_id = 0

    buf_ds = []
    buf_gene = []
    buf_expr = []

    def flush():
        nonlocal shard_id, buf_ds, buf_gene, buf_expr
        if not buf_ds:
            return

        ds_idx_t = torch.tensor(buf_ds, dtype=torch.int64)             # [M]
        gene_t = torch.stack(buf_gene, dim=0).to(torch.int32)          # [M, L]
        expr_t = torch.stack(buf_expr, dim=0).to(torch.float16)        # [M, L]

        shard_path = os.path.join(out_dir, f"se_inputs_shard_{shard_id:04d}.pt")
        torch.save({"ds_idx": ds_idx_t, "gene_ids": gene_t, "expr": expr_t}, shard_path)

        for off, ds_idx in enumerate(buf_ds):
            index[str(int(ds_idx))] = [shard_id, off]

        print(f"[SAVE] {cell_line} shard={shard_id:04d} entries={len(buf_ds)} -> {shard_path}")
        shard_id += 1
        buf_ds, buf_gene, buf_expr = [], [], []

    for i in tqdm(range(n), desc=cell_line):
        (
            _,
            gene_ids,        # [1, L]
            expressions,     # [1, L]
            _drug_smiles,
            _drug_conc,
            _cell_line_id,
            ds_level_index,
        ) = ds[i]

        gene_ids = gene_ids.squeeze(0).long().cpu()
        expressions = expressions.squeeze(0).float().cpu()

        buf_ds.append(int(ds_level_index))
        buf_gene.append(gene_ids)
        buf_expr.append(expressions)

        if len(buf_ds) >= shard_size:
            flush()

    flush()

    index_path = os.path.join(out_dir, "se_inputs_index.json")
    with open(index_path, "w") as f:
        json.dump(index, f)
    print(f"[DONE] {cell_line}: shards={shard_id} | index={index_path}")
    return cell_line


def _worker(args):
    cell_line, split_base, out_base, shard_size = args
    try:
        preprocess_one_cell_line(cell_line, split_base, out_base, shard_size)
        return (True, cell_line)
    except Exception as e:
        print(f"[ERROR] {cell_line}: {repr(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return (False, cell_line)


if __name__ == "__main__":
    CELL_LINES = ['CVCL_0023', 'CVCL_0069', 'CVCL_0480', 'CVCL_0131', 'CVCL_1056', 'CVCL_1098']
    SPLIT_BASE = "path/to/preprocessed"
    OUT_BASE = "path/to/preprocessed_se_inputs_v2"
    SHARD_SIZE = 50_000

    num_proc = min(len(CELL_LINES), cpu_count(), 4)
    tasks = [(cl, SPLIT_BASE, OUT_BASE, SHARD_SIZE) for cl in CELL_LINES]

    print(f"[INFO] Start: {len(CELL_LINES)} cell lines | processes={num_proc}")
    with Pool(processes=num_proc) as pool:
        results = pool.map(_worker, tasks)

    ok = [c for s, c in results if s]
    bad = [c for s, c in results if not s]
    print(f"[SUMMARY] success={len(ok)}/{len(CELL_LINES)} | failed={bad}")
