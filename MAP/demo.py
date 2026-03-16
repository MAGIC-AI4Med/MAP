"""
Demo: load a bulk (control cells), a drug, run inference, save inputs/outputs.
"""
import os
import argparse
import pickle
import json

import numpy as np
import torch
from omegaconf import OmegaConf
from rdkit import Chem

from model.model import MAPmodel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--cell_line", type=str, default="CVCL_0023")
    parser.add_argument("--drug_smiles", type=str, default="CC1=NC=C(C(=C1O)CO)CO")
    parser.add_argument("--drug_conc", type=float, default=0.5)
    parser.add_argument("--set_size", type=int, default=24, help="Number of control cells to use as bulk")
    parser.add_argument("--data_base_dir", type=str,
                        default="path/to/proprocessed")
    parser.add_argument("--se_inputs_base_dir", type=str,
                        default="path/to/preprocessed_se_inputs_memmap")
    parser.add_argument("--se_config", type=str,
                        default="path/to/se600m.yaml")
    parser.add_argument("--se_ckpt", type=str,
                        default="path/to/se600m.safetensors")
    parser.add_argument("--output_dir", type=str, default="./demo_output")
    return parser.parse_args()


def load_control_bulk(cell_line, data_base_dir, se_inputs_base_dir, set_size):
    """Load set_size control cells from memmap files."""
    cell_line_dir = os.path.join(data_base_dir, cell_line)

    with open(os.path.join(cell_line_dir, f"{cell_line}_meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    with open(os.path.join(cell_line_dir, "control_indices.json")) as f:
        control_ds_indices = set(json.load(f))

    # map ds_level_index -> row indices
    all_ds_indices = meta["ds_level_index"]
    control_rows = [i for i, ds_idx in enumerate(all_ds_indices) if ds_idx in control_ds_indices]

    # sample set_size rows
    chosen = np.random.choice(control_rows, set_size, replace=(len(control_rows) < set_size))
    chosen = list(map(int, chosen))

    se_dir = os.path.join(se_inputs_base_dir, cell_line)
    with open(os.path.join(se_dir, "se_shape.json")) as f:
        shape = json.load(f)
    N, L = int(shape["N"]), int(shape["pad_length"])

    gene_mm = np.memmap(os.path.join(se_dir, "se_gene_ids.npy"), dtype="int32", mode="r", shape=(N, L))
    expr_mm = np.memmap(os.path.join(se_dir, "se_expr.npy"), dtype="float16", mode="r", shape=(N, L))

    gene_ids = torch.from_numpy(gene_mm[chosen].copy())   # [S, L]
    expressions = torch.from_numpy(expr_mm[chosen].copy()).float()  # [S, L]

    return gene_ids, expressions


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ──────────────────────────────────────────────────────────
    cfg = OmegaConf.load(args.se_config)
    model = MAPmodel(
        se_ckpt=args.se_ckpt,
        se_cfg=cfg,
        smile_encoder="PrimeKG_ver1",
        hvg_info=None,
    )

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    # ── Load bulk (control cells) ────────────────────────────────────────────
    gene_ids, expressions = load_control_bulk(
        args.cell_line, args.data_base_dir, args.se_inputs_base_dir, args.set_size
    )
    # model expects [B, S, L]
    ctrl_gene = gene_ids.unsqueeze(0).to(device)        # [1, S, L]
    ctrl_expr = expressions.unsqueeze(0).to(device)     # [1, S, L]

    # ── Canonicalize SMILES ──────────────────────────────────────────────────
    mol = Chem.MolFromSmiles(args.drug_smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {args.drug_smiles}")
    canon_smiles = Chem.MolToSmiles(mol)
    smiles = [canon_smiles]
    concs = torch.tensor([args.drug_conc], dtype=torch.float32, device=device)

    print(f"Cell line : {args.cell_line}")
    print(f"Drug SMILES: {canon_smiles}")
    print(f"Drug conc  : {args.drug_conc} uM")
    print(f"Bulk size  : {args.set_size} cells")

    # ── Inference ────────────────────────────────────────────────────────────
    with torch.no_grad():
        pred_embs, pred_hvgs = model(ctrl_gene, ctrl_expr, smiles, concs)
    # pred_embs: [1, S, 2048], pred_hvgs: [1, S, 2000]

    # ── Save ─────────────────────────────────────────────────────────────────
    np.save(os.path.join(args.output_dir, "input_ctrl_gene_ids.npy"),
            ctrl_gene.cpu().numpy().astype(np.int32))
    np.save(os.path.join(args.output_dir, "input_ctrl_expressions.npy"),
            ctrl_expr.cpu().numpy().astype(np.float32))
    np.save(os.path.join(args.output_dir, "output_pred_embeddings.npy"),
            pred_embs.cpu().float().numpy())
    np.save(os.path.join(args.output_dir, "output_pred_hvg.npy"),
            pred_hvgs.cpu().float().numpy())

    meta = {
        "cell_line": args.cell_line,
        "drug_smiles": canon_smiles,
        "drug_conc_uM": args.drug_conc,
        "set_size": args.set_size,
        "pred_embs_shape": list(pred_embs.shape),
        "pred_hvgs_shape": list(pred_hvgs.shape),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {args.output_dir}/")
    print(f"  input_ctrl_gene_ids.npy    {ctrl_gene.shape}")
    print(f"  input_ctrl_expressions.npy {ctrl_expr.shape}")
    print(f"  output_pred_embeddings.npy {pred_embs.shape}")
    print(f"  output_pred_hvg.npy        {pred_hvgs.shape}")


if __name__ == "__main__":
    main()
