# eval_trainmean_sciplex.py
"""
TrainMean Baseline Evaluation for Sciplex dataset
Follows exactly the same logic as eval_trainmean.py for Tahoe.

Core Idea:
1. Compute the mean HVG vector from train split (trainMean)
2. Use trainMean as prediction for all samples on test split
3. Supports both per cell line and per drug evaluation modes
4. Uses the exact same evaluation metrics as eval.py
"""

import os
import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.ds_sciplex_lora_se import SciplexPerturbDatasetSE


def compute_mse(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute mean squared error"""
    return np.mean((pred - true) ** 2)


def compute_trainmean_vector(internal_loader, device):
    """
    Compute trainMean vector from train set

    Steps:
    1. For each batch, average HVG vectors across set_size dimension
    2. Collect all bulk means
    3. Average across all bulks to get final trainMean vector

    Args:
        internal_loader: DataLoader for train
        device: torch device

    Returns:
        trainmean_vector: numpy array of shape (G,), G = HVG dimension
    """
    print("\n" + "="*80)
    print("📊 Computing TrainMean Vector from Train Set (Sciplex)")
    print("="*80)

    all_bulk_means = []

    for batch_data in tqdm(internal_loader, desc="Processing train batches"):
        hvgs = batch_data["perturb_hvg_vectors"].to(device)

        # Average across set_size dimension
        bulk_mean = hvgs.mean(dim=1)  # (B, G)
        all_bulk_means.append(bulk_mean.cpu().numpy())

    all_bulk_means = np.concatenate(all_bulk_means, axis=0)
    trainmean_vector = all_bulk_means.mean(axis=0)  # (G,)

    print(f"✅ TrainMean vector computed:")
    print(f"   Total bulks processed: {all_bulk_means.shape[0]}")
    print(f"   HVG dimension: {trainmean_vector.shape[0]}")
    print(f"   TrainMean statistics: mean={trainmean_vector.mean():.4f}, std={trainmean_vector.std():.4f}")
    print("="*80 + "\n")

    return trainmean_vector


def compute_loss_trainmean(trainmean_vector, batch_data, device):
    """Compute MSE loss for TrainMean baseline"""
    true_hvg = batch_data["perturb_hvg_vectors"].to(device)  # (B, set_size, G)
    true_hvg_mean = true_hvg.mean(dim=1)  # (B, G)

    B = true_hvg_mean.shape[0]
    pred_hvg_mean = torch.from_numpy(trainmean_vector).to(device).unsqueeze(0).repeat(B, 1)  # (B, G)

    hvg_loss = torch.nn.functional.mse_loss(pred_hvg_mean, true_hvg_mean)
    return hvg_loss


@torch.no_grad()
def evaluate_per_cellline(
    trainmean_vector,
    test_loader,
    device,
    de_topk=50,
    verbose=True,
):
    """
    Evaluate TrainMean baseline per cell line

    Uses the exact same metrics as eval.py.
    """
    # Store data per cell line
    cell_line_data = defaultdict(lambda: {
        'hvg_loss': [],
        'pred_hvg_means': [],
        'true_hvg_means': [],
        'control_hvg_means': [],
        'hvg_dir_accs': [],
        'de_dir_accs': [],
        'de_overlaps': [],
        'de_delta_pred_all': [],
        'de_delta_true_all': [],
        'de_expr_pred_all': [],
        'de_expr_true_all': [],
    })

    for batch_data in tqdm(test_loader, desc="Evaluating TrainMean (per cell line)"):
        hvgs = batch_data["perturb_hvg_vectors"].to(device)
        cell_lines = batch_data["cell_line"]

        # Compute loss
        batch_hvg_loss = compute_loss_trainmean(trainmean_vector, batch_data, device)

        B = hvgs.shape[0]
        pred_hvg_mean = np.tile(trainmean_vector, (B, 1))  # (B, G)
        true_hvg_mean = hvgs.mean(dim=1).cpu().numpy()  # (B, G)
        control_hvg_mean = batch_data["control_hvg_vectors"].to(device).mean(dim=1).cpu().numpy()  # (B, G)

        delta_true = true_hvg_mean - control_hvg_mean
        delta_pred = pred_hvg_mean - control_hvg_mean

        B, G = delta_true.shape

        # Process each sample
        for i in range(B):
            cell_line = cell_lines[i]
            cl_data = cell_line_data[cell_line]

            cl_data['hvg_loss'].append(batch_hvg_loss.item())
            cl_data['pred_hvg_means'].append(pred_hvg_mean[i:i+1])
            cl_data['true_hvg_means'].append(true_hvg_mean[i:i+1])
            cl_data['control_hvg_means'].append(control_hvg_mean[i:i+1])

            dt = delta_true[i]
            dp = delta_pred[i]
            t_expr = true_hvg_mean[i]
            p_expr = pred_hvg_mean[i]

            # HVG direction accuracy (all genes)
            if G > 0:
                sign_true_all = np.sign(dt)
                sign_pred_all = np.sign(dp)
                hvg_dir_acc = (sign_true_all == sign_pred_all).mean()
                cl_data['hvg_dir_accs'].append(hvg_dir_acc)

            # Top-k DE processing
            if G == 0:
                continue
            k = min(de_topk, G)
            if k <= 0:
                continue

            abs_dt = np.abs(dt)
            top_idx_true = np.argsort(-abs_dt)[:k]

            abs_dp = np.abs(dp)
            top_idx_pred = np.argsort(-abs_dp)[:k]

            # DE overlap
            overlap = len(set(top_idx_true) & set(top_idx_pred)) / k
            cl_data['de_overlaps'].append(overlap)

            # DE direction accuracy (on true top-k)
            dt_top = dt[top_idx_true]
            dp_top = dp[top_idx_true]
            sign_true = np.sign(dt_top)
            sign_pred = np.sign(dp_top)
            de_dir_acc = (sign_true == sign_pred).mean()
            cl_data['de_dir_accs'].append(de_dir_acc)

            # Cache for MSE calculation
            cl_data['de_delta_true_all'].append(dt_top[None, :])
            cl_data['de_delta_pred_all'].append(dp_top[None, :])
            t_expr_top = t_expr[top_idx_true]
            p_expr_top = p_expr[top_idx_true]
            cl_data['de_expr_true_all'].append(t_expr_top[None, :])
            cl_data['de_expr_pred_all'].append(p_expr_top[None, :])

    # Compute metrics for each cell line
    per_cell_line_results = {}
    for cell_line, data in cell_line_data.items():
        results = _compute_metrics(data, de_topk)
        results['cell_line'] = cell_line
        per_cell_line_results[cell_line] = results

    # Compute overall average
    overall_avg = _compute_overall_average(per_cell_line_results)

    if verbose:
        _print_results_per_cellline(per_cell_line_results, overall_avg, de_topk)

    return {
        'per_cell_line': per_cell_line_results,
        'overall_avg': overall_avg,
    }


@torch.no_grad()
def evaluate_per_drug(
    trainmean_vector,
    test_loader,
    device,
    de_topk=50,
    verbose=True,
):
    """
    Evaluate TrainMean baseline per drug (per perturbation/SMILES)

    Uses the exact same metrics as eval.py.
    """
    # Store data per drug (per SMILES)
    drug_data = defaultdict(lambda: {
        'hvg_loss': [],
        'pred_hvg_means': [],
        'true_hvg_means': [],
        'control_hvg_means': [],
        'hvg_dir_accs': [],
        'de_dir_accs': [],
        'de_overlaps': [],
        'de_delta_pred_all': [],
        'de_delta_true_all': [],
        'cell_lines': set(),
        'drug_names': set(),
    })

    for batch_data in tqdm(test_loader, desc="Evaluating TrainMean (per drug)"):
        hvgs = batch_data["perturb_hvg_vectors"].to(device)
        smiles = batch_data["drug_smiles"]
        cell_lines = batch_data["cell_line"]
        drug_conc_strs = batch_data.get("target_drug_conc_str", ["Unknown"] * len(smiles))

        # Compute loss
        batch_hvg_loss = compute_loss_trainmean(trainmean_vector, batch_data, device)

        B = hvgs.shape[0]
        pred_hvg_mean = np.tile(trainmean_vector, (B, 1))  # (B, G)
        true_hvg_mean = hvgs.mean(dim=1).cpu().numpy()  # (B, G)
        control_hvg_mean = batch_data["control_hvg_vectors"].to(device).mean(dim=1).cpu().numpy()  # (B, G)

        delta_true = true_hvg_mean - control_hvg_mean
        delta_pred = pred_hvg_mean - control_hvg_mean

        B, G = delta_true.shape

        # Process each sample
        for i in range(B):
            smile = smiles[i]
            cell_line = cell_lines[i]
            drug_name = drug_conc_strs[i]

            d_data = drug_data[smile]
            d_data['cell_lines'].add(cell_line)
            d_data['drug_names'].add(drug_name)

            d_data['hvg_loss'].append(batch_hvg_loss.item())
            d_data['pred_hvg_means'].append(pred_hvg_mean[i:i+1])
            d_data['true_hvg_means'].append(true_hvg_mean[i:i+1])
            d_data['control_hvg_means'].append(control_hvg_mean[i:i+1])

            dt = delta_true[i]
            dp = delta_pred[i]

            # HVG direction accuracy (all genes)
            if G > 0:
                sign_true_all = np.sign(dt)
                sign_pred_all = np.sign(dp)
                hvg_dir_acc = (sign_true_all == sign_pred_all).mean()
                d_data['hvg_dir_accs'].append(hvg_dir_acc)

            # Top-k DE processing
            if G == 0:
                continue
            k = min(de_topk, G)
            if k <= 0:
                continue

            abs_dt = np.abs(dt)
            top_idx_true = np.argsort(-abs_dt)[:k]

            abs_dp = np.abs(dp)
            top_idx_pred = np.argsort(-abs_dp)[:k]

            # DE overlap
            overlap = len(set(top_idx_true) & set(top_idx_pred)) / k
            d_data['de_overlaps'].append(overlap)

            # DE direction accuracy (on true top-k)
            dt_top = dt[top_idx_true]
            dp_top = dp[top_idx_true]
            sign_true = np.sign(dt_top)
            sign_pred = np.sign(dp_top)
            de_dir_acc = (sign_true == sign_pred).mean()
            d_data['de_dir_accs'].append(de_dir_acc)

            # Cache for MSE calculation
            d_data['de_delta_true_all'].append(dt_top[None, :])
            d_data['de_delta_pred_all'].append(dp_top[None, :])

    # Compute metrics for each drug
    per_drug_results = {}
    for smile, data in drug_data.items():
        results = _compute_metrics(data, de_topk)
        results['smiles'] = smile
        results['num_cell_lines'] = len(data['cell_lines'])
        results['drug_names'] = sorted(list(data['drug_names']))
        per_drug_results[smile] = results

    # Compute overall average
    overall_avg = _compute_overall_average(per_drug_results)

    if verbose:
        _print_results_per_drug(per_drug_results, overall_avg, de_topk)

    return {
        'per_drug': per_drug_results,
        'overall_avg': overall_avg,
    }


def _compute_metrics(data, de_topk):
    """
    Compute all metrics from accumulated data, exactly the same as eval.py

    Metrics:
    - avg_hvg_loss
    - avg_hvg_dir_acc
    - avg_de_dir_acc
    - mse_hvg
    - mse_de
    - avg_de_overlap
    """
    # Average HVG loss
    avg_hvg_loss = float(np.mean(data['hvg_loss'])) if len(data['hvg_loss']) > 0 else 0.0

    # Merge all data
    pred_hvg_means = np.concatenate(data['pred_hvg_means'], axis=0)
    true_hvg_means = np.concatenate(data['true_hvg_means'], axis=0)

    # Direction accuracies
    avg_hvg_dir_acc = float(np.mean(data['hvg_dir_accs'])) if len(data['hvg_dir_accs']) > 0 else 0.0
    avg_de_dir_acc = float(np.mean(data['de_dir_accs'])) if len(data['de_dir_accs']) > 0 else 0.0
    avg_de_overlap = float(np.mean(data['de_overlaps'])) if len(data['de_overlaps']) > 0 else 0.0

    # MSE for all HVG genes
    mse_hvg = compute_mse(pred_hvg_means, true_hvg_means)

    # MSE for top-k DE genes
    if len(data['de_delta_pred_all']) > 0:
        de_delta_pred_mat = np.vstack(data['de_delta_pred_all'])
        de_delta_true_mat = np.vstack(data['de_delta_true_all'])
        mse_de = compute_mse(de_delta_pred_mat, de_delta_true_mat)
    else:
        mse_de = 0.0

    return {
        'avg_hvg_loss': avg_hvg_loss,
        'avg_hvg_dir_acc': avg_hvg_dir_acc,
        'avg_de_dir_acc': avg_de_dir_acc,
        'mse_hvg': mse_hvg,
        'mse_de': mse_de,
        'avg_de_overlap': avg_de_overlap,
        'num_samples': len(data['hvg_loss']),
    }


def _compute_overall_average(per_group_results):
    """Compute overall average across all groups (cell lines or drugs)"""
    if not per_group_results:
        return {}

    all_metrics = list(next(iter(per_group_results.values())).keys())
    # Exclude non-numeric fields from average
    exclude_fields = ['cell_line', 'cell_type', 'smiles', 'num_cell_lines', 'drug_names', 'cell_lines', 'drug_names']

    overall_avg = {}
    for metric_key in all_metrics:
        if metric_key in exclude_fields:
            continue
        values = [res[metric_key] for res in per_group_results.values()]
        overall_avg[metric_key] = np.mean(values)

    if 'num_samples' in all_metrics:
        overall_avg['total_samples'] = sum(r['num_samples'] for r in per_group_results.values())
        overall_avg['num_groups'] = len(per_group_results)

    return overall_avg


def _print_results_per_cellline(per_cell_line_results, overall_avg, de_topk):
    """Print results table for per cell line evaluation"""
    if not per_cell_line_results:
        print("No results to display.")
        return

    cell_lines = sorted(per_cell_line_results.keys())

    metrics = [
        ('HVG Loss', 'avg_hvg_loss', '.6f'),
        ('DirAcc(HVG)', 'avg_hvg_dir_acc', '.3f'),
        ('DirAcc(DE)', 'avg_de_dir_acc', '.3f'),
        ('MSE(HVG)', 'mse_hvg', '.6f'),
        (f'MSE(DE-{de_topk})', 'mse_de', '.6f'),
        (f'DE Overlap-{de_topk}', 'avg_de_overlap', '.3f'),
    ]

    print("\n" + "="*110)
    print("📊 TRAINMEAN BASELINE - PER CELL LINE RESULTS (Sciplex)")
    print("="*110)

    metric_col_width = max(len(m[0]) for m in metrics) + 2
    cell_col_width = 12
    avg_col_width = 14

    header = f"{'Metric':<{metric_col_width}}"
    for cl in cell_lines:
        header += f"{cl:>{cell_col_width}}"
    header += f"{'OVERALL AVG':>{avg_col_width}}"
    print(header)
    print("-"*110)

    for metric_name, metric_key, fmt in metrics:
        row = f"{metric_name:<{metric_col_width}}"
        for cl in cell_lines:
            value = per_cell_line_results[cl][metric_key]
            row += f"{value:>{cell_col_width}{fmt}}"
        avg_value = overall_avg[metric_key]
        row += f"{avg_value:>{avg_col_width}{fmt}}"
        print(row)

    print("="*110)
    print(f"\n📈 Summary:")
    print(f"   Number of cell lines: {len(per_cell_line_results)}")
    print(f"   Total samples evaluated: {overall_avg['total_samples']}")
    print(f"   Average samples per cell line: {overall_avg['total_samples'] / len(per_cell_line_results):.1f}")
    print()


def _print_results_per_drug(per_drug_results, overall_avg, de_topk):
    """Print results table for per drug evaluation"""
    if not per_drug_results:
        print("No results to display.")
        return

    # Sort by drug name
    sorted_smiles = sorted(
        per_drug_results.keys(),
        key=lambda s: (
            per_drug_results[s]['drug_names'][0].lower()
            if per_drug_results[s]['drug_names']
            else 'zzz'
        )
    )

    print("\n" + "="*160)
    print("📊 TRAINMEAN BASELINE - PER DRUG RESULTS (Sciplex)")
    print("="*160)

    header = (
        f"{'Drug Name':<30} {'SMILES':<40} {'#Cells':>8} {'#Samples':>10} "
        f"{'HVG Loss':>12} {'DirAcc(HVG)':>12} {'DirAcc(DE)':>12} "
        f"{'MSE(HVG)':>12} {'MSE(DE)':>12} {'DE Overlap':>12}"
    )
    print(header)
    print("-"*160)

    for smile in sorted_smiles:
        results = per_drug_results[smile]
        drug_names = results['drug_names']
        if len(drug_names) == 1:
            drug_name_str = drug_names[0]
        else:
            drug_name_str = f"{drug_names[0]} (+{len(drug_names)-1})"
        drug_name_short = drug_name_str[:27] + '...' if len(drug_name_str) > 30 else drug_name_str
        smiles_short = smile[:37] + '...' if len(smile) > 40 else smile

        row = (
            f"{drug_name_short:<30} {smiles_short:<40} {results['num_cell_lines']:>8} {results['num_samples']:>10} "
            f"{results['avg_hvg_loss']:>12.6f} {results['avg_hvg_dir_acc']:>12.3f} {results['avg_de_dir_acc']:>12.3f} "
            f"{results['mse_hvg']:>12.6f} {results['mse_de']:>12.6f} {results['avg_de_overlap']:>12.3f}"
        )
        print(row)

    print("="*160)

    total_row = (
        f"{'OVERALL AVERAGE':<30} {'-':<40} {'-':>8} {overall_avg['total_samples']:>10} "
        f"{overall_avg['avg_hvg_loss']:>12.6f} {overall_avg['avg_hvg_dir_acc']:>12.3f} {overall_avg['avg_de_dir_acc']:>12.3f} "
        f"{overall_avg['mse_hvg']:>12.6f} {overall_avg['mse_de']:>12.6f} {overall_avg['avg_de_overlap']:>12.3f}"
    )
    print(total_row)
    print("="*160)

    print(f"\n📈 Summary:")
    print(f"   Number of unique drugs: {len(per_drug_results)}")
    print(f"   Total samples evaluated: {overall_avg['total_samples']}")
    print(f"   Average samples per drug: {overall_avg['total_samples'] / len(per_drug_results):.1f}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="TrainMean Baseline Evaluation for Sciplex dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset arguments
    parser.add_argument("--data_base_dir", type=str,
                       default="/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/external_dataset_sciplex/preprocessed",
                       help="Base directory for preprocessed data")
    parser.add_argument("--se_inputs_base_dir", type=str,
                       default="/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/external_dataset_sciplex/preprocessed_se_inputs",
                       help="Base directory for precomputed SE inputs (memmap)")
    parser.add_argument("--cell_lines", "--cell_types", type=str, nargs='+',
                       default=['A549', 'K562', 'MCF7'],
                       help="Cell lines to evaluate on")
    parser.add_argument("--set_size", type=int, default=24,
                       help="Number of control cells in each sample")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of data loader workers")
    parser.add_argument("--de_topk", type=int, default=50,
                       help="Number of top DE genes for evaluation")
    parser.add_argument('--UC', action='store_true',
                       help="Use unseen cell line split")

    # Evaluation mode
    parser.add_argument("--mode", type=str, choices=['cellline', 'drug', 'both'], default='both',
                       help="Evaluation mode: per cell line, per drug, or both")

    # GPU
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU device ID to use")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    print(f"📋 Evaluation mode: {args.mode}")
    print()

    # Step 1: Load train set and compute trainMean vector
    print("📊 Loading train dataset for computing TrainMean...")
    internal_dataset = SciplexPerturbDatasetSE(
        cell_lines=args.cell_lines,
        split="train",
        base_dir=args.data_base_dir,
        se_inputs_base_dir=args.se_inputs_base_dir,
        hvg_not_yet_normed=True,
        set_size=args.set_size,
        is_train=False,
        sequential=True,
        return_control_hvg=True,
        UC=args.UC,
    )

    internal_loader = DataLoader(
        internal_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    print(f"✅ Loaded {len(internal_dataset)} train samples")

    # Compute trainMean vector
    trainmean_vector = compute_trainmean_vector(internal_loader, device)

    # Step 2: Load test set
    print("📊 Loading test dataset for evaluation...")
    external_dataset = SciplexPerturbDatasetSE(
        cell_lines=args.cell_lines,
        split="test",
        base_dir=args.data_base_dir,
        se_inputs_base_dir=args.se_inputs_base_dir,
        hvg_not_yet_normed=True,
        set_size=args.set_size,
        is_train=False,
        sequential=True,
        return_control_hvg=True,
        UC=args.UC,
    )

    external_loader = DataLoader(
        external_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    print(f"✅ Loaded {len(external_dataset)} test samples\n")

    # Step 3: Run evaluation based on mode
    print("🚀 Starting evaluation with TrainMean baseline...\n")

    if args.mode in ['cellline', 'both']:
        print("="*80)
        print("🧪 PER CELL LINE EVALUATION")
        print("="*80)
        cellline_results = evaluate_per_cellline(
            trainmean_vector=trainmean_vector,
            test_loader=external_loader,
            device=device,
            de_topk=args.de_topk,
            verbose=True,
        )
        print("✅ Per cell line evaluation completed!\n")

    if args.mode in ['drug', 'both']:
        print("="*80)
        print("🧪 PER DRUG EVALUATION")
        print("="*80)
        drug_results = evaluate_per_drug(
            trainmean_vector=trainmean_vector,
            test_loader=external_loader,
            device=device,
            de_topk=args.de_topk,
            verbose=True,
        )
        print("✅ Per drug evaluation completed!\n")

    # Print final overall summary if both modes are run
    if args.mode == 'both':
        print("="*80)
        print("📋 FINAL SUMMARY - TrainMean Baseline (Sciplex)")
        print("="*80)

        if 'cellline_results' in locals():
            print("\nPer Cell Line Overall:")
            ov = cellline_results['overall_avg']
            print(f"  HVG Loss:          {ov['avg_hvg_loss']:.6f}")
            print(f"  DirAcc (HVG):      {ov['avg_hvg_dir_acc']:.3f}")
            print(f"  DirAcc (DE):       {ov['avg_de_dir_acc']:.3f}")
            print(f"  MSE (HVG):         {ov['mse_hvg']:.6f}")
            print(f"  MSE (DE-{args.de_topk}):  {ov['mse_de']:.6f}")
            print(f"  DE Overlap:        {ov['avg_de_overlap']:.3f}")

        if 'drug_results' in locals():
            print("\nPer Drug Overall:")
            ov = drug_results['overall_avg']
            print(f"  HVG Loss:          {ov['avg_hvg_loss']:.6f}")
            print(f"  DirAcc (HVG):      {ov['avg_hvg_dir_acc']:.3f}")
            print(f"  DirAcc (DE):       {ov['avg_de_dir_acc']:.3f}")
            print(f"  MSE (HVG):         {ov['mse_hvg']:.6f}")
            print(f"  MSE (DE-{args.de_topk}):  {ov['mse_de']:.6f}")
            print(f"  DE Overlap:        {ov['avg_de_overlap']:.3f}")

        print("\n" + "="*80)

    print("\n✅ All evaluations completed!")
    return


if __name__ == '__main__':
    main()
