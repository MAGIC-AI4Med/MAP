# eval_trainmean.py
"""
TrainMean Baseline - Unified Evaluation (supports both per cell line and per drug)

Core Idea:
1. Compute the mean HVG vector from internal_test split (trainMean)
2. Use trainMean as prediction for all samples on external_test
3. Support both per-cell-line and per-drug evaluation modes
4. Uses the exact same evaluation metrics as eval.py

Metrics (same as eval.py):
- avg_hvg_loss: Average HVG MSE loss
- avg_hvg_dir_acc: Direction accuracy on all HVGs (sign match of delta)
- avg_de_dir_acc: Direction accuracy on top-k DE genes
- mse_hvg: MSE for all HVG genes
- mse_de: MSE for top-k DE genes
- avg_de_overlap: DE overlap between predicted top-k and true top-k DEGs
"""

import os
import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

from data.ds_multi_cell_lora_se import TahoePerturbDatasetSE


# ==============================
# Metric computation helpers
# ==============================

def compute_pearson_delta(pred_hvg_means: np.ndarray,
                       true_hvg_means: np.ndarray,
                       control_hvg_means: np.ndarray) -> float:
    """
    计算基于delta的Pearson相关性指标:
    - pearson_delta_mean: 逐样本的 delta Pearson correlation 的平均值
    """
    N = pred_hvg_means.shape[0]
    if N == 0:
        return float('nan')

    pred_deltas = pred_hvg_means - control_hvg_means  # [N, G]
    true_deltas = true_hvg_means - control_hvg_means  # [N, G]

    # 逐样本计算 Pearson Delta Correlation，然后平均
    per_sample_corrs = []
    for i in range(N):
        if np.any(np.isnan(pred_deltas[i])) or np.any(np.isinf(pred_deltas[i])):
            continue
        if np.any(np.isnan(true_deltas[i])) or np.any(np.isinf(true_deltas[i])):
            continue
        if np.std(pred_deltas[i]) < 1e-8 or np.std(true_deltas[i]) < 1e-8:
            continue
        r, _ = pearsonr(pred_deltas[i], true_deltas[i])
        if not np.isnan(r):
            per_sample_corrs.append(r)

    return np.mean(per_sample_corrs) if len(per_sample_corrs) > 0 else 0.0


def compute_discrimination_score_global(pred_means: np.ndarray,
                                          true_means: np.ndarray,
                                          metric: str = 'euclidean') -> float:
    """
    全局 discrimination score（归一化逆rank，越高越好。
    定义：score = 1 - 2 * mean_rank / N，其中 rank_i 是第 i 个预测到正确真实目标的相对排名。
    随机≈0，完美→1。
    """
    if pred_means.shape[0] == 0:
        return float('nan')
    dist = cdist(pred_means, true_means, metric=metric)  # [N, N]
    N = dist.shape[0]
    ranks = []
    for i in range(N):
        di = dist[i]
        d_true = di[i]
        rank = np.sum(di < d_true)  # 严格小于
        ranks.append(rank)
    mean_rank = np.mean(ranks) if len(ranks) > 0 else np.nan
    if np.isnan(mean_rank):
        return float('nan')
    return float(1.0 - 2.0 * (mean_rank / N))


def compute_wasserstein_per_gene(all_pred_hvg: np.ndarray,
                                  all_true_hvg: np.ndarray) -> np.ndarray:
    """
    计算每个基因位置的Wasserstein距离。
    all_pred_hvg: [N_samples, G_genes] - 所有样本的预测HVG表达值
    all_true_hvg: [N_samples, G_genes] - 所有样本的真实HVG表达值
    返回：长度G的数组，每个元素是该基因位置的Wasserstein距离
    """
    N_samples, G_genes = all_pred_hvg.shape
    wasserstein_per_gene = np.zeros(G_genes)
    for g in range(G_genes):
        pred_dist = all_pred_hvg[:, g]
        true_dist = all_true_hvg[:, g]
        wasserstein_per_gene[g] = wasserstein_distance(pred_dist, true_dist)
    return wasserstein_per_gene


def compute_mse(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute mean squared error"""
    return np.mean((pred - true) ** 2)


def compute_trainmean_vector(internal_loader, device):
    """
    Compute trainMean vector from internal test set

    Steps:
    1. For each batch, average HVG vectors across set_size dimension
    2. Collect all bulk means
    3. Average across all bulks to get final trainMean vector

    Args:
        internal_loader: DataLoader for internal test
        device: torch device

    Returns:
        trainmean_vector: numpy array of shape (G,), G = HVG dimension
    """
    print("\n" + "="*80)
    print("📊 Computing TrainMean Vector from Internal Test Set")
    print("="*80)

    all_bulk_means = []

    for batch_data in tqdm(internal_loader, desc="Processing internal test batches"):
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
        # Wasserstein needs all individual gene values
        'all_pred_hvg': [],
        'all_true_hvg': [],
        'all_de_indices': [],
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

            # Collect data for Wasserstein (before early continue)
            cl_data['all_pred_hvg'].append(p_expr)
            cl_data['all_true_hvg'].append(t_expr)

            # Top-k DE processing
            if G == 0:
                continue
            k = min(de_topk, G)
            if k <= 0:
                continue

            abs_dt = np.abs(dt)
            top_idx_true = np.argsort(-abs_dt)[:k]
            cl_data['all_de_indices'].append(top_idx_true)

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
        # Wasserstein needs all individual gene values
        'all_pred_hvg': [],
        'all_true_hvg': [],
        'all_de_indices': [],
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
            drug_name = drug_conc_strs[i].split('\'')[1]

            d_data = drug_data[smile]
            d_data['cell_lines'].add(cell_line)
            d_data['drug_names'].add(drug_name)

            d_data['hvg_loss'].append(batch_hvg_loss.item())
            d_data['pred_hvg_means'].append(pred_hvg_mean[i:i+1])
            d_data['true_hvg_means'].append(true_hvg_mean[i:i+1])
            d_data['control_hvg_means'].append(control_hvg_mean[i:i+1])

            dt = delta_true[i]
            dp = delta_pred[i]
            t_expr = true_hvg_mean[i]
            p_expr = pred_hvg_mean[i]

            # HVG direction accuracy (all genes)
            if G > 0:
                sign_true_all = np.sign(dt)
                sign_pred_all = np.sign(dp)
                hvg_dir_acc = (sign_true_all == sign_pred_all).mean()
                d_data['hvg_dir_accs'].append(hvg_dir_acc)

            # Collect data for Wasserstein (before early continue)
            d_data['all_pred_hvg'].append(p_expr)
            d_data['all_true_hvg'].append(t_expr)

            # Top-k DE processing
            if G == 0:
                continue
            k = min(de_topk, G)
            if k <= 0:
                continue

            abs_dt = np.abs(dt)
            top_idx_true = np.argsort(-abs_dt)[:k]
            d_data['all_de_indices'].append(top_idx_true)

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
    Compute all metrics from accumulated data (complete set of metrics)

    ===== HVG Metrics =====
    - avg_hvg_loss: Average HVG MSE loss
    - pearson_delta_hvg: Pearson correlation of expression delta (pred vs true)
    - disc_score_hvg: Global discrimination score of HVG vectors
    - mse_hvg: MSE for all HVG genes
    - wasserstein_hvg: Wasserstein distance between predicted and true HVG distributions
    - avg_hvg_dir_acc: Direction accuracy on all HVGs

    ===== DE Metrics =====
    - pearson_delta_de: Pearson correlation of expression delta for top-k DE genes
    - mse_de: MSE for top-k DE genes
    - wasserstein_de: Wasserstein distance for top-k DE genes
    - avg_de_dir_acc: Direction accuracy on top-k DE genes
    - avg_de_overlap: DE overlap between predicted top-k and true top-k DEGs
    """
    num_samples = len(data['hvg_loss'])

    # Average HVG loss
    avg_hvg_loss = float(np.mean(data['hvg_loss'])) if num_samples > 0 else 0.0

    # Merge all data
    pred_hvg_means = np.concatenate(data['pred_hvg_means'], axis=0)
    true_hvg_means = np.concatenate(data['true_hvg_means'], axis=0)
    control_hvg_means = np.concatenate(data['control_hvg_means'], axis=0)

    # Direction accuracies
    avg_hvg_dir_acc = float(np.mean(data['hvg_dir_accs'])) if len(data['hvg_dir_accs']) > 0 else 0.0
    avg_de_dir_acc = float(np.mean(data['de_dir_accs'])) if len(data['de_dir_accs']) > 0 else 0.0
    avg_de_overlap = float(np.mean(data['de_overlaps'])) if len(data['de_overlaps']) > 0 else 0.0

    # MSE for all HVG genes
    mse_hvg = compute_mse(pred_hvg_means, true_hvg_means)

    # ===== NEW: Pearson Delta (HVG) =====
    pearson_delta_hvg = compute_pearson_delta(pred_hvg_means, true_hvg_means, control_hvg_means)

    # ===== NEW: Discrimination Score (HVG) =====
    if num_samples >= 2:  # Need at least 2 samples for discrimination score
        disc_score_hvg = compute_discrimination_score_global(pred_hvg_means, true_hvg_means)
    else:
        disc_score_hvg = 0.0

    # ===== NEW: Wasserstein Distance =====
    if len(data['all_pred_hvg']) > 0 and len(data['all_true_hvg']) > 0:
        all_pred_hvg = np.array(data['all_pred_hvg'])  # [N, G]
        all_true_hvg = np.array(data['all_true_hvg'])  # [N, G]

        # Wasserstein for all HVG genes (average across genes)
        wasserstein_per_gene = compute_wasserstein_per_gene(all_pred_hvg, all_true_hvg)
        wasserstein_hvg = float(np.mean(wasserstein_per_gene))

        # Wasserstein for DE genes (average across all DE positions across all samples)
        if len(data['all_de_indices']) > 0:
            wasserstein_de_values = []
            for sample_idx, de_indices in enumerate(data['all_de_indices']):
                for gene_idx in de_indices:
                    wasserstein_de_values.append(wasserstein_per_gene[gene_idx])
            wasserstein_de = float(np.mean(wasserstein_de_values)) if len(wasserstein_de_values) > 0 else 0.0
        else:
            wasserstein_de = 0.0
    else:
        wasserstein_hvg = 0.0
        wasserstein_de = 0.0

    # ===== MSE and Pearson for DE genes =====
    if len(data['de_delta_pred_all']) > 0:
        de_delta_pred_mat = np.vstack(data['de_delta_pred_all'])
        de_delta_true_mat = np.vstack(data['de_delta_true_all'])
        mse_de = compute_mse(de_delta_pred_mat, de_delta_true_mat)

        # ===== NEW: Pearson Delta (DE) =====
        # For DE, we compute correlation across all DE positions
        pred_deltas_flat = de_delta_pred_mat.flatten()
        true_deltas_flat = de_delta_true_mat.flatten()
        if len(pred_deltas_flat) >= 2 and np.std(pred_deltas_flat) > 1e-8 and np.std(true_deltas_flat) > 1e-8:
            r, _ = pearsonr(pred_deltas_flat, true_deltas_flat)
            pearson_delta_de = float(r) if not np.isnan(r) else 0.0
        else:
            pearson_delta_de = 0.0
    else:
        mse_de = 0.0
        pearson_delta_de = 0.0

    return {
        # HVG metrics
        'avg_hvg_loss': avg_hvg_loss,
        'pearson_delta_hvg': pearson_delta_hvg,
        'disc_score_hvg': disc_score_hvg,
        'mse_hvg': mse_hvg,
        'wasserstein_hvg': wasserstein_hvg,
        'avg_hvg_dir_acc': avg_hvg_dir_acc,
        # DE metrics
        'pearson_delta_de': pearson_delta_de,
        'mse_de': mse_de,
        'wasserstein_de': wasserstein_de,
        'avg_de_dir_acc': avg_de_dir_acc,
        'avg_de_overlap': avg_de_overlap,
        # metadata
        'num_samples': num_samples,
    }


def _compute_overall_average(per_group_results):
    """Compute overall average across all groups (cell lines or drugs)"""
    if not per_group_results:
        return {}

    all_metrics = list(next(iter(per_group_results.values())).keys())
    # Exclude non-numeric fields from average
    exclude_fields = ['cell_line', 'smiles', 'num_cell_lines', 'drug_names', 'cell_lines', 'drug_names']

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

    # Complete metric list - grouped by HVG and DE
    metrics = [
        # === HVG Metrics ===
        ('HVG Loss', 'avg_hvg_loss', '.6f'),
        ('PearsonΔ(HVG)', 'pearson_delta_hvg', '.4f'),
        ('DiscScore(HVG)', 'disc_score_hvg', '.4f'),
        ('MSE(HVG)', 'mse_hvg', '.6f'),
        ('Wass(HVG)', 'wasserstein_hvg', '.6f'),
        ('DirAcc(HVG)', 'avg_hvg_dir_acc', '.3f'),
        # === DE Metrics ===
        (f'PearsonΔ(DE-{de_topk})', 'pearson_delta_de', '.4f'),
        (f'MSE(DE-{de_topk})', 'mse_de', '.6f'),
        (f'Wass(DE-{de_topk})', 'wasserstein_de', '.6f'),
        (f'DirAcc(DE-{de_topk})', 'avg_de_dir_acc', '.3f'),
        (f'DE Overlap-{de_topk}', 'avg_de_overlap', '.3f'),
    ]

    print("\n" + "="*110)
    print("📊 TRAINMEAN BASELINE - PER CELL LINE RESULTS")
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

    print("\n" + "="*230)
    print("📊 TRAINMEAN BASELINE - PER DRUG RESULTS")
    print("="*230)

    header = (
        f"{'Drug Name':<25} {'SMILES':<35} {'#Cells':>6} {'#Samples':>8} "
        f"{'HVG Loss':>10} {'PearsonΔ(HVG)':>14} {'DiscScore':>10} {'MSE(HVG)':>10} {'Wass(HVG)':>10} {'DirAcc(HVG)':>11} "
        f"{'PearsonΔ(DE)':>13} {'MSE(DE)':>9} {'Wass(DE)':>9} {'DirAcc(DE)':>11} {'DE Overlap':>10}"
    )
    print(header)
    print("-"*230)

    for smile in sorted_smiles:
        results = per_drug_results[smile]
        drug_names = results['drug_names']
        if len(drug_names) == 1:
            drug_name_str = drug_names[0]
        else:
            drug_name_str = f"{drug_names[0]} (+{len(drug_names)-1})"
        drug_name_short = drug_name_str[:22] + '...' if len(drug_name_str) > 25 else drug_name_str
        smiles_short = smile[:32] + '...' if len(smile) > 35 else smile

        row = (
            f"{drug_name_short:<25} {smiles_short:<35} {results['num_cell_lines']:>6} {results['num_samples']:>8} "
            f"{results['avg_hvg_loss']:>10.6f} {results['pearson_delta_hvg']:>14.4f} "
            f"{results['disc_score_hvg']:>10.4f} {results['mse_hvg']:>10.6f} {results['wasserstein_hvg']:>10.6f} {results['avg_hvg_dir_acc']:>11.3f} "
            f"{results['pearson_delta_de']:>13.4f} {results['mse_de']:>9.6f} {results['wasserstein_de']:>9.6f} {results['avg_de_dir_acc']:>11.3f} {results['avg_de_overlap']:>10.3f}"
        )
        print(row)

    print("="*230)

    total_row = (
        f"{'OVERALL AVERAGE':<25} {'-':<35} {'-':>6} {overall_avg['total_samples']:>8} "
        f"{overall_avg['avg_hvg_loss']:>10.6f} {overall_avg['pearson_delta_hvg']:>14.4f} "
        f"{overall_avg['disc_score_hvg']:>10.4f} {overall_avg['mse_hvg']:>10.6f} {overall_avg['wasserstein_hvg']:>10.6f} {overall_avg['avg_hvg_dir_acc']:>11.3f} "
        f"{overall_avg['pearson_delta_de']:>13.4f} {overall_avg['mse_de']:>9.6f} {overall_avg['wasserstein_de']:>9.6f} {overall_avg['avg_de_dir_acc']:>11.3f} {overall_avg['avg_de_overlap']:>10.3f}"
    )
    print(total_row)
    print("="*230)

    print(f"\n📈 Summary:")
    print(f"   Number of unique drugs: {len(per_drug_results)}")
    print(f"   Total samples evaluated: {overall_avg['total_samples']}")
    print(f"   Average samples per drug: {overall_avg['total_samples'] / len(per_drug_results):.1f}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="TrainMean Baseline Unified Evaluation (supports per cell line and per drug)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset arguments
    parser.add_argument("--data_base_dir", type=str,
                       default="/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/ours_6/preprocessed",
                       help="Base directory for preprocessed data")
    parser.add_argument("--se_inputs_base_dir", type=str, required=True,
                       help="Base directory for precomputed SE inputs (memmap)")
    parser.add_argument("--cell_lines", type=str, nargs='+',
                       default=['CVCL_0023', 'CVCL_0480', 'CVCL_0069',
                               'CVCL_0131', 'CVCL_1098', 'CVCL_1056'],
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

    # Step 1: Load internal test set and compute trainMean vector
    print("📊 Loading internal test dataset for computing TrainMean...")
    internal_dataset = TahoePerturbDatasetSE(
        cell_lines=args.cell_lines,
        split="internal_test",
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

    print(f"✅ Loaded {len(internal_dataset)} internal test samples")

    # Compute trainMean vector
    trainmean_vector = compute_trainmean_vector(internal_loader, device)

    # Step 2: Load external test set
    print("📊 Loading external test dataset for evaluation...")
    external_dataset = TahoePerturbDatasetSE(
        cell_lines=args.cell_lines,
        split="external_test",
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

    print(f"✅ Loaded {len(external_dataset)} external test samples\n")

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
        print("="*100)
        print("📋 FINAL SUMMARY - TrainMean Baseline")
        print("="*100)

        if 'cellline_results' in locals():
            print("\n===== PER CELL LINE OVERALL =====")
            ov = cellline_results['overall_avg']
            print("\n--- HVG Metrics ---")
            print(f"  HVG Loss:          {ov['avg_hvg_loss']:.6f}")
            print(f"  Pearson Delta:     {ov['pearson_delta_hvg']:.4f}")
            print(f"  Discrimination:    {ov['disc_score_hvg']:.4f}")
            print(f"  MSE (HVG):         {ov['mse_hvg']:.6f}")
            print(f"  Wasserstein:       {ov['wasserstein_hvg']:.6f}")
            print(f"  Direction Acc:     {ov['avg_hvg_dir_acc']:.3f}")

            print(f"\n--- DE-{args.de_topk} Metrics ---")
            print(f"  Pearson Delta:     {ov['pearson_delta_de']:.4f}")
            print(f"  MSE (DE):          {ov['mse_de']:.6f}")
            print(f"  Wasserstein:       {ov['wasserstein_de']:.6f}")
            print(f"  Direction Acc:     {ov['avg_de_dir_acc']:.3f}")
            print(f"  DE Overlap:        {ov['avg_de_overlap']:.3f}")

        if 'drug_results' in locals():
            print("\n===== PER DRUG OVERALL =====")
            ov = drug_results['overall_avg']
            print("\n--- HVG Metrics ---")
            print(f"  HVG Loss:          {ov['avg_hvg_loss']:.6f}")
            print(f"  Pearson Delta:     {ov['pearson_delta_hvg']:.4f}")
            print(f"  Discrimination:    {ov['disc_score_hvg']:.4f}")
            print(f"  MSE (HVG):         {ov['mse_hvg']:.6f}")
            print(f"  Wasserstein:       {ov['wasserstein_hvg']:.6f}")
            print(f"  Direction Acc:     {ov['avg_hvg_dir_acc']:.3f}")

            print(f"\n--- DE-{args.de_topk} Metrics ---")
            print(f"  Pearson Delta:     {ov['pearson_delta_de']:.4f}")
            print(f"  MSE (DE):          {ov['mse_de']:.6f}")
            print(f"  Wasserstein:       {ov['wasserstein_de']:.6f}")
            print(f"  Direction Acc:     {ov['avg_de_dir_acc']:.3f}")
            print(f"  DE Overlap:        {ov['avg_de_overlap']:.3f}")

        print("\n" + "="*100)

    print("\n✅ All evaluations completed!")
    return


if __name__ == '__main__':
    main()
