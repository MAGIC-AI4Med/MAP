# =========================
# File: chemCPA_eval_utils.py
# Evaluation utilities for chemCPA model (aligned with ours_6)
#
# Metrics for HVG:
#   - Pearson Delta Correlation
#   - Discrimination Score
#   - MSE
#   - Wasserstein Distance
#   - Direction Accuracy
#
# Metrics for DE (top-k differentially expressed genes):
#   - Pearson Delta Correlation
#   - MSE
#   - Wasserstein Distance
#   - Direction Accuracy
#   - DE Overlap (optional)
# =========================
import os
import sys
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, wasserstein_distance
from tqdm import tqdm

import torch
import torch.nn as nn


def compute_mse(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute mean squared error"""
    return np.mean((pred - true) ** 2)


def compute_discrimination_score_global(pred_means: np.ndarray, true_means: np.ndarray, metric: str = 'euclidean') -> float:
    """
    Global discrimination score (normalized inverse rank, higher is better).
    Definition: score = 1 - 2 * mean_rank / N, where rank_i is the relative rank
    of prediction i to its correct target i (strictly smaller distances).
    Random ~ 0, perfect -> 1.
    """
    if pred_means.shape[0] == 0:
        return float("nan")
    dist = cdist(pred_means, true_means, metric=metric)  # [N, N]
    N = dist.shape[0]
    ranks = []
    for i in range(N):
        di = dist[i]
        d_true = di[i]
        rank = np.sum(di < d_true)  # strictly smaller
        ranks.append(rank)
    mean_rank = np.mean(ranks) if len(ranks) > 0 else np.nan
    if np.isnan(mean_rank):
        return float("nan")
    return float(1.0 - 2.0 * (mean_rank / N))


def compute_pearson_scores(pred_hvg_means: np.ndarray,
                          true_hvg_means: np.ndarray,
                          control_hvg_means: np.ndarray):
    """
    Compute delta-based Pearson correlation:
    1. pearson_delta_mean: average of per-bulk delta Pearson correlation
    """
    N = pred_hvg_means.shape[0]
    if N == 0:
        return float("nan")

    # Compute delta vectors for all perturbations
    pred_deltas = pred_hvg_means - control_hvg_means  # [N, G]
    true_deltas = true_hvg_means - control_hvg_means  # [N, G]

    # 1. Per-bulk Pearson Delta Correlation, then average
    per_bulk_corrs = []
    for i in range(N):
        # Check for NaN/inf first
        if np.any(np.isnan(pred_deltas[i])) or np.any(np.isinf(pred_deltas[i])):
            continue
        if np.any(np.isnan(true_deltas[i])) or np.any(np.isinf(true_deltas[i])):
            continue
        if np.std(pred_deltas[i]) < 1e-8 or np.std(true_deltas[i]) < 1e-8:
            continue  # skip zero variance samples
        r, _ = pearsonr(pred_deltas[i], true_deltas[i])
        if not np.isnan(r):
            per_bulk_corrs.append(r)

    if len(per_bulk_corrs) > 0:
        pearson_delta_mean = np.mean(per_bulk_corrs)
    else:
        pearson_delta_mean = 0.0

    return pearson_delta_mean


@torch.no_grad()
def evaluate_chemCPA(
    model,
    val_loader,
    smiles_to_idx,
    device,
    de_topk=50,
    num_samples_per_comb=3,
    verbose=True,
):
    """
    Full evaluation function for chemCPA (aligned with ours_6 eval_full.py).

    For HVG:
    - Discrimination Score
    - Pearson Delta Correlation
    - MSE
    - Wasserstein Distance
    - Direction Accuracy

    For DE (top-k differentially expressed genes):
    - Pearson Delta Correlation
    - MSE
    - Wasserstein Distance
    - Direction Accuracy

    Args:
        model: chemCPA LightningModule
        val_loader: Validation DataLoader
        smiles_to_idx: dict mapping SMILES to index
        device: compute device
        de_topk: Top-K DE genes number
        num_samples_per_comb: Number of samples per combination (for Wasserstein)
        verbose: Whether to print detailed info

    Returns:
        dict: per cell line results and overall average
    """
    model.eval()

    # Group data by cell line
    cell_line_data = defaultdict(lambda: {
        'num_batches': 0,
        # Basic metrics storage
        'pred_hvg_means': [],
        'true_hvg_means': [],
        'control_hvg_means': [],
        'hvg_dir_accs': [],
        'de_dir_accs': [],
        'de_delta_pred_all': [],
        'de_delta_true_all': [],
        'de_overlaps': [],
        # For Wasserstein distance: store all predicted and true gene expressions
        'all_pred_hvg': [],   # all predicted HVG values [N, G]
        'all_true_hvg': [],   # all true HVG values [N, G]
        'all_control_hvg': [], # all control HVG values [N, G]
        'all_de_indices': [], # store top-k true DE indices per sample
    })

    total_combinations = len(val_loader.dataset)
    if verbose:
        print(f"Processing {total_combinations} cell line-perturbation combinations, "
              f"sampling {num_samples_per_comb} times each...")

    # For each cell line-perturbation combination, sample N times for Wasserstein
    for comb_idx in tqdm(range(total_combinations), disable=not verbose):
        for sample_idx in range(num_samples_per_comb):
            # Get sample: multiple random sampling of control cells for each combination
            batch_single = val_loader.dataset[comb_idx]

            # Reformat to batch (batch size=1)
            genes = batch_single["control_hvg_vectors"].unsqueeze(0).to(device)  # [1, set_size, G]
            target_hvg = batch_single["perturb_hvg_vectors"].unsqueeze(0).to(device)  # [1, set_size, G]

            # Look up drug index
            smi = batch_single["drug_smiles"]
            if smi not in smiles_to_idx:
                # If SMILES not in vocabulary (should not happen), skip
                continue
            drug_idx = smiles_to_idx[smi]

            # Build drugs_idx and dosages tensors - expand to set_size
            set_size = genes.shape[1]
            drugs_idx = torch.tensor([drug_idx] * set_size, dtype=torch.long, device=device).unsqueeze(0)  # [1, set_size]
            dosages = torch.tensor([batch_single["drug_conc"]] * set_size, dtype=torch.float32, device=device).unsqueeze(0)  # [1, set_size]

            # Build covariates (cell line one-hot)
            num_cell_lines = batch_single["cell_line_ohe"].shape[0]
            covariates = [batch_single["cell_line_ohe"].unsqueeze(0).repeat(set_size, 1).to(device)]  # list of [1*set_size, num_cell_lines]

            # Flatten inputs: batch * set_size as new batch dimension
            B, S, G = genes.shape
            genes_flat = genes.reshape(B * S, G)
            drugs_idx_flat = drugs_idx.reshape(B * S)
            dosages_flat = dosages.reshape(B * S)
            covariates_flat = [cov.reshape(B * S, num_cell_lines) for cov in covariates]

            # Forward pass: predict returns (gene_reconstructions, cell_drug_embedding)
            # where gene_reconstructions = cat([mean, var], dim=1) -> shape [B*S, 2*G]
            gene_reconstructions_flat, _ = model.model.predict(
                genes=genes_flat,
                drugs_idx=drugs_idx_flat,
                dosages=dosages_flat,
                covariates=covariates_flat,
            )

            # Split to get only mean (first G dimensions)
            dim = gene_reconstructions_flat.size(1) // 2
            pred_mean_flat = gene_reconstructions_flat[:, :dim]  # [B*S, G]

            # Reshape back: [B, S, G] -> average over set_size dimension
            pred_mean = pred_mean_flat.reshape(B, S, G)

            # Compute pseudobulk means (average over set_size)
            pred_hvg_mean_np = pred_mean.mean(dim=1).float().cpu().numpy()[0]  # [G]
            true_hvg_mean_np = target_hvg.mean(dim=1).float().cpu().numpy()[0]  # [G]
            control_hvg_mean_np = genes.mean(dim=1).float().cpu().numpy()[0]  # [G]

            cell_line = batch_single["cell_line"]
            cl_data = cell_line_data[cell_line]

            # Store all predictions and true values for Wasserstein
            cl_data['all_pred_hvg'].append(pred_hvg_mean_np)
            cl_data['all_true_hvg'].append(true_hvg_mean_np)
            cl_data['all_control_hvg'].append(control_hvg_mean_np)

            # Only accumulate basic metrics once per combination (not per sample)
            if sample_idx == 0:
                cl_data['num_batches'] += 1
                cl_data['pred_hvg_means'].append(pred_hvg_mean_np[None, :])
                cl_data['true_hvg_means'].append(true_hvg_mean_np[None, :])
                cl_data['control_hvg_means'].append(control_hvg_mean_np[None, :])

                # Pseudobulk Δ
                delta_true = true_hvg_mean_np - control_hvg_mean_np
                delta_pred = pred_hvg_mean_np - control_hvg_mean_np
                G = delta_true.shape[0]

                # HVG direction accuracy
                if G > 0:
                    sign_true_all = np.sign(delta_true)
                    sign_pred_all = np.sign(delta_pred)
                    hvg_dir_acc = (sign_true_all == sign_pred_all).mean()
                    cl_data['hvg_dir_accs'].append(hvg_dir_acc)

                # top-k DE processing
                if G > 0 and de_topk > 0:
                    k = min(de_topk, G)

                    abs_dt = np.abs(delta_true)
                    top_idx_true = np.argsort(-abs_dt)[:k]
                    cl_data['all_de_indices'].append(top_idx_true)

                    abs_dp = np.abs(delta_pred)
                    top_idx_pred = np.argsort(-abs_dp)[:k]

                    # DE overlap
                    overlap = len(set(top_idx_true) & set(top_idx_pred)) / k
                    cl_data['de_overlaps'].append(overlap)

                    # DE direction accuracy
                    dt_top = delta_true[top_idx_true]
                    dp_top = delta_pred[top_idx_true]
                    sign_true = np.sign(dt_top)
                    sign_pred = np.sign(dp_top)
                    de_dir_acc = (sign_true == sign_pred).mean()
                    cl_data['de_dir_accs'].append(de_dir_acc)

                    # Cache Δ and expression values
                    cl_data['de_delta_true_all'].append(dt_top[None, :])
                    cl_data['de_delta_pred_all'].append(dp_top[None, :])

    # ===== Compute full metrics for each cell line =====
    per_cell_line_results = {}

    for cell_line, cl_data in cell_line_data.items():
        results = _compute_all_full_metrics(cl_data, de_topk)
        per_cell_line_results[cell_line] = results

    # ===== Compute overall average =====
    overall_avg = _compute_overall_average(per_cell_line_results)

    # ===== Table-formatted output =====
    if verbose:
        _print_results_as_full_table(per_cell_line_results, overall_avg, de_topk, num_samples_per_comb)

    return {
        'per_cell_line': per_cell_line_results,
        'overall_avg': overall_avg,
    }


def _compute_all_full_metrics(data, de_topk):
    """Compute all full metrics from accumulated data"""
    num_batches = data['num_batches']
    if num_batches == 0:
        return {}

    # Merge basic metrics data
    pred_hvg_means = np.concatenate(data['pred_hvg_means'], axis=0)
    true_hvg_means = np.concatenate(data['true_hvg_means'], axis=0)
    control_hvg_means = np.concatenate(data['control_hvg_means'], axis=0)

    # Merge all data for Wasserstein
    all_pred_hvg = np.array(data['all_pred_hvg'])  # [N*samples, G]
    all_true_hvg = np.array(data['all_true_hvg'])

    # ==============================
    # 1. HVG Metrics
    # ==============================

    # 1.1 HVG Direction Accuracy
    avg_hvg_dir_acc = float(np.mean(data['hvg_dir_accs'])) if len(data['hvg_dir_accs']) > 0 else 0.0

    # 1.2 HVG MSE
    mse_hvg = compute_mse(pred_hvg_means, true_hvg_means)

    # 1.3 HVG Discrimination Score (global ranking)
    disc_hvg = compute_discrimination_score_global(pred_hvg_means, true_hvg_means, 'euclidean')

    # 1.4 HVG Pearson Delta Correlation
    pearson_hvg = compute_pearson_scores(pred_hvg_means, true_hvg_means, control_hvg_means)

    # 1.5 HVG Wasserstein Distance (per-gene, then average)
    G_genes = all_pred_hvg.shape[1] if len(all_pred_hvg) > 0 else 0
    wass_hvg_per_gene = []

    for g in range(G_genes):
        pred_dist = all_pred_hvg[:, g]
        true_dist = all_true_hvg[:, g]
        if len(pred_dist) > 0 and len(true_dist) > 0:
            wd = wasserstein_distance(pred_dist, true_dist)
            wass_hvg_per_gene.append(wd)

    wasserstein_hvg = float(np.mean(wass_hvg_per_gene)) if len(wass_hvg_per_gene) > 0 else 0.0

    # ==============================
    # 2. DE Metrics (top-k DE genes)
    # ==============================

    # 2.1 DE Direction Accuracy
    avg_de_dir_acc = float(np.mean(data['de_dir_accs'])) if len(data['de_dir_accs']) > 0 else 0.0

    # 2.2 DE Overlap
    avg_de_overlap = float(np.mean(data['de_overlaps'])) if len(data['de_overlaps']) > 0 else 0.0

    # Compute DE-specific metrics from cached data
    de_mse = 0.0
    pearson_de = 0.0
    wasserstein_de = 0.0

    if len(data['de_delta_true_all']) > 0:
        de_delta_true_all = np.concatenate(data['de_delta_true_all'], axis=0)  # [N, k]
        de_delta_pred_all = np.concatenate(data['de_delta_pred_all'], axis=0)  # [N, k]

        # 2.3 DE MSE
        de_mse = compute_mse(de_delta_pred_all, de_delta_true_all)

        # 2.4 DE Pearson Delta Correlation (at true DE positions)
        N_de = de_delta_pred_all.shape[0]
        per_sample_de_corrs = []
        for i in range(N_de):
            pred_de = de_delta_pred_all[i]
            true_de = de_delta_true_all[i]
            if np.std(pred_de) < 1e-8 or np.std(true_de) < 1e-8:
                continue
            r, _ = pearsonr(pred_de, true_de)
            if not np.isnan(r):
                per_sample_de_corrs.append(r)
        pearson_de = np.mean(per_sample_de_corrs) if len(per_sample_de_corrs) > 0 else 0.0

        # 2.5 DE Wasserstein Distance (only at DE indices)
        wass_de_list = []
        if len(data['all_de_indices']) > 0:
            all_de_indices = np.array(data['all_de_indices'])  # [N, k]
            N, k = all_de_indices.shape

            # For each DE position index (among top-k), compute Wasserstein across samples
            for de_pos_idx in range(k):
                gene_idx = all_de_indices[:, de_pos_idx]  # [N]: gene index for each sample
                # Get values at those gene indices
                pred_vals = all_pred_hvg[np.arange(N), gene_idx]
                true_vals = all_true_hvg[np.arange(N), gene_idx]
                wd = wasserstein_distance(pred_vals, true_vals)
                wass_de_list.append(wd)

            wasserstein_de = float(np.mean(wass_de_list)) if len(wass_de_list) > 0 else 0.0

    return {
        'num_batches': num_batches,
        # HVG metrics
        'hvg_dir_acc': avg_hvg_dir_acc,
        'hvg_mse': mse_hvg,
        'hvg_disc': disc_hvg,
        'hvg_pearson': pearson_hvg,
        'hvg_wasserstein': wasserstein_hvg,
        # DE metrics
        'de_dir_acc': avg_de_dir_acc,
        'de_overlap': avg_de_overlap,
        'de_mse': de_mse,
        'de_pearson': pearson_de,
        'de_wasserstein': wasserstein_de,
    }


def _compute_overall_average(per_cell_line_results):
    """Compute overall weighted average across all cell lines"""
    if len(per_cell_line_results) == 0:
        return {}

    # Initialize accumulators
    overall = {
        'hvg_dir_acc': 0.0, 'hvg_mse': 0.0, 'hvg_disc': 0.0,
        'hvg_pearson': 0.0, 'hvg_wasserstein': 0.0,
        'de_dir_acc': 0.0, 'de_overlap': 0.0, 'de_mse': 0.0,
        'de_pearson': 0.0, 'de_wasserstein': 0.0,
    }

    total_batches = sum(cl_data.get('num_batches', 0) for cl_data in per_cell_line_results.values())

    if total_batches == 0:
        return overall

    # Weighted average by number of batches per cell line
    for cell_line, cl_data in per_cell_line_results.items():
        weight = cl_data.get('num_batches', 0) / total_batches
        for key in overall:
            if key in cl_data:
                overall[key] += weight * cl_data[key]

    return overall


def _print_results_as_full_table(per_cell_line_results, overall_avg, de_topk, num_samples_per_comb):
    """Print results in formatted table (aligned with ours_6)"""
    print("\n" + "=" * 110)
    print(f"chemCPA FULL EVALUATION RESULTS (de_topk={de_topk}, samples_per_comb={num_samples_per_comb})")
    print("=" * 110)
    print()

    # Print per cell line header
    print(f"{'Metric':<25s}", end="")
    for cl in sorted(per_cell_line_results.keys()):
        print(f"{cl:<15s}", end="")
    print(f"{'Avg':<15s}")
    print("-" * 110)

    # HVG metrics
    print("📊 HVG Metrics:")
    metric_names = [
        ('hvg_pearson', 'Pearson Delta'),
        ('hvg_disc', 'Discrimination'),
        ('hvg_mse', 'MSE'),
        ('hvg_wasserstein', 'Wasserstein'),
        ('hvg_dir_acc', 'Direction Acc'),
    ]

    for key, name in metric_names:
        print(f"  {name:<23s}", end="")
        for cl in sorted(per_cell_line_results.keys()):
            val = per_cell_line_results[cl].get(key, 0.0)
            print(f"{val:<15.4f}", end="")
        avg_val = overall_avg.get(key, 0.0)
        print(f"{avg_val:<15.4f}")

    print()

    # DE metrics
    print(f"📊 DE Metrics (top-{de_topk}):")
    de_metric_names = [
        ('de_pearson', 'Pearson Delta'),
        ('de_mse', 'MSE'),
        ('de_wasserstein', 'Wasserstein'),
        ('de_dir_acc', 'Direction Acc'),
        ('de_overlap', 'DE Overlap'),
    ]

    for key, name in de_metric_names:
        print(f"  {name:<23s}", end="")
        for cl in sorted(per_cell_line_results.keys()):
            val = per_cell_line_results[cl].get(key, 0.0)
            print(f"{val:<15.4f}", end="")
        avg_val = overall_avg.get(key, 0.0)
        print(f"{avg_val:<15.4f}")

    print()
    print("=" * 110)
    print()
