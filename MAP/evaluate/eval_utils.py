import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, ttest_ind


def get_significant_deg_mask(control_hvg: np.ndarray,
                              perturb_hvg: np.ndarray,
                              pvalue_threshold: float = 0.05) -> np.ndarray:
    """
    对每个HVG做统计检验，返回显著基因的mask

    Args:
        control_hvg: shape [S_ctrl, G]，control组cell-level HVG表达
        perturb_hvg: shape [S_pert, G]，perturbed组cell-level HVG表达
        pvalue_threshold: p值阈值

    Returns:
        mask: shape [G]，True表示该基因统计显著
    """
    G = control_hvg.shape[1]
    p_values = np.zeros(G)

    for g in range(G):
        ctrl_vals = control_hvg[:, g]
        pert_vals = perturb_hvg[:, g]

        # Welch's t-test (不假设方差相等)
        _, p_val = ttest_ind(ctrl_vals, pert_vals, equal_var=False, nan_policy='omit')
        p_values[g] = p_val

    return p_values < pvalue_threshold

def compute_batch_pearson_delta_for_train(pred_hvg_mean, true_hvg_mean, control_hvg_mean):
    pred_hvg_np = pred_hvg_mean.detach().cpu().numpy()
    true_hvg_np = true_hvg_mean.detach().cpu().numpy()
    control_hvg_np = control_hvg_mean.detach().cpu().numpy()
    
    pred_deltas = pred_hvg_np - control_hvg_np
    true_deltas = true_hvg_np - control_hvg_np
    
    from scipy.stats import pearsonr
    per_sample_corrs = []
    for i in range(pred_deltas.shape[0]):
        if np.std(pred_deltas[i]) < 1e-8 or np.std(true_deltas[i]) < 1e-8:
            continue
        r, _ = pearsonr(pred_deltas[i], true_deltas[i])
        if not np.isnan(r):
            per_sample_corrs.append(r)
    
    return np.mean(per_sample_corrs) if len(per_sample_corrs) > 0 else 0.0


def compute_discrimination_score_global(pred_means: np.ndarray, true_means: np.ndarray, metric: str) -> float:
    if pred_means.shape[0] == 0:
        return float("nan")
    dist = cdist(pred_means, true_means, metric=metric)
    N = dist.shape[0]
    ranks = []
    for i in range(N):
        di = dist[i]
        d_true = di[i]
        rank = np.sum(di < d_true)
        ranks.append(rank)
    mean_rank = np.mean(ranks) if len(ranks) > 0 else np.nan
    if np.isnan(mean_rank):
        return float("nan")
    return float(1.0 - 2.0 * (mean_rank / N))


def compute_pearson_scores(pred_hvg_means: np.ndarray, 
                          true_hvg_means: np.ndarray,
                          control_hvg_means: np.ndarray):
    N = pred_hvg_means.shape[0]
    if N == 0:
        return float("nan")
    
    pred_deltas = pred_hvg_means - control_hvg_means 
    true_deltas = true_hvg_means - control_hvg_means 
    
    per_bulk_corrs = []
    for i in range(N):
        if np.std(pred_deltas[i]) < 1e-8 or np.std(true_deltas[i]) < 1e-8:
            continue
        r, _ = pearsonr(pred_deltas[i], true_deltas[i])
        per_bulk_corrs.append(r)
    
    pearson_delta_mean = np.mean(per_bulk_corrs) if len(per_bulk_corrs) > 0 else float("nan")
    
    return pearson_delta_mean
