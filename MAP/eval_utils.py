import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

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