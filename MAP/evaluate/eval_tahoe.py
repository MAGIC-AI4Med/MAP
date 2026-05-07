# =========================
# File: eval_full.py
# Full evaluation script with ALL metrics:
# - HVG: Discrimination, Pearson delta, MSE, Wasserstein, Direction Accuracy
# - DEG: Pearson delta, MSE, Wasserstein, Direction Accuracy
# =========================
import os
import argparse
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import wasserstein_distance

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.ds_multi_cell_lora_se import TahoePerturbDatasetSE
from model.model import MAPmodel
from eval_utils import compute_discrimination_score_global, compute_pearson_scores


def compute_mse(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute mean squared error"""
    return np.mean((pred - true) ** 2)


@torch.no_grad()
def evaluate_model_full(
    model,
    test_loader,
    loss_fn_emb,
    loss_fn_hvg,
    hvg_weight,
    device,
    de_topk=50,
    num_samples_per_comb=1,
    verbose=True,
):
    """
    完整的多细胞系评估函数，包含所有指标：

    For HVG:
    - Discrimination Score
    - Pearson Delta Correlation
    - MSE
    - Wasserstein Distance
    - Direction Accuracy

    For DEG (top-k):
    - Pearson Delta Correlation
    - MSE
    - Wasserstein Distance
    - Direction Accuracy

    Args:
        model: 模型
        test_loader: DataLoader
        loss_fn_emb: Embedding损失函数
        loss_fn_hvg: HVG损失函数
        hvg_weight: HVG损失权重
        device: 计算设备
        de_topk: Top-K DE基因数量
        num_samples_per_comb: 每个cell line-perturbation组合采样多少次（用于Wasserstein）
        verbose: 是否打印详细信息

    Returns:
        dict: 每个细胞系的评测结果和总体平均
    """
    model.eval()

    # 按细胞系分组存储数据
    cell_line_data = defaultdict(lambda: {
        'total_loss': 0.0,
        'emb_loss': 0.0,
        'hvg_loss': 0.0,
        'num_batches': 0,
        # 基础metrics存储
        'pred_hvg_means': [],
        'true_hvg_means': [],
        'control_hvg_means': [],
        'hvg_dir_accs': [],
        'de_dir_accs': [],
        'de_delta_pred_all': [],
        'de_delta_true_all': [],
        'de_expr_pred_all': [],
        'de_expr_true_all': [],
        'de_overlaps': [],
        # 为Wasserstein距离存储：所有预测和真实的基因表达值
        'all_pred_hvg': [],   # 所有预测HVG表达值 [N, G]
        'all_true_hvg': [],   # 所有真实HVG表达值 [N, G]
        'all_control_hvg': [], # 所有control HVG表达值 [N, G]
        'all_de_indices': [], # 存储每个样本的top-k true DE indices
    })

    total_combinations = len(test_loader.dataset)
    print(f"Processing {total_combinations} cell line-perturbation combinations, "
          f"sampling {num_samples_per_comb} times each...")

    # 对每个cell line-perturbation组合，采样N次用于Wasserstein
    for comb_idx in tqdm(range(total_combinations), disable=not verbose):
        for sample_idx in range(num_samples_per_comb):
            # 获取样本：每个组合多次随机采样control cells
            batch = test_loader.dataset[comb_idx]

            # 整理成batch格式 (batch size=1)
            ctrl_gene = batch["control_gene_ids"].unsqueeze(0).to(device)
            ctrl_expr = batch["control_expressions"].unsqueeze(0).to(device)
            perturb_embeddings = batch["perturb_embeddings"].unsqueeze(0).to(device).float()
            perturb_hvg_vectors = batch["perturb_hvg_vectors"].unsqueeze(0).to(device)
            smiles = [batch["drug_smiles"]]
            conc = torch.tensor([batch["drug_conc"]], dtype=torch.float32, device=device)
            cell_line = batch["cell_line"]

            # 前向传播
            pred_embs, pred_hvg = model(ctrl_gene, ctrl_expr, smiles, conc)

            # 计算损失
            pred_emb_mean = pred_embs.mean(dim=1)
            true_emb_mean = perturb_embeddings.mean(dim=1)
            emb_loss = loss_fn_emb(pred_emb_mean, true_emb_mean)

            # HVG loss
            true_hvg = perturb_hvg_vectors.to(device)
            pred_hvg_mean = pred_hvg.mean(dim=1)
            true_hvg_mean = true_hvg.mean(dim=1)

            if loss_fn_hvg is None:
                hvg_loss = loss_fn_emb(pred_hvg_mean, true_hvg_mean)
            else:
                hvg_loss = loss_fn_hvg(pred_hvg_mean, true_hvg_mean)

            total_loss = emb_loss + hvg_weight * hvg_loss

            # 转换为numpy
            pred_hvg_mean_np = pred_hvg_mean.float().cpu().numpy()[0]  # [G]
            true_hvg_mean_np = true_hvg_mean.float().cpu().numpy()[0]  # [G]
            control_hvg_mean_np = batch["control_hvg_vectors"].to(device).mean(dim=0).float().cpu().numpy()  # [G]

            # pseudobulk Δ
            delta_true = true_hvg_mean_np - control_hvg_mean_np
            delta_pred = pred_hvg_mean_np - control_hvg_mean_np

            G = delta_true.shape[0]

            cl_data = cell_line_data[cell_line]

            # 累积损失（只累积一次每个组合，不是每个采样）
            if sample_idx == 0:
                cl_data['num_batches'] += 1
                cl_data['total_loss'] += total_loss.item()
                cl_data['emb_loss'] += emb_loss.item()
                cl_data['hvg_loss'] += hvg_loss.item()

            # 存储所有预测和真实值
            cl_data['all_pred_hvg'].append(pred_hvg_mean_np)
            cl_data['all_true_hvg'].append(true_hvg_mean_np)
            cl_data['all_control_hvg'].append(control_hvg_mean_np)

            # 只在第一次采样时存储基础metrics（避免重复）
            if sample_idx == 0:
                cl_data['pred_hvg_means'].append(pred_hvg_mean_np[None, :])
                cl_data['true_hvg_means'].append(true_hvg_mean_np[None, :])
                cl_data['control_hvg_means'].append(control_hvg_mean_np[None, :])

                # HVG方向准确率
                if G > 0:
                    sign_true_all = np.sign(delta_true)
                    sign_pred_all = np.sign(delta_pred)
                    hvg_dir_acc = (sign_true_all == sign_pred_all).mean()
                    cl_data['hvg_dir_accs'].append(hvg_dir_acc)

                # top-k DE 处理
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

                    # 缓存 Δ 和表达
                    cl_data['de_delta_true_all'].append(dt_top[None, :])
                    cl_data['de_delta_pred_all'].append(dp_top[None, :])
                    t_expr_top = true_hvg_mean_np[top_idx_true]
                    p_expr_top = pred_hvg_mean_np[top_idx_true]
                    cl_data['de_expr_true_all'].append(t_expr_top[None, :])
                    cl_data['de_expr_pred_all'].append(p_expr_top[None, :])

    # ===== 计算每个细胞系的完整指标 =====
    per_cell_line_results = {}

    for cell_line, cl_data in cell_line_data.items():
        results = _compute_all_full_metrics(cl_data, de_topk)
        per_cell_line_results[cell_line] = results

    # ===== 计算总体平均 =====
    overall_avg = _compute_overall_average(per_cell_line_results)

    # ===== 表格形式输出 =====
    if verbose:
        _print_results_as_full_table(per_cell_line_results, overall_avg, de_topk, num_samples_per_comb)

    return {
        'per_cell_line': per_cell_line_results,
        'overall_avg': overall_avg,
    }


def _compute_all_full_metrics(data, de_topk):
    """从累积的数据中计算所有完整指标"""
    num_batches = data['num_batches']

    # 平均损失
    avg_total_loss = data['total_loss'] / num_batches if num_batches > 0 else 0.0
    avg_emb_loss = data['emb_loss'] / num_batches if num_batches > 0 else 0.0
    avg_hvg_loss = data['hvg_loss'] / num_batches if num_batches > 0 else 0.0

    # 合并基础metrics需要的数据
    pred_hvg_means = np.concatenate(data['pred_hvg_means'], axis=0)
    true_hvg_means = np.concatenate(data['true_hvg_means'], axis=0)
    control_hvg_means = np.concatenate(data['control_hvg_means'], axis=0)

    # ==============================
    # 1. HVG 指标
    # ==============================

    # 1.1 HVG Direction Accuracy
    avg_hvg_dir_acc = float(np.mean(data['hvg_dir_accs'])) if len(data['hvg_dir_accs']) > 0 else 0.0

    # 1.2 HVG MSE
    mse_hvg = compute_mse(pred_hvg_means, true_hvg_means)

    # 1.3 HVG Discrimination Score (全局排序)
    disc_hvg = compute_discrimination_score_global(pred_hvg_means, true_hvg_means, 'euclidean')

    # 1.4 HVG Pearson Delta Correlation
    pearson_delta_hvg = compute_pearson_scores(pred_hvg_means, true_hvg_means, control_hvg_means)

    # ==============================
    # 2. DEG 指标 (top-k)
    # ==============================

    mse_de = 0.0
    avg_de_dir_acc = 0.0
    avg_de_overlap = 0.0
    pearson_delta_de = 0.0

    if len(data['de_delta_pred_all']) > 0:
        # DE MSE
        de_delta_pred_mat = np.vstack(data['de_delta_pred_all'])
        de_delta_true_mat = np.vstack(data['de_delta_true_all'])
        mse_de = compute_mse(de_delta_pred_mat, de_delta_true_mat)

        # DE Direction Accuracy
        avg_de_dir_acc = float(np.mean(data['de_dir_accs'])) if len(data['de_dir_accs']) > 0 else 0.0

        # DE Overlap
        avg_de_overlap = float(np.mean(data['de_overlaps'])) if len(data['de_overlaps']) > 0 else 0.0

        # DE Pearson Delta Correlation (在DE基因位置上计算)
        # 这里我们重新收集所有样本的delta值，然后在DE位置上计算
        N = pred_hvg_means.shape[0]
        pred_deltas = pred_hvg_means - control_hvg_means
        true_deltas = true_hvg_means - control_hvg_means

        de_indices_all = data['all_de_indices']  # [N, k]
        per_sample_de_pearson = []
        for i in range(N):
            de_idx = de_indices_all[i]
            pred_de_delta = pred_deltas[i][de_idx]
            true_de_delta = true_deltas[i][de_idx]
            if np.std(pred_de_delta) < 1e-8 or np.std(true_de_delta) < 1e-8:
                continue
            from scipy.stats import pearsonr
            r, _ = pearsonr(pred_de_delta, true_de_delta)
            if not np.isnan(r):
                per_sample_de_pearson.append(r)
        pearson_delta_de = np.mean(per_sample_de_pearson) if len(per_sample_de_pearson) > 0 else 0.0

    # ==============================
    # 3. Wasserstein 距离
    # ==============================

    all_pred_hvg = np.array(data['all_pred_hvg'])  # [N*samples, G]
    all_true_hvg = np.array(data['all_true_hvg'])  # [N*samples, G]
    all_control_hvg = np.array(data['all_control_hvg'])  # [N*samples, G]

    wasserstein_hvg = 0.0
    wasserstein_de = 0.0

    if len(all_pred_hvg) > 0:
        N_samples, G_genes = all_pred_hvg.shape

        # 3.1 HVG Wasserstein - 每个基因位置计算Wasserstein距离
        wass_hvg_per_gene = []
        for g in range(G_genes):
            pred_dist = all_pred_hvg[:, g]
            true_dist = all_true_hvg[:, g]
            wd = wasserstein_distance(pred_dist, true_dist)
            wass_hvg_per_gene.append(wd)
        wasserstein_hvg = float(np.mean(wass_hvg_per_gene)) if G_genes > 0 else 0.0

        # 3.2 DE Wasserstein - DE基因位置的Wasserstein距离平均
        if len(data['all_de_indices']) > 0:
            wass_de_list = []
            # 对每个样本的每个DE位置，取出该基因的Wasserstein距离
            for de_indices in data['all_de_indices']:
                for g in de_indices:
                    wass_de_list.append(wass_hvg_per_gene[g])
            wasserstein_de = float(np.mean(wass_de_list)) if len(wass_de_list) > 0 else 0.0

    return {
        'num_batches': num_batches,
        # Loss metrics
        'avg_total_loss': avg_total_loss,
        'avg_emb_loss': avg_emb_loss,
        'avg_hvg_loss': avg_hvg_loss,
        # ============================
        # HVG metrics
        # ============================
        'hvg_disc': disc_hvg,
        'hvg_pearson_delta': pearson_delta_hvg,
        'hvg_mse': mse_hvg,
        'hvg_wasserstein': wasserstein_hvg,
        'hvg_dir_acc': avg_hvg_dir_acc,
        # ============================
        # DEG metrics (top-k)
        # ============================
        'de_pearson_delta': pearson_delta_de,
        'de_mse': mse_de,
        'de_wasserstein': wasserstein_de,
        'de_dir_acc': avg_de_dir_acc,
        'de_overlap': avg_de_overlap,  # 额外保留
    }


def _compute_overall_average(per_cell_line_results):
    """计算所有细胞系的总体平均"""
    if not per_cell_line_results:
        return {}

    all_metrics = list(next(iter(per_cell_line_results.values())).keys())
    overall_avg = {}

    for metric_key in all_metrics:
        values = [res[metric_key] for res in per_cell_line_results.values()]
        overall_avg[metric_key] = np.mean(values)

    return overall_avg


def _print_results_as_full_table(per_cell_line_results, overall_avg, de_topk, num_samples_per_comb):
    """以表格形式打印完整的评测结果"""

    if not per_cell_line_results:
        print("No results to display.")
        return

    cell_lines = sorted(per_cell_line_results.keys())

    # 分两部分显示：HVG指标 和 DEG指标
    hvg_metrics = [
        ('Disc(HVG)', 'hvg_disc', '.4f'),
        ('Pearson Δ(HVG)', 'hvg_pearson_delta', '.4f'),
        ('MSE(HVG)', 'hvg_mse', '.6f'),
        ('Wass(HVG)', 'hvg_wasserstein', '.6f'),
        ('DirAcc(HVG)', 'hvg_dir_acc', '.4f'),
    ]

    deg_metrics = [
        ('Pearson Δ(DE)', 'de_pearson_delta', '.4f'),
        ('MSE(DE)', 'de_mse', '.6f'),
        ('Wass(DE)', 'de_wasserstein', '.6f'),
        ('DirAcc(DE)', 'de_dir_acc', '.4f'),
        (f'Overlap(DE-{de_topk})', 'de_overlap', '.4f'),
    ]

    print("\n" + "="*150)
    print("📊 FULL EVALUATION RESULTS (ALL METRICS)")
    print(f"   Dataset: {len(cell_lines)} cell lines, {num_samples_per_comb} samples per combination for Wasserstein")
    print("="*150)

    metric_col_width = max(len(m[0]) for m in hvg_metrics + deg_metrics) + 2
    cell_col_width = 14
    avg_col_width = 16

    # ============================
    # HVG指标部分
    # ============================
    print("\n[ HVG Metrics ]")
    print("-"*150)
    header = f"{'Metric':<{metric_col_width}}"
    for cl in cell_lines:
        header += f"{cl:>{cell_col_width}}"
    header += f"{'OVERALL AVG':>{avg_col_width}}"
    print(header)
    print("-"*150)

    for metric_name, metric_key, fmt in hvg_metrics:
        row = f"{metric_name:<{metric_col_width}}"
        for cl in cell_lines:
            value = per_cell_line_results[cl][metric_key]
            row += f"{value:>{cell_col_width}{fmt}}"
        avg_value = overall_avg[metric_key]
        row += f"{avg_value:>{avg_col_width}{fmt}}"
        print(row)

    # ============================
    # DEG指标部分
    # ============================
    print("\n[ DEG Metrics (top-{}) ]".format(de_topk))
    print("-"*150)
    header = f"{'Metric':<{metric_col_width}}"
    for cl in cell_lines:
        header += f"{cl:>{cell_col_width}}"
    header += f"{'OVERALL AVG':>{avg_col_width}}"
    print(header)
    print("-"*150)

    for metric_name, metric_key, fmt in deg_metrics:
        row = f"{metric_name:<{metric_col_width}}"
        for cl in cell_lines:
            value = per_cell_line_results[cl][metric_key]
            row += f"{value:>{cell_col_width}{fmt}}"
        avg_value = overall_avg[metric_key]
        row += f"{avg_value:>{avg_col_width}{fmt}}"
        print(row)

    print("="*150 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Full evaluation with ALL metrics for drug response prediction.")

    # Required arguments
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Experiment name (checkpoint will be loaded from checkpoints/{exp_name})")
    parser.add_argument("--checkpoint_epoch", type=int, required=True,
                        help="Epoch number of the checkpoint to load")

    # Dataset arguments
    parser.add_argument("--data_base_dir", type=str, required=True)
    parser.add_argument("--se_inputs_base_dir", type=str, required=True)
    parser.add_argument("--cell_lines", type=str, nargs='+', required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--set_size", type=int, default=128)
    parser.add_argument('--UC', action='store_true')

    # Model arguments
    parser.add_argument("--smile_encoder", type=str, default='PrimeKG_ver1')
    parser.add_argument("--se_config", type=str, required=True)
    parser.add_argument("--se_ckpt", type=str, required=True)
    parser.add_argument("--hvg_info", type=str, default=None)

    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--hvg_loss_weight", type=float, default=0.1)
    parser.add_argument("--de_topk", type=int, default=50)

    # Wasserstein specific arguments
    parser.add_argument("--num_samples_per_comb", type=int, default=1,
                        help="Number of random cell set samples per cell line-perturbation combination for Wasserstein")

    # GPU
    parser.add_argument("--gpu", type=int, default=0, help="GPU device to use")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    ckpt_path = os.path.join("./checkpoints", args.exp_name, f"epoch_{args.checkpoint_epoch}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    print(f"Loading checkpoint from: {ckpt_path}")

    # Dataset - external test split
    print(f"📊 Loading test dataset...")
    ds_test = TahoePerturbDatasetSE(
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

    test_loader = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    print(f"✅ Test set loaded: {len(ds_test)} unique (cell_line, perturbation) combinations")
    if args.num_samples_per_comb > 1:
        print(f"   Will sample {args.num_samples_per_comb} times per combination for Wasserstein distance estimation")

    # Model initialization
    print("🏗️ Initializing model...")
    cfg = OmegaConf.load(args.se_config)
    model = MAPmodel(
        se_ckpt=args.se_ckpt,
        se_cfg=cfg,
        smile_encoder=args.smile_encoder,
        hvg_info=args.hvg_info
    )
    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✅ Checkpoint loaded (epoch {checkpoint['epoch'] + 1})")

    # Total parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total:     {total:,} ({total/1e6:.2f}M)")
    print(f"  Trainable: {trainable:,} ({trainable/1e6:.2f}M)")

    # Loss functions
    loss_fn_emb = nn.MSELoss()
    loss_fn_hvg = None

    # Run full evaluation
    print("\n🚀 Starting FULL evaluation...")
    eval_results = evaluate_model_full(
        model, test_loader, loss_fn_emb, loss_fn_hvg,
        args.hvg_loss_weight, device,
        de_topk=args.de_topk,
        num_samples_per_comb=args.num_samples_per_comb,
        verbose=True
    )

    print("✅ Full evaluation completed!")

    return eval_results


if __name__ == '__main__':
    main()
