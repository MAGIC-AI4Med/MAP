# evaluate_with_DE_multicell.py

import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn

from eval_utils import (
    compute_discrimination_score_global,
    compute_pearson_scores,
)


def compute_loss(
    pred_embs,
    pred_hvg,
    tgt_embs,
    batch_data,
    loss_fn_emb,
    loss_fn_hvg,
    hvg_weight,
    device,
):
    """
    计算损失：对 bulk 均值做 MSE（或 Cosine）.
    
    Args:
        pred_embs: [B, S, emb_dim] 预测的embeddings
        pred_hvg: [B, S, 2000] 预测的HVG向量
        tgt_embs: [B, S, emb_dim] 目标embeddings（真实的perturb embeddings）
        batch_data: 包含真实HVG向量的字典
        loss_fn_emb: Embedding损失函数
        loss_fn_hvg: HVG损失函数
        hvg_weight: HVG损失权重
        device: 计算设备
    """
    # Embedding loss
    pred_emb_mean = pred_embs.mean(dim=1)  # [B, emb_dim]
    true_emb_mean = tgt_embs.mean(dim=1)  # [B, emb_dim]
    emb_loss = loss_fn_emb(pred_emb_mean, true_emb_mean)

    # HVG loss
    true_hvg = batch_data["perturb_hvg_vectors"].to(device)  # [B, S, 2000]
    pred_hvg_mean = pred_hvg.mean(dim=1)  # [B, 2000]
    true_hvg_mean = true_hvg.mean(dim=1)  # [B, 2000]

    if loss_fn_hvg is None:
        hvg_loss = loss_fn_emb(pred_hvg_mean, true_hvg_mean)
    else:
        hvg_loss = loss_fn_hvg(pred_hvg_mean, true_hvg_mean)

    total_loss = emb_loss + hvg_weight * hvg_loss
    return total_loss, emb_loss, hvg_loss


@torch.no_grad()
def evaluate_model_multicell(
    model,
    test_loader,
    loss_fn_emb,
    loss_fn_hvg,
    hvg_weight,
    device,
    de_topk=50,
    verbose=True,
):
    """
    多细胞系评估函数（适配新框架：SE模型在线编码）
    
    Args:
        model: PerturbationEncoderLoRASE模型（包含可训练的SE）
        test_loader: DataLoader（返回SE模型输入：gene_ids和expressions）
        loss_fn_emb: Embedding损失函数
        loss_fn_hvg: HVG损失函数
        hvg_weight: HVG损失权重
        device: 计算设备
        de_topk: Top-K DE基因数量
        verbose: 是否打印详细信息
    
    Returns:
        dict: {
            'per_cell_line': {  # 每个细胞系的指标
                'CVCL_0023': {...},
                'CVCL_0480': {...},
                ...
            }
        }
    """
    model.eval()

    # 按细胞系分组存储数据
    cell_line_data = defaultdict(lambda: {
        'total_loss': 0.0,
        'emb_loss': 0.0,
        'hvg_loss': 0.0,
        'num_batches': 0,
        'pred_hvg_means': [],
        'true_hvg_means': [],
        'control_hvg_means': [],
        'hvg_dir_accs': [],
        'de_dir_accs': [],
        'de_delta_pred_all': [],
        'de_delta_true_all': [],
        'de_expr_pred_all': [],
        'de_expr_true_all': [],
    })

    for batch_data in tqdm(test_loader, disable=not verbose):
        # 🔥 新框架：加载SE模型输入
        ctrl_gene = batch_data["control_gene_ids"].to(device)
        ctrl_expr = batch_data["control_expressions"].to(device)
        
        hvgs = batch_data["perturb_hvg_vectors"].to(device)
        smiles = batch_data["drug_smiles"]
        concs = torch.tensor(
            batch_data["drug_conc"], dtype=torch.float32, device=device
        )
        cell_lines = batch_data["cell_line"]  # List of cell line names

        # ✅ 添加这一行：加载真实的 target embeddings
        tgt_emb_real = batch_data["perturb_embeddings"].to(device).float()

        # 🔥 前向传播：模型内部会调用SE编码
        pred_embs, pred_hvg = model(
            ctrl_gene, ctrl_expr, smiles, concs
        )

        # 计算损失
        # print('fefwf:', pred_embs, pred_embs.shape)
        batch_total_loss, batch_emb_loss, batch_hvg_loss = compute_loss(
            pred_embs,
            pred_hvg,
            tgt_emb_real,
            batch_data,
            loss_fn_emb,
            loss_fn_hvg,
            hvg_weight,
            device,
        )

        # 🔥 修复：转换为numpy之前先转为float32
        pred_hvg_mean = pred_hvg.mean(dim=1).float().cpu().numpy()  # [B, G]
        true_hvg_mean = hvgs.mean(dim=1).float().cpu().numpy()  # [B, G]
        control_hvg_mean = batch_data["control_hvg_vectors"].to(device).mean(dim=1).float().cpu().numpy()  # [B, G]

        # print('fwfw:', pred_hvg_mean)
        # print('wfwf:', true_hvg_mean)
        # print('gege:', control_hvg_mean)

        # pseudobulk Δ
        delta_true = true_hvg_mean - control_hvg_mean   # [B, G]
        delta_pred = pred_hvg_mean - control_hvg_mean   # [B, G]

        B, G = delta_true.shape

        # 按样本处理（每个样本对应一个细胞系）
        for i in range(B):
            cell_line = cell_lines[i]
            
            # 获取该细胞系的数据容器
            cl_data = cell_line_data[cell_line]
            
            # 累积损失（batch级别，需要除以batch中该细胞系的样本数）
            if cl_data['num_batches'] == 0:
                cl_data['total_loss'] = batch_total_loss.item()
                cl_data['emb_loss'] = batch_emb_loss.item()
                cl_data['hvg_loss'] = batch_hvg_loss.item()
            cl_data['num_batches'] += 1
            
            # 存储预测和真实值
            cl_data['pred_hvg_means'].append(pred_hvg_mean[i:i+1])
            cl_data['true_hvg_means'].append(true_hvg_mean[i:i+1])
            cl_data['control_hvg_means'].append(control_hvg_mean[i:i+1])
            
            dt = delta_true[i]           # [G]
            dp = delta_pred[i]           # [G]
            t_expr = true_hvg_mean[i]    # [G]
            p_expr = pred_hvg_mean[i]    # [G]

            # 1) HVG 级别方向准确率（全基因）
            if G > 0:
                sign_true_all = np.sign(dt)
                sign_pred_all = np.sign(dp)
                hvg_dir_acc = (sign_true_all == sign_pred_all).mean()
                cl_data['hvg_dir_accs'].append(hvg_dir_acc)

            # 2) top-k DE
            if G == 0:
                continue
            k = min(de_topk, G)
            if k <= 0:
                continue

            abs_dt = np.abs(dt)
            top_idx = np.argsort(-abs_dt)[:k]

            dt_top = dt[top_idx]        # [k]
            dp_top = dp[top_idx]        # [k]

            # 2a) DE direction accuracy
            sign_true = np.sign(dt_top)
            sign_pred = np.sign(dp_top)
            de_dir_acc = (sign_true == sign_pred).mean()
            cl_data['de_dir_accs'].append(de_dir_acc)

            # 2b) 缓存 Δ(top-k)
            cl_data['de_delta_true_all'].append(dt_top[None, :])
            cl_data['de_delta_pred_all'].append(dp_top[None, :])

            # 2c) 缓存 表达(top-k)
            t_expr_top = t_expr[top_idx]
            p_expr_top = p_expr[top_idx]
            cl_data['de_expr_true_all'].append(t_expr_top[None, :])
            cl_data['de_expr_pred_all'].append(p_expr_top[None, :])

    # ===== 计算每个细胞系的指标 =====
    per_cell_line_results = {}
    
    for cell_line, cl_data in cell_line_data.items():
        results = _compute_metrics_from_data(cl_data, de_topk)
        per_cell_line_results[cell_line] = results

    # ===== 表格形式输出 =====
    if verbose:
        _print_results_as_table(per_cell_line_results, de_topk)

    return {
        'per_cell_line': per_cell_line_results,
    }


def _compute_metrics_from_data(data, de_topk):
    """从累积的数据中计算所有指标"""
    num_batches = data['num_batches']
    
    # 平均损失
    avg_total_loss = data['total_loss'] / num_batches if num_batches > 0 else 0.0
    avg_emb_loss = data['emb_loss'] / num_batches if num_batches > 0 else 0.0
    avg_hvg_loss = data['hvg_loss'] / num_batches if num_batches > 0 else 0.0
    
    # 合并所有数据
    pred_hvg_means = np.concatenate(data['pred_hvg_means'], axis=0)
    true_hvg_means = np.concatenate(data['true_hvg_means'], axis=0)
    control_hvg_means = np.concatenate(data['control_hvg_means'], axis=0)
    
    # Discrimination scores (只计算HVG的)
    disc_hvg = compute_discrimination_score_global(
        pred_hvg_means, true_hvg_means, metric="cityblock"
    )
    
    # Pearson scores (只返回pearson_delta)
    pearson_delta = compute_pearson_scores(
        pred_hvg_means, true_hvg_means, control_hvg_means
    )
    
    # HVG / DE 方向准确率
    avg_hvg_dir_acc = float(np.mean(data['hvg_dir_accs'])) if len(data['hvg_dir_accs']) > 0 else 0.0
    avg_de_dir_acc = float(np.mean(data['de_dir_accs'])) if len(data['de_dir_accs']) > 0 else 0.0
    
    # DE discrimination + PrΔ(top-k DE)
    if len(data['de_delta_pred_all']) > 0:
        de_delta_pred_mat = np.vstack(data['de_delta_pred_all'])
        de_delta_true_mat = np.vstack(data['de_delta_true_all'])
        de_expr_pred_mat = np.vstack(data['de_expr_pred_all'])
        de_expr_true_mat = np.vstack(data['de_expr_true_all'])
        
        disc_de = compute_discrimination_score_global(
            de_expr_pred_mat, de_expr_true_mat, metric="cityblock"
        )
        
        dt_flat = de_delta_true_mat.reshape(-1)
        dp_flat = de_delta_pred_mat.reshape(-1)
        if np.std(dt_flat) > 0 and np.std(dp_flat) > 0:
            pearson_delta_de = float(np.corrcoef(dt_flat, dp_flat)[0, 1])
        else:
            pearson_delta_de = 0.0
    else:
        disc_de = 0.0
        pearson_delta_de = 0.0
    
    return {
        'avg_total_loss': avg_total_loss,
        'avg_emb_loss': avg_emb_loss,
        'avg_hvg_loss': avg_hvg_loss,
        'disc_hvg': disc_hvg,
        'pearson_delta': pearson_delta,
        'avg_hvg_dir_acc': avg_hvg_dir_acc,
        'disc_de': disc_de,
        'avg_de_dir_acc': avg_de_dir_acc,
        'pearson_delta_de': pearson_delta_de,
    }


def _print_results_as_table(per_cell_line_results, de_topk):
    """以紧凑的表格形式打印多细胞系的评测结果"""
    
    if not per_cell_line_results:
        print("No results to display.")
        return
    
    # 获取所有细胞系（按字母顺序排序）
    cell_lines = sorted(per_cell_line_results.keys())
    
    # 定义要显示的指标（按行）
    metrics = [
        ('Total Loss', 'avg_total_loss', '.6f'),
        ('Emb Loss', 'avg_emb_loss', '.6f'),
        ('HVG Loss', 'avg_hvg_loss', '.6f'),
        ('Disc(HVG)', 'disc_hvg', '.3f'),
        ('Pearson Δ(HVG)', 'pearson_delta', '.3f'),
        ('DirAcc(HVG)', 'avg_hvg_dir_acc', '.3f'),
        (f'Disc(DE-{de_topk})', 'disc_de', '.3f'),
        (f'Pearson Δ(DE-{de_topk})', 'pearson_delta_de', '.3f'),
        (f'DirAcc(DE-{de_topk})', 'avg_de_dir_acc', '.3f'),
    ]
    
    # 计算每个指标在所有细胞系上的平均值
    avg_metrics = {}
    for metric_name, metric_key, fmt in metrics:
        values = [per_cell_line_results[cl][metric_key] for cl in cell_lines]
        avg_metrics[metric_key] = np.mean(values)
    
    print("\n" + "="*120)
    print("📊 EVALUATION RESULTS (Per Cell Line)")
    print("="*120)
    
    # 计算列宽
    metric_col_width = max(len(m[0]) for m in metrics) + 2
    cell_col_width = 12
    avg_col_width = 12
    
    # 打印表头
    header = f"{'Metric':<{metric_col_width}}"
    for cl in cell_lines:
        header += f"{cl:>{cell_col_width}}"
    header += f"{'AVERAGE':>{avg_col_width}}"  # 添加AVERAGE列
    print(header)
    print("-"*120)
    
    # 打印每一行指标
    for metric_name, metric_key, fmt in metrics:
        row = f"{metric_name:<{metric_col_width}}"
        
        # 打印每个细胞系的值
        for cl in cell_lines:
            value = per_cell_line_results[cl][metric_key]
            row += f"{value:>{cell_col_width}{fmt}}"
        
        # 打印平均值
        avg_value = avg_metrics[metric_key]
        row += f"{avg_value:>{avg_col_width}{fmt}}"
        
        print(row)
    
    print("="*120 + "\n")
