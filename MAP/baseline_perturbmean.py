#!/usr/bin/env python
import numpy as np
from scipy.stats import wasserstein_distance

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.ds_combosciplex_lora_se import ComboSciplexPerturbDatasetSE


class TrainMeanModel(nn.Module):
    def __init__(self, train_hvg_mean, train_emb_mean):
        super().__init__()
        self.train_hvg_mean = torch.tensor(train_hvg_mean, dtype=torch.float32)
        self.train_emb_mean = torch.tensor(train_emb_mean, dtype=torch.float32)

    def forward(self, ctrl_gene, ctrl_expr, drug1_smiles, drug2_smiles):
        B = len(drug1_smiles)
        S = ctrl_gene.shape[1]
        device = ctrl_gene.device

        # 广播到所有 batch 和 set 维度
        pred_embs = self.train_emb_mean.to(device).unsqueeze(0).unsqueeze(0).expand(B, S, -1)
        pred_hvgs = self.train_hvg_mean.to(device).unsqueeze(0).unsqueeze(0).expand(B, S, -1)

        return pred_embs, pred_hvgs


@torch.no_grad()
def evaluate_baseline(model, dataset, loss_fn_emb, loss_fn_hvg, hvg_weight,
                      device, de_topk=50, verbose=True):
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    all_pred_hvg = []  # [N, 2000]
    all_true_hvg = []  # [N, 2000]
    all_control_hvg = []  # [N, 2000]
    all_de_indices = []

    total_loss_sum = 0.0
    emb_loss_sum = 0.0
    hvg_loss_sum = 0.0
    num_batches = 0
    hvg_dir_accs = []
    de_dir_accs = []

    for batch in loader:
        ctrl_gene = batch["control_gene_ids"].to(device)
        ctrl_expr = batch["control_expressions"].to(device)
        perturb_emb = batch["perturb_embeddings"].to(device)
        true_hvg = batch["perturb_hvg_vectors"].to(device)
        d1 = batch["drug1_smiles"]
        d2 = batch["drug2_smiles"]

        pred_emb, pred_hvg = model(ctrl_gene, ctrl_expr, d1, d2)

        # 计算损失
        pred_emb_mean = pred_emb.mean(dim=1)
        true_emb_mean = perturb_emb.mean(dim=1)
        emb_loss = loss_fn_emb(pred_emb_mean, true_emb_mean)

        pred_hvg_mean = pred_hvg.mean(dim=1)
        true_hvg_mean = true_hvg.mean(dim=1)
        hvg_loss = loss_fn_hvg(pred_hvg_mean, true_hvg_mean)
        total_loss = emb_loss + hvg_weight * hvg_loss

        total_loss_sum += total_loss.item()
        emb_loss_sum += emb_loss.item()
        hvg_loss_sum += hvg_loss.item()
        num_batches += 1

        # 收集数据
        pred_hvg_mean_np = pred_hvg_mean.float().cpu().numpy()[0]
        true_hvg_mean_np = true_hvg_mean.float().cpu().numpy()[0]
        control_hvg_mean_np = batch["control_hvg_vectors"][0].mean(dim=0).float().cpu().numpy()

        all_pred_hvg.append(pred_hvg_mean_np)
        all_true_hvg.append(true_hvg_mean_np)
        all_control_hvg.append(control_hvg_mean_np)

        # 方向准确率
        delta_true = true_hvg_mean_np - control_hvg_mean_np
        delta_pred = pred_hvg_mean_np - control_hvg_mean_np
        hvg_dir_acc = (np.sign(delta_true) == np.sign(delta_pred)).mean()
        hvg_dir_accs.append(hvg_dir_acc)

        # DE 基因位置
        abs_dt = np.abs(delta_true)
        top_idx = np.argsort(-abs_dt)[:de_topk]
        all_de_indices.append(top_idx)

        # DE 方向准确率
        de_dir_acc = (np.sign(delta_true[top_idx]) == np.sign(delta_pred[top_idx])).mean()
        de_dir_accs.append(de_dir_acc)

    # 转换为numpy数组
    all_pred_hvg = np.array(all_pred_hvg)  # [N, 2000]
    all_true_hvg = np.array(all_true_hvg)
    N, G = all_pred_hvg.shape

    # 常规指标
    avg_total_loss = total_loss_sum / num_batches
    avg_emb_loss = emb_loss_sum / num_batches
    avg_hvg_loss = hvg_loss_sum / num_batches
    avg_hvg_dir_acc = float(np.mean(hvg_dir_accs))
    avg_de_dir_acc = float(np.mean(de_dir_accs))
    mse_hvg = np.mean((all_pred_hvg - all_true_hvg) ** 2)

    # Wasserstein距离
    wasserstein_hvg = []
    for g in range(G):
        pred_dist = all_pred_hvg[:, g]
        true_dist = all_true_hvg[:, g]
        wd = wasserstein_distance(pred_dist, true_dist)
        wasserstein_hvg.append(wd)
    avg_wasserstein_hvg = float(np.mean(wasserstein_hvg))

    # Wasserstein DE
    wasserstein_de = []
    for i, de_indices in enumerate(all_de_indices):
        for g in de_indices:
            wasserstein_de.append(wasserstein_hvg[g])
    avg_wasserstein_de = float(np.mean(wasserstein_de)) if len(wasserstein_de) > 0 else 0.0

    metrics = {
        'avg_total_loss': avg_total_loss,
        'avg_emb_loss': avg_emb_loss,
        'avg_hvg_loss': avg_hvg_loss,
        'mse_hvg': mse_hvg,
        'avg_hvg_dir_acc': avg_hvg_dir_acc,
        'avg_de_dir_acc': avg_de_dir_acc,
        'wasserstein_hvg': avg_wasserstein_hvg,
        'wasserstein_de': avg_wasserstein_de,
        'num_combinations': N,
    }

    if verbose:
        print(f"\n✅ Evaluation complete ({N} unique combinations):")
        print(f"   Total Loss:        {avg_total_loss:.6f}")
        print(f"   HVG MSE:           {mse_hvg:.6f}")
        print(f"   HVG Dir Acc:       {avg_hvg_dir_acc:.3f}")
        print(f"   DE Dir Acc:        {avg_de_dir_acc:.3f}")
        print(f"   Wasserstein HVG:   {avg_wasserstein_hvg:.6f}")
        print(f"   Wasserstein DE:    {avg_wasserstein_de:.6f}")

    return metrics


def main():
    DATA_DIR = "/path/to/external_dataset_combosciplex/preprocessed"
    SET_SIZE = 24
    DE_TOPK = 50
    HVG_LOSS_WEIGHT = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")


    ds_train = ComboSciplexPerturbDatasetSE(
        cell_lines=["K562"], split="train", base_dir=DATA_DIR,
        se_inputs_base_dir=DATA_DIR, set_size=SET_SIZE,
        is_train=False, sequential=True,
    )
    train_loader = DataLoader(ds_train, batch_size=1, shuffle=False, num_workers=4)

    all_hvg = []
    all_emb = []
    for batch in train_loader:
        perturb_hvg = batch["perturb_hvg_vectors"][0]  # [S, 2000]
        perturb_emb = batch["perturb_embeddings"][0]   # [S, emb_dim]
        all_hvg.append(perturb_hvg.mean(dim=0).numpy())  # 每个条件取均值
        all_emb.append(perturb_emb.mean(dim=0).numpy())

    global_hvg_mean = np.mean(all_hvg, axis=0)  # [2000]
    global_emb_mean = np.mean(all_emb, axis=0)  # [emb_dim]

    print(f"  训练集扰动条件数: {len(all_hvg)}")
    print(f"  HVG 均值 L2 norm: {np.linalg.norm(global_hvg_mean):.3f}")
    print(f"  Emb 均值 L2 norm: {np.linalg.norm(global_emb_mean):.3f}")

    # 创建常数预测模型
    model = TrainMeanModel(global_hvg_mean, global_emb_mean)
    model = model.to(device)
    model.eval()

    loss_fn_emb = nn.MSELoss()
    loss_fn_hvg = nn.MSELoss()

    # Step 2: 在三个测试集上评测
    splits = ["both_seen", "one_seen", "both_unseen"]
    all_results = {}

    for split in splits:
        print(f"\n{'='*100}")
        print(f"📊 Evaluating: {split.upper()}")
        print(f"{'='*100}")

        ds = ComboSciplexPerturbDatasetSE(
            cell_lines=["K562"], split=split, base_dir=DATA_DIR,
            se_inputs_base_dir=DATA_DIR, set_size=SET_SIZE,
            is_train=False, sequential=True,
        )

        metrics = evaluate_baseline(
            model, ds, loss_fn_emb, loss_fn_hvg, HVG_LOSS_WEIGHT,
            device, de_topk=DE_TOPK, verbose=True
        )
        all_results[split] = metrics

    # 汇总表格
    print(f"\n{'='*100}")
    print(f"📊 TRAIN MEAN BASELINE FINAL SUMMARY")
    print(f"{'='*100}")

    keys = ['avg_total_loss', 'avg_emb_loss', 'avg_hvg_loss', 'mse_hvg',
            'avg_hvg_dir_acc', 'avg_de_dir_acc', 'wasserstein_hvg', 'wasserstein_de']

    short_names = {
        'avg_total_loss': 'Total Loss',
        'avg_emb_loss': 'Emb Loss',
        'avg_hvg_loss': 'HVG Loss',
        'mse_hvg': 'MSE HVG',
        'avg_hvg_dir_acc': 'DirAcc HVG',
        'avg_de_dir_acc': 'DirAcc DE',
        'wasserstein_hvg': 'Wass HVG',
        'wasserstein_de': 'Wass DE',
    }

    print(f"{'Metric':<15}", end="")
    for split in splits:
        print(f"{split:>28}", end="")
    print()
    print("-" * 99)

    for k in keys:
        print(f"{short_names[k]:<15}", end="")
        for split in splits:
            print(f"{all_results[split][k]:>28.6f}", end="")
        print()

    print(f"\n✅ Train Mean Baseline Evaluation Complete!")


if __name__ == "__main__":
    main()
