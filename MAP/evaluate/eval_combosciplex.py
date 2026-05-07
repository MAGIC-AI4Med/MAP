#!/usr/bin/env python
import csv
import argparse
import numpy as np
from scipy.stats import wasserstein_distance

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from data.ds_combosciplex_lora_se import ComboSciplexPerturbDatasetSE


def load_smiles_to_name():
    csv_path = "/path/to/combosciplex_drug_smiles.csv"
    mapping = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["SMILES"]] = row["drug_name"]
    return mapping


def get_drug_name(smiles, mapping):
    if not smiles:
        return "Control"
    if smiles in mapping:
        return mapping[smiles]
    return smiles[:15] + "..."


@torch.no_grad()
def evaluate_combosciplex_v2(
    model,
    dataset,
    loss_fn_emb,
    loss_fn_hvg,
    hvg_weight,
    device,
    de_topk=50,
    verbose=True,
):
    """
    ComboSciplex 评测函数 v2:
    - 每个扰动条件只采样1个cell set
    - 同时计算常规指标 + Wasserstein距离
    """
    model.eval()

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # 收集所有样本数据
    all_pred_hvg = []  # [N, 2000]，每个组合的pseudobulk预测
    all_true_hvg = []  # [N, 2000]，每个组合的pseudobulk真实值
    all_control_hvg = []  # [N, 2000]，每个组合的对照组
    all_de_indices = []  # 存储每个样本的top-k DE indices

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
        drug1_smiles = batch["drug1_smiles"]
        drug2_smiles = batch["drug2_smiles"]

        pred_emb, pred_hvg = model(ctrl_gene, ctrl_expr, drug1_smiles, drug2_smiles)

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

        # DE 基因位置（用于Wasserstein DE）
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

    # 返回所有指标
    metrics = {
        # 常规指标
        'avg_total_loss': avg_total_loss,
        'avg_emb_loss': avg_emb_loss,
        'avg_hvg_loss': avg_hvg_loss,
        'mse_hvg': mse_hvg,
        'avg_hvg_dir_acc': avg_hvg_dir_acc,
        'avg_de_dir_acc': avg_de_dir_acc,
        # Wasserstein指标
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
    parser = argparse.ArgumentParser(description="Evaluate ComboSciplex with Wasserstein distance")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--scheme", type=str, default="avg_emb", choices=["avg_emb", "two_tokens"])
    parser.add_argument("--data_base_dir", type=str)
    parser.add_argument("--se_inputs_base_dir", type=str)
    parser.add_argument("--se_config", type=str)
    parser.add_argument("--se_ckpt", type=str)
    parser.add_argument("--checkpoint", type=str, default=None, help="如果不指定，默认用 checkpoints/{exp_name}/best.pt")
    parser.add_argument("--set_size", type=int, default=24)
    parser.add_argument("--de_topk", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Scheme: {args.scheme}")
    print(f"Set size: {args.set_size}")

    # 根据scheme选模型
    if args.scheme == "avg_emb":
        from model.model_combosciplex_avg_emb import MAPmodel
    else:
        from model.model_combosciplex_two_tokens import MAPmodel

    # 初始化模型
    cfg = OmegaConf.load(args.se_config)
    model = MAPmodel(se_ckpt=args.se_ckpt, se_cfg=cfg, smile_encoder="PrimeKG_ver1")
    model = model.to(device)

    # 加载checkpoint
    ckpt_path = args.checkpoint or f"./checkpoints/{args.exp_name}/best.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"\n✅ Loaded model from: {ckpt_path}")

    loss_fn_emb = nn.MSELoss()
    loss_fn_hvg = nn.MSELoss()

    smiles_to_name = load_smiles_to_name()

    # 评测三个split
    splits = ["both_seen", "one_seen", "both_unseen"]
    all_results = {}

    for split in splits:
        print(f"\n{'='*100}")
        print(f"📊 Evaluating: {split.upper()}")
        print(f"{'='*100}")

        ds = ComboSciplexPerturbDatasetSE(
            cell_lines=["K562"], split=split, base_dir=args.data_base_dir,
            se_inputs_base_dir=args.se_inputs_base_dir, set_size=args.set_size,
            is_train=False, sequential=True,
        )

        # both_unseen时打印所有药物组合
        if split == "both_unseen":
            loader_tmp = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
            pairs = set()
            for batch in loader_tmp:
                d1 = get_drug_name(batch["drug1_smiles"][0], smiles_to_name)
                d2 = get_drug_name(batch["drug2_smiles"][0], smiles_to_name)
                pairs.add(f"{d1} + {d2}")
            print(f"\n🧪 {split.upper()} 药物组合列表 ({len(pairs)}个):")
            for p in sorted(pairs):
                print(f"   - {p}")

        metrics = evaluate_combosciplex_v2(
            model, ds, loss_fn_emb, loss_fn_hvg, 0.1,
            device, de_topk=args.de_topk,
            verbose=True
        )
        all_results[split] = metrics

    # 汇总表格
    print(f"\n{'='*100}")
    print(f"📊 FINAL SUMMARY - {args.scheme.upper()}")
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

    print(f"\n{'='*100}")
    print(f"✅ Evaluation complete!")


if __name__ == "__main__":
    main()
