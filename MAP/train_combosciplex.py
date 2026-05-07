# =========================
# File: train_combosciplex.py
# Training script for ComboSciplex dataset (single drug train, dual drug test)
# Two schemes:
#   - avg_emb: average two drug embeddings (minimal modification)
#   - two_tokens: two separate drug tokens in sequence
# =========================
import os
import time
import argparse

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from data.ds_combosciplex_lora_se import ComboSciplexPerturbDatasetSE
from eval_multi_cell import evaluate_model_combosciplex


def setup_ddp():
    """初始化DDP环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed mode")
        return -1, 1, -1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return rank, world_size, local_rank


def cleanup_ddp():
    """清理DDP环境"""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def print_model_parameters(model, rank, scheme):
    """打印模型参数统计（仅rank 0）"""
    if rank != 0:
        return

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*80}")
    print(f"MODEL PARAMETERS (ComboSciplex - {scheme.upper()})")
    print(f"{'='*80}")
    print(f"Total:     {total:,} ({total/1e6:.2f}M)")
    print(f"Trainable: {trainable:,} ({trainable/1e6:.2f}M)")
    print(f"Frozen:    {total-trainable:,} ({(total-trainable)/1e6:.2f}M)")
    print(f"{'='*80}\n")


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """Cosine学习率调度器（带warmup）"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def compute_loss(pred_emb, true_emb, pred_hvg, true_hvg, loss_fn_emb, loss_fn_hvg, hvg_weight):
    """计算loss"""
    pred_emb_mean = pred_emb.mean(dim=1)
    true_emb_mean = true_emb.mean(dim=1)
    emb_loss = loss_fn_emb(pred_emb_mean, true_emb_mean)

    pred_hvg_mean = pred_hvg.mean(dim=1)
    true_hvg_mean = true_hvg.mean(dim=1)

    if loss_fn_hvg is None:
        hvg_loss = loss_fn_emb(pred_hvg_mean, true_hvg_mean)
    else:
        hvg_loss = loss_fn_hvg(pred_hvg_mean, true_hvg_mean)

    total_loss = emb_loss + hvg_weight * hvg_loss
    return total_loss, emb_loss, hvg_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Training on ComboSciplex dataset with DDP.")
    parser.add_argument("--exp_name", type=str, required=True)

    # 方案选择
    parser.add_argument("--scheme", type=str, default="avg_emb", choices=["avg_emb", "two_tokens"],
                        help="ComboSciplex training scheme: 'avg_emb' (drug embedding averaging) or 'two_tokens' (two separate drug tokens)")

    # 数据集参数
    parser.add_argument("--data_base_dir", type=str, required=True)
    parser.add_argument("--se_inputs_base_dir", type=str, required=True)
    parser.add_argument("--cell_lines", type=str, nargs='+', default=["K562"],
                        help="Cell lines (ComboSciplex only has K562)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--set_size", type=int, default=24)
    parser.add_argument('--UC', action='store_true')

    # 模型参数
    parser.add_argument("--smile_encoder", type=str, default='PrimeKG_ver1')
    parser.add_argument("--se_config", type=str, required=True)
    parser.add_argument("--se_ckpt", type=str, required=True)
    parser.add_argument("--hvg_info", type=str, default=None)

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--hvg_loss_weight", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_warmup_steps", type=int, default=10000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_every_k_epochs", type=int, default=3)
    parser.add_argument("--eval_every_steps", type=int, default=500)
    parser.add_argument("--de_topk", type=int, default=50)

    # Mixed precision
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_ddp()
    if local_rank >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 根据方案选择model
    if args.scheme == "avg_emb":
        from model.model_combosciplex_avg_emb import MAPmodel
    else:  # two_tokens
        from model.model_combosciplex_two_tokens import MAPmodel

    # AMP配置
    use_amp = bool(args.amp)
    if use_amp:
        if args.amp_dtype == "bf16":
            amp_dtype = torch.bfloat16
            use_scaler = False
            if rank == 0:
                print("✅ AMP enabled: bfloat16 (no GradScaler)")
        else:
            amp_dtype = torch.float16
            use_scaler = True
            if rank == 0:
                print("✅ AMP enabled: float16 (with GradScaler)")
    else:
        amp_dtype = torch.float32
        use_scaler = False
        if rank == 0:
            print("ℹ️  AMP disabled: float32")

    scaler = GradScaler() if use_scaler else None

    # Checkpoint目录
    ckpt_dir = os.path.join("./checkpoints", args.exp_name)
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"📂 Checkpoints will be saved to: {ckpt_dir}")
        print(f"🧪 ComboSciplex scheme: {args.scheme.upper()}")

    # 训练数据集（单药扰动，train split = DMSO + Drug）
    if rank == 0:
        print(f"📊 Loading ComboSciplex training dataset (single drug perturbations)...")

    ds_train = ComboSciplexPerturbDatasetSE(
        cell_lines=args.cell_lines,
        split="train",
        base_dir=args.data_base_dir,
        se_inputs_base_dir=args.se_inputs_base_dir,
        hvg_not_yet_normed=True,
        set_size=args.set_size,
        is_train=True,
        sequential=False,
        return_control_hvg=True,
        UC=args.UC,
    )

    train_sampler = DistributedSampler(
        ds_train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # 测试数据集（双药组合，both_seen）
    external_loader = None
    if rank == 0:
        print("📊 Loading ComboSciplex test dataset (both_seen drug combinations)...")
        ds_external_test = ComboSciplexPerturbDatasetSE(
            cell_lines=args.cell_lines,
            split="both_seen",
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
            ds_external_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )
        print(f"✅ Test set: {len(ds_external_test)} unique (cell_line, drug1, drug2) combinations")

    # 模型
    if rank == 0:
        print("🏗️ Initializing model...")

    cfg = OmegaConf.load(args.se_config)
    model = MAPmodel(
        se_ckpt=args.se_ckpt,
        se_cfg=cfg,
        smile_encoder=args.smile_encoder,
        hvg_info=args.hvg_info,
    )
    model = model.to(device)

    model = DDP(
        model,
        device_ids=[local_rank] if torch.cuda.is_available() else None,
        output_device=local_rank if torch.cuda.is_available() else None,
        find_unused_parameters=True,
    )

    print_model_parameters(model, rank, args.scheme)

    # 优化器：仅更新Pert Encoder部分（SE frozen）
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    # 学习率调度器
    num_training_steps = args.epochs * len(train_loader) // args.gradient_accumulation_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.num_warmup_steps, num_training_steps
    )

    # 损失函数
    loss_fn_emb = nn.MSELoss()
    loss_fn_hvg = nn.MSELoss()

    # 训练循环
    global_step = 0
    best_loss = float('inf')

    if rank == 0:
        print(f"🚀 Starting training for {args.epochs} epochs...")
        print(f"   Total steps: {num_training_steps}")
        print(f"   Gradient accumulation steps: {args.gradient_accumulation_steps}")

    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch+1}/{args.epochs}")
            print(f"{'='*80}")

        epoch_loss = 0.0
        epoch_emb_loss = 0.0
        epoch_hvg_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", disable=(rank != 0))

        for batch in pbar:
            global_step += 1

            ctrl_gene = batch["control_gene_ids"].to(device)
            ctrl_expr = batch["control_expressions"].to(device)
            perturb_gene = batch["perturb_gene_ids"].to(device)
            perturb_expr = batch["perturb_expressions"].to(device)
            perturb_emb = batch["perturb_embeddings"].to(device)
            true_hvg = batch["perturb_hvg_vectors"].to(device)

            drug1_smiles = batch["drug1_smiles"]
            drug2_smiles = batch["drug2_smiles"]

            with autocast(enabled=use_amp, dtype=amp_dtype):
                pred_emb, pred_hvg = model(ctrl_gene, ctrl_expr, drug1_smiles, drug2_smiles)
                total_loss, emb_loss, hvg_loss = compute_loss(
                    pred_emb, perturb_emb, pred_hvg, true_hvg,
                    loss_fn_emb, loss_fn_hvg, args.hvg_loss_weight
                )

                total_loss = total_loss / args.gradient_accumulation_steps

            if use_scaler:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            if global_step % args.gradient_accumulation_steps == 0:
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            epoch_loss += total_loss.item() * args.gradient_accumulation_steps
            epoch_emb_loss += emb_loss.item()
            epoch_hvg_loss += hvg_loss.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{total_loss.item()*args.gradient_accumulation_steps:.4f}",
                "emb": f"{emb_loss.item():.4f}",
                "hvg": f"{hvg_loss.item():.4f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
            })

            # 验证 (both_seen 双药组合测试集)
            if rank == 0 and global_step % args.eval_every_steps == 0:
                model.eval()
                with torch.no_grad():
                    metrics = evaluate_model_combosciplex(
                        model, external_loader, loss_fn_emb, loss_fn_hvg, args.hvg_loss_weight,
                        device, de_topk=args.de_topk, verbose=True
                    )
                    if metrics["avg_total_loss"] < best_loss:
                        best_loss = metrics["avg_total_loss"]
                        ckpt_path = os.path.join(ckpt_dir, "best.pt")
                        torch.save(model.module.state_dict(), ckpt_path)
                        print(f"  ✅ New best model saved!")
                model.train()

        avg_loss = epoch_loss / num_batches
        avg_emb_loss = epoch_emb_loss / num_batches
        avg_hvg_loss = epoch_hvg_loss / num_batches

        if rank == 0:
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"   Avg total loss: {avg_loss:.4f}")
            print(f"   Avg emb loss:   {avg_emb_loss:.4f}")
            print(f"   Avg hvg loss:   {avg_hvg_loss:.4f}")

            # 定期保存checkpoint
            if (epoch + 1) % args.save_every_k_epochs == 0:
                ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt")
                torch.save(model.module.state_dict(), ckpt_path)
                print(f"💾 Checkpoint saved to: {ckpt_path}")

    # 训练结束
    if rank == 0:
        ckpt_path = os.path.join(ckpt_dir, "final.pt")
        torch.save(model.module.state_dict(), ckpt_path)
        print(f"\n{'='*80}")
        print(f"✅ TRAINING COMPLETE")
        print(f"   Final checkpoint saved to: {ckpt_path}")
        print(f"   Best loss: {best_loss:.4f}")
        print(f"{'='*80}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
