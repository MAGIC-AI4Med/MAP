# =========================
# File: train_sciplex.py
# Training script for Sciplex dataset - follows the same DDP training logic as train.py
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

from data.ds_sciplex_lora_se import SciplexPerturbDatasetSE
from model.model import MAPmodel
from eval_multi_cell import evaluate_model_multicell


def setup_ddp():
    """初始化DDP环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed mode")
        return -1, 1, -1

    # Fix for cluster environments: ensure CUDA is properly initialized
    # Handle case where CUDA_VISIBLE_DEVICES is already set by srun
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(local_rank)
        except RuntimeError:
            # If set_device fails, try re-initializing CUDA
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return rank, world_size, local_rank


def cleanup_ddp():
    """清理DDP环境"""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def print_model_parameters(model, rank):
    """打印模型参数统计（仅rank 0）"""
    if rank != 0:
        return

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*80}")
    print(f"MODEL PARAMETERS (Sciplex)")
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
        hvg_loss = loss_fn_emb(pred_hvg_mean, true_hvg_mean)

    total_loss = emb_loss + hvg_weight * hvg_loss
    return total_loss, emb_loss, hvg_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Training on Sciplex dataset with DDP.")
    parser.add_argument("--exp_name", type=str, required=True)

    # 数据集参数
    parser.add_argument("--data_base_dir", type=str, required=True)
    parser.add_argument("--se_inputs_base_dir", type=str, required=True)
    parser.add_argument("--cell_lines", "--cell_types", type=str, nargs='+', required=True)
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

    # 训练数据集
    if rank == 0:
        print(f"📊 Loading Sciplex training dataset...")

    ds_train = SciplexPerturbDatasetSE(
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

    # 测试数据集（仅rank 0）
    external_loader = None
    if rank == 0:
        print("📊 Loading Sciplex test dataset...")
        ds_external_test = SciplexPerturbDatasetSE(
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
            ds_external_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )
        print(f"✅ Sciplex test set: {len(ds_external_test)} unique (cell_line, drug) combinations")

    # 模型
    if rank == 0:
        print("🏗️ Initializing model...")

    cfg = OmegaConf.load(args.se_config)
    model = MAPmodel(
        se_ckpt=args.se_ckpt,
        se_cfg=cfg,
        smile_encoder=args.smile_encoder,
        hvg_info=args.hvg_info
    )
    model = model.to(device)

    model = DDP(
        model,
        device_ids=[local_rank] if torch.cuda.is_available() else None,
        output_device=local_rank if torch.cuda.is_available() else None,
        find_unused_parameters=False,
    )

    if rank == 0:
        print("✅ Model distributed (DDP).")

    print_model_parameters(model, rank)

    # Optimizer & Scheduler
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    num_training_steps = args.epochs * (len(train_loader) // max(1, args.gradient_accumulation_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=0.1,
    )

    # Loss functions
    loss_fn_emb = nn.MSELoss()
    loss_fn_hvg = None

    global_step = 0
    loss_accum = emb_loss_accum = hvg_loss_accum = 0.0
    accum_count = 0

    # 训练循环
    for epoch in range(args.epochs):
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{args.epochs} [Sciplex]")
            print(f"{'='*80}")

        # ===== Epoch开始时的evaluation =====
        if rank == 0 and external_loader is not None:
            print("\n" + "="*80)
            print(f"[Eval @ epoch start] epoch={epoch+1} [Sciplex]")
            print("="*80)

            if accum_count > 0:
                print(f"  Avg train loss (last {accum_count} steps):")
                print(f"    Total: {loss_accum/accum_count:.4f}")
                print(f"    Emb:   {emb_loss_accum/accum_count:.4f}")
                print(f"    HVG:   {hvg_loss_accum/accum_count:.4f}")
                loss_accum = emb_loss_accum = hvg_loss_accum = 0.0
                accum_count = 0

            t0 = time.time()
            with autocast(dtype=amp_dtype, enabled=use_amp):
                eval_results = evaluate_model_multicell(
                    model, external_loader, loss_fn_emb, loss_fn_hvg,
                    args.hvg_loss_weight, device, de_topk=args.de_topk, verbose=True
                )
            print(f"[Eval done] {time.time() - t0:.1f}s")
            print("="*80 + "\n")

        train_sampler.set_epoch(epoch)
        model.train()

        for batch_idx, batch in enumerate(tqdm(train_loader, disable=(rank != 0))):
            ctrl_gene = batch["control_gene_ids"].to(device, non_blocking=True)
            ctrl_expr = batch["control_expressions"].to(device, non_blocking=True)
            tgt_emb = batch["perturb_embeddings"].to(device, non_blocking=True).float()
            hvgs = batch["perturb_hvg_vectors"].to(device, non_blocking=True)
            smiles = batch["drug_smiles"]
            concs = torch.tensor(batch["drug_conc"], device=device, dtype=torch.float32)

            is_acc_step = (batch_idx + 1) % args.gradient_accumulation_steps != 0
            ctx = model.no_sync() if is_acc_step else torch.enable_grad()

            with ctx:
                with autocast(dtype=amp_dtype, enabled=use_amp):
                    pred_embs, pred_hvgs = model(ctrl_gene, ctrl_expr, smiles, concs)

                    total_loss, emb_loss, hvg_loss = compute_loss(
                        pred_emb=pred_embs,
                        true_emb=tgt_emb,
                        pred_hvg=pred_hvgs,
                        true_hvg=hvgs,
                        loss_fn_emb=loss_fn_emb,
                        loss_fn_hvg=loss_fn_hvg,
                        hvg_weight=args.hvg_loss_weight,
                    )

                    loss_for_backward = total_loss / args.gradient_accumulation_steps

                if use_scaler:
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

            if not is_acc_step:
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if rank == 0:
                    loss_accum += float(total_loss.item())
                    emb_loss_accum += float(emb_loss.item())
                    hvg_loss_accum += float(hvg_loss.item())
                    accum_count += 1

            # ===== Mid-epoch evaluation =====
            if rank == 0 and external_loader is not None and args.eval_every_steps > 0:
                if (global_step + 1) % args.eval_every_steps == 0:
                    model.eval()
                    print("\n" + "="*80)
                    print(f"[Eval mid] epoch={epoch+1}, step={global_step} [Sciplex]")
                    print("="*80)

                    if accum_count > 0:
                        print(f"  Avg train loss (last {accum_count} steps):")
                        print(f"    Total: {loss_accum/accum_count:.4f}")
                        print(f"    Emb:   {emb_loss_accum/accum_count:.4f}")
                        print(f"    HVG:   {hvg_loss_accum/accum_count:.4f}")
                        loss_accum = emb_loss_accum = hvg_loss_accum = 0.0
                        accum_count = 0

                    t0 = time.time()
                    with autocast(dtype=amp_dtype, enabled=use_amp):
                        eval_results = evaluate_model_multicell(
                            model, external_loader, loss_fn_emb, loss_fn_hvg,
                            args.hvg_loss_weight, device, de_topk=args.de_topk, verbose=True
                        )
                    print(f"[Eval mid done] {time.time() - t0:.1f}s")
                    print("="*80 + "\n")

                    model.train()

        # Epoch结束：打印平均loss
        if rank == 0 and accum_count > 0:
            print(f"\n[Epoch {epoch+1}] Avg Train Loss [Sciplex]:")
            print(f"  Total: {loss_accum/accum_count:.4f}")
            print(f"  Emb:   {emb_loss_accum/accum_count:.4f}")
            print(f"  HVG:   {hvg_loss_accum/accum_count:.4f}")
            loss_accum = emb_loss_accum = hvg_loss_accum = 0.0
            accum_count = 0

        # 保存checkpoint
        if rank == 0 and (epoch + 1) % args.save_every_k_epochs == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    cleanup_ddp()


if __name__ == '__main__':
    main()
