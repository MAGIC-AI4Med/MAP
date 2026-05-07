# train_crisp_multicell_sciplex.py
# -*- coding: utf-8 -*-

import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

# ✅ 改动 1：从 Sciplex 版本的脚本导入 build_crisp_dataloader
# 你需要确保 sciplex_crisp.py 在 PYTHONPATH 或同目录
from ds_sciplex_crisp import build_crisp_dataloader

from CRISP.model import PertAE


@dataclass
class CRISPMultiCellConfig:
    # =========================
    # Sciplex data paths (✅ 改动 2：替换 Tahoe 的 base_dir/UC/cell_lines 等字段)
    # =========================
    h5ad_path: str
    shards_dir: str
    hvg_root: str
    split_dir: str

    # smiles embedding (rdkit2d, morgan, etc.)
    drug_emb_pt_path: str

    # Sciplex 的“cell line”语义是 cell_type（字符串列表）
    cell_types: List[str]

    # =========================
    # Training settings
    # =========================
    batch_size: int = 2
    set_size: int = 128
    UC: bool = True
    num_workers: int = 4
    hvg_not_yet_normed: bool = False
    seed: int = 0
    device: str = "cuda"

    # =========================
    # Training hyperparameters
    # =========================
    num_epochs: int = 50
    max_steps: Optional[int] = None
    log_every: int = 50
    save_every_epochs: int = 5

    # =========================
    # Model hyperparameters
    # =========================
    mmd_co: float = 0.1
    celltype_co: float = 1.0
    hparams: str = ""

    # =========================
    # Output
    # =========================
    out_dir: str = "./checkpoints_crisp_multicell_sciplex"
    exp_name: str = "sciplex_multicell_run1"


class CRISPMultiCellTrainer:
    def __init__(self, cfg: CRISPMultiCellConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.run_dir = os.path.join(cfg.out_dir, cfg.exp_name)
        os.makedirs(self.run_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("Initializing CRISP Multi-Cell Trainer (Sciplex-3)")
        print("=" * 80)

        # ✅ 改动 3：build_crisp_dataloader 参数换成 Sciplex 版本
        (
            self.train_loader,
            self.num_genes,
            self.fm_dim,
            self.drug_emb_dim,
            self.cell_line_to_idx,
        ) = build_crisp_dataloader(
            cell_types=cfg.cell_types,
            split="train",
            drug_emb_pt_path=cfg.drug_emb_pt_path,
            batch_size=cfg.batch_size,
            set_size=cfg.set_size,
            UC=cfg.UC,
            num_workers=cfg.num_workers,
            device=str(self.device),
            hvg_not_yet_normed=cfg.hvg_not_yet_normed,
            is_train=True,
            sequential=False,
            seed=cfg.seed,
            # Sciplex paths
            h5ad_path=cfg.h5ad_path,
            shards_dir=cfg.shards_dir,
            hvg_root=cfg.hvg_root,
            split_dir=cfg.split_dir,
        )

        self.num_celltypes = len(self.cell_line_to_idx)

        print("\nDataset Statistics:")
        print(f"   Number of genes (HVG): {self.num_genes}")
        print(f"   FM embedding dim:      {self.fm_dim}")
        print(f"   Drug embedding dim:    {self.drug_emb_dim}")
        print(f"   Number of cell types:  {self.num_celltypes}")
        print(f"   Cell type mapping:     {self.cell_line_to_idx}")

        # Model init（保持你原脚本一致）
        self.model = PertAE(
            num_genes=int(self.num_genes),
            num_drugs=0,  # 使用预计算embedding，不需要离散 drug id
            num_celltypes=int(self.num_celltypes),
            num_covariates=[0],
            drug_embeddings=None,
            drug_emb_dim=int(self.drug_emb_dim),
            mmd_co=cfg.mmd_co,
            celltype_co=cfg.celltype_co,
            device=str(self.device),
            seed=int(cfg.seed),
            hparams=cfg.hparams,
            FM_ndim=int(self.fm_dim),
        ).to(self.device)

        self.global_step = 0

        # Save config（保持一致）
        config_dict = {
            **cfg.__dict__,
            "cell_line_to_idx": self.cell_line_to_idx,
            "num_genes": self.num_genes,
            "fm_dim": self.fm_dim,
            "drug_emb_dim": self.drug_emb_dim,
            "num_celltypes": self.num_celltypes,
        }
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        print(f"\nTrainer initialized. Checkpoint dir: {self.run_dir}")
        print("=" * 80 + "\n")

    def save(self, tag: str) -> str:
        """保存模型检查点（保持你原脚本一致）"""
        path = os.path.join(self.run_dir, f"ckpt_{tag}.pt")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer_autoencoder": self.model.optimizer_autoencoder.state_dict(),
                "optimizer_cell": self.model.optimizer_cell.state_dict(),
                "optimizer_dosers": self.model.optimizer_dosers.state_dict()
                if hasattr(self.model, "optimizer_dosers")
                else None,
                "scheduler_autoencoder": self.model.scheduler_autoencoder.state_dict(),
                "scheduler_cell": self.model.scheduler_cell.state_dict(),
                "scheduler_dosers": self.model.scheduler_dosers.state_dict()
                if hasattr(self.model, "scheduler_dosers")
                else None,
                "global_step": self.global_step,
                "iteration": getattr(self.model, "iteration", None),
                "cfg": self.cfg.__dict__,
                "cell_line_to_idx": self.cell_line_to_idx,
                "num_genes": self.num_genes,
                "fm_dim": self.fm_dim,
                "drug_emb_dim": self.drug_emb_dim,
                "num_celltypes": self.num_celltypes,
            },
            path,
        )
        return path

    def train(self):
        """训练主循环（保持你原脚本一致）"""
        self.model.train()

        last_log_t = time.time()
        running: Dict[str, float] = {}

        print("\n" + "=" * 80)
        print("Starting Training (Sciplex)")
        print("=" * 80 + "\n")

        for epoch in range(1, self.cfg.num_epochs + 1):
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}/{self.cfg.num_epochs}",
                leave=True,
            )

            for batch_data in pbar:
                # Unpack batch (17 elements from collator)
                (
                    genes,
                    paired_cell_embeddings,
                    drugs_idx,  # None
                    dosages,
                    degs,
                    celltype_idx,
                    neg_genes,
                    neg_paired_cell_embeddings,
                    neg_drugs_idx,  # None
                    neg_dosages,
                    neg_degs,
                    neg_celltype_idx,
                    covariates,  # None
                    neg_covariates,  # None
                    drugs_pre,  # ✅ 预计算药物embedding
                    neg_drugs_pre,  # ✅ 负样本药物embedding
                    meta,
                ) = batch_data

                # Move to device
                genes = genes.to(self.device, non_blocking=True)
                paired_cell_embeddings = paired_cell_embeddings.to(self.device, non_blocking=True)
                dosages = dosages.to(self.device, non_blocking=True)
                degs = degs.to(self.device, non_blocking=True)
                celltype_idx = celltype_idx.to(self.device, non_blocking=True)
                drugs_pre = drugs_pre.to(self.device, non_blocking=True)

                neg_genes = neg_genes.to(self.device, non_blocking=True)
                neg_paired_cell_embeddings = neg_paired_cell_embeddings.to(self.device, non_blocking=True)
                neg_dosages = neg_dosages.to(self.device, non_blocking=True)
                neg_degs = neg_degs.to(self.device, non_blocking=True)
                neg_celltype_idx = neg_celltype_idx.to(self.device, non_blocking=True)
                neg_drugs_pre = neg_drugs_pre.to(self.device, non_blocking=True)

                # Model update
                stats = self.model.iter_update(
                    genes=genes,
                    cell_embeddings=paired_cell_embeddings,
                    drugs_idx=None,
                    dosages=dosages,
                    degs=degs,
                    celltype_idx=celltype_idx,
                    covariates=None,
                    neg_genes=neg_genes,
                    neg_cell_embeddings=neg_paired_cell_embeddings,
                    neg_drugs_idx=None,
                    neg_dosages=neg_dosages,
                    neg_degs=neg_degs,
                    neg_celltype_idx=neg_celltype_idx,
                    neg_covariates=None,
                    drugs_pre=drugs_pre,
                    neg_drugs_pre=neg_drugs_pre,
                )

                self.global_step += 1

                # Accumulate stats
                for k, v in stats.items():
                    running[k] = running.get(k, 0.0) + float(v)

                # Log
                if self.global_step % self.cfg.log_every == 0:
                    now = time.time()
                    avg = {k: running[k] / self.cfg.log_every for k in running}
                    avg["epoch"] = epoch
                    avg["step"] = self.global_step
                    avg["sec_per_log"] = now - last_log_t
                    tqdm.write(json.dumps(avg, ensure_ascii=False))
                    running = {}
                    last_log_t = now

                # Early stop
                if self.cfg.max_steps is not None and self.global_step >= self.cfg.max_steps:
                    ckpt = self.save(f"step{self.global_step}_earlystop")
                    tqdm.write(f"[Saved] {ckpt}")
                    print("\nTraining done (early stop by max_steps).")
                    return

            # Scheduler step
            self.model.scheduler_autoencoder.step()
            self.model.scheduler_cell.step()
            if hasattr(self.model, "scheduler_dosers") and self.model.scheduler_dosers is not None:
                self.model.scheduler_dosers.step()

            # Save checkpoint
            if self.cfg.save_every_epochs and (epoch % self.cfg.save_every_epochs == 0):
                ckpt = self.save(f"epoch{epoch}")
                tqdm.write(f"[Saved] {ckpt}")

        # Final save
        ckpt = self.save("final")
        print(f"\nTraining done. Final checkpoint: {ckpt}")


if __name__ == "__main__":
    # ✅ 示例配置：把路径换成你 Sciplex 的实际路径即可
    cfg = CRISPMultiCellConfig(
        # Sciplex paths
        h5ad_path="external_dataset_sciplex/raw/sciplex3_pp_hvgenes_scFM_resplit_filtered_ESM2.h5ad",
        shards_dir="shards/sciplex",
        hvg_root="external_dataset_sciplex/preprocessed/sciplex3_hvg_by_celltype_seurat_top2000",
        split_dir="external_dataset_sciplex/split",

        # smiles embedding
        drug_emb_pt_path="baselines/CRISP/data/drug_embeddings/rdkit2d_smiles_embeddings_sciplex.pt",

        # Sciplex cell types（示例；请替换为你实际想训练的 cell_type 列表）
        cell_types=["A549", "K562", "MCF7"],

        batch_size=2,
        set_size=128,
        UC=False,
        num_epochs=50,
        save_every_epochs=5,
        log_every=20,
        max_steps=None,

        mmd_co=0.1,
        celltype_co=1.0,

        device="cuda",
        out_dir="./checkpoints_crisp_multicell_sciplex",
        exp_name="6cell_rdkit2d_scFM_hvg2000",
        hparams="",
        hvg_not_yet_normed=False,
        seed=0,
        num_workers=4,
    )

    trainer = CRISPMultiCellTrainer(cfg)
    trainer.train()
