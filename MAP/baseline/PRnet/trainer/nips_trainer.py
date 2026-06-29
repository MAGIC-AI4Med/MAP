# trainer/nips_trainer.py
import os
import math
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.distributions import NegativeBinomial, normal

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

# NIPS dataset + collator (your full script)
from data.nips_6ct import NIPSPerturbDataset, NIPS2PRnetAdaptCollator
from models.PRnet import PRnet


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


class PRnetTrainer:
    """
    NIPS version of PRnetTrainer.
    This is a minimal-diff port of tahoe_trainer.PRnetTrainer:
      - Keep model / loss / training loop / checkpointing behavior the same
      - Replace TahoePerturbDataset + Tahoe2PRnetAdaptCollator with NIPS equivalents
      - Remove hard-coded Tahoe paths and move them into args/kwargs
    """

    def __init__(
        self,
        adata=None,
        batch_size=32,
        comb_num=2,
        shuffle=True,
        split_key="random_split",
        model_save_dir="./checkpoint/",
        results_save_dir="./results/",
        x_dimension=5000,
        hidden_layer_sizes=[128],
        z_dimension=64,
        adaptor_layer_sizes=[128],
        comb_dimension=64,
        drug_dimension=1024,
        n_genes=50,
        dr_rate=0.05,
        loss=["GUSS"],
        obs_key="cov_drug",
        # ---- NIPS-specific knobs (new) ----
        cell_types=None,
        set_size=128,
        h5ad_path=None,
        shards_dir=None,
        hvg_root=None,
        split_dir=None,
        hvg_not_yet_normed=True,
        seed=2024,
        is_train=True,
        sequential=False,
        **kwargs,
    ):
        assert set(loss).issubset(["NB", "GUSS", "KL", "MSE"]), \
            "loss should be subset of ['NB', 'GUSS', 'KL', 'MSE']"

        # ---- keep the same core attributes (as tahoe_trainer) ----
        self.x_dim = x_dimension
        self.split_key = split_key
        self.z_dimension = z_dimension
        self.comb_dimension = comb_dimension

        self.model = PRnet(
            None,
            x_dimension=self.x_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            z_dimension=z_dimension,
            adaptor_layer_sizes=adaptor_layer_sizes,
            comb_dimension=comb_dimension,
            comb_num=comb_num,
            drug_dimension=drug_dimension,
            dr_rate=dr_rate,
        )

        self.model_save_dir = model_save_dir
        self.results_save_dir = results_save_dir
        self.loss = loss
        self.modelPGM = self.model.get_PGM()

        # ---- seed / device / DP behavior: keep same as tahoe_trainer :contentReference[oaicite:2]{index=2} ----
        self.seed = seed
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            if torch.cuda.device_count() > 1:
                self.modelPGM = nn.DataParallel(
                    self.modelPGM,
                    device_ids=[i for i in range(torch.cuda.device_count())],
                )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelPGM = self.modelPGM.to(self.device)

        self.modelPGM.apply(self.weight_init)
        print(self.modelPGM)

        self.de_n_genes = n_genes

        # =========================
        # Build NIPS dataloader (replacing Tahoe hard-coded block) :contentReference[oaicite:3]{index=3}
        # =========================
        if cell_types is None or len(cell_types) == 0:
            raise ValueError("NIPS trainer requires non-empty cell_types.")
        if h5ad_path is None or shards_dir is None or hvg_root is None or split_dir is None:
            raise ValueError("NIPS trainer requires h5ad_path, shards_dir, hvg_root, split_dir.")

        dataset = NIPSPerturbDataset(
            cell_types=cell_types,
            split="train",
            h5ad_path=h5ad_path,
            shards_dir=shards_dir,
            hvg_root=hvg_root,
            split_dir=split_dir,
            set_size=set_size,
            is_train=is_train,
            sequential=sequential,
            hvg_not_yet_normed=hvg_not_yet_normed,
            seed=seed,
        )
        collator = NIPS2PRnetAdaptCollator(num_bits=drug_dimension, set_size=set_size)

        # Keep the same DataLoader style: dataset already samples set internally,
        # so typical usage is batch_size=1; but we preserve your param.
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
            shuffle=shuffle,
        )

        # ---- losses: keep same as tahoe_trainer :contentReference[oaicite:4]{index=4} ----
        if set(["NB"]).issubset(loss):
            self.criterion = NBLoss()
        if set(["GUSS"]).issubset(loss):
            self.criterion = nn.GaussianNLLLoss()
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        self.shuffle = shuffle
        self.batch_size = batch_size

        # training bookkeeping (same as tahoe_trainer)
        self.epoch = -1
        self.best_state_dictPGM = None

        self.PGM_losses = []
        self.r2_score_mean = []
        self.r2_score_var = []
        self.mse_score = []
        self.r2_score_mean_de = []
        self.r2_score_var_de = []
        self.mse_score_de = []
        self.best_mse = np.inf
        self.patient = 0

    def train(
        self,
        n_epochs=100,
        lr=0.001,
        weight_decay=1e-8,
        scheduler_factor=0.5,
        scheduler_patience=10,
        save_every=10,
        **extras_kwargs,
    ):
        # Keep as-is from tahoe_trainer: no val-based scheduler :contentReference[oaicite:5]{index=5}
        self.n_epochs = n_epochs
        paramsPGM = filter(lambda p: p.requires_grad, self.modelPGM.parameters())

        self.optimPGM = torch.optim.Adam(paramsPGM, lr=lr, weight_decay=weight_decay)

        os.makedirs(self.model_save_dir, exist_ok=True)

        for self.epoch in range(self.n_epochs):
            loop = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
            for i, data in loop:
                self.modelPGM.zero_grad()
                (control, target) = data["features"]
                encode_label = data["label"]

                control = control.to(self.device, dtype=torch.float32)
                if set(["NB"]).issubset(self.loss):
                    control = torch.log1p(control)
                target = target.to(self.device, dtype=torch.float32)

                encode_label = encode_label.to(self.device, dtype=torch.float32)
                b_size = control.size(0)

                noise = self.make_noise(b_size, 10)

                gene_reconstructions = self.modelPGM(control, encode_label, noise)
                dim = gene_reconstructions.size(1) // 2
                gene_means = gene_reconstructions[:, :dim]
                gene_vars = gene_reconstructions[:, dim:]
                gene_vars = F.softplus(gene_vars)

                if set(["GUSS"]).issubset(self.loss):
                    reconstruction_loss = self.criterion(input=gene_means, target=target, var=gene_vars)

                    dist = normal.Normal(
                        torch.clamp(torch.Tensor(gene_means), min=1e-3, max=1e3),
                        torch.clamp(torch.Tensor(gene_vars.sqrt()), min=1e-3, max=1e3),
                    )

                if set(["NB"]).issubset(self.loss):
                    reconstruction_loss = self.criterion(gene_means, target, gene_vars)

                    counts, logits = self._convert_mean_disp_to_counts_logits(
                        torch.clamp(torch.Tensor(gene_means), min=1e-3, max=1e3),
                        torch.clamp(torch.Tensor(gene_vars), min=1e-3, max=1e3),
                    )

                    dist = NegativeBinomial(total_count=counts, logits=logits)

                nb_sample = dist.sample()

                if set(["MSE"]).issubset(self.loss):
                    mse_loss = self.mse_loss(nb_sample, target)
                    reconstruction_loss += mse_loss * 10

                if set(["KL"]).issubset(self.loss):
                    kl_loss = self.kl_loss(nb_sample, target)
                    reconstruction_loss += kl_loss * 0.01

                reconstruction_loss.backward()
                self.optimPGM.step()

                self.PGM_losses.append(reconstruction_loss.item())
                loop.set_description(f"Epoch [{self.epoch+1}/{self.n_epochs}] [{i+1}/{len(self.train_dataloader)}]")
                loop.set_postfix(Loss=reconstruction_loss.item())

            if (self.epoch + 1) % save_every == 0:
                print(f"\n💾 Saving checkpoint at epoch {self.epoch + 1}...")
                checkpoint_path = os.path.join(self.model_save_dir, f"{self.split_key}_epoch_{self.epoch + 1}.pt")
                if isinstance(self.modelPGM, nn.DataParallel):
                    torch.save(self.modelPGM.module.state_dict(), checkpoint_path)
                else:
                    torch.save(self.modelPGM.state_dict(), checkpoint_path)
                print(f"✅ Checkpoint saved to {checkpoint_path}\n")

        print(f"\n💾 Saving final model...")
        final_model_path = os.path.join(self.model_save_dir, f"{self.split_key}_final_epoch_{self.n_epochs}.pt")
        if isinstance(self.modelPGM, nn.DataParallel):
            torch.save(self.modelPGM.module.state_dict(), final_model_path)
        else:
            torch.save(self.modelPGM.state_dict(), final_model_path)
        print(f"✅ Final model saved to {final_model_path}\n")

        loss_df = pd.DataFrame({"Loss_PGM": self.PGM_losses})
        loss_csv_path = os.path.join(self.model_save_dir, f"{self.split_key}_loss_comb.csv")
        loss_df.to_csv(loss_csv_path, index=False)
        print(f"📊 Training losses saved to {loss_csv_path}")

    def make_noise(self, batch_size, shape, volatile=False):
        tensor = torch.randn(batch_size, shape)
        noise = Variable(tensor, volatile)
        noise = noise.to(self.device, dtype=torch.float32)
        return noise

    def weight_init(self, m):
        if isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.normal_(1, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

    @staticmethod
    def pearson_mean(data1, data2):
        sum_pearson_1 = 0
        sum_pearson_2 = 0
        for i in range(data1.shape[0]):
            pearsonr_ = pearsonr(data1[i], data2[i])
            sum_pearson_1 += pearsonr_[0]
            sum_pearson_2 += pearsonr_[1]
        return sum_pearson_1 / data1.shape[0], sum_pearson_2 / data1.shape[0]

    @staticmethod
    def r2_mean(data1, data2):
        sum_r2_1 = 0
        for i in range(data1.shape[0]):
            r2_score_ = r2_score(data1[i], data2[i])
            sum_r2_1 += r2_score_
        return sum_r2_1 / data1.shape[0]

    @staticmethod
    def _convert_mean_disp_to_counts_logits(mu, theta, eps=1e-6):
        assert (mu is None) == (theta is None)
        logits = (mu + eps).log() - (theta + eps).log()
        total_count = theta
        return total_count, logits

    @staticmethod
    def _sample_z(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps


class NBLoss(torch.nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, yhat, y, eps=1e-8):
        dim = yhat.size(1) // 2
        mu = yhat[:, :dim]
        theta = yhat[:, dim:]

        if theta.ndimension() == 1:
            theta = theta.view(1, theta.size(0))

        t1 = (
            torch.lgamma(theta + eps)
            + torch.lgamma(y + 1.0)
            - torch.lgamma(y + theta + eps)
        )
        t2 = (theta + y) * torch.log(1.0 + (mu / (theta + eps))) + (
            y * (torch.log(theta + eps) - torch.log(mu + eps))
        )
        final = _nan2inf(t1 + t2)
        return torch.mean(final)
