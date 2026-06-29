"""
Train chemCPA on NIPS dataset, FULLY ALIGNED with ours_6:
- Same 6 cell types for both train and test
- Same split files: train / external_test
- Same UC (Unseen Combination) setting
- Uses chemCPA model architecture
"""
import argparse
import os
import sys

# Suppress rdkit MorganGenerator deprecation warnings
import warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore", message=".*please use MorganGenerator.*")

import lightning as L
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from chemCPA.embedding import get_chemical_representation
from chemCPA.model import ComPert

# NIPS dataset class
from ds_nips_lora_se import NIPSPerturbDatasetSE

# Evaluation utilities (aligned with ours_6)
from chemCPA_eval_utils import evaluate_chemCPA


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train chemCPA on NIPS dataset (aligned with ours_6)"
    )

    # Dataset arguments (same as ours_6/train_nips.py)
    parser.add_argument("--data_base_dir", type=str,
                        default="/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/external_dataset_nips/preprocessed",
                        help="Base directory for preprocessed NIPS data")
    parser.add_argument("--se_inputs_base_dir", type=str,
                        default="/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/external_dataset_nips/preprocessed_se_inputs_memmap/nips",
                        help="Base directory for SE input memmaps")
    parser.add_argument("--cell_types", type=str, nargs="+",
                        default=["B cells", "Myeloid cells", "NK cells", "T cells CD4+", "T cells CD8+", "T regulatory cells"],
                        help="Cell types (ALL 6, same for train AND test!)")
    parser.add_argument("--UC", action="store_true", default=False,
                        help="Use UC (Unseen Combination) split files (*_indices_UC.json)")
    parser.add_argument("--set_size", type=int, default=24,
                        help="Number of cells per perturbation set")
    parser.add_argument("--hvg_not_yet_normed", action="store_true", default=True,
                        help="Whether HVG needs log1p normalization")
    parser.add_argument("--num_genes", type=int, default=2000,
                        help="Number of HVG genes")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader workers")

    # Drug embedding
    parser.add_argument("--embedding_model", type=str, default="random",
                        help="Drug embedding model name: random (computes online) or name of precomputed")
    parser.add_argument("--embedding_path", type=str, default="",
                        help="Path to embedding file (optional)")

    # Model hyperparameters (chemCPA defaults)
    parser.add_argument("--dim", type=int, default=256,
                        help="Latent dimension")
    parser.add_argument("--autoencoder_width", type=int, default=512,
                        help="Autoencoder hidden layer width")
    parser.add_argument("--autoencoder_depth", type=int, default=4,
                        help="Number of autoencoder layers")
    parser.add_argument("--adversary_width", type=int, default=128,
                        help="Adversary hidden layer width")
    parser.add_argument("--adversary_depth", type=int, default=3,
                        help="Number of adversary layers")
    parser.add_argument("--adversary_steps", type=int, default=3,
                        help="Update adversary every N steps")
    parser.add_argument("--reg_adversary", type=float, default=5.0,
                        help="Adversary regularization weight")
    parser.add_argument("--penalty_adversary", type=float, default=3.0,
                        help="Gradient penalty weight")
    parser.add_argument("--dosers_width", type=int, default=64,
                        help="Dosers hidden layer width")
    parser.add_argument("--dosers_depth", type=int, default=2,
                        help="Number of dosers layers")
    parser.add_argument("--doser_type", type=str, default="logsigm",
                        help="Doser type: logsigm, sigm, mlp")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (number of perturbations per batch)")
    parser.add_argument("--autoencoder_lr", type=float, default=1e-3,
                        help="Learning rate for autoencoder")
    parser.add_argument("--adversary_lr", type=float, default=3e-4,
                        help="Learning rate for adversary")
    parser.add_argument("--dosers_lr", type=float, default=1e-3,
                        help="Learning rate for dosers")
    parser.add_argument("--autoencoder_wd", type=float, default=1e-6,
                        help="Weight decay for autoencoder")
    parser.add_argument("--adversary_wd", type=float, default=1e-4,
                        help="Weight decay for adversary")
    parser.add_argument("--dosers_wd", type=float, default=1e-7,
                        help="Weight decay for dosers")
    parser.add_argument("--step_size_lr", type=int, default=45,
                        help="Step size for LR scheduler")
    parser.add_argument("--embedding_encoder_width", type=int, default=512,
                        help="Embedding encoder width")
    parser.add_argument("--embedding_encoder_depth", type=int, default=0,
                        help="Number of embedding encoder layers")

    # Training
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--save_dir", type=str, default="./outputs_chemCPA_nips",
                        help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_freq", type=int, default=5,
                        help="Checkpoint every N epochs")
    parser.add_argument("--eval_freq_steps", type=int, default=10000,
                        help="Run full evaluation every N training steps")

    # WandB
    parser.add_argument("--wandb_project", type=str, default="chemCPA-nips-ours6-aligned",
                        help="WandB project name")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="WandB run name")

    # GPU
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID to use")

    return parser.parse_args()


class ChemCPAModule(L.LightningModule):
    """ChemCPA LightningModule - fully aligned with original training logic."""

    def __init__(self, hparams, canonical_smiles, num_genes, num_cell_types, val_loader=None, smiles_to_idx=None, eval_freq_steps=10000):
        super().__init__()
        self.save_hyperparameters(ignore=['val_loader', 'smiles_to_idx'])
        self.automatic_optimization = False  # Required for multi-optimizer setup

        self.hparams_config = hparams
        self.num_genes = num_genes
        self.num_cell_types = num_cell_types
        self.canonical_smiles = canonical_smiles
        self.num_drugs = len(canonical_smiles)

        # For periodic evaluation
        self.val_loader = val_loader
        self.smiles_to_idx = smiles_to_idx
        self.eval_freq_steps = eval_freq_steps
        self.global_step_counter = 0

        # Load drug embeddings - compute ECFP4 on-the-fly if needed
        try:
            self.drug_embeddings = get_chemical_representation(
                smiles=canonical_smiles,
                embedding_model=hparams["embedding_model"],
                data_path=hparams["embedding_path"] if hparams["embedding_path"] else None,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            print(f"✅ Loaded precomputed drug embeddings")
        except (AssertionError, FileNotFoundError) as e:
            print(f"⚠️ Precomputed embedding not found ({e}), computing ECFP4 fingerprints...")
            from rdkit import Chem
            from rdkit.Chem import AllChem
            import numpy as np

            fps = []
            for smi in canonical_smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    print(f"  ⚠️ Invalid SMILES: {smi[:50]}...")
                    fp = np.zeros(1024, dtype=np.float32)
                else:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                    fp = np.array(list(fp), dtype=np.float32)
                fps.append(fp)

            emb = torch.tensor(np.stack(fps), dtype=torch.float32)
            self.drug_embeddings = torch.nn.Embedding.from_pretrained(emb, freeze=True)
        print(f"✅ Drug embeddings shape: {self.drug_embeddings.weight.shape}")

        # Create ComPert model
        self.model = ComPert(
            num_genes=num_genes,
            num_drugs=self.num_drugs,
            num_covariates=[num_cell_types],  # one covariate (cell type)
            device="cuda" if torch.cuda.is_available() else "cpu",
            hparams=hparams,
            drug_embeddings=self.drug_embeddings,
            use_drugs_idx=True,
            doser_type=hparams["doser_type"],
        )

    def forward(self, genes, drugs_idx, dosages, covariates, return_latent_basal=False):
        """Wrap ComPert forward.
        model.predict returns gene_reconstructions = [mean, var] concatenated along dim=1.
        We split and return (mean, var).
        """
        result = self.model.predict(
            genes=genes,
            drugs_idx=drugs_idx,
            dosages=dosages,
            covariates=covariates,
            return_latent_basal=return_latent_basal
        )

        if return_latent_basal:
            gene_reconstructions, cell_drug_embedding, latents = result
            dim = gene_reconstructions.size(1) // 2
            mean = gene_reconstructions[:, :dim]
            var = gene_reconstructions[:, dim:]
            return (mean, var), latents
        else:
            gene_reconstructions, cell_drug_embedding = result
            dim = gene_reconstructions.size(1) // 2
            mean = gene_reconstructions[:, :dim]
            var = gene_reconstructions[:, dim:]
            return mean, var

    def _common_step(self, batch, batch_idx):
        """Shared step for train/val."""
        genes = batch["genes"]  # [batch_size * set_size, num_genes]
        drugs_idx = batch["drugs_idx"]  # [batch_size * set_size]
        dosages = batch["dosages"]  # [batch_size * set_size]
        target_genes = batch["target_genes"]  # [batch_size * set_size, num_genes]
        covariates = batch["covariates"]  # list of [batch_size * set_size, num_cell_types]

        # Forward
        mean, var = self(genes, drugs_idx, dosages, covariates)

        # Reconstruction loss (predict perturbed expression from control)
        recon_loss = self.model.loss_autoencoder(mean, target_genes, var)

        return recon_loss

    def training_step(self, batch, batch_idx):
        # Retrieve our 3 optimizers
        optimizers = self.optimizers()
        optimizer_autoencoder = optimizers[0]
        optimizer_adversaries = optimizers[1]
        optimizer_dosers = optimizers[2]

        # Get data
        genes = batch["genes"]
        drugs_idx = batch["drugs_idx"]
        dosages = batch["dosages"]
        target_genes = batch["target_genes"]
        covariates = batch["covariates"]  # list of tensors (one per covariate type)

        # 1. Forward pass with latent_basal
        # predict returns: gene_reconstructions, cell_drug_embedding, (latent_basal, drug_embedding, latent_treated)
        gene_reconstructions, _, latents = self.model.predict(
            genes=genes,
            drugs_idx=drugs_idx,
            dosages=dosages,
            covariates=covariates,
            return_latent_basal=True
        )
        dim = gene_reconstructions.size(1) // 2
        mean = gene_reconstructions[:, :dim]
        var = gene_reconstructions[:, dim:]
        latent_basal = latents[0] if latents is not None else None

        # 2. Reconstruction loss
        reconstruction_loss = self.model.loss_autoencoder(mean, target_genes, var)

        # 3. Adversary losses
        adversary_drugs_loss = torch.tensor(0.0, device=self.device)
        adversary_covariates_loss = torch.tensor(0.0, device=self.device)

        if self.model.num_drugs > 0 and latent_basal is not None:
            adversary_drugs_predictions = self.model.adversary_drugs(latent_basal)
            # drugs_idx is single index -> convert to multi-hot
            batch_size = drugs_idx.size(0)
            multi_hot_targets = torch.zeros(batch_size, self.model.num_drugs, device=drugs_idx.device)
            multi_hot_targets[torch.arange(batch_size), drugs_idx] = 1.0
            adversary_drugs_loss = self.model.loss_adversary_drugs(
                adversary_drugs_predictions, multi_hot_targets
            )

        # Covariates adversary
        if self.model.num_covariates[0] > 0 and latent_basal is not None:
            for i, adv in enumerate(self.model.adversary_covariates):
                pred_cov = adv(latent_basal)
                # covariates is a list, covariates[i] is the i-th covariate tensor
                # each row is one-hot, so argmax gives class index
                adversary_covariates_loss += self.model.loss_adversary_covariates[i](
                    pred_cov, covariates[i].argmax(dim=1)
                )

        # 4. Decide whether to optimize adversaries or autoencoder/doser
        if (self.global_step_counter % self.model.hparams["adversary_steps"]) == 0:
            # ---- Adversary update ----
            optimizer_adversaries.zero_grad()

            # Gradient penalty
            adv_drugs_grad_penalty = torch.tensor(0.0, device=self.device)
            adv_covs_grad_penalty = torch.tensor(0.0, device=self.device)

            if latent_basal is not None:
                def compute_gradient_penalty(out, x):
                    grads = torch.autograd.grad(out, x, create_graph=True)[0]
                    return (grads ** 2).mean()

                if self.model.num_drugs > 0:
                    adv_drugs_grad_penalty = compute_gradient_penalty(
                        adversary_drugs_predictions.sum(), latent_basal
                    )

                if self.model.num_covariates[0] > 0:
                    for i, adv in enumerate(self.model.adversary_covariates):
                        out = adv(latent_basal).sum()
                        adv_covs_grad_penalty += compute_gradient_penalty(out, latent_basal)

            penalty_scale = self.model.hparams["penalty_adversary"]
            loss_adversary_total = (
                adversary_drugs_loss
                + adversary_covariates_loss
                + penalty_scale * adv_drugs_grad_penalty
                + penalty_scale * adv_covs_grad_penalty
            )

            self.manual_backward(loss_adversary_total)
            optimizer_adversaries.step()

        else:
            # ---- Autoencoder & doser update ----
            optimizer_autoencoder.zero_grad()
            optimizer_dosers.zero_grad()

            loss_ae = (
                reconstruction_loss
                - self.model.hparams["reg_adversary"] * adversary_drugs_loss
                - self.model.hparams.get("reg_adversary_cov", 1.0) * adversary_covariates_loss
            )

            self.manual_backward(loss_ae)
            self.clip_gradients(optimizer_autoencoder, gradient_clip_val=1, gradient_clip_algorithm="norm")
            self.clip_gradients(optimizer_dosers, gradient_clip_val=1, gradient_clip_algorithm="norm")
            optimizer_autoencoder.step()
            optimizer_dosers.step()

        # Logging
        self.log("train/recon_loss", reconstruction_loss, prog_bar=True)
        self.log("train/adv_drugs_loss", adversary_drugs_loss)
        self.log("train/adv_cov_loss", adversary_covariates_loss)

        # Periodic evaluation
        self.global_step_counter += 1
        if self.global_step_counter > 0 and self.global_step_counter % self.eval_freq_steps == 0:
            if self.val_loader is not None and self.smiles_to_idx is not None:
                print(f"\n{'='*80}")
                print(f"📊 Running evaluation at step {self.global_step_counter}...")
                print(f"{'='*80}")
                eval_results = evaluate_chemCPA(
                    model=self,
                    val_loader=self.val_loader,
                    smiles_to_idx=self.smiles_to_idx,
                    device=self.device,
                    de_topk=50,
                    num_samples_per_comb=3,
                    verbose=True,
                )
                # Log overall average metrics
                overall = eval_results.get('overall_avg', {})
                for metric_name, value in overall.items():
                    self.log(f"eval/{metric_name}", value, prog_bar=False, sync_dist=True)
                # Ensure model is back in training mode after evaluation
                self.train()

        return reconstruction_loss

    def validation_step(self, batch, batch_idx):
        # Just compute recon loss for validation (simplified)
        genes = batch["genes"]
        drugs_idx = batch["drugs_idx"]
        dosages = batch["dosages"]
        target_genes = batch["target_genes"]
        covariates = batch["covariates"]

        with torch.no_grad():
            mean, var = self(genes, drugs_idx, dosages, covariates)
            recon_loss = self.model.loss_autoencoder(mean, target_genes, var)

        self.log("val/recon_loss", recon_loss, prog_bar=True, sync_dist=True)
        return recon_loss

    def on_train_epoch_end(self):
        # Step LR schedulers at epoch end (manual step for stored schedulers)
        self.scheduler_ae.step()
        self.scheduler_adv.step()
        self.scheduler_dosers.step()

    def configure_optimizers(self):
        """Set up three optimizers for manual optimization."""
        has_covariates = self.model.num_covariates[0] > 0

        def get_params(module_, condition):
            return list(module_.parameters()) if condition else []

        # 1) Autoencoder
        autoencoder_parameters = (
            get_params(self.model.encoder, True)
            + get_params(self.model.decoder, True)
            + get_params(self.model.drug_embedding_encoder, self.model.drug_embedding_encoder is not None)
        )
        for emb in getattr(self.model, "covariates_embeddings", []):
            autoencoder_parameters.extend(get_params(emb, has_covariates))

        optimizer_autoencoder = torch.optim.Adam(
            autoencoder_parameters,
            lr=self.model.hparams["autoencoder_lr"],
            weight_decay=self.model.hparams["autoencoder_wd"],
        )

        # 2) Adversaries
        adversaries_parameters = []
        if self.model.num_drugs > 0:
            adversaries_parameters += list(self.model.adversary_drugs.parameters())
        if has_covariates:
            for adv in self.model.adversary_covariates:
                adversaries_parameters.extend(list(adv.parameters()))

        optimizer_adversaries = torch.optim.Adam(
            adversaries_parameters,
            lr=self.model.hparams["adversary_lr"],
            weight_decay=self.model.hparams["adversary_wd"],
        )

        # 3) Dosers
        if hasattr(self.model, "dosers"):
            optimizer_dosers = torch.optim.Adam(
                self.model.dosers.parameters(),
                lr=self.model.hparams["dosers_lr"],
                weight_decay=self.model.hparams["dosers_wd"],
            )
        else:
            optimizer_dosers = torch.optim.Adam([], lr=0.0)

        # Store schedulers for manual stepping in on_train_epoch_end
        self.scheduler_ae = torch.optim.lr_scheduler.StepLR(optimizer_autoencoder, step_size=self.model.hparams["step_size_lr"], gamma=0.9)
        self.scheduler_adv = torch.optim.lr_scheduler.StepLR(optimizer_adversaries, step_size=self.model.hparams["step_size_lr"], gamma=0.9)
        self.scheduler_dosers = torch.optim.lr_scheduler.StepLR(optimizer_dosers, step_size=self.model.hparams["step_size_lr"], gamma=0.9)

        # For manual optimization with multi-optimizer, return list of optimizers
        return [optimizer_autoencoder, optimizer_adversaries, optimizer_dosers]


def nips_collate_fn(batch, smiles_to_idx):
    """Collate function - converts NIPSPerturbDatasetSE batch to chemCPA format.

    Args:
        batch: list of samples from NIPSPerturbDatasetSE
        smiles_to_idx: dict mapping SMILES to index

    Returns:
        dict with tensors formatted for chemCPA
    """
    batch_size = len(batch)
    set_size = batch[0]["perturb_hvg_vectors"].shape[0]

    # Control (baseline) and perturbed HVG expression
    control_hvg_list = [sample["control_hvg_vectors"] for sample in batch]
    perturb_hvg_list = [sample["perturb_hvg_vectors"] for sample in batch]
    control_hvg = torch.cat(control_hvg_list, dim=0)  # [batch_size * set_size, num_genes]
    perturb_hvg = torch.cat(perturb_hvg_list, dim=0)  # [batch_size * set_size, num_genes]

    # Build drugs_idx and dosages
    drugs_idx_list = []
    dosages_list = []
    cell_type_ohe_list = []

    for sample in batch:
        smi = sample["drug_smiles"]
        # All SMILES should already be in vocabulary (collected at init time)
        if smi not in smiles_to_idx:
            raise ValueError(f"SMILES not in vocabulary: {smi}. "
                           f"This should not happen - all SMILES should be pre-collected!")
        drug_idx = smiles_to_idx[smi]
        conc = sample["drug_conc"]
        ct_ohe = sample["cell_line_ohe"]

        # Repeat for all cells in this set
        drugs_idx_list.extend([drug_idx] * set_size)
        dosages_list.extend([conc] * set_size)
        cell_type_ohe_list.extend([ct_ohe] * set_size)

    drugs_idx = torch.tensor(drugs_idx_list, dtype=torch.long)
    dosages = torch.tensor(dosages_list, dtype=torch.float32)
    covariates = [torch.stack(cell_type_ohe_list, dim=0)]  # [batch_size * set_size, num_cell_types]

    return {
        "genes": control_hvg,           # input baseline expression (control)
        "drugs_idx": drugs_idx,         # drug indices
        "dosages": dosages,             # drug concentrations
        "target_genes": perturb_hvg,    # target perturbed expression
        "covariates": covariates,       # cell type one-hot
    }


def main():
    args = parse_args()

    # Print configuration
    print("\n" + "=" * 80)
    print("ChemCPA Training (NIPS, aligned with ours_6)")
    print("=" * 80)
    print(f"Cell types: {args.cell_types}")
    print(f"UC (Unseen Combination): {args.UC}")
    print(f"Set size: {args.set_size}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Data dir: {args.data_base_dir}")
    print("=" * 80 + "\n")

    # First, collect all unique SMILES across ALL cell types AND BOTH splits!
    print("📊 Collecting all SMILES across train + external_test splits...")

    import pickle
    from rdkit import Chem
    all_smiles = set()

    for cell_type in args.cell_types:
        print(f"  Processing {cell_type}...")
        cell_type_dir = os.path.join(args.data_base_dir, cell_type)
        meta_path = os.path.join(cell_type_dir, f"{cell_type}_meta.pkl")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        # Deduplicate within cell type to avoid redundant rdkit work
        unique_in_cell = set(smi for smi in meta["drug_smiles"] if smi is not None)
        print(f"    Found {len(unique_in_cell)} unique SMILES")

        # Apply the SAME canonicalization as NIPSPerturbDatasetSE does!
        for smi in unique_in_cell:
            try:
                canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
                all_smiles.add(canon_smi)
            except:
                # If rdkit fails, just add the original string
                all_smiles.add(smi)
                print(f"    ⚠️ RDKit failed for SMILES, using original: {smi[:50]}...")

    # Also include all original (non-canonical) SMILES to be safe
    for cell_type in args.cell_types:
        cell_type_dir = os.path.join(args.data_base_dir, cell_type)
        meta_path = os.path.join(cell_type_dir, f"{cell_type}_meta.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        unique_in_cell = set(smi for smi in meta["drug_smiles"] if smi is not None)
        all_smiles.update(unique_in_cell)

    canonical_smiles = sorted(list(all_smiles))
    smiles_to_idx = {smi: i for i, smi in enumerate(canonical_smiles)}
    num_drugs = len(canonical_smiles)
    print(f"✅ Total: {num_drugs} unique SMILES from all meta files (canonical + original)")

    # Build datasets - SAME cell types for train AND test (different splits!)
    print("\n🏗️ Building training dataset (split='train')...")
    train_dataset = NIPSPerturbDatasetSE(
        cell_lines=args.cell_types,
        split="train",
        base_dir=args.data_base_dir,
        se_inputs_base_dir=args.se_inputs_base_dir,
        UC=args.UC,
        set_size=args.set_size,
        is_train=True,
        sequential=False,
        hvg_not_yet_normed=args.hvg_not_yet_normed,
    )
    print(f"✅ Train dataset size: {len(train_dataset)} batches")

    print("\n🏗️ Building validation dataset (split='test')...")
    val_dataset = NIPSPerturbDatasetSE(
        cell_lines=args.cell_types,
        split="test",
        base_dir=args.data_base_dir,
        se_inputs_base_dir=args.se_inputs_base_dir,
        UC=args.UC,
        set_size=args.set_size,
        is_train=False,
        sequential=True,
        hvg_not_yet_normed=args.hvg_not_yet_normed,
    )
    print(f"✅ Val dataset size: {len(val_dataset)} batches")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: nips_collate_fn(b, smiles_to_idx),
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: nips_collate_fn(b, smiles_to_idx),
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # Build hparams dict
    hparams = {
        "dim": args.dim,
        "autoencoder_width": args.autoencoder_width,
        "autoencoder_depth": args.autoencoder_depth,
        "adversary_width": args.adversary_width,
        "adversary_depth": args.adversary_depth,
        "adversary_steps": args.adversary_steps,
        "reg_adversary": args.reg_adversary,
        "penalty_adversary": args.penalty_adversary,
        "dosers_width": args.dosers_width,
        "dosers_depth": args.dosers_depth,
        "dosers_lr": args.dosers_lr,
        "dosers_wd": args.dosers_wd,
        "autoencoder_lr": args.autoencoder_lr,
        "adversary_lr": args.adversary_lr,
        "autoencoder_wd": args.autoencoder_wd,
        "adversary_wd": args.adversary_wd,
        "batch_size": args.batch_size,
        "step_size_lr": args.step_size_lr,
        "embedding_encoder_width": args.embedding_encoder_width,
        "embedding_encoder_depth": args.embedding_encoder_depth,
        "embedding_model": args.embedding_model,
        "embedding_path": args.embedding_path,
        "doser_type": args.doser_type,
    }

    # Create model
    print("\n🤖 Creating ChemCPA model...")
    model = ChemCPAModule(
        hparams=hparams,
        canonical_smiles=canonical_smiles,
        num_genes=args.num_genes,
        num_cell_types=len(args.cell_types),
        val_loader=val_loader,
        smiles_to_idx=smiles_to_idx,
        eval_freq_steps=args.eval_freq_steps,
    )
    print(f"✅ Model created with {model.num_drugs} drugs, {model.num_genes} genes, {model.num_cell_types} cell types")
    print(f"   Evaluation frequency: every {args.eval_freq_steps} steps")

    # Callbacks
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        save_last=True,
        save_top_k=3,
        monitor="val/recon_loss",
        mode="min",
        filename="epoch={epoch:02d}-val_loss={val/recon_loss:.4f}",
        auto_insert_metric_name=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Trainer
    trainer = L.Trainer(
        accelerator="cuda",
        devices=[args.gpu],
        max_epochs=args.num_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        check_val_every_n_epoch=args.checkpoint_freq,
        # gradient_clip_val - done manually in training_step for manual optimization
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    # Start training
    print("\n🚀 Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("\n✅ Training completed!")


if __name__ == "__main__":
    main()
