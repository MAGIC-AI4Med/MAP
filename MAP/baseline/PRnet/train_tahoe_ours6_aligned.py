#!/usr/bin/env python
"""
Train PRnet on Tahoe dataset, FULLY ALIGNED with ours_6

Key features:
- All 6 cell lines in both training and testing
- UC (Unseen Cell-Drug Combination) mode support
- Full evaluation with all 9 metrics matching ours_6
- Regular evaluation during training
"""
import os
import sys
import argparse
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Normal
from tqdm import tqdm

# Filter rdkit deprecation warnings
warnings.filterwarnings('ignore', message='.*MorganGenerator.*')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.PRnet import PGM
from data.tahoe_6cl import TahoePerturbDataset
from prnet_eval_utils import evaluate_PRnet


def parse_args():
    parser = argparse.ArgumentParser(description='Train PRnet on Tahoe dataset (ours_6 aligned)')

    # Data parameters
    parser.add_argument('--data_base_dir', type=str,
                        default='/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/ours_3/preprocessed',
                        help='Directory containing preprocessed Tahoe data')
    parser.add_argument('--cell_lines', type=str, nargs='+',
                        default=['CVCL_0023', 'CVCL_0480', 'CVCL_0069',
                                 'CVCL_0131', 'CVCL_1098', 'CVCL_1056'],
                        help='Cell lines to use for training')
    parser.add_argument('--set_size', type=int, default=24,
                        help='Number of cells sampled per perturbation condition')
    parser.add_argument('--UC', action='store_true',
                        help='Use UC (Unseen Combination) split instead of random split')

    # Model parameters
    parser.add_argument('--x_dimension', type=int, default=2000,
                        help='Dimension of gene expression (HVG) vector')
    parser.add_argument('--hidden_layer_sizes', type=int, nargs='+', default=[128],
                        help='Hidden layer sizes for encoder/decoder')
    parser.add_argument('--z_dimension', type=int, default=64,
                        help='Latent space dimension')
    parser.add_argument('--comb_dimension', type=int, default=64,
                        help='Drug combination embedding dimension')
    parser.add_argument('--drug_dimension', type=int, default=1024,
                        help='Drug fingerprint dimension')
    parser.add_argument('--dr_rate', type=float, default=0.05,
                        help='Dropout rate')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (should be 1 for set_size-based batching)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-8,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')

    # Loss parameters
    parser.add_argument('--loss_type', type=str, default='gaussian',
                        choices=['gaussian', 'mse'],
                        help='Loss type: gaussian (NLL) or mse')

    # Evaluation parameters
    parser.add_argument('--eval_freq_steps', type=int, default=500,
                        help='Evaluate every N training steps')
    parser.add_argument('--de_topk', type=int, default=50,
                        help='Top-K DE genes for evaluation')
    parser.add_argument('--num_eval_samples', type=int, default=3,
                        help='Number of samples per combination for evaluation')

    # Checkpoint and logging
    parser.add_argument('--checkpoint_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--save_dir', type=str, default='./outputs_prnet_tahoe',
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')

    # Device
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device id')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()
    return args


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """
    Collate function for training.
    Converts dataset output to PRnet input format.

    Returns:
        control_hvgs: [set_size, G]
        target_hvgs: [set_size, G]
        encoded_drug: [set_size, drug_dim]
        cell_line: str
    """
    batch = batch[0]  # Dataset returns single sample

    control_hvgs = batch['control_hvg_vectors']  # [set_size, G]
    target_hvgs = batch['perturb_hvg_vectors']   # [set_size, G]
    drug_smiles = batch['drug_smiles']
    drug_conc = batch['drug_conc']
    cell_line = batch['cell_line']
    set_size = control_hvgs.shape[0]

    # Encode drug-dose
    mol = __import__('rdkit.Chem').Chem.MolFromSmiles(drug_smiles)
    if mol is None:
        encoded_single = np.zeros(1024, dtype=np.float32)
    else:
        fcfp4 = __import__('rdkit.Chem').Chem.AllChem.GetMorganFingerprintAsBitVect(
            mol, 2, useFeatures=True, nBits=1024
        ).ToBitString()
        encoded_single = np.array(list(fcfp4), dtype=np.float32)
        encoded_single = encoded_single * np.log10(drug_conc + 1)

    encoded_drug = torch.from_numpy(np.tile(encoded_single, (set_size, 1)))  # [set_size, 1024]

    return {
        'control_hvgs': control_hvgs,
        'target_hvgs': target_hvgs,
        'encoded_drug': encoded_drug,
        'cell_line': cell_line,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"prnet_tahoe_{'UC' if args.UC else 'random'}_{timestamp}"
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training PRnet on Tahoe dataset")
    print(f"UC mode: {args.UC}")
    print(f"Cell lines: {args.cell_lines}")
    print(f"Set size: {args.set_size}")
    print(f"Run directory: {run_dir}")
    print(f"{'='*60}\n")

    # Create datasets
    print("Creating training dataset...")
    train_dataset = TahoePerturbDataset(
        cell_lines=args.cell_lines,
        split='train',
        base_dir=args.data_base_dir,
        hvg_not_yet_normed=True,
        set_size=args.set_size,
        is_train=True,
        sequential=False,
        UC=args.UC,
        num_bits=args.drug_dimension,
    )

    print("\nCreating validation dataset (external_test)...")
    val_dataset = TahoePerturbDataset(
        cell_lines=args.cell_lines,
        split='external_test',
        base_dir=args.data_base_dir,
        hvg_not_yet_normed=True,
        set_size=args.set_size,
        is_train=False,
        sequential=True,  # Sequential mode for evaluation
        UC=args.UC,
        num_bits=args.drug_dimension,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x[0],  # No processing needed for sequential mode
    )

    print(f"\nTrain dataset size: {len(train_dataset)} batches")
    print(f"Val dataset size: {len(val_dataset)} combinations")

    # Create model
    print("\nCreating PRnet model...")
    model = PGM(
        x_dim=args.x_dimension,
        c_dim=args.comb_dimension,
        n_dim=10,  # Noise dimension
        hidden_layer_sizes=args.hidden_layer_sizes,
        z_dimension=args.z_dimension,
        adaptor_layer_sizes=[128],
        comb_adapt_dim=args.drug_dimension,
        dr_rate=args.dr_rate,
    )
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint.get('global_step', 0)
        print(f"Resumed at epoch {start_epoch}, step {global_step}")

    # Training loop
    print("\nStarting training...")
    best_pearson = -float('inf')

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for batch in pbar:
            optimizer.zero_grad()

            # Move data to device
            control_hvgs = batch['control_hvgs'].to(device)  # [set_size, G]
            target_hvgs = batch['target_hvgs'].to(device)    # [set_size, G]
            encoded_drug = batch['encoded_drug'].to(device)  # [set_size, drug_dim]

            set_size = control_hvgs.shape[0]

            # Generate noise
            noise = torch.randn(set_size, 10, device=device)

            # Forward pass
            gene_reconstructions = model(control_hvgs, encoded_drug, noise)

            # Split into mean and variance
            dim = gene_reconstructions.size(1) // 2
            pred_mean = gene_reconstructions[:, :dim]
            pred_var = gene_reconstructions[:, dim:]
            pred_var = F.softplus(pred_var) + 1e-6  # Ensure positive variance

            # Compute loss
            if args.loss_type == 'gaussian':
                # Gaussian NLL loss
                dist = Normal(pred_mean, torch.sqrt(pred_var))
                loss = -dist.log_prob(target_hvgs).mean()
            else:
                # MSE loss
                loss = F.mse_loss(pred_mean, target_hvgs)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update progress
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{epoch_loss/num_batches:.4f}",
            })

            # Periodic evaluation
            if global_step % args.eval_freq_steps == 0:
                print(f"\n{'='*60}")
                print(f"Evaluating at step {global_step} (epoch {epoch+1})...")
                model.eval()
                with torch.no_grad():
                    eval_results = evaluate_PRnet(
                        model=model,
                        val_loader=val_loader,
                        device=device,
                        de_topk=args.de_topk,
                        num_samples_per_comb=args.num_eval_samples,
                        verbose=True,
                    )

                # Save best model based on HVG Pearson Delta
                current_pearson = eval_results['overall_avg']['hvg_pearson']
                if current_pearson > best_pearson:
                    best_pearson = current_pearson
                    best_path = checkpoint_dir / 'best_model.pt'
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'eval_results': eval_results,
                        'hvg_pearson': current_pearson,
                    }, best_path)
                    print(f"✨ New best model saved with HVG Pearson Delta: {current_pearson:.4f}")

                model.train()
                print(f"{'='*60}\n")

        # End of epoch
        avg_epoch_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch+1}/{args.num_epochs} completed - Average loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")

    # Final evaluation
    print(f"\n{'='*60}")
    print("Training completed! Running final evaluation...")
    model.eval()
    with torch.no_grad():
        final_results = evaluate_PRnet(
            model=model,
            val_loader=val_loader,
            device=device,
            de_topk=args.de_topk,
            num_samples_per_comb=args.num_eval_samples,
            verbose=True,
        )

    # Save final model
    final_path = checkpoint_dir / 'final_model.pt'
    torch.save({
        'epoch': args.num_epochs,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_results': final_results,
    }, final_path)
    print(f"💾 Final model saved: {final_path}")

    print("\nTraining and evaluation completed successfully!")
    print(f"Run directory: {run_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
