#!/usr/bin/env python
"""
Adapter class to adapt TahoePerturbDatasetSE to XPert MyDataset interface.

Does NOT modify original TahoePerturbDatasetSE class, just wraps it.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any

from unimol_tools import UniMolRepr


def digitize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
    """
    Digitize the data into bins. This method spreads data uniformly when bins have same values.
    Copied from XPert MyDataset.
    """
    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_difits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))
    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits


def assign_pert_dose_corrected(pert_dose: float) -> int:
    """
    Bin drug concentration into 10 bins. Copied from XPert.
    """
    if 0 <= pert_dose < 0.21:
        return 0
    elif 0.21 <= pert_dose < 0.41:
        return 1
    elif 0.41 <= pert_dose < 0.71:
        return 2
    elif 0.71 <= pert_dose < 1.01:
        return 3
    elif 1.01 <= pert_dose < 1.51:
        return 4
    elif 1.51 <= pert_dose < 3.1:
        return 5
    elif 3.1 <= pert_dose < 4.1:
        return 6
    elif 4.1 <= pert_dose < 7.1:
        return 7
    elif 7.1 <= pert_dose < 12.1:
        return 8
    elif pert_dose >= 12.1:
        return 9
    else:
        print(f'{pert_dose} is out of boundary!')
        return 0


class TahoeXPertDatasetAdapter(Dataset):
    """
    Adapter that wraps TahoePerturbDatasetSE to match XPert MyDataset interface.

    Args:
        tahoe_dataset: the original TahoePerturbDatasetSE instance
        drug_feat: pre-computed drug features (UniMol format: [max_atoms, 513])
                   dict mapping drug_name -> drug_feature
        n_bins: number of bins for expression digitization, from XPert config
        max_value: max expression value for quantile binning
        min_value: min expression value for quantile binning
        gene_names: list of all gene names (for getting full gene expression)
        gene_to_idx: dict mapping gene name to index in full expression array
    """
    def __init__(
        self,
        tahoe_dataset,
        drug_feat: Dict[str, np.ndarray],
        n_bins: int = 10,
        max_value: float = 10.0,
        min_value: float = 0.0,
    ):
        self.tahoe_dataset = tahoe_dataset
        self.drug_feat = drug_feat
        self.n_bins = n_bins
        self.max_value = max_value
        self.min_value = min_value

        # Create drug -> drug_idx mapping from drug_feat
        self.drug_name_to_idx = {name: idx for idx, name in enumerate(drug_feat.keys())}

        # Pre-compute bins from quantiles
        self.bins = np.quantile(np.array([min_value, max_value]),
                              np.linspace(0, 1, n_bins - 1))

        print(f"[TahoeXPertAdapter] Initialized:")
        print(f"  total samples: {len(self.tahoe_dataset)}")
        print(f"  unique drugs: {len(self.drug_feat)}")
        print(f"  n_bins: {n_bins}")
        print(f"  expression range: [{min_value}, {max_value}]")

    def __len__(self):
        return len(self.tahoe_dataset)

    def __getitem__(self, index):
        """
        Returns tuple matching XPert MyDataset format (train mode):
        (
            trt_raw_data:      tensor(num_genes) float - average perturbed full gene expression across set_size cells
            ctl_raw_data:      tensor(num_genes) float - average control full gene expression
            trt_raw_binned:    tensor(num_genes) int64 - binned perturbed expression
            ctl_raw_binned:    tensor(num_genes) int64 - binned control expression
            drug_feat:         tensor(max_atoms, 513) float - UniMol drug feature
            pert_dose_idx:     tensor() int64 - binned drug concentration
            pert_time_idx:     tensor() int64 - binned time (we always use 0 for Tahoe)
            pert_idx:          tensor() long - drug index for HG pre-trained embeddings
            cell_idx:          tensor() long - cell line index
            tissue_idx:        tensor() long - tissue index (we always use 0 for Tahoe)
        )
        """
        sample = self.tahoe_dataset[index]

        # Extract from Tahoe format
        control_expr = sample['control_expressions']      # [set_size, pad_length]
        perturb_expr = sample['perturb_expressions']    # [set_size, pad_length]
        drug_smiles = sample['drug_smiles']
        drug_conc = sample['drug_conc']
        cell_line_idx = sample['cell_line_idx']

        # For Tahoe dataset, we don't have multiple time points - use 0
        pert_time_idx = 0

        # Get drug feature from pre-computed dict
        # Note: drug_smiles is already the key in drug_feat
        if drug_smiles in self.drug_feat:
            drug_feat = self.drug_feat[drug_smiles]
        else:
            # Fallback: should not happen if we preprocessed correctly
            drug_feat = next(iter(self.drug_feat.values()))

        # XPert requires format: [max_atoms, 514]
        # column 0: atom mask (1=existing atom, 0=padding)
        # column 1: atom symbol id
        # columns 2-513: 512 features
        # Current drug_feat shape: [max_atoms, 513] = (atom_id + 512 features)
        # We need to insert mask as first column
        atom_ids = drug_feat[:, 0].copy()  # first column is atom id from our generation
        features = drug_feat[:, 1:].copy()
        # mask = 1 where atom id > 0, else 0
        mask = (atom_ids > 0).astype(np.float32)
        # new array: [max_atoms, 514]
        new_drug_feat = np.zeros((drug_feat.shape[0], 514), dtype=np.float32)
        new_drug_feat[:, 0] = mask
        new_drug_feat[:, 1] = atom_ids
        new_drug_feat[:, 2:] = features

        drug_feat = torch.tensor(new_drug_feat, dtype=torch.float32)

        # Get drug index
        pert_idx = self.drug_name_to_idx.get(drug_smiles, 0)

        # We need FULL gene expression vector for all genes
        # Tahoe stores only the genes that are present in this cell (sparse)
        # We need to expand to full gene expression vector with zeros for missing genes
        # Get total number of genes from HG pre-training
        num_genes = 19790  # from our HG pre-training on Tahoe
        trt_raw_data = torch.zeros(num_genes, dtype=torch.float32)
        ctl_raw_data = torch.zeros(num_genes, dtype=torch.float32)

        # Average across set_size cells
        # We need the gene expression matrix to get expression from gene_ids
        # control_expr[cell_i] gives expressions for pad_length genes
        # control_gene_ids gives the actual gene ids for each position

        # Note: Tahoe does not store full expression matrix for all genes
        # We need to accumulate into full matrix
        set_size = control_expr.shape[0]
        for cell_i in range(set_size):
            gene_ids_cell_i = sample['control_gene_ids'][cell_i]
            expr_cell_i = control_expr[cell_i]
            for gene_id, expr in zip(gene_ids_cell_i, expr_cell_i):
                if gene_id < num_genes:  # gene_id is 0-based from original data
                    ctl_raw_data[gene_id] += expr.item() / set_size

        for cell_i in range(set_size):
            gene_ids_cell_i = sample['perturb_gene_ids'][cell_i]
            expr_cell_i = perturb_expr[cell_i]
            for gene_id, expr in zip(gene_ids_cell_i, expr_cell_i):
                if gene_id < num_genes:
                    trt_raw_data[gene_id] += expr.item() / set_size

        # Digitize/bin the expression for XPert Transformer
        trt_raw_binned = digitize(trt_raw_data.numpy(), self.bins, side="one")
        ctl_raw_binned = digitize(ctl_raw_data.numpy(), self.bins, side="one")

        trt_raw_binned = torch.tensor(trt_raw_binned, dtype=torch.int64)
        ctl_raw_binned = torch.tensor(ctl_raw_binned, dtype=torch.int64)

        # Bin drug concentration
        pert_dose_idx = assign_pert_dose_corrected(drug_conc)

        # Convert to tensors
        pert_dose_idx = torch.tensor(pert_dose_idx, dtype=torch.int64)
        pert_time_idx = torch.tensor(pert_time_idx, dtype=torch.int64)
        pert_idx = torch.tensor(pert_idx, dtype=torch.long)
        cell_idx = torch.tensor(cell_line_idx, dtype=torch.long)
        tissue_idx = torch.tensor(0, dtype=torch.long)  # Tahoe has one tissue by default

        return (
            trt_raw_data,
            ctl_raw_data,
            trt_raw_binned,
            ctl_raw_binned,
            drug_feat,
            pert_dose_idx,
            pert_time_idx,
            pert_idx,
            cell_idx,
            tissue_idx,
        )
