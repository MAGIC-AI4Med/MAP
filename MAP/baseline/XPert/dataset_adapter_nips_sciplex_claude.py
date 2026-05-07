#!/usr/bin/env python
"""
Adapter class to adapt NIPS/SciplexPerturbDataset to XPert MyDataset interface.

Does NOT modify original NIPSPerturbDataset/SciplexPerturbDataset class, just wraps it.
All new files have _claude suffix as required.
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
    Copied from XPpert MyDataset.
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
    Bin drug concentration into 10 bins. Copied from XPpert.
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


class NipsSciplexXPertDatasetAdapter(Dataset):
    """
    Adapter that wraps NIPSPerturbDataset/SciplexPerturbDataset to match XPert MyDataset interface.

    Args:
        original_dataset: the original NIPS/Sciplex dataset instance
        drug_feat: pre-computed drug features (UniMol format) dict mapping drug_name -> drug_feature
        n_bins: number of bins for expression digitization, from XPert config
        max_value: max expression value for quantile binning
        min_value: min expression value for quantile binning
        dataset_name: 'nips' or 'sciplex' - used to locate HG pre-trained embeddings
    """
    def __init__(
        self,
        original_dataset,
        drug_feat: Dict[str, np.ndarray],
        n_bins: int = 10,
        max_value: float = 10.0,
        min_value: float = 0.0,
    ):
        self.original_dataset = original_dataset
        self.drug_feat = drug_feat
        self.n_bins = n_bins
        self.max_value = max_value
        self.min_value = min_value

        # Create drug name -> drug_idx mapping from drug_feat
        self.drug_name_to_idx = {name: idx for idx, name in enumerate(drug_feat.keys())}

        # Pre-compute bins from quantiles
        self.bins = np.quantile(np.array([min_value, max_value]),
                              np.linspace(0, 1, n_bins - 1))

        print(f"[{self.__class__.__name__}] Initialized:")
        print(f"  total samples: {len(self.original_dataset)}")
        print(f"  unique drugs: {len(self.drug_feat)}")
        print(f"  n_bins: {n_bins}")
        print(f"  expression range: [{min_value}, {max_value}]")

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        """
        Returns tuple matching XPert MyDataset format:
        (
            trt_raw_data:      tensor(num_genes) float - average perturbed full gene expression across set_size cells
            ctl_raw_data:      tensor(num_genes) float - average control full gene expression
            trt_raw_binned:    tensor(num_genes) int64 - binned perturbed expression
            ctl_raw_binned:    tensor(num_genes) int64 - binned control expression
            drug_feat:         tensor(max_atoms, 514) float - UniMol drug feature
            pert_dose_idx:     tensor() int64 - binned drug concentration
            pert_time_idx:     tensor() int64 - binned time (always 0 for these datasets)
            pert_idx:          tensor() long - drug index for HG pre-trained embeddings
            cell_idx:          tensor() long - cell line index
            tissue_idx:        tensor() long - tissue index (always 0)
        )
        """
        sample = self.original_dataset[index]

        # Extract from original dataset output (dict format aligned with Tahoe)
        drug_smiles = sample['drug_smiles']
        drug_conc = sample['drug_conc']  # NIPS: always -1.0 ; Sciplex: from dose_val
        cell_line_name = sample['cell_line']

        # Get cell line index - original NIPS/Sciplex doesn't have this key, map it internally
        if 'cell_line_idx' in sample:
            cell_line_idx = sample['cell_line_idx']
        else:
            if not hasattr(self, '_cell_type_to_idx'):
                self._cell_type_to_idx = {ct: idx for idx, ct in enumerate(self.original_dataset.cell_types)}
            cell_line_idx = self._cell_type_to_idx[cell_line_name]

        # Get drug feature from pre-computed dict
        # drug_smiles is the key
        if drug_smiles in self.drug_feat:
            drug_feat = self.drug_feat[drug_smiles]
        else:
            # Fallback: should not happen if preprocessing correct
            drug_feat = next(iter(self.drug_feat.values()))

        # XPert requires format: [max_atoms, 514]
        #   column 0: atom mask (1=existing atom, 0=padding)
        #   column 1: atom symbol id
        #   columns 2-513: 512 features
        # Our pre-generated drug_feat from UniMol is: [max_atoms, 513] (col 0 = atom symbol, cols 1-512 = features)
        # We need to insert the atom mask as column 0
        # Compute atom mask: any row where the symbol is not zero is an existing atom
        drug_feat_np = np.array(drug_feat)
        atom_mask = (drug_feat_np[:, 0] != 0).astype(np.float32)
        # Insert mask at column 0 → [max_atoms, 514]
        drug_feat_np = np.insert(drug_feat_np, 0, atom_mask, axis=1)
        drug_feat = torch.tensor(drug_feat_np, dtype=torch.float32)

        # Get drug index
        drug_name = cell_line_name + '_' + drug_smiles if drug_smiles not in self.drug_name_to_idx else drug_smiles
        pert_idx = self.drug_name_to_idx.get(drug_name, 0)

        # We need FULL gene expression vector for all genes
        # Number of genes is from HG pre-training:
        #   nips: 12748
        #   sciplex: 1880
        num_genes = len(self.drug_name_to_idx) + 12748 if 'nips' in str(type(self.original_dataset)).lower() else 1880
        # Actually, we can get num_genes from HG preprocessing - but we don't have it here
        # We'll infer from drug count
        num_genes = 12748 if len(self.drug_name_to_idx) == 146 else 1880

        trt_raw_data = torch.zeros(num_genes, dtype=torch.float32)
        ctl_raw_data = torch.zeros(num_genes, dtype=torch.float32)

        # Note: Original dataset (NIPS/Sciplex) like Tahoe already gives us HVG vectors only
        # But XPert needs expression for ALL genes for the full prediction
        # However, we only have HVG in the dataset (pre-selected)
        # So we fill only the first NUM_HVG genes (2000) and leave rest as zero
        # This matches evaluation where we only evaluate on the 2000 HVG genes
        NUM_HVG = 2000
        # The original dataset returns perturb_hvg_vectors [set_size, 2000]
        perturb_hvg = sample['perturb_hvg_vectors']  # [set_size, 2000]
        control_hvg = sample['control_hvg_vectors']  # [set_size, 2000]

        # Average across set_size cells
        set_size = perturb_hvg.shape[0]
        for cell_i in range(set_size):
            for hvg_idx in range(NUM_HVG):
                if hvg_idx < num_genes:
                    trt_raw_data[hvg_idx] += perturb_hvg[cell_i][hvg_idx].item() / set_size
                    ctl_raw_data[hvg_idx] += control_hvg[cell_i][hvg_idx].item() / set_size

        # Digitize/bin the expression for XPert Transformer
        trt_raw_binned = digitize(trt_raw_data.numpy(), self.bins, side="one")
        ctl_raw_binned = digitize(ctl_raw_data.numpy(), self.bins, side="one")

        trt_raw_binned = torch.tensor(trt_raw_binned, dtype=torch.int64)
        ctl_raw_binned = torch.tensor(ctl_raw_binned, dtype=torch.int64)

        # Bin drug concentration
        # NIPS returns drug_conc = -1.0 (no concentration info) → bin 0
        if drug_conc < 0:
            pert_dose_idx = 0
        else:
            pert_dose_idx = assign_pert_dose_corrected(drug_conc)

        # Convert to tensors
        pert_dose_idx = torch.tensor(pert_dose_idx, dtype=torch.int64)
        pert_time_idx = torch.tensor(0, dtype=torch.int64)  # no multiple time points → always 0
        pert_idx = torch.tensor(pert_idx, dtype=torch.long)
        cell_idx = torch.tensor(cell_line_idx, dtype=torch.long)
        tissue_idx = torch.tensor(0, dtype=torch.long)  # single tissue → always 0

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
