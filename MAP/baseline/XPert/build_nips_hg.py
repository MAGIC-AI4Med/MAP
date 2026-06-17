#!/usr/bin/env python
"""
Build heterogeneous graph data for XPert HG pre-training on NIPS dataset.
Processes genes, drugs, and extracts edges from existing knowledge graph data.

Output files will be saved to baselines/XPert/HG_data/
All new files have _claude suffix as required.
"""

import os
import json
import torch
import scanpy as sc
import numpy as np
import pandas as pd
from collections import defaultdict

# ============================================================
# Step 1: Load gene information and ESM embeddings from dataset
# ============================================================

def load_genes_from_h5ad(h5ad_path: str, embedding_path: str):
    """Load gene symbols from NIPS h5ad and get ESM embeddings"""
    print(f"Loading h5ad from {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path, backed='r')

    # Get gene symbols from var
    if "gene_id" not in adata.var.columns:
        raise KeyError("h5ad does not have 'gene_id' in var columns")

    gene_list = list(adata.var["gene_id"].astype(str).values)
    print(f"[NIPS] Genes in dataset: {len(gene_list)}")

    # Load ESM embeddings dict
    print(f"Loading ESM gene embeddings from {embedding_path}")
    gene_emb_dict = torch.load(embedding_path, map_location="cpu")
    if isinstance(gene_emb_dict, (list, tuple)) and len(gene_emb_dict) > 0:
        gene_emb_dict = gene_emb_dict[0]

    # Filter genes that have ESM embeddings
    valid_genes = []
    emb_list = []
    missing = 0
    for gene in gene_list:
        if gene in gene_emb_dict:
            valid_genes.append(gene)
            emb = gene_emb_dict[gene].numpy()
            emb_list.append(emb)
        else:
            missing += 1

    gene_emb_matrix = np.stack(emb_list, axis=0)
    gene_to_idx = {gene: idx for idx, gene in enumerate(valid_genes)}

    print(f"  Valid with ESM embedding: {len(valid_genes)}, missing: {missing}")
    print(f"  Final gene embedding shape: {gene_emb_matrix.shape}")

    return valid_genes, gene_to_idx, gene_emb_matrix

# ============================================================
# Step 2: Load unique drugs and SMILES from dataset
# ============================================================

def load_unique_drugs_from_h5ad(h5ad_path: str):
    """Extract unique drugs with SMILES from NIPS h5ad"""
    print(f"Extracting unique drugs from {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path, backed='r')

    if "SMILES" not in adata.obs.columns:
        raise KeyError("h5ad does not have 'SMILES' in obs columns")

    if "condition" not in adata.obs.columns:
        raise KeyError("h5ad does not have 'condition' in obs columns")

    # Get unique (drug condition, SMILES) pairs
    # Remove DMSO (control) from drugs
    drug_set = set()
    for cond, smi in zip(adata.obs["condition"], adata.obs["SMILES"]):
        cond = str(cond)
        smi = str(smi)
        if cond.lower() == "dmso":
            continue
        if pd.isna(smi) or smi == "nan" or smi == "":
            continue
        drug_set.add((cond, smi))

    drug_list = sorted(list(drug_set), key=lambda x: x[0])
    print(f"Unique non-control drugs: {len(drug_list)}")

    drug_names = [name for name, smi in drug_list]
    drug_smiles = [smi for name, smi in drug_list]
    drug_to_idx = {name: idx for idx, (name, smi) in enumerate(drug_list)}

    return drug_list, drug_names, drug_smiles, drug_to_idx

# ============================================================
# Build HGNC to gene symbol mapping from hgnc_complete_set.txt
# ============================================================

def build_hgnc_mapping(hgnc_path: str):
    """Build mapping from HGNC:XXXX to gene symbol"""
    print(f"Building HGNC mapping from {hgnc_path}")
    df = pd.read_csv(hgnc_path, sep='\t', low_memory=False)

    hgnc_to_symbol = {}
    for _, row in df.iterrows():
        hgnc_id = str(row['hgnc_id'])
        if not hgnc_id.startswith('HGNC:'):
            hgnc_id = f'HGNC:{hgnc_id}'
        symbol = str(row['symbol'])
        hgnc_to_symbol[hgnc_id] = symbol

    print(f"Built mapping for {len(hgnc_to_symbol)} HGNC entries")
    return hgnc_to_symbol

# ============================================================
# Step 3: Extract PPI edges (gene-gene) from PPI.csv
# ============================================================

def extract_ppi(ppi_path: str, gene_to_idx: dict, hgnc_to_symbol: dict):
    """Extract PPI edges where both genes are in our gene list"""
    print(f"Extracting PPI edges from {ppi_path}")
    df = pd.read_csv(ppi_path)
    print(f"Original PPI entries: {len(df)}")

    edges = []
    edge_weights = []
    missing_hgnc = 0
    missing_gene = 0

    for _, row in df.iterrows():
        hgnc1 = str(row.iloc[0]).strip()
        hgnc2 = str(row.iloc[1]).strip()
        score = float(row.iloc[2])

        # Map HGNC to symbol
        if hgnc1 not in hgnc_to_symbol or hgnc2 not in hgnc_to_symbol:
            missing_hgnc += 1
            continue

        symbol1 = hgnc_to_symbol[hgnc1]
        symbol2 = hgnc_to_symbol[hgnc2]

        if symbol1 not in gene_to_idx or symbol2 not in gene_to_idx:
            missing_gene += 1
            continue

        idx1 = gene_to_idx[symbol1]
        idx2 = gene_to_idx[symbol2]
        edges.append((idx1, idx2))
        edge_weights.append(score)

    print(f"Found valid PPI edges: {len(edges)}")
    print(f"Missing HGNC entries (skipped): {missing_hgnc}")
    print(f"Missing genes (skipped): {missing_gene}")

    if len(edges) > 0:
        edge_index = np.array(edges).T  # shape [2, num_edges]
        print(f"Final PPI edge_index shape: {edge_index.shape}")
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return edge_index, np.array(edge_weights)

# ============================================================
# Step 4: Extract DTI edges (drug-gene) from PrimeKG
# ============================================================

def extract_dti(primekg_path: str, drug_to_idx: dict, gene_to_idx: dict):
    """Extract drug-target interaction edges from PrimeKG"""
    print(f"Extracting DTI edges from {primekg_path}")
    df = pd.read_csv(primekg_path, low_memory=False)
    print(f"Original PrimeKG entries: {len(df)}")

    # Filter for drug-gene (protein) interactions
    # PrimeKG format: relation, display_relation, x_index, x_id, x_type, x_name, x_source, y_index, y_id, y_type, y_name, y_source
    dti_edges = []
    missing_drug = 0
    missing_gene = 0
    valid = 0

    for _, row in df.iterrows():
        x_type = str(row['x_type'])
        y_type = str(row['y_type'])

        # We want drug -> gene/protein interactions
        is_dti = False
        drug_name = None
        gene_name = None

        if 'drug' in x_type.lower() and 'gene' in y_type.lower():
            is_dti = True
            drug_name = str(row['x_name']).strip()
            gene_name = str(row['y_name']).strip()
        elif 'drug' in y_type.lower() and 'gene' in x_type.lower():
            is_dti = True
            drug_name = str(row['y_name']).strip()
            gene_name = str(row['x_name']).strip()

        if not is_dti:
            continue

        # Match drug by name (fuzzy match - find any containing)
        drug_idx = None
        for d_name, d_idx in drug_to_idx.items():
            d_name_lower = d_name.lower()
            drug_name_lower = drug_name.lower()
            # Check if one contains the other
            if drug_name_lower in d_name_lower or d_name_lower in drug_name_lower:
                drug_idx = d_idx
                break

        if drug_idx is None:
            missing_drug += 1
            continue

        if gene_name not in gene_to_idx:
            missing_gene += 1
            continue

        gene_idx = gene_to_idx[gene_name]
        dti_edges.append((drug_idx, gene_idx))
        valid += 1

    print(f"Found valid DTI edges: {valid}")
    print(f"Missing drugs (skipped): {missing_drug}")
    print(f"Missing genes (skipped): {missing_gene}")

    if len(dti_edges) > 0:
        edge_index = np.array(dti_edges).T
        print(f"Final DTI edge_index shape: {edge_index.shape}")
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return edge_index

# ============================================================
# Step 5: Extract DDS edges (drug-drug) from PrimeKG
# ============================================================

def extract_dds(primekg_path: str, drug_to_idx: dict):
    """Extract drug-drug interaction/association edges from PrimeKG"""
    print(f"Extracting DDS edges from {primekg_path}")
    df = pd.read_csv(primekg_path, low_memory=False)
    print(f"Original PrimeKG entries: {len(df)}")

    dds_edges = []
    edge_weights = []
    missing = 0
    valid = 0

    for _, row in df.iterrows():
        x_type = str(row['x_type'])
        y_type = str(row['y_type'])

        # We want drug-drug interactions
        is_dds = False
        drug1_name = None
        drug2_name = None

        if 'drug' in x_type.lower() and 'drug' in y_type.lower():
            is_dds = True
            drug1_name = str(row['x_name']).strip()
            drug2_name = str(row['y_name']).strip()

        if not is_dds:
            continue

        # Find drug indices
        idx1 = None
        idx2 = None
        for d_name, d_idx in drug_to_idx.items():
            d_name_lower = d_name.lower()
            if drug1_name.lower() in d_name_lower or d_name_lower in drug1_name.lower():
                idx1 = d_idx
                break
        for d_name, d_idx in drug_to_idx.items():
            d_name_lower = d_name.lower()
            if drug2_name.lower() in d_name_lower or d_name_lower in drug2_name.lower():
                idx2 = d_idx
                break

        if idx1 is not None and idx2 is not None:
            # Get relation score - use confidence if available, else 1.0
            if 'score' in row and not pd.isna(row['score']):
                score = float(row['score'])
            else:
                score = 1.0
            dds_edges.append((idx1, idx2))
            edge_weights.append(score)
            valid += 1
        else:
            missing += 1

    print(f"Found valid DDS edges: {valid}")
    print(f"Missing drugs (skipped): {missing}")

    if len(dds_edges) > 0:
        edge_index = np.array(dds_edges).T
        print(f"Final DDS edge_index shape: {edge_index.shape}")
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_weights = np.array([])

    return edge_index, np.array(edge_weights)

# ============================================================
# Main
# ============================================================

def main():
    # Paths
    gene_emb_path = "/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/base/other_data/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"
    nips_h5ad = "/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/external_dataset_nips/raw/nips_pp_scFM_resplit.filtered_ESM2.h5ad"
    ppi_path = "/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/knowledge/GeneKG_Encoder/data_csvs/PPI.csv"
    hgnc_path = "/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/base/other_data/hgnc_complete_set.txt"
    primekg_path = "/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/knowledge/Gene_Drug_KG/preprocessed/primekg_filtered.csv"

    # Output directory
    output_dir = "/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/baselines/XPert/HG_data"
    os.makedirs(output_dir, exist_ok=True)

    # 0. Build HGNC mapping
    hgnc_to_symbol = build_hgnc_mapping(hgnc_path)

    # 1. Load genes from h5ad - keep full 5120 dimension
    gene_list, gene_to_idx, gene_emb_matrix = load_genes_from_h5ad(nips_h5ad, gene_emb_path)
    num_genes = len(gene_list)
    out_gene_feat = os.path.join(output_dir, f"nips_all_gene_node_feat_{num_genes}.npy")
    np.save(out_gene_feat, gene_emb_matrix)
    print(f"Saved gene node features to {out_gene_feat}")

    # Save gene list
    with open(os.path.join(output_dir, f"nips_gene_list_{num_genes}.json"), 'w') as f:
        json.dump(gene_list, f, indent=2)

    # 2. Load drugs from h5ad
    drug_list, drug_names, drug_smiles, drug_to_idx = load_unique_drugs_from_h5ad(nips_h5ad)
    num_drugs = len(drug_list)

    # Save mappings
    drug_smiles_map = {idx: smi for idx, (name, smi) in enumerate(drug_list)}
    with open(os.path.join(output_dir, f"nips_drugs_smiles_{num_drugs}.json"), 'w') as f:
        json.dump(drug_smiles_map, f, indent=2)
    drug_name_map = {idx: name for idx, (name, smi) in enumerate(drug_list)}
    with open(os.path.join(output_dir, f"nips_drugs_names_{num_drugs}.json"), 'w') as f:
        json.dump(drug_name_map, f, indent=2)

    print(f"Saved drug mappings to {output_dir}")

    # 3. Extract PPI edges
    ppi_edge_index, ppi_weights = extract_ppi(ppi_path, gene_to_idx, hgnc_to_symbol)
    ppi_out = os.path.join(output_dir, f"nips_PPI_edge_index_all_{ppi_edge_index.shape[1]}_pairs_weighted.npy")
    np.save(ppi_out, {'edge_index': ppi_edge_index, 'edge_attr': ppi_weights}, allow_pickle=True)
    print(f"Saved PPI edges to {ppi_out}")

    # 4. Extract DTI edges
    dti_edge_index = extract_dti(primekg_path, drug_to_idx, gene_to_idx)
    dti_out = os.path.join(output_dir, f"nips_DTI_edge_index_all_{dti_edge_index.shape[1]}_pairs.npy")
    np.save(dti_out, {'edge_index': dti_edge_index}, allow_pickle=True)
    print(f"Saved DTI edges to {dti_out}")

    # 5. Extract DDS edges
    dds_edge_index, dds_weights = extract_dds(primekg_path, drug_to_idx)
    dds_out = os.path.join(output_dir, f"nips_DDS_edge_index_all_{dds_edge_index.shape[1]}_pairs_weighted.npy")
    np.save(dds_out, {'edge_index': dds_edge_index, 'edge_attr': dds_weights}, allow_pickle=True)
    print(f"Saved DDS edges to {dds_out}")

    # 6. Save complete mappings
    mappings = {
        'gene_to_idx': gene_to_idx,
        'drug_to_idx': drug_to_idx,
        'num_genes': num_genes,
        'num_drugs': num_drugs,
        'stats': {
            'ppi_edges': ppi_edge_index.shape[1],
            'dti_edges': dti_edge_index.shape[1],
            'dds_edges': dds_edge_index.shape[1],
        }
    }
    with open(os.path.join(output_dir, f"nips_mappings.json"), 'w') as f:
        json.dump(mappings, f, indent=2)

    print("\n" + "="*60)
    print("NIPS HG PREPROCESSING SUMMARY:")
    print("="*60)
    print(f"Genes: {num_genes} (full {gene_emb_matrix.shape[1]} dimensions)")
    print(f"Drugs: {num_drugs}")
    print(f"PPI edges: {ppi_edge_index.shape[1]}")
    print(f"DTI edges: {dti_edge_index.shape[1]}")
    print(f"DDS edges: {dds_edge_index.shape[1]}")
    print(f"\nAll files saved to: {output_dir}")
    print("Next step: generate drug node features with UniMol")

if __name__ == "__main__":
    main()
