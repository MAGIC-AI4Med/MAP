import warnings
warnings.filterwarnings("ignore")

import math
import logging
import numpy as np
import pandas as pd
import scanpy as sc
import torch.nn.functional as F
import torch
import lightning as L
import sys
import time

from tqdm.auto import tqdm
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, BCEWithLogitsLoss

sys.path.append("./")
sys.path.append("../")
sys.path.append("model")

from flash_transformer import FlashTransformerEncoderLayer, FlashTransformerEncoder
from data.utils import (
    get_embedding_cfg,
    get_dataset_cfg,
)


class SkipBlock(nn.Module):
    def __init__(self, in_features):
        """
        Given input X of size in_features
        - out = layernorm(x + MLP(MLP(X))

        """
        super().__init__()
        self.dim = in_features
        self.intermediate_dense = nn.Linear(in_features, in_features * 2, bias=True)
        self.dense = nn.Linear(in_features * 2, in_features, bias=True)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x):
        residual = x
        x = self.intermediate_dense(x)
        x = self.activation(x)
        x = self.dense(x)
        x = self.layer_norm(x + residual)
        return x
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1536):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
    

def nanstd(x):
    return torch.sqrt(torch.nanmean(torch.pow(x - torch.nanmean(x, dim=-1).unsqueeze(-1), 2), dim=-1))


class StateEmbeddingModel(L.LightningModule):
    def __init__(
        self,
        token_dim: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        output_dim: int,
        dropout: float = 0.0,
        warmup_steps: int = 0,
        compiled: bool = False,
        max_lr=4e-4,
        emb_cnt=145469,
        emb_size=5120,
        cfg=None,
        collater=None,
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.save_hyperparameters(
            "token_dim", "d_model", "nhead", "d_hid", "nlayers", "output_dim", "dropout",
            "warmup_steps", "compiled", "max_lr", "emb_cnt", "emb_size"
        )
        self.cfg = cfg
        self.compiled = compiled
        self.model_type = "Transformer"
        self.cls_token = nn.Parameter(torch.randn(1, token_dim))

        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.dropout = dropout
        self.max_lr = max_lr
        self.collater = collater

        # Encodes Tokens
        if self.cfg.embeddings.current == 'esm2-cellxgene-gene-kg-bert':
            token_dim_1 = get_embedding_cfg(self.cfg).size_esm
            token_dim_2 = get_embedding_cfg(self.cfg).size_kg
            self.encoder_esm = nn.Sequential(
                nn.Linear(token_dim_1, d_model, bias=True),
                nn.LayerNorm(d_model),  # Moved before activation
                nn.SiLU(),  # Changed to SiLU
            )
            self.encoder_kg = nn.Sequential(
                nn.Linear(token_dim_2, d_model, bias=True),
                nn.LayerNorm(d_model),  # Moved before activation
                nn.SiLU(),  # Changed to SiLU
            )
            self.fusion_layer = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU()
            )
        else:   
            self.encoder = nn.Sequential(
                nn.Linear(token_dim, d_model, bias=True),
                nn.LayerNorm(d_model),  # Moved before activation
                nn.SiLU(),  # Changed to SiLU
            )



        # Check the configuration flag whether to use Flash Attention
        use_flash = getattr(self.cfg.model, "use_flash_attention", False)
        if use_flash and FlashTransformerEncoderLayer is not None:
            print("!!! Using Flash Attention !!!")
            # Create a list of FlashTransformerEncoderLayer instances
            layers = [FlashTransformerEncoderLayer(d_model, nhead, d_hid, dropout=dropout) for _ in range(nlayers)]
            self.transformer_encoder = FlashTransformerEncoder(layers)
        else:
            # Fallback to the standard PyTorch TransformerEncoderLayer
            encoder_layer = TransformerEncoderLayer(
                d_model, nhead, d_hid, dropout=dropout, batch_first=True, activation="gelu"
            )
            self.transformer_encoder = TransformerEncoder(encoder_layer, nlayers)

        if compiled:
            self.transformer_encoder = torch.compile(self.transformer_encoder)

        self.d_model = d_model
        self.dropout = dropout

        self.decoder = nn.Sequential(
            SkipBlock(d_model),
            nn.Linear(d_model, output_dim, bias=True),
        )

        if compiled:
            self.decoder = torch.compile(self.decoder)

        self.z_dim_rd = 1 if self.cfg.model.rda else 0
        self.z_dim_ds = 10 if self.cfg.model.get("dataset_correction", False) else 0
        self.z_dim = self.z_dim_rd + self.z_dim_ds

        self.binary_decoder = nn.Sequential(
            SkipBlock(output_dim + d_model + self.z_dim),
            SkipBlock(output_dim + d_model + self.z_dim),
            nn.Linear(output_dim + d_model + self.z_dim, 1, bias=True),
        )

        if self.cfg.model.counts:
            self.bin_encoder = nn.Embedding(10, d_model)
            self.count_encoder = nn.Sequential(
                nn.Linear(1, 512, bias=True),
                nn.LeakyReLU(),
                nn.Linear(512, 10),
            )

        if compiled:
            self.binary_decoder = torch.compile(self.binary_decoder)

        # Encodes Tokens for Decoder
        if self.cfg.embeddings.current == 'esm2-cellxgene-gene-kg-bert':
            self.gene_embedding_layer = self.fusion_layer  # reuse fusion layer for gene embeddings
        else:
            self.gene_embedding_layer = self.encoder  # reuse this layer


        if compiled:
            self.gene_embedding_layer = torch.compile(self.gene_embedding_layer)

        self.pe_embedding = (
            None  # TODO: make this cleaner for the type checker, right now it gets set externally after model init
        )
        self.step_ctr = 0

        self.true_top_genes = None
        self.protein_embeds = None

        self._last_val_de_check = 0
        self._last_val_perturbation_check = 0

        if getattr(self.cfg.model, "dataset_correction", False):
            self.dataset_token = nn.Parameter(torch.randn(1, token_dim))
            self.dataset_embedder = nn.Linear(output_dim, 10)

            # Assume self.cfg.model.num_datasets is set to the number of unique datasets.
            num_dataset = get_dataset_cfg(self.cfg).num_datasets
            self.dataset_encoder = nn.Sequential(
                nn.Linear(output_dim, d_model),
                nn.SiLU(),
                nn.LayerNorm(d_model),
                nn.Dropout(0.1),
                nn.Linear(d_model, num_dataset),
            )

            # this should be a classification label loss
            self.dataset_loss = nn.CrossEntropyLoss()
        else:
            self.dataset_token = None

        if self.cfg.model.counts:
            self.register_buffer('_bin_indices_cached', torch.arange(10))

    def infer(self, batch):
        embedding, dataset_emb = self._compute_embedding_for_batch(batch)
        return embedding

    def get_gene_embedding(self, genes):
        if self.protein_embeds is None:
            self.protein_embeds = torch.load(get_embedding_cfg(self.cfg).all_embeddings, weights_only=False)

        device = next(self.parameters()).device

        if self.cfg.embeddings.current == 'esm2-cellxgene-gene-kg-bert':
            token_dim_1 = get_embedding_cfg(self.cfg).size_esm
            
            protein_embeds_esm = []
            protein_embeds_kg = []
            
            for x in genes:
                if x in self.protein_embeds:
                    full_embed = self.protein_embeds[x]
                    esm_embed = full_embed[:token_dim_1]
                    kg_embed = full_embed[token_dim_1:]
                    protein_embeds_esm.append(esm_embed)
                    protein_embeds_kg.append(kg_embed)
                else:
                    protein_embeds_esm.append(torch.zeros(token_dim_1))
                    protein_embeds_kg.append(torch.zeros(get_embedding_cfg(self.cfg).size_kg))
            
            protein_embeds_esm = torch.stack(protein_embeds_esm).to(device)
            protein_embeds_kg = torch.stack(protein_embeds_kg).to(device)
            
            if protein_embeds_esm.sum() == 0 and protein_embeds_kg.sum() == 0:
                raise ValueError("No gene embeddings found")
            
            # 编码并融合
            esm_encoded = self.encoder_esm(protein_embeds_esm)
            kg_encoded = self.encoder_kg(protein_embeds_kg)
            fused = torch.cat([esm_encoded, kg_encoded], dim=-1)
            return self.fusion_layer(fused)

        else:
            protein_embeds = [
                self.protein_embeds[x] if x in self.protein_embeds else torch.zeros(get_embedding_cfg(self.cfg).size)
                for x in genes
            ]
            protein_embeds = torch.stack(protein_embeds).to(device)
            if protein_embeds.sum() == 0:
                raise ValueError("No gene embeddings found")

            return self.gene_embedding_layer(protein_embeds)
        
    def _compute_embedding_for_batch(self, batch):
        device = next(self.parameters()).device
        
        with torch.cuda.amp.autocast():
            batch_sentences = batch[0].to(device, non_blocking=True)
            batch_sentences_counts = batch[1].to(device, non_blocking=True)

            batch_sentences = self.pe_embedding(batch_sentences)

            batch_sentences = nn.functional.normalize(batch_sentences, dim=2)
            
            cls_tokens = self.cls_token.expand(batch_sentences.size(0), 1, -1)
            batch_sentences = torch.cat([
                cls_tokens,
                batch_sentences[:, 1:, :]
            ], dim=1)
            
            if self.dataset_token is not None:
                dataset_token = self.dataset_token.expand(batch_sentences.size(0), 1, -1)
                batch_sentences = torch.cat((batch_sentences, dataset_token), dim=1)

            _, embedding, dataset_emb = self.forward(
                src=batch_sentences, counts=batch_sentences_counts
            )

        return embedding, dataset_emb

    @staticmethod
    def resize_batch(cell_embeds, task_embeds, task_counts=None, sampled_rda=None, ds_emb=None):
        A = task_embeds.unsqueeze(0).repeat(cell_embeds.size(0), 1, 1)
        B = cell_embeds.unsqueeze(1).repeat(1, task_embeds.size(0), 1)
        if sampled_rda is not None:
            # your code here that computes mu and std dev from Y
            reshaped_counts = sampled_rda.unsqueeze(1)
            reshaped_counts = reshaped_counts.repeat(1, A.shape[1], 1)
            combine = torch.cat((A, B, reshaped_counts), dim=2)
        elif task_counts is not None:
            reshaped_counts = task_counts.unsqueeze(1).unsqueeze(2)
            reshaped_counts = reshaped_counts.repeat(1, A.shape[1], 1)

            # Concatenate all three tensors along the third dimension
            combine = torch.cat((A, B, reshaped_counts), dim=2)
        else:
            # Original behavior if total_counts is None
            combine = torch.cat((A, B), dim=2)

        if ds_emb is not None:
            # ds_emb is a tensor of shape (batch_size, 10). concatenate it to the combine tensor
            ds_emb = ds_emb.unsqueeze(1).repeat(1, A.shape[1], 1)
            combine = torch.cat((combine, ds_emb), dim=2)

        return combine
    
    def forward(self, src: Tensor, counts=None, dataset_nums=None, profile: bool = False, **kwargs):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, ntoken]
        Returns:
            output Tensor of shape [batch_size, seq_len, ntoken]
        """
        start_time = time.time()
        sqrt_d_model = math.sqrt(self.d_model)
        
        if self.cfg.embeddings.current == 'esm2-cellxgene-gene-kg-bert':
            token_dim_1 = get_embedding_cfg(self.cfg).size_esm
            src_esm, src_kg = torch.split(src, [token_dim_1, src.size(-1) - token_dim_1], dim=-1)
            src_esm_encoded = self.encoder_esm(src_esm)
            src_kg_encoded = self.encoder_kg(src_kg)
            src = self.fusion_layer(torch.cat([src_esm_encoded, src_kg_encoded], dim=-1)) * sqrt_d_model
        else:
            src = self.encoder(src) * sqrt_d_model

        encoder_time = time.time()


        counts_time = encoder_time
        if counts is not None:
            counts = counts.unsqueeze(-1)
            bin_weights = F.softmax(self.count_encoder(counts), dim=-1)
            bin_embeddings = self.bin_encoder(self._bin_indices_cached)
            count_emb = torch.matmul(bin_weights, bin_embeddings)

            if self.dataset_token is not None:
                device = next(self.parameters()).device
                if not hasattr(self, '_dataset_count_emb_template'):
                    self._dataset_count_emb_template = torch.zeros(
                        1, 1, count_emb.size(2), 
                        device=device,
                        dtype=count_emb.dtype
                    )
                dataset_count_emb = self._dataset_count_emb_template.expand(count_emb.size(0), -1, -1)
                count_emb = torch.cat((count_emb, dataset_count_emb), dim=1)

            src = src + count_emb
            counts_time = time.time()


        output = self.transformer_encoder(src, src_key_padding_mask=None)
        transformer_time = time.time()
        

        gene_output = self.decoder(output)
        embedding = nn.functional.normalize(gene_output[:, 0, :], dim=1)
        
        dataset_emb = None
        if self.dataset_token is not None:
            dataset_emb = gene_output[:, -1, :]

        decoder_time = time.time()
        if profile:
            print(
                f"[SE.forward] Encoder: {encoder_time - start_time:.4f}s, "
                f"Counts: {counts_time - encoder_time:.4f}s, "
                f"Transformer: {transformer_time - counts_time:.4f}s, "
                f"Decoder: {decoder_time - transformer_time:.4f}s, "
                f"Total: {decoder_time - start_time:.4f}s"
            )

        return gene_output, embedding, dataset_emb