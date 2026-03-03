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


from flash_transformer import FlashTransformerEncoderLayer, FlashTransformerEncoder
from data.utils import (
    # compute_gene_overlap_cross_pert,
    get_embedding_cfg,
    get_dataset_cfg,
    # compute_pearson_delta,
    # compute_perturbation_ranking_score,
)
from data.tahoe import create_dataloader


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
        # 判断是否需要多个token降维头来降维
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
        # 【新增代码】：根据嵌入类型选择基因嵌入层
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

    # def _compute_embedding_for_batch(self, batch):
    #     # print('in _compute_embedding_for_batch -- device:', self.device)
    #     t0 = time.time()
    #     batch_sentences = batch[0].to(self.device, non_blocking=True)
    #     batch_sentences_counts = batch[1].to(self.device, non_blocking=True)
    #     print('batch_fetch:', time.time()-t0)

    #     # convert the cell sentence and task sentence into embeddings
    #     t0 = time.time()
    #     batch_sentences = self.pe_embedding(batch_sentences)
    #     print('pe_embedding:', time.time()-t0)

    #     t0 = time.time()
    #     batch_sentences = batch_sentences.contiguous()
    #     batch_sentences = nn.functional.normalize(batch_sentences, dim=2)
        

    #     # Add a learnable CLS token to the beginning of the sentence
    #     batch_sentences_new = batch_sentences.clone()
    #     batch_sentences_new[:, 0, :] = self.cls_token.expand(batch_sentences.size(0), -1)
    #     batch_sentences = batch_sentences_new
        

    #     # Optionally add a learnable dataset token to the end of the sentence
    #     if self.dataset_token is not None:
    #         dataset_token = self.dataset_token.expand(batch_sentences.size(0), -1).unsqueeze(1)
    #         batch_sentences = torch.cat((batch_sentences, dataset_token), dim=1)
    #     print('batch_sentences:', time.time()-t0)

    #     t0 = time.time()
    #     _, embedding, dataset_emb = self.forward(
    #         src=batch_sentences, counts=batch_sentences_counts
    #     )
    #     print('forward:', time.time()-t0)

    #     return embedding, dataset_emb

    def infer(self, batch):
        embedding, dataset_emb = self._compute_embedding_for_batch(batch)
        return embedding

    def get_gene_embedding(self, genes):
        if self.protein_embeds is None:
            self.protein_embeds = torch.load(get_embedding_cfg(self.cfg).all_embeddings, weights_only=False)

        # 【新增代码】：根据嵌入类型处理基因嵌入
        if self.cfg.embeddings.current == 'esm2-cellxgene-gene-kg-bert':
            # 假设protein_embeds包含两种嵌入的拼接
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
            
            protein_embeds_esm = torch.stack(protein_embeds_esm).to(self.device)
            protein_embeds_kg = torch.stack(protein_embeds_kg).to(self.device)
            
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
            protein_embeds = torch.stack(protein_embeds).to(self.device)
            if protein_embeds.sum() == 0:
                raise ValueError("No gene embeddings found")

            return self.gene_embedding_layer(protein_embeds)
        
    def _compute_embedding_for_batch(self, batch):
        with torch.cuda.amp.autocast():
            # t0 = time.time()
            batch_sentences = batch[0].to(self.device, non_blocking=True)
            batch_sentences_counts = batch[1].to(self.device, non_blocking=True)
            # print('batch_fetch:', time.time()-t0)

            # t0 = time.time()
            batch_sentences = self.pe_embedding(batch_sentences)
            # print('pe_embedding:', time.time()-t0)

            # t0 = time.time()
            batch_sentences = nn.functional.normalize(batch_sentences, dim=2)
            
            # 🚀 避免原地操作：用cat替代直接赋值
            cls_tokens = self.cls_token.expand(batch_sentences.size(0), 1, -1)
            batch_sentences = torch.cat([
                cls_tokens,
                batch_sentences[:, 1:, :]
            ], dim=1)
            
            if self.dataset_token is not None:
                dataset_token = self.dataset_token.expand(batch_sentences.size(0), 1, -1)
                batch_sentences = torch.cat((batch_sentences, dataset_token), dim=1)
            # print('batch_sentences:', time.time()-t0)

            # t0 = time.time()
            _, embedding, dataset_emb = self.forward(
                src=batch_sentences, counts=batch_sentences_counts
            )
            # print('forward:', time.time()-t0)

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

    def _predict_exp_for_adata(self, adata, dataset_name, pert_col):
        dataloader = create_dataloader(
            self.cfg,
            adata=adata,
            adata_name=dataset_name,
            shuffle=False,
            sentence_collator=self.collater,
        )
        try:
            gene_embeds = self.get_gene_embedding(adata.var.index)
        except:
            gene_embeds = self.get_gene_embedding(adata.var["gene_symbols"])
        emb_batches = []
        ds_emb_batches = []
        logprob_batches = []
        for batch in tqdm(
            dataloader,
            position=0,
            leave=True,
            ncols=100,
            desc=f"Embeddings for {dataset_name}",
        ):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            _, _, _, emb, ds_emb = self._compute_embedding_for_batch(batch)

            # now decode from the embedding
            task_counts = None
            sampled_rda = None
            if self.z_dim_rd == 1:
                Y = batch[2].to(self.device)
                nan_y = Y.masked_fill(Y == 0, float("nan"))[:, : self.cfg.dataset.P + self.cfg.dataset.N]
                task_counts = torch.nanmean(nan_y, dim=1) if self.cfg.model.rda else None
                sampled_rda = None

            ds_emb = None
            if self.dataset_token is not None:
                ds_emb = self.dataset_embedder(ds_emb)

            emb_batches.append(emb.detach().cpu().numpy())
            ds_emb_batches.append(ds_emb.detach().cpu().numpy())

            merged_embs = StateEmbeddingModel.resize_batch(emb, gene_embeds, task_counts, sampled_rda, ds_emb)
            logprobs_batch = self.binary_decoder(merged_embs)
            logprobs_batch = logprobs_batch.detach().cpu().numpy()
            logprob_batches.append(logprobs_batch.squeeze())

        logprob_batches = np.vstack(logprob_batches)
        adata.obsm["X_emb"] = np.vstack(emb_batches)
        adata.obsm["X_ds_emb"] = np.vstack(ds_emb_batches)
        adata.obsm["X_emb"] = np.concatenate([adata.obsm["X_emb"], adata.obsm["X_ds_emb"]], axis=-1)

        # Free up memory from logprob_batches if possible
        probs_df = pd.DataFrame(logprob_batches)
        del logprob_batches
        torch.cuda.empty_cache()
        probs_df[pert_col] = adata.obs[pert_col].values

        # Read config properties
        k = self.cfg.validations.diff_exp.top_k_rank
        pert_col = self.cfg.validations.diff_exp.obs_pert_col
        non_targating_label = self.cfg.validations.diff_exp.obs_filter_label

        probs_df = probs_df.groupby(pert_col).mean()
        ctrl = probs_df.loc[non_targating_label].values
        pert_effects = np.abs(probs_df - ctrl)
        top_k_indices = np.argsort(pert_effects.values, axis=1)[:, -k:][:, ::-1]
        top_k_genes = np.array(adata.var.index)[top_k_indices]
        de_genes = pd.DataFrame(top_k_genes)
        de_genes.index = pert_effects.index.values

        return de_genes

    def forward_o(self, src: Tensor, counts=None, dataset_nums=None):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, ntoken]
        Returns:
            output Tensor of shape [batch_size, seq_len, ntoken]
        """

        # print('cp1:', src.shape, counts.shape)
        # 【新增代码】：根据嵌入类型处理输入编码
        sqrt_d_model = math.sqrt(self.d_model)

        if self.cfg.embeddings.current == 'esm2-cellxgene-gene-kg-bert':
            # 假设src包含两种嵌入的拼接
            token_dim_1 = get_embedding_cfg(self.cfg).size_esm

            # 🚀 优化3: 避免多次切片，一次性处理
            # src_esm = src[:, :, :token_dim_1]
            # src_kg = src[:, :, token_dim_1:]
            src_esm, src_kg = torch.split(src, [token_dim_1, src.size(-1) - token_dim_1], dim=-1)
    
            src_esm_encoded = self.encoder_esm(src_esm)
            src_kg_encoded = self.encoder_kg(src_kg)
            src_fused = torch.cat([src_esm_encoded, src_kg_encoded], dim=-1)
            src = self.fusion_layer(src_fused) * sqrt_d_model
        else:
            src = self.encoder(src) * sqrt_d_model
        # src = self.encoder(src) * sqrt_d_model

        if counts is not None:
            # print('hi')
            # scFoundation-style soft binning for counts
            counts = counts.unsqueeze(-1)  # now B x H x 1

            # 🚀 优化6: 预计算bin相关数据
            if not hasattr(self, '_bin_indices_cached'):
                self._bin_indices_cached = torch.arange(10, device=self.device)

            # Step 1: Transform count values into bin distribution
            bin_weights = self.count_encoder(counts)  # B x H x 10
            bin_weights = F.softmax(bin_weights, dim=-1)  # Convert to probabilities over bins

            # Step 2: Get bin embeddings
            # bin_indices = torch.arange(10, device=self.device)  # 10 bins
            bin_embeddings = self.bin_encoder(self._bin_indices_cached)  # 10 x d_model

            # Step 3: Compute weighted sum of bin embeddings
            count_emb = torch.matmul(bin_weights, bin_embeddings)

            if self.dataset_token is not None:
                # print('hello')
                # append B x 1 x d_model to count_emb of all zeros
                if not hasattr(self, '_dataset_count_emb_template'):
                    # 缓存一个模板，只创建一次
                    self._dataset_count_emb_template = torch.zeros(1, 1, count_emb.size(2), 
                                                     device=self.device, dtype=count_emb.dtype)
                # dataset_count_emb = torch.zeros(count_emb.size(0), 1, count_emb.size(2), device=self.device)
                dataset_count_emb = self._dataset_count_emb_template.expand(count_emb.size(0), -1, -1)
                count_emb = torch.cat((count_emb, dataset_count_emb), dim=1)  # B x H x d_model

            # print('cp2:', src.shape, count_emb.shape)
            # Add count embeddings to token embeddings
            src = (
                src + count_emb
            )  # should both be B x H x self.d_model, or B x (H + 1) x self.d_model if dataset correction

        output = self.transformer_encoder(src, src_key_padding_mask=None)
        gene_output = self.decoder(output)  # batch x seq_len x 128
        # In the new format, the cls token, which is at the 0 index mark, is the output.
        embedding = gene_output[:, 0, :]  # select only the CLS token.
        embedding = nn.functional.normalize(embedding, dim=1)  # Normalize.

        # we must be in train mode to use dataset correction
        dataset_emb = None
        if self.dataset_token is not None:
            dataset_emb = gene_output[:, -1, :]

        return gene_output, embedding, dataset_emb
    
    def forward(self, src: Tensor, counts=None, dataset_nums=None):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, ntoken]
        Returns:
            output Tensor of shape [batch_size, seq_len, ntoken]
        """
        
        # 🚀 优化1: 预计算常量
        sqrt_d_model = math.sqrt(self.d_model)
        
        # 🚀 优化2: 根据嵌入类型处理输入编码
        if self.cfg.embeddings.current == 'esm2-cellxgene-gene-kg-bert':
            token_dim_1 = get_embedding_cfg(self.cfg).size_esm
            
            # 🚀 优化3: 一次性split，避免多次切片
            src_esm, src_kg = torch.split(src, [token_dim_1, src.size(-1) - token_dim_1], dim=-1)
            
            # 🚀 优化4: 并行计算编码器（如果可能的话）
            src_esm_encoded = self.encoder_esm(src_esm)
            src_kg_encoded = self.encoder_kg(src_kg)
            
            # 🚀 优化5: 直接融合，减少中间变量
            src = self.fusion_layer(torch.cat([src_esm_encoded, src_kg_encoded], dim=-1)) * sqrt_d_model
        else:
            src = self.encoder(src) * sqrt_d_model

        # 🚀 优化6: 计数处理优化
        if counts is not None:
            counts = counts.unsqueeze(-1)  # B x H x 1

            # 🚀 优化7: 缓存bin_indices，避免重复创建
            if not hasattr(self, '_bin_indices_cached'):
                self._bin_indices_cached = torch.arange(10, device=self.device)

            # 🚀 优化8: 合并softmax操作
            bin_weights = F.softmax(self.count_encoder(counts), dim=-1)  # B x H x 10
            
            # 🚀 优化9: 预计算bin_embeddings（如果bins不变的话）
            bin_embeddings = self.bin_encoder(self._bin_indices_cached)  # 10 x d_model
            count_emb = torch.matmul(bin_weights, bin_embeddings)

            # 🚀 优化10: 缓存dataset模板，避免重复创建zeros
            if self.dataset_token is not None:
                if not hasattr(self, '_dataset_count_emb_template'):
                    self._dataset_count_emb_template = torch.zeros(
                        1, 1, count_emb.size(2), 
                        device=self.device, 
                        dtype=count_emb.dtype
                    )
                
                # 🚀 优化11: 使用expand代替zeros创建
                dataset_count_emb = self._dataset_count_emb_template.expand(count_emb.size(0), -1, -1)
                count_emb = torch.cat((count_emb, dataset_count_emb), dim=1)

            # 🚀 优化12: 原地加法操作
            src = src + count_emb

        # 🚀 优化13: Transformer计算
        output = self.transformer_encoder(src, src_key_padding_mask=None)
        
        # 🚀 优化14: 解码器计算
        gene_output = self.decoder(output)  # batch x seq_len x 128
        
        # 🚀 优化15: 一次性提取所有需要的输出，避免重复索引
        embedding = nn.functional.normalize(gene_output[:, 0, :], dim=1)  # CLS token
        
        dataset_emb = None
        if self.dataset_token is not None:
            dataset_emb = gene_output[:, -1, :]  # dataset token

        return gene_output, embedding, dataset_emb