import json
import torch
import torch.nn as nn
from safetensors.torch import load_file
from collections import OrderedDict

from model.se import StateEmbeddingModel
from model.pert_ca import PerturbationEncoder_ca
from model.pert import PerturbationEncoder
from data.utils import get_embedding_cfg

class MAPmodel(nn.Module):
    def __init__(
        self,
        se_ckpt,
        se_cfg,
        smile_encoder,
        hvg_info=None,
        freeze_se: bool = True,
    ):
        super().__init__()
        self.se_ckpt = se_ckpt
        self.freeze_se = freeze_se
        self.smile_encoder = smile_encoder
        self.hvg_info = hvg_info

        self.se = StateEmbeddingModel(
            token_dim=5120,
            d_model=2048,
            nhead=16,
            d_hid=2048,
            nlayers=16,
            output_dim=2048,
            dropout=0.1,
            warmup_steps=0,
            compiled=False,
            max_lr=4e-4,
            emb_size=5120,
            cfg=se_cfg,
            collater=None,
        )

        all_pe = torch.load(get_embedding_cfg(se_cfg).all_embeddings, weights_only=False)
        if isinstance(all_pe, dict):
            all_pe = torch.vstack(list(all_pe.values()))
        all_pe.requires_grad = False
        self.se.pe_embedding = nn.Embedding.from_pretrained(all_pe)
        
        self.load_se_ckpt()

        if self.freeze_se:
            for param in self.se.parameters():
                param.requires_grad = False
            self.se.eval()

        self.pert_model = PerturbationEncoder(
            dim_emb=1024,
            encode_conc=False,
            num_heads=8,
            dropout=0.1,
            is_hvg_space=False,
            detach_decoder=False,
            transformer_backbone_key='llama',
            smile_encoder=self.smile_encoder
        )


    def load_se_ckpt(self):
        state = load_file(self.se_ckpt)
        want = set(self.se.state_dict().keys())
        new_state = OrderedDict()

        pairs = [
            ("encoder.0.weight", "gene_embedding_layer.0.weight"),
            ("encoder.0.bias",   "gene_embedding_layer.0.bias"),
            ("encoder.1.weight", "gene_embedding_layer.1.weight"),
            ("encoder.1.bias",   "gene_embedding_layer.1.bias"),
        ]
        
        for k, v in state.items():
            if not k.startswith('pe_embedding'):
                new_state[k] = v

        for a, b in pairs:
            if a in state and b in want and b not in new_state:
                new_state[b] = state[a]
            if b in state and a in want and a not in new_state:
                new_state[a] = state[b]
        
        missing, unexpected = self.se.load_state_dict(new_state, strict=False)
        
        expected_missing = {
            '_bin_indices_cached', 
            'pe_embedding.weight', 
        }
        
        critical_missing = [k for k in missing if k not in expected_missing]
        

        if critical_missing:
            print(f"⚠️  missing: {critical_missing}")
        if unexpected:
            print(f"⚠️  unexpected: {unexpected}")
    
    def forward(self, ctrl_src, ctrl_counts, smiles, concs=None):
        """
        Args:
            ctrl_src: [B, S, sentence_len]
            ctrl_counts: [B, S, sentence_len]
            smiles: List[str], len = B
            concs: [B]
        Returns:
            perturbed_embs: [B, S, d_emb]
            perturbed_hvgs: [B, S, 2000]
        """
        B, S, L = ctrl_src.shape
        gene_ids = ctrl_src.reshape(B * S, L)
        expressions = ctrl_counts.reshape(B * S, L)

        if self.se.pe_embedding is not None:
            src = self.se.pe_embedding(gene_ids)

            if not self.pert_model_hvg_ids_already_set and self.hvg_info is not None:
                hvg_ids_tensor = torch.tensor(self.hvg_ids, device=self.se.pe_embedding.weight.device).reshape(1, -1)
                self.pert_model.hvg_esm_tokens = self.se.pe_embedding(hvg_ids_tensor)
                self.pert_model_hvg_ids_already_set = True
         
        esm_tokens = src[:, 1:, :].clone().detach()
        
        src = torch.nn.functional.normalize(src, dim=2)
        cls_tokens = self.se.cls_token.expand(src.size(0), 1, -1)
        src = torch.cat([
            cls_tokens,
            src[:, 1:, :]
        ], dim=1)

        if self.se.dataset_token is not None:
            dataset_token = self.se.dataset_token.expand(src.size(0), 1, -1)
            src = torch.cat((src, dataset_token), dim=1)

        if self.freeze_se:
            with torch.no_grad():
                gene_output, embedding, _ = self.se(
                    src=src,
                    counts=expressions,
                    dataset_nums=None,
                    profile=False,
                ) 
        else:
            gene_output, embedding, _ = self.se(
                src=src,
                counts=expressions,
                dataset_nums=None,
                profile=False,
            )
        
        gene_output = gene_output[:, 1:-1, :]
        pred_embs, pred_hvgs = self.pert_model(gene_output, embedding, esm_tokens, smiles, concs)

        return pred_embs, pred_hvgs