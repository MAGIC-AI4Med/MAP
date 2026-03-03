# =========================
# File: model/pert_ddp.py
# =========================
import torch
import torch.nn as nn

from model.gene_decoders import LatentToGeneDecoder
from model.transformer_encoder import get_transformer_backbone, ST_small_transformer_backbone_kwargs
from model.smiles_kg_encoder_v3 import KGSmilesEncoder_v3
from model.mega_molbart.STMencoder_ddp import MolSTM_Extractor
from model.components import ResidualProjector, DrugProjector, CellProjector, ProjectOut, ConcentrationEmbedding


class PerturbationEncoder(nn.Module):
    def __init__(
        self,
        dim_emb=1024,
        hvg_dim = 2000,
        encode_conc=False,
        num_heads=8,
        dropout=0.1,
        is_hvg_space=False,
        detach_decoder=False,
        transformer_backbone_key='llama',
        smile_encoder='MAP-KG',
    ):
        super().__init__()
        self.dim_emb = dim_emb
        self.hvg_dim = hvg_dim
        self.encode_conc = encode_conc
        self.num_heads = num_heads
        self.dropout = dropout
        self.is_hvg_space = is_hvg_space
        self.detach_decoder = detach_decoder
        self.transformer_backbone_key = transformer_backbone_key
        self.smile_encoder = smile_encoder


        self.cell_projector = CellProjector(n_layers=4, d_cell=2048, d_out=self.dim_emb, dropout=self.dropout)

        ckpt_path = "path/to/your/pretrained/kg/encoder/best_model.pt"

        self.kg_smiles_encoder = KGSmilesEncoder_v3(
            d_smiles=256,
            d_model=self.dim_emb,
        )
        self.kg_smiles_encoder.load_from_full_model(ckpt_path)

        self.kg_smiles_encoder.eval()
        for param in self.kg_smiles_encoder.parameters():
            param.requires_grad = False

        self.gene_tokens_projector = ResidualProjector(
            input_dim=2048+5120,
            output_dim=self.dim_emb,
            hidden_dim=2048
        )

        transformer_backbone_kwargs = ST_small_transformer_backbone_kwargs.copy()
        transformer_backbone_kwargs["max_position_embeddings"] = 2049
        transformer_backbone_kwargs["hidden_size"] = self.dim_emb

        self.transformer_backbone, self.transformer_model_dim = get_transformer_backbone(
            key=self.transformer_backbone_key,
            kwargs=transformer_backbone_kwargs,
        )

        self.project_out = ProjectOut(
            hidden_dim=self.dim_emb,
            output_dim=self.cell_projector.d_cell,
            dropout=self.dropout
        )

        # HVG decoder
        self.gene_decoder = LatentToGeneDecoder(
            latent_dim=2048,
            gene_dim=self.hvg_dim,
            hidden_dims=[512, 1024],
            dropout=self.dropout
        )


    def forward(self, gene_tokens, cls_tokens, esm_tokens, smiles, concs):        
        B = len(smiles)
        BS = cls_tokens.shape[0]
        S = BS // B

        cell_tokens = self.encode_cells(cls_tokens) # B*S, dim_emb
        drug_tokens = self.encode_drug(smiles)  # B, dim_emb
        gene_tokens = self.encode_genes(gene_tokens, esm_tokens) # B*S, 2047, dim_emb

        drug_tokens = drug_tokens.unsqueeze(1).expand(B, S, -1).reshape(BS, -1)  # [B*S, dim_emb]
        
        # [CLS | Drug | Genes]
        combined_tokens = torch.cat([
            cell_tokens.unsqueeze(1),
            drug_tokens.unsqueeze(1),
            gene_tokens
        ], dim=1)  # [B*S, 2049, dim_emb]

        res_pred = self.transformer_backbone(inputs_embeds=combined_tokens).last_hidden_state
        res_pred = res_pred[:, 0, :]

        pred_tokens = self.project_out(cell_tokens + res_pred)

        pert_cell_counts_preds = self.gene_decoder(pred_tokens)

        pred_tokens = pred_tokens.reshape(B,S,-1)
        pert_cell_counts_preds = pert_cell_counts_preds.reshape(B,S,-1)
        
        return pred_tokens, pert_cell_counts_preds
        

    def encode_drug(self, smiles):
        assert ('' not in smiles) and ('xxx' not in smiles) and (None not in smiles), smiles

        if self.smile_encoder == 'MAP-KG':
            with torch.no_grad():
                drug_vecs = self.kg_smiles_encoder(smiles)
        else:
            print("ERROR! in pert.py")
            exit()

        return drug_vecs

    def encode_cells(self, ctrl_embs):
        """
        ctrl_embs: B * S * d_emb
        """
        return self.cell_projector(ctrl_embs)
    
    def encode_genes(self, gene_tokens, esm_tokens):
        """
        gene_tokens: B * S, 2047, 2048
        esm_tokens: B * S, 2047, 5120
        """
        combined = torch.cat([gene_tokens, esm_tokens], dim=-1) # B*S, 2047, 2048+5120
        return self.gene_tokens_projector(combined) # B*S, 2047, self.dim_emb
