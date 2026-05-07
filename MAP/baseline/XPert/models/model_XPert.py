import sys
import numpy as np
import torch
import torch.nn as nn
from models.model_utils import (Embeddings, Encoder, cell_Embeddings,
                                crossEncoder, unimol_Embeddings)
from torch_geometric.nn import HeteroConv, SAGEConv


def get_unimol_drug_feat(input_features):

    atom_musk_raw = input_features[:, :, 0].long()
    atom_symbol = input_features[:, :, 1].long()
    atom_feat = input_features[:, :, 2:].to(input_features.dtype)

    atom_musk = atom_musk_raw.unsqueeze(1).unsqueeze(2)
    atom_musk = (1.0 - atom_musk) * -10000.0

    return atom_feat, atom_symbol, atom_musk




class AttnEncoder(nn.Module):
    def __init__(self, config, logger, structure):
        super(AttnEncoder, self).__init__()
        
        hidden_size = config['model']['ATTN']['hidden_size']
        intermediate_size = hidden_size * 2
        num_attention_heads = config['model']['ATTN']['n_heads']
        topk_cell = config['model']['ATTN']['topk_cell']
        topk_drug = config['model']['ATTN']['topk_drug']
        attention_probs_dropout_prob = config['model']['ATTN']['attention_probs_dropout_prob']
        hidden_dropout_prob = config['model']['ATTN']['hidden_dropout_prob']
        self.sparse_flag = config['model']['ATTN']['sparse_flag']

        print(f'Using sparse attn: {self.sparse_flag}')
        
        self.structure = structure
        self.layers = self.structure.split('+')
        CA_N = self.layers.count('CA')
        SA_N = self.layers.count('SA')        

        self.crossEncoders = nn.ModuleList(
            [crossEncoder(hidden_size, intermediate_size, num_attention_heads, 
                          attention_probs_dropout_prob, hidden_dropout_prob, 
                          topk_cell, topk_drug) for _ in range(CA_N)]
        )

        self.selfEncoders = nn.ModuleList(
            [Encoder(hidden_size, intermediate_size, num_attention_heads, 
                          attention_probs_dropout_prob, hidden_dropout_prob, 
                          topk_cell) for _ in range(SA_N)]
        )
        
        
    def forward(self, cell_embed, drug_embed, cell_attention_mask, drug_attention_mask, output_attention=False, drug_specific_gene_embedding=None):
        
        CA_layer_idx = 0
        SA_layer_idx = 0
        attention_dict = {}
        
        for layer_count, layer_type in enumerate(self.layers):
            if layer_type == 'CA':
                cell_embed, drug_embed, attention = self.crossEncoders[CA_layer_idx](cell_embed, drug_embed, 
                                                                        drug_attention_mask, 
                                                                        cell_attention_mask,
                                                                        self.sparse_flag,
                                                                        output_attention)
                if (drug_specific_gene_embedding is not None) and (layer_count < len(self.layers) - 1):
                    lamda = 1
                    cell_embed = cell_embed + lamda * drug_specific_gene_embedding
                if output_attention:
                    attention_dict[f'CA_{layer_count}'] = attention
                CA_layer_idx += 1                    

            elif layer_type == 'SA':
                cell_embed, attention = self.selfEncoders[SA_layer_idx](cell_embed, cell_attention_mask, self.sparse_flag, output_attention)
                if output_attention:
                    attention_dict[f'SA_{layer_count}'] = attention
                SA_layer_idx += 1 
        
        if output_attention:
            return cell_embed, drug_embed, attention_dict
        else:
            return cell_embed, drug_embed, None






class XPertNet(torch.nn.Module):
    def __init__(self, args, config, device, logger):
        super(XPertNet, self).__init__()

        self.args = args
        self.config = config
        self.device = device
        self.logger = logger

        # args for cell
        max_gene_length = config['dataset']['gene_num']
        exp_vocab_size = config['dataset']['n_bins']
        
        # args for drug
        atom_num = config['dataset']['atom_num']
        max_atom_size = config['dataset']['max_atom_size']

        # args for embedding
        hidden_size = config['model']['ATTN']['hidden_size']
        self.hidden_size = hidden_size
        cell_input_hidden_dropout_prob = config['model']['ATTN']['cell_input_hidden_dropout_prob']
        drug_input_hidden_dropout_prob = config['model']['ATTN']['drug_input_hidden_dropout_prob']

        # cell_embedding
        print('For cell embedding: using ppi_gene_vector + exp embedding')
        pretrained_ppi_embed_path = config['model']['ATTN']['ppi_gene_vector_path']
        self.cell_emb = cell_Embeddings(exp_vocab_size, hidden_size, max_gene_length, cell_input_hidden_dropout_prob, pretrained_ppi_embed_path, args)

        # drug_embedding
        # print('Using HG_embed_pretrained + unimol_molecule  + unimol_atom embedding')
        if 'sdst' in args.dataset or args.dataset in ['tahoe', 'nips', 'sciplex']:
            # tahoe/nips/sciplex are all single-drug single-time (like sdst)
            self.drug_emb = unimol_Embeddings(atom_num, hidden_size, max_atom_size, drug_input_hidden_dropout_prob, args)
        elif 'mdmt' in args.dataset:
            print('Using dose and time embedding')
            # define dose and time embedding
            pert_dose_emb = nn.Embedding(config['dataset'][args.dataset.split('_')[0]]['num_pert_dose'], hidden_size)
            pert_time_emb = nn.Embedding(config['dataset'][args.dataset.split('_')[0]]['num_pert_time'], hidden_size)
            self.drug_emb = unimol_Embeddings(atom_num, hidden_size, max_atom_size, drug_input_hidden_dropout_prob, args, pert_dose_emb, pert_time_emb)


        # drug_HG_embedding
        drug_hg_embed_path = config['model']['HG']['drug_hg_pretrained_embed_path']
        self.drug_HG_embed = torch.tensor(np.load(drug_hg_embed_path, allow_pickle=True), dtype=torch.float, device=device)
        
            
        if args.pretrained_mode == 'specific':
            drug_specific_hg_embed_path = config['model']['HG']['specific_pretrained_embed_path']
            self.drug_specific_gene_embed = torch.tensor(np.load(drug_specific_hg_embed_path, allow_pickle=True), dtype=torch.float, device=device)
            self.transform_cell = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(p=0.1),
                )


        ctl_structure = config['model']['ATTN']['ctl_structure']
        trt_structure = config['model']['ATTN']['trt_structure']
        self.attnEncoder_ctl = AttnEncoder(config, logger, ctl_structure)
        self.attnEncoder_trt = AttnEncoder(config, logger, trt_structure)


        latent_size = 64
        self.ctl_fc = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(latent_size, 1),
        )
        self.trt_fc = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(latent_size, 1),
        )
        self.deg_fc = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(latent_size, 1),
        )





        if args.include_cell_idx:
            num_cell_id = config['dataset'][args.dataset.split('_')[0]]['num_cell_id']
            num_tissue_id = config['dataset'][args.dataset.split('_')[0]]['num_tissue_id']
            
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
            self.class_fc = nn.Linear(hidden_size, num_cell_id)


    def forward(self, data, mode='ST'):

        trt_raw_data, ctl_raw_data, trt_raw_data_binned, ctl_raw_data_binned, drug_feat, pert_dose_idx, pert_time_idx, drug_idx, cell_idx, tissue_idx = data
        trt_raw_data = trt_raw_data.to(self.device)
        ctl_raw_data = ctl_raw_data.to(self.device)
        trt_raw_data_binned = trt_raw_data_binned.to(self.device)
        ctl_raw_data_binned = ctl_raw_data_binned.to(self.device)
        drug_feat = drug_feat.to(self.device)
        pert_dose_idx = pert_dose_idx.to(self.device)
        pert_time_idx = pert_time_idx.to(self.device)
        drug_idx = drug_idx.to(self.device)
        cell_idx = cell_idx.to(self.device)
        tissue_idx = tissue_idx.to(self.device)

        cell_class_true = cell_idx
        num_samples = cell_idx.shape[0]

        if self.args.drug_feat == 'unimol':
            drug_unimol_embed, drug_atom_symbols, drug_attention_mask = get_unimol_drug_feat(drug_feat)
        else:
            drug_unimol_embed = drug_feat
            drug_atom_symbols = None
            drug_attention_mask = None

        drug_HG_embed = self.drug_HG_embed[drug_idx]
        drug_embed = self.drug_emb(drug_unimol_embed, drug_HG_embed, drug_atom_symbols, pert_dose_idx, pert_time_idx)


        if self.args.pretrained_mode == 'global':
            cell_embed = self.cell_emb(ctl_raw_data_binned)
        elif self.args.pretrained_mode == 'specific':
            cell_specific_gene_embed = self.drug_specific_gene_embed[drug_idx]
            cell_embed = self.cell_emb(ctl_raw_data_binned)

        if self.args.include_cell_idx:
            cls_tokens = self.cls_token.expand(num_samples, -1, -1)
            cell_embed = torch.cat([cls_tokens, cell_embed], dim=1)
            

        trt_cell_embed_attn, drug_embed, trt_attention_dict = self.attnEncoder_trt(cell_embed, drug_embed, None, drug_attention_mask, self.args.output_attention)
        ctl_cell_embed_attn, _, ctl_attention_dict = self.attnEncoder_ctl(cell_embed, None, None, None, self.args.output_attention)

        if self.args.include_cell_idx:
            trt_cell_embed = trt_cell_embed_attn[:,1:, :]
            ctl_cell_embed = ctl_cell_embed_attn[:,1:, :]
            trt_cls_embed = trt_cell_embed_attn[:, 0, :]
            ctl_cls_embed = ctl_cell_embed_attn[:, 0, :]  # [batch, hidden_size]
            cell_class_predict_1 = self.class_fc(ctl_cls_embed)
            cell_class_predict_2 = self.class_fc(trt_cls_embed)
            cell_class_predict = (cell_class_predict_1, cell_class_predict_2)

            cls_embed = torch.cat((trt_cls_embed.unsqueeze(1), ctl_cls_embed.unsqueeze(1)), dim=1) #[batch, 2, hidden_dim]
        else:
            trt_cell_embed = trt_cell_embed_attn
            ctl_cell_embed = ctl_cell_embed_attn
            cell_class_predict = None
            trt_cls_embed = None
            ctl_cls_embed = None
            cls_embed = None
            
        trt_output = self.trt_fc(trt_cell_embed).squeeze(-1)
        ctl_output = self.ctl_fc(ctl_cell_embed).squeeze(-1)
        deg_output = self.deg_fc(trt_cell_embed-ctl_cell_embed).squeeze(-1)


        attention_dict = (trt_attention_dict, ctl_attention_dict)
        

        if self.args.mode == 'infer':
            return trt_output, ctl_output, deg_output, trt_raw_data, ctl_raw_data, attention_dict, cell_class_true, cell_class_predict, cls_embed
        else:
            return trt_output, ctl_output, deg_output, trt_raw_data, ctl_raw_data, attention_dict, cell_class_true, cell_class_predict

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)



