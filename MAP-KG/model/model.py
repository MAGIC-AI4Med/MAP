import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, BertConfig, AutoTokenizer

from model.mega_molbart.STMencoder_ddp import MolSTM_Extractor

import numpy as np
from rdkit import Chem


class esm2_mapping():
    def __init__(self,
            dict_path="/path/to/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
        ):
        self.esm_dict = torch.load(dict_path)

    def fetch_esm(self, hgnc_id):
        return self.esm_dict[hgnc_id]
    
    def fetch_esm_batch(self, hgnc_ids):
        return torch.stack([self.esm_dict[hid] for hid in hgnc_ids])

    
class ResidualProjector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self.residual_proj = nn.Linear(input_dim, output_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        transformed = self.mlp(x)
        residual = self.residual_proj(x)
        gate_value = self.gate(x)
        
        return gate_value * transformed + (1 - gate_value) * residual


class GatedFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.fusion_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, source, relation):
        concat_feat = torch.cat([source, relation], dim=-1)
        
        gate = self.gate_net(concat_feat)
        
        gated_relation = gate * relation
        
        fusion_input = torch.cat([source, gated_relation], dim=-1)
        fused = self.fusion_net(fusion_input)
        
        return fused
    

class DrugGeneModel(nn.Module):
    def __init__(self,
            bert_model_name,
            empty_relation,
            d_model=1024,
            d_esm=5120,  
            d_smiles=256,  
            mode_threshold=0.5,
            finetune_smiles_encoder = False,
            finetune_bert = False,
            freeze_bert_layers = 0,
            pretrained_mol_stm = True
        ):
        super().__init__()

        self.empty_relation = empty_relation
        self.d_model = d_model
        self.d_esm = d_esm
        self.d_smiles = d_smiles
        self.mode_threshold = mode_threshold

        self.esm2_mapping = esm2_mapping()

        self.smiles_encoder = MolSTM_Extractor(pretrained=pretrained_mol_stm)
        for param in self.smiles_encoder.parameters():
            param.requires_grad = finetune_smiles_encoder

        config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)
        self.text_encoder = AutoModel.from_pretrained(bert_model_name, config=config)
        print(bert_model_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = finetune_bert

        if finetune_bert and freeze_bert_layers > 0:
            self._freeze_bert_layers(freeze_bert_layers)

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

        self.d_bert = self.text_encoder.config.hidden_size

        self.smiles_projector = ResidualProjector(
            input_dim=self.d_smiles,
            output_dim=d_model,
            hidden_dim=512
        )
        
        self.gene_projector = ResidualProjector(
            input_dim=self.d_esm,
            output_dim=d_model,
            hidden_dim=1536,
        )

        self.text_projector = ResidualProjector(
            input_dim=self.d_bert,
            output_dim=d_model,
            hidden_dim=768,
        )

        self.fusion_module = GatedFusion(d_model)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self._device = None


    def _freeze_bert_layers(self, num_layers_to_freeze):
        total_layers = len(self.text_encoder.encoder.layer)
        num_layers_to_freeze = min(num_layers_to_freeze, total_layers)
        
        for param in self.text_encoder.embeddings.parameters():
            param.requires_grad = False
        
        for i in range(num_layers_to_freeze):
            for param in self.text_encoder.encoder.layer[i].parameters():
                param.requires_grad = False
        
        total_params = sum(p.numel() for p in self.text_encoder.parameters())
        trainable_params = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
    def _get_device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device
    
    def _ensure_smiles_encoder_device(self):
        target_device = self._get_device()
        
        smiles_encoder_device = next(self.smiles_encoder.parameters()).device
        
        if smiles_encoder_device != target_device:
            self.smiles_encoder = self.smiles_encoder.to(target_device)
    
    def encode_text(self, edge_sentences):
        target_device = self._get_device()
        
        inputs = self.tokenizer(
            edge_sentences,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            truncation=True,
            return_tensors='pt'
        ).to(target_device)

        output = self.text_encoder(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

        _, pooler_output, _ = output[0], output[1], output[2]
        return self.text_projector(pooler_output)
    
    def encode_smiles(self, smiles_list):
        normalized_smiles = []
        for s in smiles_list:
            try:
                mol = Chem.MolFromSmiles(s)
                if mol is not None:
                    normalized_smiles.append(Chem.MolToSmiles(mol))
                else:
                    normalized_smiles.append(s)
            except:
                normalized_smiles.append(s)
        
        smiles_list = normalized_smiles
        valid_indices = [i for i, s in enumerate(smiles_list) if s != 'xxx']
        
        if len(valid_indices) == 0:
            print(f"ERROR: invalid SMILES exist: {smiles_list}")
            exit()
        
        valid_smiles = [smiles_list[i] for i in valid_indices]
        
        smiles_emb = self.smiles_encoder(valid_smiles)

        smiles_emb = self.smiles_projector(smiles_emb)
        
        batch_size = len(smiles_list)
        device = smiles_emb.device
        full_emb = torch.zeros(batch_size, self.d_model, device=device, dtype=smiles_emb.dtype)
        full_emb[valid_indices] = smiles_emb
        
        return full_emb, valid_indices

    def _encode_properties_batched(self, properties, types, output_tensor):
        """
        按类型分组批处理编码
        
        Args:
            properties: list of property values
            types: list of property types ('text', 'smiles', 'esm')
            output_tensor: 用于存储结果的tensor (batch_size, d_model)
        """
        device = self._get_device()
        
        text_indices = [i for i, t in enumerate(types) if t == 'text']
        smiles_indices = [i for i, t in enumerate(types) if t == 'smiles']
        esm_indices = [i for i, t in enumerate(types) if t == 'esm']
        
        if text_indices:
            text_properties = [properties[i] for i in text_indices]
            text_embs = self.encode_text(text_properties)
            for idx, text_idx in enumerate(text_indices):
                output_tensor[text_idx] = text_embs[idx]
        
        if smiles_indices:
            smiles_properties = [properties[i] for i in smiles_indices]
            smiles_embs, valid_indices = self.encode_smiles(smiles_properties)
            for idx, smiles_idx in enumerate(smiles_indices):
                output_tensor[smiles_idx] = smiles_embs[idx]
        
        if esm_indices:
            esm_properties = [properties[i] for i in esm_indices]
            esm_embs = self.esm2_mapping.fetch_esm_batch(esm_properties)
            esm_embs = esm_embs.to(device)
            esm_embs = self.gene_projector(esm_embs)
            for idx, esm_idx in enumerate(esm_indices):
                output_tensor[esm_idx] = esm_embs[idx]

    def SoftCrossEntropy(self, inputs, target, reduction='average'):
        log_likelihood = -F.log_softmax(inputs, dim=1)
        batch = inputs.shape[0]
        if reduction == 'average':
            loss = torch.sum(torch.mul(log_likelihood, target)) / batch
        else:
            loss = torch.sum(torch.mul(log_likelihood, target))
        return loss

    def forward(self, batch):
        batch_size = len(batch['property_1s'])
        device = self._get_device()

        dummy = self.text_projector.mlp[0].weight.new_zeros(1)
        current_dtype = dummy.dtype
        
        property_1_embs = torch.zeros(batch_size, self.d_model, device=device, dtype=current_dtype)
        property_2_embs = torch.zeros(batch_size, self.d_model, device=device, dtype=current_dtype)
        
        self._encode_properties_batched(
            batch['property_1s'], 
            batch['type_1s'], 
            property_1_embs
        )
        
        self._encode_properties_batched(
            batch['property_2s'], 
            batch['type_2s'], 
            property_2_embs
        )
        
        relation_embs = self.encode_text(batch['relations'])  # (batch_size, d_model)
        
        is_node = torch.tensor([r == self.empty_relation for r in batch['relations']], 
                            device=device)  # (batch_size,)
        
        start_embs = property_1_embs.clone()
        target_embs = property_2_embs.clone()
        
        edge_indices = torch.where(~is_node)[0]
        
        if len(edge_indices) > 0:
            use_mode1 = torch.rand(len(edge_indices), device=device) < self.mode_threshold
            
            for idx_in_edges, batch_idx in enumerate(edge_indices):
                batch_idx = batch_idx.item()
                
                if use_mode1[idx_in_edges]:
                    fused = self.fusion_module(
                        property_1_embs[batch_idx:batch_idx+1], 
                        relation_embs[batch_idx:batch_idx+1]
                    )
                    start_embs[batch_idx] = fused.squeeze(0)
                else:
                    fused = self.fusion_module(
                        relation_embs[batch_idx:batch_idx+1],
                        property_2_embs[batch_idx:batch_idx+1]
                    )
                    target_embs[batch_idx] = fused.squeeze(0)

        start_norm = nn.functional.normalize(start_embs, dim=-1)  
        target_norm = nn.functional.normalize(target_embs, dim=-1)  
        
        logits_per_text1 = self.logit_scale * start_norm @ target_norm.T
        logits_per_text2 = self.logit_scale * target_norm @ start_norm.T
    
        labels = torch.zeros(batch_size, batch_size, device=device)
        
        for i in range(batch_size):
            for j in range(batch_size):                
                if is_node[i]:
                    target_id_i = batch['id_2s'][i]
                else:
                    edge_idx_in_list = (edge_indices == i).nonzero(as_tuple=True)[0]
                    if len(edge_idx_in_list) > 0 and use_mode1[edge_idx_in_list[0]]:
                        target_id_i = batch['id_2s'][i]
                    else:
                        target_id_i = batch['id_1s'][i]
                
                if is_node[j]:
                    target_id_j = batch['id_2s'][j]
                else:
                    edge_idx_in_list = (edge_indices == j).nonzero(as_tuple=True)[0]
                    if len(edge_idx_in_list) > 0 and use_mode1[edge_idx_in_list[0]]:
                        target_id_j = batch['id_2s'][j]
                    else:
                        target_id_j = batch['id_1s'][j]
                
                if target_id_i == target_id_j:
                    labels[i, j] = 1.0
        
        labels = F.normalize(labels, dim=0)
    
        loss = (
            self.SoftCrossEntropy(logits_per_text1, labels) +
            self.SoftCrossEntropy(logits_per_text2, labels)
        ) / 2
        
        return {
            'loss': loss,
            'logits': (logits_per_text1 + logits_per_text2) * 0.5,
            'labels': labels
        }
