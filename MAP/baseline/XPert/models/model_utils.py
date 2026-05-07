import sys

import math

import numpy as np
import torch
import torch.nn.functional as F
from flash_attn.flash_attn_interface import flash_attn_func
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        # Normalize input_tensor
        s = (x - u).pow(2).mean(-1, keepdim=True)
        # Apply scaling and bias
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate, args=None):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        # input_ids = input_ids.unsqueeze(0)
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        # embeddings = torch.cat((words_embeddings, position_embeddings), dim=-1)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings



class cell_Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate, pretrained_embed_path, args=None):
        super(cell_Embeddings, self).__init__()

        self.args = args
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)

        # load gene vector
        pretrained_embed = np.load(pretrained_embed_path, allow_pickle=True)
        pretrained_embed = torch.tensor(pretrained_embed).float()
        if hidden_size != pretrained_embed.size(-1):
            self.linear = nn.Linear(pretrained_embed.size(-1), hidden_size) 
            pretrained_embed = self.linear(pretrained_embed)

        self.pretrained_embed = nn.Embedding.from_pretrained(pretrained_embed, freeze=False)

        if self.args.use_gene_pos_emed:
            self.position_embed = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

            
    def forward(self, input_ids):
        # input_ids = input_ids.unsqueeze(0)        
        words_embeddings = self.word_embeddings(input_ids) # [64,978,128]

        if self.args.wo_ppi:
            embeddings = words_embeddings
        else:
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            pretrained_embed = self.pretrained_embed(position_ids)
            embeddings = words_embeddings + pretrained_embed

        if self.args.use_gene_pos_emed:
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            pos_embeddings = self.position_embed(position_ids)
            embeddings = embeddings + pos_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings





class unimol_Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate, args=None, dose_embed=None, time_embed=None):
        super(unimol_Embeddings, self).__init__()
        

        self.args = args


        if 'mdmt' in args.dataset:
            self.pert_dose_emb = dose_embed
            self.pert_time_emb = time_embed
            self.position_embeddings = nn.Embedding(max_position_size+2, hidden_size)
        else:
            self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_size = hidden_size

        if args.drug_feat == 'unimol':
            self.linear = nn.Linear(512, hidden_size)
        elif args.drug_feat == 'KPGT':
            self.linear = nn.Linear(2304, hidden_size)
        elif args.drug_feat == 'morgan':
            self.linear = nn.Linear(1024, hidden_size)


    def forward(self, input_embed, HG_embed, atom_symbols, pert_dose_idx, pert_time_idx):

        input_embeddings = self.linear(input_embed)

        if self.args.drug_feat == 'unimol':
            input_embeddings[:,0,:] = HG_embed
            if self.args.wo_HG:
                input_embeddings = input_embeddings[:,1:,:] 
            if self.args.wo_atom:
                input_embeddings = input_embeddings[:,:2,:]
            if self.args.wo_atom_HG:
                input_embeddings = input_embeddings[:,1,:].unsqueeze(1)
            if self.args.wo_unimol:
                input_embeddings = input_embeddings[:,0,:].unsqueeze(1)
        elif self.args.drug_feat in ['KPGT','morgan']:
            input_embeddings = torch.stack([HG_embed,input_embeddings], dim=1)         

        if 'mdmt' in self.args.dataset:
            pert_dose_embed = self.pert_dose_emb(pert_dose_idx).unsqueeze(1)
            pert_time_embed = self.pert_time_emb(pert_time_idx).unsqueeze(1)
            input_embeddings = torch.cat([pert_dose_embed, pert_time_embed, input_embeddings], dim=1)
        
        seq_length = input_embeddings.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embed.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_embeddings[:,:,0])  # [64,200]
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings




class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, topk):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention heads ({num_attention_heads})"
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout_p = attention_probs_dropout_prob


    def forward(self, hidden_states, attention_mask=None, sparse_flag=False, output_attention=False):
        
        batch_size, seq_len, hidden_size = hidden_states.size()

        # Linear projections
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        # Flash Attention expects inputs as [batch_size, seq_len, num_heads, head_size]
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        key = key.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        value = value.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)

        if output_attention:
            # Calculate using the traditional attention mechanism to obtain the attention score

            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2) # [batch, num_heads, seq_len, head_size]


            attention_scores = torch.matmul(query, key.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)


            if attention_mask is not None:
                if attention_scores.size(-2) != attention_mask.size(-2):
                    attention_mask_pad = torch.ones((attention_scores.size(0), 2), device=attention_scores.device).unsqueeze(1).unsqueeze(2)
                    attention_mask = torch.cat([attention_mask_pad, attention_mask], dim=-1)
                else:
                    attention_scores = attention_scores + attention_mask

            attention_probs = F.softmax(attention_scores, dim=-1)  # [batch, num_heads, seq_len, seq_len]
            context = torch.matmul(attention_probs, value)
            context = context.transpose(1, 2).contiguous()

            return context.view(batch_size, seq_len, hidden_size), attention_probs

        else:
            context = flash_attn_func(
                query, key, value,
                dropout_p=self.dropout_p if self.training else 0.0
            )
            return context.view(batch_size, seq_len, hidden_size), None



class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, topk):
        super(CrossAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention heads ({num_attention_heads})"
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout_p = attention_probs_dropout_prob


    def forward(self, cell, drug, cell_attention_mask=None, drug_attention_mask=None, sparse_flag=False, output_attention=False):
        batch_size, seq_len_A, hidden_size = cell.size()
        _, seq_len_B, _ = drug.size()


        # Linear projections
        query = self.query(cell)
        key = self.key(drug)
        value = self.value(drug)

        # Flash Attention expects inputs as [batch_size, seq_len, num_heads, head_size]
        query = query.view(batch_size, seq_len_A, self.num_attention_heads, self.attention_head_size)
        key = key.view(batch_size, seq_len_B, self.num_attention_heads, self.attention_head_size)
        value = value.view(batch_size, seq_len_B, self.num_attention_heads, self.attention_head_size)

        if output_attention:    
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2) # [batch, num_heads, seq_len, head_size]

            attention_scores = torch.matmul(query, key.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            if cell_attention_mask is not None:
                attention_scores = attention_scores + cell_attention_mask
            attention_probs = F.softmax(attention_scores, dim=-1)
            context = torch.matmul(attention_probs, value)
            context = context.transpose(1, 2).contiguous()
            return context.view(batch_size, seq_len_A, hidden_size), attention_probs
        else:
            context = flash_attn_func(
                query, key, value,
                dropout_p=self.dropout_p if self.training else 0.0
            )
            return context.view(batch_size, seq_len_A, hidden_size), None







class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.ReLU()
        self.dense_2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        return hidden_states


class SublayerConnection(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SublayerConnection, self).__init__()
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dropout(self.LayerNorm(hidden_states))
        return hidden_states + input_tensor
        


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob)
        self.sublayer = SublayerConnection(hidden_size, hidden_dropout_prob)      

    def forward(self, input_tensor):
        hidden_states = self.feed_forward(input_tensor)
        hidden_states = self.sublayer(hidden_states, input_tensor)
        return hidden_states


# Drug self-attention encoder
class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob, topk):
        super(Encoder, self).__init__()
        self.attention = SelfAttention(hidden_size, num_attention_heads,
                                   attention_probs_dropout_prob, topk)
        self.sublayer = SublayerConnection(hidden_size, hidden_dropout_prob)
        self.output = SelfOutput(hidden_size, intermediate_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask, sparse_flag=False, output_attention=False):
        attention_output, attention_probs_0 = self.attention(input_tensor, attention_mask, sparse_flag, output_attention)
        attention_output = self.sublayer(attention_output, input_tensor)
        layer_output = self.output(attention_output)

        return layer_output, attention_probs_0  


class crossEncoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob, topk_cell=128, topk_drug=64):
        super(crossEncoder, self).__init__()
        self.LayerNorm = LayerNorm(hidden_size)
        self.attention_CA = CrossAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob, topk_drug)
        self.attention = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob, topk_cell)
        self.sublayer = SublayerConnection(hidden_size, hidden_dropout_prob)
        self.output = SelfOutput(hidden_size, intermediate_size, hidden_dropout_prob)

        self.drug_SA = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob, topk_drug)
    
    def forward(self, cell, drug, drug_attention_mask, cell_attention_mask=None, sparse_flag=False, output_attention=False):
        
        # drug:SA
        drug_SA_embed, _ = self.drug_SA(drug, drug_attention_mask, sparse_flag, output_attention)
        
        # cell:SA
        cell_attention_out_0, cell_attention_probs_0 = self.attention(cell, cell_attention_mask, sparse_flag, output_attention)
        cell_embed = self.sublayer(cell_attention_out_0, cell)
        
        # cell:CA
        cell_attention_output_1, cell_attention_probs_1 = self.attention_CA(cell_embed, drug_SA_embed, cell_attention_mask, drug_attention_mask, sparse_flag, output_attention)
        cell_embed = self.sublayer(cell_attention_output_1, cell_embed)
        # cell_output
        cell_intermediate_output = self.output(cell_embed)

        return cell_intermediate_output, drug_SA_embed, cell_attention_probs_1

