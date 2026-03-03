# n
import sys
from rdkit import Chem
from pathlib import Path

import torch
import torch.nn as nn

from model.mega_molbart.tokenizer import MolEncTokenizer
from model.mega_molbart.util import REGEX, DEFAULT_CHEM_TOKEN_START, DEFAULT_VOCAB_PATH
from model.mega_molbart.decoder import DecodeSampler
from model.mega_molbart.megatron_bart import MegatronBART

class MolSTM_Extractor(nn.Module):
    def __init__(
        self,
        ckpt_path = "/mnt/petrelfs/fengjinghao/molecule_model.pth",
        vocab_path = "/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/ours_v1/drug/MoleculeSTM/MoleculeSTM/bart_vocab.txt",  # 确保指向实际 vocab 文件 
        pretrained=True,   # ← 新增参数
    ):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.vocab_path = vocab_path
        self.pretrained = pretrained

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = MolEncTokenizer.from_vocab_file(self.vocab_path, REGEX, DEFAULT_CHEM_TOKEN_START)
        self._load_model()

    def _load_model(self):
        # 加载 state_dict
        state_dict = torch.load(self.ckpt_path, map_location="cpu")

        # tokenizer 和超参
        vocab_size = len(self.tokenizer)
        pad_token_idx = self.tokenizer.vocab[self.tokenizer.pad_token]

        d_model = state_dict["emb.weight"].shape[1]                        # 256
        self.dim_model = d_model
        
        num_layers = max(int(k.split(".")[2]) for k in state_dict if k.startswith("encoder.layers.")) + 1  # 4
        d_feedforward = state_dict["encoder.layers.0.fc1.weight"].shape[0] # 1024
        num_heads = 8                                                      # 由权重形状确定
        self.max_seq_len = state_dict["pos_emb"].shape[0]                       # 512

        sampler = DecodeSampler(self.tokenizer, max_seq_len=self.max_seq_len)
        self._model = MegatronBART(
            decode_sampler=sampler,
            pad_token_idx=pad_token_idx,
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_feedforward=d_feedforward,
            max_seq_len=self.max_seq_len,
            dropout=0.1,
        )
        
        if self.pretrained:
            missing, unexpected = self._model.load_state_dict(state_dict, strict=True)
            print("[MolSTM_Extractor] Missing keys:", missing)
            print("[MolSTM_Extractor] Unexpected keys:", unexpected)
            assert len(missing) == len(unexpected) == 0, "[MolSTM_Extractor] Load Error! incomplete ckpt loaded"
        else:
            print("[MolSTM_Extractor] Skipping pretrained weight loading.")
            

    def forward(self, drugs):
        '''
        假设输入的形状是batch_size长度的一个SMILES字符串列表
        '''
        drugs = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in drugs] # 对齐MoleculeSTM的与处理方法
        
        tok = self.tokenizer.tokenize(drugs, pad=True)
        ids = self.tokenizer.convert_tokens_to_ids(tok['original_tokens'])  # List[List[int]]

        encoder_input = torch.tensor(ids, dtype=torch.long).T                 # (seq_len, 1)
        encoder_pad_mask = torch.tensor(tok['masked_pad_masks'], dtype=torch.bool).T  # (seq_len, 1)

        # 截断到模型最大长度
        encoder_input = encoder_input[:self.max_seq_len] # seq_len * batch
        encoder_pad_mask = encoder_pad_mask[:self.max_seq_len] # seq_len * batch

        memory = self._model.encode({
            "encoder_input": encoder_input,
            "encoder_pad_mask": encoder_pad_mask,
        }) # seq_len * batch * d_model

        valid = (~encoder_pad_mask).float()
        weights = valid / (valid.sum(dim=0, keepdim=True) + 1e-9)
        weights = weights.unsqueeze(-1) # seq_len * batch * 1
        # mol_vec = (memory.squeeze(1) * weights).sum(dim=0) # batch * d_model
        mol_vec = (memory * weights).sum(dim=0) # batch * d_model
        
        return mol_vec


# a = MolSTM_Extractor()
# print(a.max_seq_len)
# drugs = ["C1=CC2=C(C(=C1)O)N=CC=C2", "C1=CC2=C(C(=C1)O)N=CC=C2"]

# b = a.forward(drugs)





