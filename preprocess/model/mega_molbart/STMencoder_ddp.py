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
        ckpt_path = "molecule_model.pth",
        vocab_path = "bart_vocab.txt",
        pretrained=True,
    ):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.vocab_path = vocab_path
        self.pretrained = pretrained

        self.tokenizer = MolEncTokenizer.from_vocab_file(
            self.vocab_path, REGEX, DEFAULT_CHEM_TOKEN_START
        )
        
        self._model = self._load_model()

    def _load_model(self):
        state_dict = torch.load(self.ckpt_path, map_location="cpu")

        vocab_size = len(self.tokenizer)
        pad_token_idx = self.tokenizer.vocab[self.tokenizer.pad_token]

        d_model = state_dict["emb.weight"].shape[1]
        self.dim_model = d_model
        
        num_layers = max(
            int(k.split(".")[2]) for k in state_dict if k.startswith("encoder.layers.")
        ) + 1
        d_feedforward = state_dict["encoder.layers.0.fc1.weight"].shape[0]
        num_heads = 8
        self.max_seq_len = state_dict["pos_emb"].shape[0]

        sampler = DecodeSampler(self.tokenizer, max_seq_len=self.max_seq_len)
        model = MegatronBART(
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
            missing, unexpected = model.load_state_dict(state_dict, strict=True)
            print("[MolSTM_Extractor] Missing keys:", missing)
            print("[MolSTM_Extractor] Unexpected keys:", unexpected)
            assert len(missing) == len(unexpected) == 0, \
                "[MolSTM_Extractor] Load Error! incomplete ckpt loaded"
        else:
            print("[MolSTM_Extractor] Skipping pretrained weight loading.")
        
        return model

    def forward(self, drugs):
        device = next(self.parameters()).device
        
        processed_drugs = []
        for s in drugs:
            if s == 'xxx':
                processed_drugs.append('C')
            else:
                mol = Chem.MolFromSmiles(s)
                if mol is not None:
                    processed_drugs.append(Chem.MolToSmiles(mol))
                else:
                    processed_drugs.append('C')
        drugs = processed_drugs
        
        tok = self.tokenizer.tokenize(drugs, pad=True)
        ids = self.tokenizer.convert_tokens_to_ids(tok['original_tokens'])


        encoder_input = torch.tensor(ids, dtype=torch.long, device=device).T
        encoder_pad_mask = torch.tensor(
            tok['masked_pad_masks'], dtype=torch.bool, device=device
        ).T


        encoder_input = encoder_input[:self.max_seq_len]
        encoder_pad_mask = encoder_pad_mask[:self.max_seq_len]

        memory = self._model.encode({
            "encoder_input": encoder_input,
            "encoder_pad_mask": encoder_pad_mask,
        })

        valid = (~encoder_pad_mask).float()
        weights = valid / (valid.sum(dim=0, keepdim=True) + 1e-9)
        weights = weights.unsqueeze(-1)
        mol_vec = (memory * weights).sum(dim=0)
        
        return mol_vec