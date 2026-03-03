import torch
from torch import nn
from omegaconf import OmegaConf
from safetensors.torch import load_file
from collections import OrderedDict

from model.se import StateEmbeddingModel
from data.utils import get_embedding_cfg

def get_embeddings(cfg):
    # Load in ESM2 embeddings and special tokens
    all_pe = torch.load(get_embedding_cfg(cfg).all_embeddings, weights_only=False)
    if isinstance(all_pe, dict):
        all_pe = torch.vstack(list(all_pe.values()))

    all_pe = all_pe.cuda()
    return all_pe

class CellEmbeddingExtractor():
    def __init__(self, 
        config = "../configs/se600m.yaml",
        se_model_path = "checkpoints/se600m.safetensors"
    ):
        self.cfg = OmegaConf.load(config) 

        self.se = self._load_se(se_model_path)
        self.se.eval()
        for p in self.se.parameters():
            p.requires_grad_(False)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"initialized pretrained SE model on device: {device}")
        self.se.to(device)

    def _load_se(self, model_path):
        model = StateEmbeddingModel(
            token_dim=5120,
            d_model=2048,
            nhead=16,
            d_hid=2048,
            nlayers=16,
            output_dim=2048,
            dropout=0.1,
            warmup_steps=0,
            compiled=False,
            max_lr=self.cfg.optimizer.max_lr,
            emb_size=5120,
            collater=None,
            cfg=self.cfg,
        )

        all_pe = get_embeddings(self.cfg)
        all_pe.requires_grad = False
        model.pe_embedding = nn.Embedding.from_pretrained(all_pe)

        state = load_file(model_path)

        want = set(model.state_dict().keys())
        new_state = OrderedDict(state)

        pairs = [
            ("encoder.0.weight", "gene_embedding_layer.0.weight"),
            ("encoder.0.bias",   "gene_embedding_layer.0.bias"),
            ("encoder.1.weight", "gene_embedding_layer.1.weight"),
            ("encoder.1.bias",   "gene_embedding_layer.1.bias"),
        ]

        for a, b in pairs:
            if a in state and b in want and b not in state:
                new_state[b] = state[a]
            if b in state and a in want and a not in state:
                new_state[a] = state[b]
    
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        print(f"loaded SE model from {model_path}")
        print("missing:", missing)
        print("unexpected:", unexpected)
        assert len(missing) == len(unexpected) == 0, "[ERROR] SE model checkpoint has inconsistent state_dict with model object!"
        
        return model