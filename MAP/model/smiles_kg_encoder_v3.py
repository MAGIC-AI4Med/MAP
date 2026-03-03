import torch
import torch.nn as nn
from rdkit import Chem
from model.mega_molbart.STMencoder_ddp import MolSTM_Extractor

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

class KGSmilesEncoder_v3(nn.Module):
    def __init__(self, d_smiles=256, d_model=1024):
        super().__init__()
        self.smiles_encoder = MolSTM_Extractor(pretrained=True)
        self.smiles_projector = ResidualProjector(
            input_dim=d_smiles,
            output_dim=d_model,
            hidden_dim=512
        )

    def forward(self, smiles_list):
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
        assert 'xxx' not in smiles_list, f"something wrong with smiles_list, 'xxx' in it! smiles_list:{smiles_list}"

        smiles_emb = self.smiles_encoder(smiles_list)
        proj = self.smiles_projector(smiles_emb)
        return proj

    def load_from_full_model(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        sub_state = {}

        for k, v in state.items():
            if k.startswith("smiles_encoder.") or k.startswith("smiles_projector."):
                new_k = k.replace("module.", "")
                sub_state[new_k] = v

        missing, unexpected = self.load_state_dict(sub_state, strict=False)

        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
