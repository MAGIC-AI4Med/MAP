import math
from rdkit import Chem

import numpy as np
import torch
import torch.nn as nn



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



class DrugProjector(nn.Module):
    def __init__(
        self,
        d_in=256,
        d_out=1024,
        dropout=0.1,
    ):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.hidden_dim = (d_in + d_out) // 2
        self.activation = nn.GELU()

        layers = [
            nn.Linear(d_in, self.hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, d_out)
        ]
        self.projector = nn.Sequential(*layers)

    def forward(self, x):
        return self.projector(x)
    


class CellProjector(nn.Module):
    def __init__(self, n_layers: int, d_cell: int, d_out: int, hidden_dim: int = None, dropout: float = 0.1, activation: nn.Module = nn.GELU):
        """
        Args:
            d_cell: Input dimension of cell embeddings
            d_out: Output dimension
            hidden_dim: Hidden layer dimension. If None, defaults to max(d_cell, d_out)
            dropout: Dropout rate
            activation: Activation function class (not instance)
        """
        super().__init__()

        self.d_cell = d_cell
        if hidden_dim is None:
            hidden_dim = max(d_cell, d_out)

        layers = []
        layers.append(nn.Linear(d_cell, hidden_dim))
        layers.append(activation())
        layers.append(nn.Dropout(dropout))

        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, d_out))

        self.projector = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, S, d_cell]

        Returns:
            Output tensor of shape [B, S, d_out]
        """
        return self.projector(x)



class ProjectOut(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        act_cls = nn.GELU

        layers = []
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(act_cls())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)




class ConcentrationEmbedding(nn.Module):
    def __init__(
        self,
        d_model=1024,
        dropout=0.1,
        log_transform=True,
        min_concentration=1e-3,  # 0.001 uM
        max_concentration=1e3,   # 1000 uM
    ):
        super().__init__()

        self.d_model = d_model
        self.log_transform = log_transform
        self.min_concentration = min_concentration
        self.max_concentration = max_concentration

        hidden_dim = d_model // 4
        self.activation = nn.GELU()

        layers = [
            nn.Linear(1, hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model)
        ]
        self.embedding_net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _preprocess_concentration(self, concentration):
        concentration = torch.clamp(
            concentration,
            min=self.min_concentration,
            max=self.max_concentration
        )

        if self.log_transform:
            concentration = torch.log10(concentration)
            log_min = math.log10(self.min_concentration)
            log_max = math.log10(self.max_concentration)
            concentration = 2 * (concentration - log_min) / (log_max - log_min) - 1
        else:
            concentration = (concentration - self.min_concentration) / \
                (self.max_concentration - self.min_concentration)
        return concentration

    def forward(self, concentration):
        if concentration.dim() == 1:
            concentration = concentration.unsqueeze(-1)  # (batch_size, 1)

        concentration = self._preprocess_concentration(concentration)
        embedding = self.embedding_net(concentration)
        return embedding