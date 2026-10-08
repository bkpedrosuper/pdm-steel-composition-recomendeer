"""Rede neural do surrogate: uma MLP multi-saída (LE, LR, AL)."""
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, hidden: list[int], dropout: float = 0.0):
        super().__init__()
        layers, width = [], n_inputs
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
