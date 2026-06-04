"""
Top-K Sparse Autoencoder.
Architecture: Linear encoder → Top-K hard gate → Linear decoder.
Loss: MSE reconstruction only (Top-K handles sparsity).
"""

import torch
import torch.nn as nn
import numpy as np


class TopKSAE(nn.Module):
    def __init__(self, input_dim: int, hidden_factor: int = 4, k: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim * hidden_factor
        self.k = k

        self.encoder = nn.Linear(input_dim, self.hidden_dim, bias=True)
        self.decoder = nn.Linear(self.hidden_dim, input_dim, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        # Decoder columns start as normalised copies of encoder rows
        with torch.no_grad():
            w = self.encoder.weight.clone().T
            w = w / (w.norm(dim=0, keepdim=True) + 1e-8)
            self.decoder.weight.copy_(w)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return pre-gate hidden activations."""
        return self.encoder(x)

    def top_k_gate(self, h: torch.Tensor) -> torch.Tensor:
        """Hard Top-K: zero out all but the k largest values per sample."""
        topk_vals, topk_idx = torch.topk(h, self.k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, topk_idx, 1.0)
        # Keep only positive activations in the top-k (ReLU-like)
        return h * mask * (h > 0).float()

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h)

    def forward(self, x: torch.Tensor):
        h_pre = self.encode(x)
        h = self.top_k_gate(h_pre)
        x_hat = self.decode(h)
        return x_hat, h

    def loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        return ((x - x_hat) ** 2).mean()

    @torch.no_grad()
    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Return sparse hidden activations without gradient."""
        h_pre = self.encode(x)
        return self.top_k_gate(h_pre)

    def normalize_decoder(self):
        """Keep decoder columns unit-norm for stability (call after each batch)."""
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1.0)
            self.decoder.weight.div_(norms)
