"""
TopKSAE v2 — adds feature resampling (nudging) to prevent dead features.
Any feature inactive for `resample_threshold` consecutive batches gets its
encoder/decoder directions reinitialised from high-loss input samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TopKSAEv2(nn.Module):
    def __init__(self, input_dim: int, hidden_factor: int = 2, k: int = 32,
                 resample_threshold: int = 500):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim * hidden_factor
        self.k = k
        self.resample_threshold = resample_threshold

        self.encoder = nn.Linear(input_dim, self.hidden_dim, bias=True)
        self.decoder = nn.Linear(self.hidden_dim, input_dim, bias=True)

        # Track batches since last activation for each feature
        self.register_buffer("inactive_batches", torch.zeros(self.hidden_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        with torch.no_grad():
            w = F.normalize(self.encoder.weight.clone().T, dim=0)
            self.decoder.weight.copy_(w)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def top_k_gate(self, h: torch.Tensor) -> torch.Tensor:
        topk_vals, topk_idx = torch.topk(h, self.k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, topk_idx, 1.0)
        return h * mask * (h > 0).float()

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h)

    def forward(self, x: torch.Tensor):
        h_pre = self.encode(x)
        h = self.top_k_gate(h_pre)
        x_hat = self.decode(h)

        # Update per-feature inactivity counter
        with torch.no_grad():
            active_any = (h > 0).any(0)           # (hidden_dim,) bool
            self.inactive_batches += 1
            self.inactive_batches[active_any] = 0

        return x_hat, h

    def loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        return ((x - x_hat) ** 2).mean()

    def resample_dead_features(self, x_batch: torch.Tensor) -> int:
        """
        Reinitialise decoder + encoder directions for dead features.
        Called every N batches during training.
        Uses input samples from the current batch as new feature directions.
        Returns number of features resampled.
        """
        dead_mask = self.inactive_batches >= self.resample_threshold
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0:
            return 0

        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        n_resample = min(n_dead, x_batch.shape[0])
        dead_indices = dead_indices[:n_resample]

        with torch.no_grad():
            # Use random input samples as new feature directions
            perm = torch.randperm(x_batch.shape[0], device=x_batch.device)[:n_resample]
            new_dirs = F.normalize(x_batch[perm].float(), dim=1)  # (n_resample, input_dim)

            self.decoder.weight.data[:, dead_indices] = new_dirs.T
            self.encoder.weight.data[dead_indices] = new_dirs
            self.encoder.bias.data[dead_indices] = 0.0
            self.inactive_batches[dead_indices] = 0

        return n_dead

    def normalize_decoder(self):
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1.0)
            self.decoder.weight.div_(norms)

    @torch.no_grad()
    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        h_pre = self.encode(x)
        return self.top_k_gate(h_pre)
