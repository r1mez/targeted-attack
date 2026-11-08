"""Structural graph autoencoder (the existing B architecture).

The parameter names and tensor shapes deliberately match historical B checkpoints.
Only unordered, non-loop node pairs are decoded; loops are copied from the input.
"""

import torch
from torch import nn


class StructuralAutoencoder(nn.Module):
    def __init__(self, features, auxiliary, hidden, latent, layers, edge_hidden):
        super().__init__()
        for name, value in (("features", features), ("hidden", hidden),
                            ("latent", latent), ("layers", layers),
                            ("edge_hidden", edge_hidden)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(auxiliary, int) or isinstance(auxiliary, bool) or auxiliary < 0:
            raise ValueError("auxiliary must be a non-negative integer")
        self.auxiliary = auxiliary
        self.input = nn.Linear(features + auxiliary + 1, hidden)
        self.updates = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.project = nn.Linear(hidden, latent)
        self.edge = nn.Sequential(
            nn.Linear(3 * latent, edge_hidden), nn.GELU(),
            nn.Linear(edge_hidden, edge_hidden), nn.GELU(), nn.Linear(edge_hidden, 1),
        )

    def encode(self, x, adjacency, auxiliary):
        a = adjacency.to(dtype=x.dtype).clone()
        a.diagonal(dim1=-2, dim2=-1).zero_()
        degree = a.sum(-1, keepdim=True)
        h = self.input(torch.cat((x, degree / (a.size(-1) - 1), auxiliary), dim=-1))
        a = a / degree.clamp_min(1).sqrt()
        for update, norm in zip(self.updates, self.norms):
            h = norm(h + update(torch.cat((h, a @ h), dim=-1)))
        return self.project(h)

    def edge_logits(self, z, rows, cols):
        left, right = z[:, rows], z[:, cols]
        pair = torch.cat((left + right, (left - right).abs(), left * right), dim=-1)
        return self.edge(pair).squeeze(-1)


def restore_adjacency(original, values, rows, cols):
    """Mirror upper-triangular edge values and copy the original diagonal.

    Assignment preserves the gradient from ``values`` to the returned float
    adjacency. Neither thresholding nor discrete sampling is performed.
    """
    if original.ndim != 3 or original.size(-1) != original.size(-2):
        raise ValueError("original must have shape [batch, nodes, nodes]")
    if values.ndim != 2 or values.shape != (original.size(0), rows.numel()):
        raise ValueError("values must have shape [batch, number_of_pairs]")
    if rows.ndim != 1 or cols.shape != rows.shape:
        raise ValueError("rows and cols must be equally sized vectors")
    adjacency = torch.diag_embed(original.diagonal(dim1=-2, dim2=-1)).to(values.dtype)
    adjacency[:, rows, cols] = values
    adjacency[:, cols, rows] = values
    return adjacency
