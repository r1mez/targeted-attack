"""Continuous BA3Motif prototype generation and training-graph structure prior.

This module optimizes edge weights against a frozen surrogate. It does not
discretize graphs or supply the counterfactual training objective L_boundary.
"""

from dataclasses import dataclass
from hashlib import sha256
from time import monotonic

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F


NODES = 25
UPPER = torch.triu_indices(NODES, NODES, offset=1)


@dataclass
class SeedGraph:
    parent_id: int
    label: int
    features: torch.Tensor
    adjacency: torch.Tensor

    @classmethod
    def from_data(cls, graph):
        x, edges = graph.x, graph.edge_index
        if x.shape != (NODES, 10) or x.dtype != torch.float32 or not x.isfinite().all():
            raise ValueError("Unexpected training feature shape, dtype, or values")
        if edges.ndim != 2 or edges.size(0) != 2 or edges.dtype != torch.long:
            raise ValueError("Unexpected training edge index shape or dtype")
        if edges.numel() and (edges.min() < 0 or edges.max() >= NODES):
            raise ValueError("Training edge index out of range")
        adjacency = torch.zeros((NODES, NODES), dtype=torch.bool)
        adjacency[edges[0], edges[1]] = True
        if not torch.equal(adjacency, adjacency.T) or adjacency.sum() != edges.size(1):
            raise ValueError("Training edges must be symmetric and unique")
        return cls(int(graph.idx), int(graph.y), x.clone(), adjacency)


def graph_hash(features, adjacency):
    return sha256(features.detach().cpu().contiguous().numpy().tobytes()
                  + adjacency.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def structure_vector(adjacency):
    """Keep the original continuous edge-count, degree-moment, triangle prior."""
    degree = adjacency.sum(dim=-1)
    variance = (adjacency * (1 - adjacency)).sum(dim=-1)
    return torch.stack((degree.sum() / 2, (degree.square() + variance).mean(),
                        (adjacency @ adjacency @ adjacency).trace() / 6))


def _graph_structure(adjacency):
    off_diagonal = adjacency.clone()
    off_diagonal.fill_diagonal_(False)
    vector = structure_vector(off_diagonal.to(torch.float64)).tolist()
    network = nx.from_numpy_array(off_diagonal.numpy())
    degrees = [degree for _, degree in network.degree]
    return {
        "vector": vector,
        "degree_histogram": np.bincount(degrees, minlength=NODES).tolist(),
        "clustering": nx.average_clustering(network),
        "bridge_fraction": sum(1 for _ in nx.bridges(network)) / max(1, network.size()),
        "connected": nx.is_connected(network),
    }


class StructureReference:
    """Nearest training-graph structure distance with the existing normalization."""

    def __init__(self, sources):
        self.rows = [
            {"parent_id": graph.parent_id, "label": graph.label,
             **_graph_structure(graph.adjacency)}
            for graph in sources
        ]
        self.vectors = torch.tensor([row["vector"] for row in self.rows], dtype=torch.float64)
        self.scale = self.vectors.std(dim=0, unbiased=False).clamp_min(1)
        normalized = self.vectors / self.scale
        distances = (normalized[:, None] - normalized[None, :]).square().mean(dim=-1)
        distances.fill_diagonal_(float("inf"))
        self.rho = max(1.0, float(np.percentile(distances.min(dim=1).values.numpy(), 95)))
        self.size_budget = max(24, int(np.ceil(np.percentile(self.vectors[:, 0], 99))) + 2)

    def distance(self, vector):
        distances = ((vector - self.vectors) / self.scale).square().mean(dim=-1)
        index = int(distances.argmin())
        return distances[index], self.rows[index]["parent_id"]

    def record(self):
        return {
            "training_graphs": self.rows,
            "structure_vector": ["undirected_non_loop_edges", "mean_squared_degree", "triangles"],
            "scales": self.scale.tolist(), "B_size": self.size_budget, "rho": self.rho,
        }


def boundary_stats(probabilities, pair, gap=0.05, margin=0.10):
    left, right = (probabilities[index] for index in pair)
    third = next(index for index in range(3) if index not in pair)
    delta = abs(left - right)
    gamma = min(left, right) - probabilities[third]
    return {"delta": delta, "gamma": gamma, "passed": delta <= gap and gamma >= margin}


class _WeightedGraph:
    """A deterministic complete off-diagonal support with fixed source loops."""

    def __init__(self, source):
        self.original = source.adjacency[UPPER[0], UPPER[1]].float()
        self.loops = source.adjacency.diag().nonzero().flatten()
        edges = torch.cat((UPPER, UPPER.flip(0), self.loops.repeat(2, 1)), dim=1)
        self.inputs = (source.features, edges, torch.zeros(NODES, dtype=torch.long))

    def matrix(self, edges, loops):
        adjacency = edges.new_zeros((NODES, NODES))
        adjacency[UPPER[0], UPPER[1]] = edges
        adjacency[UPPER[1], UPPER[0]] = edges
        if loops:
            adjacency[self.loops, self.loops] = 1
        return adjacency

    def message_weights(self, edges):
        return torch.cat((edges, edges, edges.new_ones(len(self.loops))))

    def logits(self, model, edges):
        return model(*self.inputs, edge_weight=self.message_weights(edges))[0]


@torch.no_grad()
def proxy_probabilities(model, source, *, weighted):
    """Compare the original graph through sparse and continuous model inputs."""
    if weighted:
        graph = _WeightedGraph(source)
        logits = graph.logits(model, graph.original)
    else:
        edges = source.adjacency.nonzero().t().contiguous()
        logits = model(source.features, edges, torch.zeros(NODES, dtype=torch.long))[0]
    return logits.softmax(dim=-1).tolist()


def check_deadline(deadline):
    if monotonic() >= deadline:
        raise TimeoutError("Approved wall-clock limit reached")


def optimize_soft_boundary(source, model, reference, pair, config, deadline, log):
    """Return the selected adjacency, surrogate embedding and optimization record.

    The caller supplies an eval-mode model with frozen parameters. Gradients
    apply only to 300 undirected non-loop weights, projected to [0, 1].
    """
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Soft prototypes require an eval-mode, frozen surrogate")
    graph = _WeightedGraph(source)
    anchor = graph.original
    weights = torch.nn.Parameter(anchor.clone())
    optimizer = torch.optim.Adam(
        [weights], lr=config["learning_rate"], betas=tuple(config["adam_betas"]),
        eps=config["adam_eps"],
    )
    i, j = pair
    third = next(k for k in range(3) if k not in pair)
    best_key = None
    best_weights = None
    best_record = None
    for step in range(config["steps"] + 1):
        check_deadline(deadline)
        optimizer.zero_grad(set_to_none=True)
        logits = graph.logits(model, weights)
        probabilities = logits.detach().softmax(-1).tolist()
        boundary = boundary_stats(probabilities, pair, config["probability_gap"],
                                  config["third_class_margin"])
        delta_excess = max(0.0, boundary["delta"] - config["probability_gap"])
        gamma_shortfall = max(0.0, config["third_class_margin"] - boundary["gamma"])
        distance, _ = reference.distance(structure_vector(graph.matrix(weights, loops=False)))
        anchor_distance = (weights - anchor).abs().sum()
        loss = (
            (logits[i] - logits[j]).square()
            + F.relu(config["third_logit_margin"] + logits[third]
                     - torch.minimum(logits[i], logits[j])).square()
            + config["structure_weight"] * distance
            + config["anchor_weight"] * anchor_distance
        )
        if not loss.isfinite():
            raise RuntimeError(f"Non-finite loss at step {step}")

        eps = config["fractional_eps"]
        fractional = int(((weights > eps) & (weights < 1 - eps)).sum())
        record = {
            "step": step, "proxy_probs": probabilities, "proxy_boundary": boundary,
            "fractional_edges": fractional, "weight_sum": float(weights.detach().sum()),
            "structure_distance": float(distance.detach()),
            "anchor_l1": float(anchor_distance.detach()), "loss": float(loss.detach()),
        }
        key = (fractional == 0, not boundary["passed"], delta_excess + gamma_shortfall,
               record["structure_distance"], record["anchor_l1"], step)
        if best_key is None or key < best_key:
            best_key = key
            best_weights = weights.detach().clone()
            best_record = record
        log(record)

        if step == config["steps"]:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([weights], config["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            if not weights.isfinite().all():
                raise RuntimeError(f"Non-finite edge weights after step {step}")
            weights.clamp_(0, 1)

    with torch.no_grad():
        embedding = model.get_graph_rep(
            *graph.inputs, edge_weight=graph.message_weights(best_weights),
        )[0]
    if not embedding.isfinite().all():
        raise RuntimeError("Non-finite selected proxy embedding")
    return graph.matrix(best_weights, loops=True), embedding, best_record
