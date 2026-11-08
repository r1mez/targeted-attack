"""Frozen BA3Motif classifiers: differentiable proxy and probability-only target."""

from hashlib import sha256
import json

import torch
from torch.nn import functional as F

from src.data.ba3motif_dataset import CLASS_NAMES
from src.models.ba3motif_gnn import BA3MotifLEConv


def load_frozen(checkpoint):
    """Load the existing checkpoint format without changing class order or kwargs."""
    payload = torch.load(checkpoint, map_location="cpu")
    if tuple(payload["class_names"]) != CLASS_NAMES:
        raise ValueError(f"Unexpected output classes in {checkpoint}")
    model = BA3MotifLEConv(**payload["model_kwargs"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval().requires_grad_(False)


def tensor_hash(tensor):
    """Hash values together with dtype and shape; device does not change identity."""
    tensor = tensor.detach().cpu().contiguous()
    state = sha256()
    state.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
    state.update(b"\0")
    state.update(tensor.numpy().tobytes())
    return state.hexdigest()


def state_hash(model):
    state = sha256()
    for name, value in sorted(model.state_dict().items()):
        state.update(name.encode())
        state.update(b"\0")
        state.update(bytes.fromhex(tensor_hash(value)))
    return state.hexdigest()


class DenseProxy:
    """Evaluate a frozen proxy on complete directed graphs, retaining input gradients.

    Inputs have shapes ``[B, N, F]`` and ``[B, N, N]``. Dense LEConv sums use
    the same frozen weights as the PyG classifier, avoiding CUDA scatter_add
    (unsupported by strict determinism in the pinned PyTorch). All node pairs,
    including zero-weight edges, retain gradients through the adjacency.
    """

    def __init__(self, model, nodes):
        if not isinstance(nodes, int) or isinstance(nodes, bool) or nodes <= 0:
            raise ValueError("nodes must be a positive integer")
        self.model = model.eval().requires_grad_(False)
        self.nodes = nodes
        if not isinstance(model, BA3MotifLEConv):
            raise TypeError("DenseProxy implements the existing BA3Motif LEConv classifier")
        self.forwards = 0

    def __call__(self, x, adjacency):
        if (x.ndim != 3 or x.size(1) != self.nodes or x.size(0) == 0
                or adjacency.shape != (x.size(0), self.nodes, self.nodes)):
            raise ValueError("Expected x[B,N,F] and adjacency[B,N,N] for this proxy")
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("DenseProxy requires frozen parameters in evaluation mode")
        parameter = next(self.model.parameters())
        x = x.to(device=parameter.device, dtype=parameter.dtype)
        adjacency = adjacency.to(device=parameter.device, dtype=parameter.dtype)
        self.forwards += x.size(0)
        h = self.model.node_emb(x)
        incoming = adjacency.transpose(-1, -2)
        degree = incoming.sum(-1, keepdim=True)
        for conv in self.model.convs:
            # PyG LEConv: sum_j A[j,i]*(lin1(h_j)-lin2(h_i))+lin3(h_i).
            # Dropout is the identity because this classifier is frozen in eval.
            h = F.relu(incoming @ conv.lin1(h) - degree * conv.lin2(h) + conv.lin3(h))
        pooled = h.mean(dim=1)
        return self.model.lin2(F.relu(self.model.lin1(pooled)))


class ProbabilityOracle:
    """Expose only target probabilities, with no gradient through target queries."""

    def __init__(self, checkpoint):
        self._model = load_frozen(checkpoint)
        self._initial_state = state_hash(self._model)
        self._dense = None

    @torch.no_grad()
    def probabilities(self, features, adjacency):
        if features.ndim != 2:
            raise ValueError("Expected node features with shape [N,F]")
        n = features.size(0)
        if (n == 0 or features.size(1) != self._model.in_channels
                or not torch.isfinite(features).all() or adjacency.shape != (n, n)
                or not torch.equal(adjacency, adjacency.T)):
            raise ValueError("Expected finite node features and a symmetric square adjacency")
        if self._model.training or any(p.requires_grad for p in self._model.parameters()):
            raise RuntimeError("The target must remain frozen and in evaluation mode")
        parameter = next(self._model.parameters())
        features = features.to(device=parameter.device, dtype=parameter.dtype)
        adjacency = adjacency.to(parameter.device)
        if adjacency.dtype == torch.bool:
            edges = adjacency.nonzero().t().contiguous()
            batch = torch.zeros(n, dtype=torch.long, device=parameter.device)
            logits = self._model(features, edges, batch)
        elif (adjacency.dtype == torch.float32 and torch.isfinite(adjacency).all()
              and not (adjacency < 0).any() and not (adjacency > 1).any()):
            if self._dense is None or self._dense.nodes != n:
                self._dense = DenseProxy(self._model, n)
            logits = self._dense(features.unsqueeze(0), adjacency.unsqueeze(0))
        else:
            raise ValueError("Expected bool or float32 adjacency with weights in [0,1]")
        probabilities = logits.softmax(dim=-1)[0]
        if (not torch.isfinite(probabilities).all() or (probabilities < 0).any()
                or abs(float(probabilities.sum()) - 1) > 1e-5):
            raise RuntimeError("Invalid target probabilities")
        return probabilities.cpu().tolist()

    def check_frozen(self):
        if (self._model.training or state_hash(self._model) != self._initial_state
                or any(p.requires_grad or p.grad is not None for p in self._model.parameters())):
            raise RuntimeError("The target model changed or acquired gradients")
