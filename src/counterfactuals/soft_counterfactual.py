"""Frozen graph encoding, differentiable transport, and soft-graph objectives.

The encoder is evaluated under ``no_grad``. The frozen decoder and proxy retain
input gradients so both transport legs can update the shared flow. No boundary
prototype, threshold, top-k projection, or hard sample enters this module.
"""

import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from src.counterfactuals.graph_ae import StructuralAutoencoder, restore_adjacency
from src.counterfactuals.unpaired_flow import ClassFlow, leap
from src.models.ba3motif_model_access import state_hash


SCHEMA_VERSION = 2


def fixed_auxiliary(ids, nodes, dimension, seed):
    """Reproduce one node-feature draw for each ``ae_seed + graph_id`` on CPU."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("ae_seed must be a non-negative integer")
    if not isinstance(nodes, int) or nodes <= 1 or not isinstance(dimension, int) or dimension < 0:
        raise ValueError("nodes must exceed one and auxiliary dimension must be non-negative")
    graph_ids = torch.as_tensor(ids)
    if graph_ids.ndim != 1 or graph_ids.numel() == 0:
        raise ValueError("ids must be a nonempty vector")
    if graph_ids.dtype == torch.bool or graph_ids.is_floating_point() or graph_ids.is_complex():
        raise ValueError("graph ids must be integers")
    if bool((graph_ids < 0).any()):
        raise ValueError("graph ids must be non-negative")
    if any(seed + int(graph_id) >= 2 ** 63 for graph_id in graph_ids.tolist()):
        raise ValueError("ae_seed + graph_id must fit in a signed 64-bit integer")
    return torch.stack([
        torch.randn(nodes, dimension, generator=torch.Generator().manual_seed(seed + int(graph_id)))
        for graph_id in graph_ids.tolist()
    ])


def freeze(module):
    """Freeze parameters and clear stale gradients while keeping input autograd."""
    module.eval().requires_grad_(False)
    for parameter in module.parameters():
        parameter.grad = None
    return module


def validate_graph_batch(x, adjacency, ids=None):
    if x.ndim != 3 or not x.is_floating_point() or x.size(0) == 0 or x.size(1) <= 1:
        raise ValueError("x must be a nonempty float [batch, nodes>1, features] tensor")
    if adjacency.shape != (x.size(0), x.size(1), x.size(1)):
        raise ValueError("adjacency must have shape [batch, nodes, nodes]")
    if x.device != adjacency.device:
        raise ValueError("x and adjacency must be on the same device")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(adjacency).all()):
        raise ValueError("graph tensors must be finite")
    if not torch.equal(adjacency, adjacency.transpose(-1, -2)):
        raise ValueError("original adjacency must be symmetric")
    if not bool(((adjacency == 0) | (adjacency == 1)).all()):
        raise ValueError("original adjacency must contain binary values")
    if ids is not None and torch.as_tensor(ids).shape != (x.size(0),):
        raise ValueError("ids must have one value per graph")


def validate_normalization(mean, std, latent):
    if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        raise ValueError("normalization mean/std must be tensors")
    if mean.shape != (latent,) or std.shape != (latent,):
        raise ValueError("normalization mean/std must have shape [latent]")
    if (not mean.is_floating_point() or not std.is_floating_point()
            or not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all())
            or bool((std <= 0).any())):
        raise ValueError("normalization must be finite with strictly positive std")


def normalization_hash(mean, std):
    """Fingerprint the saved normalization without changing its dtype."""
    digest = hashlib.sha256()
    for tensor in (mean, std):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _architecture(checkpoint, external, name):
    stored = checkpoint.get("architecture")
    if external is not None and stored is not None and stored != external:
        raise RuntimeError(f"{name} checkpoint architecture differs from the expected architecture")
    architecture = stored if stored is not None else external
    if not isinstance(architecture, dict):
        raise RuntimeError(f"{name} checkpoint requires an explicit architecture")
    return dict(architecture)


def _metadata(checkpoint, architecture):
    keys = ("schema_version", "stage", "group", "selection", "epoch", "provenance",
            "normalization", "initialization", "seeds", "training")
    return {**{key: checkpoint[key] for key in keys if key in checkpoint},
            "architecture": architecture}


def _check_provenance(checkpoint, expected):
    if expected is None:
        return
    actual = checkpoint.get("provenance")
    if not isinstance(actual, dict):
        raise RuntimeError("Checkpoint does not contain the expected provenance")
    for key, value in expected.items():
        if key not in actual or actual[key] != value:
            raise RuntimeError(f"Checkpoint provenance differs for {key}")


def load_autoencoder(path, architecture=None, *, expected_provenance=None):
    """Load B strictly, including legacy B-selected files lacking architecture.

    Legacy ``selection.threshold`` is retained only as provenance. It never
    changes the soft decoder output. Full historical run checks are centralized
    in :func:`load_ae`.
    """
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("group", "B") != "B":
        raise RuntimeError("Only the structural B autoencoder is supported")
    if "schema_version" in checkpoint:
        if checkpoint["schema_version"] != SCHEMA_VERSION or checkpoint.get("stage") != "ae_pretrain":
            raise RuntimeError("Unsupported autoencoder checkpoint schema/stage")
    spec = _architecture(checkpoint, architecture, "AE")
    _check_provenance(checkpoint, expected_provenance)
    ae = StructuralAutoencoder(**spec)
    ae.load_state_dict(checkpoint["state_dict"], strict=True)
    return freeze(ae), _metadata(checkpoint, spec)


def load_flow(path, architecture=None, *, expected_provenance=None):
    """Load a completed single-stage joint-loss flow for evaluation only."""
    checkpoint = torch.load(path, map_location="cpu")
    spec = _architecture(checkpoint, architecture, "Flow")
    if (checkpoint.get("schema_version") != SCHEMA_VERSION
            or checkpoint.get("stage") != "joint_flow"
            or checkpoint.get("initialization") != "random"):
        raise RuntimeError("Current evaluation requires a single-stage joint_flow checkpoint")
    _check_provenance(checkpoint, expected_provenance)
    mean, std = checkpoint["mean"].detach(), checkpoint["std"].detach()
    validate_normalization(mean, std, spec["latent"])
    if "schema_version" in checkpoint:
        if checkpoint.get("normalization") != normalization_hash(mean, std):
            raise RuntimeError("Flow normalization fingerprint mismatch")
    flow = ClassFlow(**spec)
    flow.load_state_dict(checkpoint["state_dict"], strict=True)
    return freeze(flow), mean, std, _metadata(checkpoint, spec)


def _read(path):
    return json.loads(Path(path).read_text())


def _compare_inputs(ancestor_inputs, current_inputs, names):
    for name in names:
        if name not in ancestor_inputs or current_inputs.get(name) != ancestor_inputs[name]:
            raise RuntimeError(f"Checkpoint provenance mismatch for input {name}")


def _seed_check(saved, expected, names):
    for name in names:
        if name not in saved or saved[name] != expected.get(name):
            raise RuntimeError(f"Checkpoint seed mismatch for {name}")


def _manifest_ids(value):
    return torch.as_tensor(value).cpu().tolist()


def load_ae(config, paths, *, training_ids=None):
    """Load only the frozen AE, retaining historical B provenance checks."""
    if config.get("fixed_auxiliary_seed") != config.get("seeds", {}).get("ae"):
        raise RuntimeError("Fixed auxiliary seed must match the saved AE seed")
    ae, ae_meta = load_autoencoder(paths["ae_checkpoint"], config["ae"])
    ae_samples = None
    if "schema_version" not in ae_meta:
        summary, parent = _read(paths["ae_summary"]), _read(paths["ae_run"])
        ae_samples = _read(paths["ae_samples"])
        group = summary.get("groups", {}).get("B", {})
        ancestor = parent["config"]
        if (summary.get("status") != "completed" or summary.get("selected_group") != "B"
                or not group.get("gate_passed") or ae_meta.get("group") != "B"
                or ae_meta.get("selection") != group.get("selection")
                or parent.get("signature") != summary.get("signature")
                or ancestor.get("enhanced") != config["ae"]):
            raise RuntimeError("Structural AE does not match the accepted historical B selection")
        _seed_check(ancestor["seeds"], config["seeds"], ("ae",))
        _compare_inputs(ancestor["inputs"], config["inputs"], ("data", "target", "proxy", "cache"))
        for name in ("training_ids", "selection_ids", "audit_ids"):
            ids = ae_samples.get(name)
            if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
                raise RuntimeError(f"Invalid historical AE manifest field {name}")
        if (set(ae_samples["selection_ids"]) & set(ae_samples["audit_ids"])
                or set(ae_samples["training_ids"]) & (set(ae_samples["selection_ids"])
                                                        | set(ae_samples["audit_ids"]))):
            raise RuntimeError("Historical AE sample partitions overlap")
        if training_ids is not None and ae_samples["training_ids"] != _manifest_ids(training_ids):
            raise RuntimeError("AE training manifest differs from the current training IDs")
        ae_meta["source_run_signature"] = parent["signature"]
    else:
        provenance = ae_meta.get("provenance", {})
        _compare_inputs(provenance.get("inputs", {}), config["inputs"], ("data",))
        _seed_check(provenance.get("seeds", ae_meta.get("seeds", {})), config["seeds"], ("ae",))
        if provenance.get("fixed_auxiliary_seed") != config["fixed_auxiliary_seed"]:
            raise RuntimeError("AE provenance auxiliary seed differs from the current configuration")
    return ae, ae_meta


def load_evaluation_models(config, paths, *, training_ids=None):
    """Load the frozen AE and the explicitly configured trained joint-loss flow.

    Training never calls this function. Flow input belongs to evaluation only;
    historical CFM-only weights are not initialization for the current method.
    """
    ae, ae_meta = load_ae(config, paths, training_ids=training_ids)
    if config["flow"]["latent"] != config["ae"]["latent"]:
        raise RuntimeError("AE and flow latent dimensions differ")
    flow, mean, std, flow_meta = load_flow(paths["flow_checkpoint"], config["flow"])
    provenance = flow_meta.get("provenance", {})
    _compare_inputs(provenance.get("inputs", {}), config["inputs"],
                    ("data", "target", "proxy", "cache", "ae_checkpoint"))
    _seed_check(provenance.get("seeds", {}), config["seeds"], ("ae",))
    if provenance.get("fixed_auxiliary_seed") != config["fixed_auxiliary_seed"]:
        raise RuntimeError("Flow auxiliary seed differs from the current AE")
    if provenance.get("ae_architecture") != config["ae"]:
        raise RuntimeError("Flow provenance refers to a different AE architecture")
    if provenance.get("ae_state_hash") != state_hash(ae):
        raise RuntimeError("Flow provenance refers to different frozen AE weights")
    if provenance.get("normalization_source") != "frozen_ae_original_training_latents":
        raise RuntimeError("Flow must use normalization fitted on original training AE latents")
    ids = provenance.get("normalization_training_ids")
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
        raise RuntimeError("Flow lacks its original normalization training IDs")
    if training_ids is not None and not set(ids).issubset(set(_manifest_ids(training_ids))):
        raise RuntimeError("Flow normalization used graphs outside the training split")
    return ae, flow, mean, std, {"ae": ae_meta, "flow": flow_meta, "boundary_weight": 0.0}


def generate_soft(ae, flow, mean, std, x, adjacency, ids, source, target, *,
                  ae_seed, steps, multiplier, check_time=None):
    """Generate one symmetric float soft graph per source/target task.

    Encoding is detached once. Both RK4 legs and decoding stay differentiable.
    ``source`` must be the fixed target-GNN prediction for the ORIGINAL graph;
    the caller supplies targets according to its explicitly approved policy.
    """
    validate_graph_batch(x, adjacency, ids)
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if flow.output.out_features != ae.project.out_features:
        raise ValueError("AE and flow latent dimensions differ")
    if x.size(-1) != ae.input.in_features - ae.auxiliary - 1:
        raise ValueError("Graph feature dimension differs from the AE architecture")
    multiplier = torch.as_tensor(multiplier, dtype=x.dtype, device=x.device)
    if (multiplier.ndim not in (0, 1, 3)
            or multiplier.numel() not in (1, x.size(0))
            or not bool(torch.isfinite(multiplier).all()) or bool((multiplier <= 0).any())):
        raise ValueError("multiplier must be finite and strictly positive, scalar or per graph")
    if multiplier.numel() == 1:
        multiplier = multiplier.reshape(())
    for name, values in (("source", source), ("target", target)):
        labels = torch.as_tensor(values)
        if (labels.ndim > 1 or (labels.ndim == 1 and labels.numel() != x.size(0))
                or labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex()
                or bool(((labels < 0) | (labels >= flow.classes)).any())):
            raise ValueError(f"{name} must be an integer class or one class per graph")
    freeze(ae)
    validate_normalization(mean, std, ae.project.out_features)
    if check_time is not None:
        check_time()
    auxiliary = fixed_auxiliary(ids, x.size(1), ae.auxiliary, ae_seed).to(x)
    with torch.no_grad():
        original = ae.encode(x, adjacency, auxiliary)
        if not bool(torch.isfinite(original).all()):
            raise RuntimeError("Non-finite original graph latent")
        center, scale = mean.detach().to(original), std.detach().to(original)
        normalized = (original - center) / scale
    transported = leap(flow, normalized, source, target, multiplier, steps,
                       check_time=check_time)
    rows, cols = torch.triu_indices(x.size(1), x.size(1), offset=1, device=x.device)
    logits = ae.edge_logits(transported * scale + center, rows, cols)
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("Non-finite decoder edge logits")
    probability = logits.sigmoid()
    soft = restore_adjacency(adjacency, probability, rows, cols)
    return {"adjacency": soft, "edge_probabilities": probability,
            "normalized_source": normalized, "normalized_target": transported,
            "source": torch.as_tensor(source, device=x.device).long(),
            "target": torch.as_tensor(target, device=x.device).long()}


def counterfactual_loss(proxy, x, soft_adjacency, target, original_adjacency):
    """Mean target CE and mean per-graph unordered-pair L1 against ORIGINAL A."""
    if soft_adjacency.shape != original_adjacency.shape or soft_adjacency.ndim != 3:
        raise ValueError("soft and original adjacency must have the same batch shape")
    if not soft_adjacency.is_floating_point() or not bool(torch.isfinite(soft_adjacency).all()):
        raise ValueError("soft adjacency must be finite and floating point")
    if bool(((soft_adjacency < 0) | (soft_adjacency > 1)).any()):
        raise ValueError("soft adjacency weights must be in [0, 1]")
    if not torch.equal(soft_adjacency, soft_adjacency.transpose(-1, -2)):
        raise ValueError("soft adjacency must be symmetric")
    if not torch.equal(soft_adjacency.diagonal(dim1=-2, dim2=-1),
                       original_adjacency.diagonal(dim1=-2, dim2=-1).to(soft_adjacency)):
        raise ValueError("soft adjacency must preserve original loops")
    if not hasattr(proxy, "model"):
        raise TypeError("proxy must expose its frozen model (use DenseProxy)")
    freeze(proxy.model)
    logits = proxy(x, soft_adjacency)
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("Non-finite proxy logits")
    targets = torch.as_tensor(target, device=logits.device)
    if targets.dtype == torch.bool or targets.is_floating_point() or targets.is_complex():
        raise ValueError("target must contain integer class indices")
    if targets.ndim == 0:
        targets = targets.expand(logits.size(0))
    if targets.shape != (logits.size(0),):
        raise ValueError("target must have one class per graph")
    if bool(((targets < 0) | (targets >= logits.size(-1))).any()):
        raise ValueError("target contains an invalid class index")
    rows, cols = torch.triu_indices(soft_adjacency.size(-1), soft_adjacency.size(-1),
                                   offset=1, device=soft_adjacency.device)
    per_graph = (soft_adjacency[:, rows, cols]
                 - original_adjacency[:, rows, cols].to(soft_adjacency)).abs().sum(-1)
    return {"cf": F.cross_entropy(logits, targets.long(), reduction="mean"),
            "dist": per_graph.mean(), "per_graph_distance": per_graph,
            "proxy_logits": logits}
