"""B-AE pretraining and single-stage soft counterfactual flow training.

The frozen AE supplies the latent coordinates. A freshly initialized ClassFlow
optimizes CFM + lambda_cf*CE + lambda_dist*soft-L1 from its first update.
Callers supply approved settings and explicit target policy; no flow checkpoint
is loaded for training. Model execution remains subject to experiment approval.
"""

import copy
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from src.counterfactuals.graph_ae import StructuralAutoencoder
from src.counterfactuals.soft_counterfactual import (
    SCHEMA_VERSION, counterfactual_loss, fixed_auxiliary, freeze, generate_soft,
    normalization_hash, validate_graph_batch, validate_normalization,
)
from src.counterfactuals.unpaired_flow import ClassFlow, matching_loss
from src.models.ba3motif_model_access import state_hash


def _positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite_number(value, name, *, positive=False):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or (value <= 0 if positive else value < 0)):
        qualifier = "strictly positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")


def _validate_settings(epochs, batch_size, lr, seed, log, check_time):
    _positive_integer(epochs, "epochs")
    _positive_integer(batch_size, "batch_size")
    _finite_number(lr, "lr", positive=True)
    if (not isinstance(seed, int) or isinstance(seed, bool)
            or seed < 0 or seed >= 2 ** 63 - 2001):
        raise ValueError("seed must be an integer in [0, 2**63-2001)")
    if not callable(log) or not callable(check_time):
        raise TypeError("log and check_time must be callable")


def _validate_data(data, classes=None, *, targets=False):
    for key in ("ids", "x", "adjacency"):
        if key not in data:
            raise ValueError(f"Missing graph data field {key}")
    validate_graph_batch(data["x"], data["adjacency"], data["ids"])
    count = data["x"].size(0)
    if classes is None:
        return
    for key in (("classes", "targets") if targets else ("classes",)):
        if key not in data:
            raise ValueError(f"Missing graph data field {key}")
        values = torch.as_tensor(data[key])
        if (values.shape != (count,) or values.dtype == torch.bool
                or values.is_floating_point() or values.is_complex()
                or bool(((values < 0) | (values >= classes)).any())):
            raise ValueError(f"{key} must be one integer in [0, classes) per graph")
    if targets and bool((data["classes"] == data["targets"]).any()):
        raise ValueError("Counterfactual targets must differ from original predictions")
    if "probabilities" in data:
        probabilities = torch.as_tensor(data["probabilities"])
        if (probabilities.shape != (count, classes)
                or not bool(torch.isfinite(probabilities).all())
                or bool(((probabilities < 0) | (probabilities > 1)).any())
                or not torch.allclose(probabilities.sum(-1), probabilities.new_ones(count),
                                      atol=1e-5, rtol=1e-5)):
            raise ValueError("Original target probabilities must be a normalized [batch, classes] tensor")
        if not torch.equal(probabilities.argmax(-1).cpu(), data["classes"].cpu()):
            raise ValueError("classes must remain the original target-GNN predictions")


def _compatible_splits(train, validation):
    if train["x"].device.type != "cuda":
        raise ValueError("Model training requires a local CUDA GPU; CPU fallback is not allowed")
    if (train["x"].shape[1:] != validation["x"].shape[1:]
            or train["x"].device != validation["x"].device
            or train["x"].dtype != validation["x"].dtype):
        raise ValueError("Training and validation graph dimensions, device, and dtype must match")


def _subset(data, indices):
    """Keep every aligned field, including original target probabilities."""
    result = {}
    count = len(data["ids"])
    positions = indices.cpu().tolist()
    for name, value in data.items():
        if isinstance(value, torch.Tensor):
            if value.ndim == 0 or value.size(0) != count:
                raise ValueError(f"Data field {name} is not aligned to graph IDs")
            result[name] = value[indices.to(value.device)]
        elif isinstance(value, (list, tuple)) and len(value) == count:
            result[name] = type(value)(value[index] for index in positions)
        else:
            raise ValueError(f"Data field {name} is not an aligned tensor/list")
    return result


def expand_target_tasks(data, policy, *, num_classes=None):
    """Expand ORIGINAL graphs into explicitly requested all-other-class tasks.

    The source classes and every other field are copied with the same indices.
    No source/target graph pairing is constructed. A probability matrix supplies
    the class count, or callers must pass it explicitly for a tensor-only input.
    """
    if policy != "all_other":
        raise ValueError("An explicitly approved target_policy='all_other' is required")
    if "targets" in data:
        raise ValueError("Input already contains targets; refusing to expand tasks twice")
    if num_classes is None:
        probabilities = data.get("probabilities")
        if not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 2:
            raise ValueError("num_classes is required when no original probability matrix is available")
        num_classes = probabilities.size(1)
    _positive_integer(num_classes, "num_classes")
    if num_classes < 2:
        raise ValueError("Counterfactual tasks require at least two classes")
    _validate_data(data, num_classes)
    source = data["classes"].cpu()
    indices, targets = [], []
    for index, original in enumerate(source.tolist()):
        for target in range(num_classes):
            if target != original:
                indices.append(index)
                targets.append(target)
    result = _subset(data, torch.tensor(indices, dtype=torch.long))
    result["targets"] = torch.tensor(targets, dtype=torch.long, device=data["classes"].device)
    return result


def batches(size, batch_size, generator=None):
    _positive_integer(size, "size")
    _positive_integer(batch_size, "batch_size")
    indices = torch.arange(size) if generator is None else torch.randperm(size, generator=generator)
    return indices.split(batch_size)


def _directory(directory, names):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (directory / name).exists():
            raise FileExistsError(f"Refusing to overwrite an existing stage checkpoint: {directory / name}")
    return directory


def _sample(shape, generator, reference, *, uniform=False):
    draw = torch.rand(shape, generator=generator) if uniform else torch.randn(shape, generator=generator)
    return draw.to(device=reference.device, dtype=reference.dtype)


def _rng_state(generator, validation_generator=None):
    state = {"torch_cpu": torch.get_rng_state(), "training_generator": generator.get_state()}
    if validation_generator is not None:
        state["validation_generator"] = validation_generator.get_state()
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        state["torch_cuda"] = {"device": device, "state": torch.cuda.get_rng_state(device)}
    return state


def _step(optimizer, loss):
    if loss.ndim or not bool(torch.isfinite(loss)):
        raise RuntimeError("Training objective must be a finite scalar")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError("Non-finite flow/autoencoder parameter gradient")
    optimizer.step()


def _provenance(provenance, *, ae_seed, ae_architecture=None):
    result = copy.deepcopy(provenance or {})
    if "fixed_auxiliary_seed" in result and result["fixed_auxiliary_seed"] != ae_seed:
        raise ValueError("Provenance fixed_auxiliary_seed differs from ae_seed")
    if "ae" in result.get("seeds", {}) and result["seeds"]["ae"] != ae_seed:
        raise ValueError("Provenance AE seed differs from fixed auxiliary seed")
    result["fixed_auxiliary_seed"] = ae_seed
    result.setdefault("seeds", {})["ae"] = ae_seed
    if ae_architecture is not None:
        if result.get("ae_architecture", ae_architecture) != ae_architecture:
            raise ValueError("Provenance AE architecture mismatch")
        result["ae_architecture"] = ae_architecture
    return result


def _assert_frozen(module, initial_hash, name):
    if (module.training or state_hash(module) != initial_hash
            or any(parameter.requires_grad or parameter.grad is not None for parameter in module.parameters())):
        raise RuntimeError(f"Frozen {name} parameters or state changed during training")


def _ae_architecture(ae):
    return {"features": ae.input.in_features - ae.auxiliary - 1,
            "auxiliary": ae.auxiliary, "hidden": ae.input.out_features,
            "latent": ae.project.out_features, "layers": len(ae.updates),
            "edge_hidden": ae.edge[0].out_features}


@torch.no_grad()
def encode_data(ae, data, *, ae_seed, batch_size, check_time):
    """Encode ORIGINAL graphs with fixed per-graph auxiliary features."""
    _validate_data(data)
    freeze(ae)
    auxiliary = fixed_auxiliary(data["ids"], data["x"].size(1), ae.auxiliary, ae_seed).to(data["x"])
    latent = []
    for index in batches(len(data["ids"]), batch_size):
        check_time()
        block = _subset(data, index)
        latent.append(ae.encode(block["x"], block["adjacency"], auxiliary[index.to(auxiliary.device)]))
    result = torch.cat(latent)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("Non-finite original graph encoding")
    return result


def train_autoencoder(train, validation, *, architecture, epochs, batch_size, lr, seed,
                      ae_seed, directory, log, check_time, provenance=None):
    """Train only B with independent upper-pair BCE; select by soft validation BCE.

    No proxy objective or hard reconstruction gate is used. Historical B model
    weights remain untouched. ``autoencoder.pt`` is the best soft-BCE checkpoint;
    ``autoencoder-last.pt`` also saves optimizer and RNG at the last full epoch.
    """
    _validate_settings(epochs, batch_size, lr, seed, log, check_time)
    _validate_data(train)
    _validate_data(validation)
    _compatible_splits(train, validation)
    provenance = _provenance(provenance, ae_seed=ae_seed, ae_architecture=architecture)
    directory = _directory(directory, ("autoencoder.pt", "autoencoder-last.pt"))
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1000)
    ae = StructuralAutoencoder(**architecture).to(train["x"].device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    rows, cols = torch.triu_indices(train["x"].size(1), train["x"].size(1), offset=1,
                                   device=train["x"].device)
    val_aux = fixed_auxiliary(validation["ids"], validation["x"].size(1), ae.auxiliary, ae_seed).to(validation["x"])
    best_loss, best_epoch, best_state = float("inf"), None, None
    for epoch in range(1, epochs + 1):
        ae.train()
        train_total = 0.0
        for index in batches(len(train["ids"]), batch_size, generator):
            check_time()
            block = _subset(train, index)
            # Preserve B's original auxiliary augmentation during AE pretraining.
            # Joint flow training uses fixed seed+graph_id coordinates after AE freezing.
            auxiliary = _sample((*block["x"].shape[:2], ae.auxiliary), generator, block["x"])
            logits = ae.edge_logits(ae.encode(block["x"], block["adjacency"], auxiliary), rows, cols)
            loss = F.binary_cross_entropy_with_logits(logits, block["adjacency"][:, rows, cols].to(logits))
            _step(optimizer, loss)
            train_total += float(loss.detach()) * len(index)
        ae.eval()
        val_total = 0.0
        with torch.no_grad():
            for index in batches(len(validation["ids"]), batch_size):
                check_time()
                block = _subset(validation, index)
                logits = ae.edge_logits(ae.encode(block["x"], block["adjacency"],
                                                  val_aux[index.to(val_aux.device)]), rows, cols)
                loss = F.binary_cross_entropy_with_logits(logits, block["adjacency"][:, rows, cols].to(logits))
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Non-finite validation reconstruction BCE")
                val_total += float(loss) * len(index)
        val_loss = val_total / len(validation["ids"])
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(ae.state_dict())
        payload = {"schema_version": SCHEMA_VERSION, "stage": "ae_pretrain", "group": "B",
                   "architecture": dict(architecture), "state_dict": ae.state_dict(), "epoch": epoch,
                   "selection": {"criterion": "soft_bce", "epoch": best_epoch, "value": best_loss},
                   "provenance": provenance, "optimizer": optimizer.state_dict(),
                   "rng": _rng_state(generator), "seeds": {"training": seed, "ae": ae_seed},
                   "training": {"epochs": epochs, "batch_size": batch_size, "lr": lr,
                                "auxiliary_training": "resampled Gaussian per batch",
                                "auxiliary_validation_seed": ae_seed}}
        torch.save(payload, directory / "autoencoder-last.pt")
        if best_epoch == epoch:
            torch.save(payload, directory / "autoencoder.pt")
        log({"stage": "ae_pretrain", "epoch": epoch, "bce": train_total / len(train["ids"]),
             "validation_bce": val_loss, "selected_epoch": best_epoch})
    ae.load_state_dict(best_state, strict=True)
    return freeze(ae), {"selection": {"criterion": "soft_bce", "epoch": best_epoch, "value": best_loss},
                        "checkpoint": str(directory / "autoencoder.pt"), "stage": "ae_pretrain"}


def _save_flow(path, flow, mean, std, optimizer, epoch, *, stage, architecture,
               generator, validation_generator, provenance, seeds, settings, initialization):
    torch.save({"schema_version": SCHEMA_VERSION, "stage": stage,
                "architecture": architecture, "state_dict": flow.state_dict(),
                "mean": mean.detach().cpu().clone(), "std": std.detach().cpu().clone(),
                "normalization": normalization_hash(mean, std), "epoch": epoch,
                "optimizer": optimizer.state_dict(), "rng": _rng_state(generator, validation_generator),
                "provenance": provenance, "seeds": seeds, "training": settings,
                "initialization": initialization}, path)


def _task_loss(ae, flow, mean, std, proxy, block, real_z, noise, time, *,
               ae_seed, steps, multiplier, lambda_cf, lambda_dist, check_time):
    generated = generate_soft(ae, flow, mean, std, block["x"], block["adjacency"], block["ids"],
                              block["classes"], block["targets"], ae_seed=ae_seed,
                              steps=steps, multiplier=multiplier, check_time=check_time)
    losses = counterfactual_loss(proxy, block["x"], generated["adjacency"],
                                 block["targets"], block["adjacency"])
    # This branch uses REAL original-graph latents and ORIGINAL predicted classes.
    # Noise/time are sampled independently of the deterministic CF transport.
    cfm = matching_loss(flow, real_z, noise, time, block["classes"].to(real_z.device))
    total = cfm + lambda_cf * losses["cf"] + lambda_dist * losses["dist"]
    hit = (losses["proxy_logits"].argmax(-1) == block["targets"].to(losses["proxy_logits"].device)).sum()
    return total, {"loss": total, "cfm": cfm, "cf": losses["cf"], "dist": losses["dist"],
                   "proxy_target_hits": hit}


def train_flow(ae, proxy, train, validation, *, architecture, target_policy,
               epochs, batch_size, lr, seed, ae_seed, std_floor, steps, multiplier,
               lambda_cf, lambda_dist, directory, log, check_time, provenance=None):
    """Train a randomly initialized flow with all three losses in every update.

    Inputs are ORIGINAL graphs, before target expansion. Fit normalization once
    from their frozen AE training latents, then keep it fixed. The AE and proxy
    parameters stay frozen while both lift and land retain their input gradients.
    """
    _validate_settings(epochs, batch_size, lr, seed, log, check_time)
    _positive_integer(steps, "steps")
    _finite_number(std_floor, "std_floor", positive=True)
    _finite_number(multiplier, "multiplier", positive=True)
    _finite_number(lambda_cf, "lambda_cf")
    _finite_number(lambda_dist, "lambda_dist")
    _validate_data(train, architecture["classes"])
    _validate_data(validation, architecture["classes"])
    _compatible_splits(train, validation)
    for data in (train, validation):
        ids = data["ids"].cpu().tolist()
        if "targets" in data or len(set(ids)) != len(ids):
            raise ValueError("Flow training requires unique ORIGINAL graphs before target expansion")
    if architecture["latent"] != ae.project.out_features:
        raise ValueError("Flow and AE latent dimensions differ")
    if not hasattr(proxy, "model"):
        raise TypeError("proxy must expose its frozen model (use DenseProxy)")
    provenance = _provenance(provenance, ae_seed=ae_seed, ae_architecture=_ae_architecture(ae))
    artifact = provenance.get("inputs", {}).get("ae_checkpoint")
    if not isinstance(artifact, dict) or not artifact.get("sha256"):
        raise ValueError("Flow training requires the frozen AE checkpoint SHA-256")
    provenance["normalization_training_ids"] = train["ids"].cpu().tolist()
    provenance["validation_ids"] = validation["ids"].cpu().tolist()
    provenance["normalization_source"] = "frozen_ae_original_training_latents"
    provenance["target_policy"] = target_policy
    # Expansion supplies target CLASSES, never paired target graphs.
    task_train = expand_target_tasks(train, target_policy, num_classes=architecture["classes"])
    task_validation = expand_target_tasks(validation, target_policy, num_classes=architecture["classes"])
    directory = _directory(directory, ("flow.pt",))
    freeze(ae.to(train["x"].device))
    freeze(proxy.model.to(train["x"].device))
    initial_ae, initial_proxy = state_hash(ae), state_hash(proxy.model)
    provenance["ae_state_hash"] = initial_ae
    provenance["proxy_state_hash"] = initial_proxy
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1000)
    validation_generator = torch.Generator().manual_seed(seed + 2000)
    original_train_z = encode_data(ae, train, ae_seed=ae_seed, batch_size=batch_size,
                                   check_time=check_time)
    original_val_z = encode_data(ae, validation, ae_seed=ae_seed, batch_size=batch_size,
                                 check_time=check_time)
    # Only original TRAINING graphs contribute to these statistics.
    mean = original_train_z.mean(dim=(0, 1)).detach()
    std = original_train_z.std(dim=(0, 1), unbiased=False).clamp_min(std_floor).detach()
    validate_normalization(mean, std, architecture["latent"])
    normalization = normalization_hash(mean, std)
    train_positions = {int(graph_id): i for i, graph_id in enumerate(train["ids"])}
    val_positions = {int(graph_id): i for i, graph_id in enumerate(validation["ids"])}
    train_indices = torch.tensor([train_positions[int(i)] for i in task_train["ids"]],
                                 device=original_train_z.device)
    val_indices = torch.tensor([val_positions[int(i)] for i in task_validation["ids"]],
                               device=original_val_z.device)
    train_z = ((original_train_z - mean) / std)[train_indices]
    val_z = ((original_val_z - mean) / std)[val_indices]
    train, validation = task_train, task_validation
    val_noise = _sample(val_z.shape, validation_generator, val_z)
    val_time = _sample((len(val_z),), validation_generator, val_z, uniform=True)
    # This is the ONLY flow initialization; no prior flow weights are loaded.
    flow = ClassFlow(**architecture).to(train["x"].device)
    provenance["initial_flow_state_hash"] = state_hash(flow)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    settings = {"epochs": epochs, "batch_size": batch_size, "lr": lr,
                "std_floor": std_floor, "steps": steps, "multiplier": multiplier,
                "lambda_cf": lambda_cf, "lambda_dist": lambda_dist,
                "L_CFM_coefficient": 1, "L_boundary": 0, "target_policy": target_policy}
    final = None
    for epoch in range(1, epochs + 1):
        flow.train()
        totals = dict.fromkeys(("loss", "cfm", "cf", "dist", "proxy_target_hits"), 0.0)
        for index in batches(len(train_z), batch_size, generator):
            check_time()
            block = _subset(train, index)
            real_z = train_z[index.to(train_z.device)]
            noise = _sample(real_z.shape, generator, real_z)
            time = _sample((len(index),), generator, real_z, uniform=True)
            loss, values = _task_loss(ae, flow, mean, std, proxy, block, real_z, noise, time,
                                      ae_seed=ae_seed, steps=steps, multiplier=multiplier,
                                      lambda_cf=lambda_cf, lambda_dist=lambda_dist, check_time=check_time)
            _step(optimizer, loss)
            for name, value in values.items():
                totals[name] += float(value.detach()) * (1 if name == "proxy_target_hits" else len(index))
        flow.eval()
        val_totals = dict.fromkeys(totals, 0.0)
        with torch.no_grad():
            for index in batches(len(val_z), batch_size):
                check_time()
                positions = index.to(val_z.device)
                _, values = _task_loss(ae, flow, mean, std, proxy, _subset(validation, index),
                                       val_z[positions], val_noise[positions], val_time[positions],
                                       ae_seed=ae_seed, steps=steps, multiplier=multiplier,
                                       lambda_cf=lambda_cf, lambda_dist=lambda_dist, check_time=check_time)
                for name, value in values.items():
                    if not bool(torch.isfinite(value)):
                        raise RuntimeError(f"Non-finite validation {name}")
                    val_totals[name] += float(value) * (1 if name == "proxy_target_hits" else len(index))
        if normalization_hash(mean, std) != normalization:
            raise RuntimeError("Frozen normalization changed during flow training")
        _save_flow(directory / "flow.pt", flow, mean, std, optimizer, epoch,
                   stage="joint_flow", architecture=dict(architecture), generator=generator,
                   validation_generator=validation_generator, provenance=provenance,
                   seeds={"training": seed, "ae": ae_seed}, settings=settings,
                   initialization="random")
        final = {"stage": "joint_flow", "epoch": epoch,
                 **{name: value / len(train_z) for name, value in totals.items()},
                 **{f"validation_{name}": value / len(val_z) for name, value in val_totals.items()}}
        log(final)
    _assert_frozen(ae, initial_ae, "autoencoder")
    _assert_frozen(proxy.model, initial_proxy, "proxy")
    return freeze(flow), {"checkpoint": str(directory / "flow.pt"), "stage": "joint_flow",
                          "normalization": normalization, "final_metrics": final,
                          "initialization": "random", "training_tasks": len(train_z),
                          "validation_tasks": len(val_z)}
