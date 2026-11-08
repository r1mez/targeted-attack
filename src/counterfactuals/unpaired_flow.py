"""Node-aligned, class-conditional CFM and differentiable two-leg transport."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ClassFlow(nn.Module):
    def __init__(self, latent, hidden, heads, feedforward, layers, classes):
        super().__init__()
        for name, value in (("latent", latent), ("hidden", hidden), ("heads", heads),
                            ("feedforward", feedforward), ("layers", layers),
                            ("classes", classes)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.classes = classes
        self.input = nn.Linear(latent + classes + 1, hidden)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                hidden, heads, feedforward, dropout=0, activation="gelu",
                batch_first=True,
            )
            for _ in range(layers)
        )
        self.output = nn.Linear(hidden, latent)

    def forward(self, z, tau, labels):
        condition = F.one_hot(labels, self.classes).to(dtype=z.dtype)
        condition = torch.cat((tau.reshape(-1, 1), condition), dim=-1)
        condition = condition.unsqueeze(1).expand(-1, z.size(1), -1)
        state = self.input(torch.cat((z, condition), dim=-1))
        for block in self.blocks:
            state = block(state)
        return self.output(state)


def _labels(values, count, z, classes, name):
    labels = torch.as_tensor(values, device=z.device)
    if labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex():
        raise ValueError(f"{name} must contain integer class indices")
    if labels.ndim == 0:
        labels = labels.expand(count)
    if labels.shape != (count,):
        raise ValueError(f"{name} must be scalar or have shape [batch]")
    labels = labels.long()
    if bool(((labels < 0) | (labels >= classes)).any()):
        raise ValueError(f"{name} contains an invalid class index")
    return labels


def matching_loss(flow, z1, z0, tau, labels):
    """Independent real-graph/noise CFM objective, averaged over all entries."""
    if z1.ndim != 3 or z0.shape != z1.shape or not z1.is_floating_point():
        raise ValueError("z1 and z0 must have the same [batch, nodes, latent] shape")
    if tau.shape != (z1.size(0),) or not bool(torch.isfinite(tau).all()):
        raise ValueError("tau must be a finite [batch] vector")
    if bool(((tau < 0) | (tau > 1)).any()):
        raise ValueError("CFM interpolation times must be in [0, 1]")
    labels = _labels(labels, z1.size(0), z1, flow.classes, "labels")
    time = tau.reshape(-1, 1, 1)
    z = (1 - time) * z0 + time * z1
    return F.mse_loss(flow(z, tau, labels), z1 - z0)


def rk4(flow, z, labels, start, end, steps, multiplier, *, check_time=None):
    """Integrate a batch without detaching any RK4 stage.

    ``multiplier`` is either scalar or has one entry per graph. Parameters and
    input latents may require gradients; the complete integration is retained.
    """
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if z.ndim != 3 or not z.is_floating_point() or z.size(0) == 0:
        raise ValueError("z must be a nonempty float [batch, nodes, latent] tensor")
    if not math.isfinite(float(start)) or not math.isfinite(float(end)):
        raise ValueError("integration endpoints must be finite")
    labels = _labels(labels, z.size(0), z, flow.classes, "labels")
    scale = torch.as_tensor(multiplier, dtype=z.dtype, device=z.device)
    if scale.ndim == 0:
        scale = scale.expand(z.size(0))
    if scale.numel() != z.size(0) or scale.ndim not in (1, 3):
        raise ValueError("multiplier must be scalar or one value per graph")
    scale = scale.reshape(-1, 1, 1)
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise ValueError("multipliers must be finite and strictly positive")
    dt = (end - start) / steps

    def velocity(state, time):
        if check_time is not None:
            check_time()
        return scale * flow(state, state.new_full((state.size(0),), time), labels)

    for index in range(steps):
        time = start + index * dt
        k1 = velocity(z, time)
        k2 = velocity(z + (dt / 2) * k1, time + dt / 2)
        k3 = velocity(z + (dt / 2) * k2, time + dt / 2)
        k4 = velocity(z + dt * k3, time + dt)
        z = z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    if not bool(torch.isfinite(z).all()):
        raise RuntimeError("Non-finite integrated latent")
    return z


def leap(flow, z, source, target, multipliers, steps, *, check_time=None):
    """Lift under source labels (1→0), then land under target labels (0→1).

    A batch ``z[B,N,D]`` accepts scalar or batch source/target/multipliers.
    Historical single-graph ``z[N,D]`` accepts a multiplier vector and returns
    one transported latent per multiplier. The two legs share ``flow`` and
    remain in one autograd graph.
    """
    if z.ndim == 2:
        scale = torch.as_tensor(multipliers, dtype=z.dtype, device=z.device)
        if scale.ndim == 0:
            scale = scale.reshape(1)
        if scale.ndim != 1 or not scale.numel():
            raise ValueError("single-graph multipliers must be a nonempty vector")
        batch = z.unsqueeze(0).expand(scale.numel(), -1, -1)
    elif z.ndim == 3:
        batch, scale = z, multipliers
    else:
        raise ValueError("z must have shape [nodes, latent] or [batch, nodes, latent]")
    source = _labels(source, batch.size(0), batch, flow.classes, "source")
    target = _labels(target, batch.size(0), batch, flow.classes, "target")
    lifted = rk4(flow, batch, source, 1.0, 0.0, steps, scale, check_time=check_time)
    return rk4(flow, lifted, target, 0.0, 1.0, steps, scale, check_time=check_time)
