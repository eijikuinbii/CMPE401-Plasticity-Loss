"""
Plasticity diagnostics: internal signals that may predict loss of plasticity.

All functions take a model built as in `networks.py` (an nn.Sequential with a
distinct nn.ReLU after each hidden layer) and a batch of inputs drawn from the
data the model is currently training on.

References
----------
- Dormant units: Sokar et al., "The Dormant Neuron Phenomenon in Deep RL", ICML 2023.
- Effective rank: Roy & Vetterli, 2007 (as used by Dohare et al., Nature 2024).
- srank: Kumar et al., "Implicit Under-Parameterization", ICLR 2021.
- Plasticity probe: Lyle et al., "Understanding and Preventing Capacity Loss in RL", ICLR 2022.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


@torch.no_grad()
def hidden_activations(model: nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    """Post-ReLU activations of every hidden layer, in forward order."""
    acts: list[torch.Tensor] = []
    hooks = [
        m.register_forward_hook(lambda _m, _i, out: acts.append(out))
        for m in model.modules()
        if isinstance(m, nn.ReLU)
    ]
    try:
        model(x)
    finally:
        for h in hooks:
            h.remove()
    return acts


def unit_scores(act: torch.Tensor) -> torch.Tensor:
    """Per-unit dormancy score s_i = E|h_i| / mean_j E|h_j| (Sokar et al., 2023).

    Expectation is over the batch and, for conv layers, over spatial positions,
    so each conv channel counts as one unit.
    """
    dims = [0] + list(range(2, act.dim()))
    s = act.abs().mean(dim=dims)
    return s / (s.mean() + 1e-8)


def dormant_fraction(acts: list[torch.Tensor], tau: float) -> float:
    """Fraction of hidden units (over all layers) whose score is <= tau."""
    total = sum(a.shape[1] for a in acts)
    dormant = sum(int((unit_scores(a) <= tau).sum()) for a in acts)
    return dormant / total


@torch.no_grad()
def effective_rank(features: torch.Tensor) -> tuple[float, float]:
    """(effective rank, srank) of a (batch, dim) feature matrix.

    effective rank = exp(entropy of normalised singular values) (Roy & Vetterli).
    srank = smallest k whose top-k singular values hold 99% of the mass (Kumar et al.).
    """
    sv = torch.linalg.svdvals(features.float())
    p = sv / sv.sum().clamp_min(1e-12)
    erank = torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum())
    srank = int((torch.cumsum(p, 0) < 0.99).sum()) + 1
    return float(erank), float(srank)


@torch.no_grad()
def weight_norm(model: nn.Module) -> float:
    return float(torch.sqrt(sum(p.pow(2).sum() for p in model.parameters())))


@torch.no_grad()
def grad_norm(model: nn.Module) -> float:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.sqrt(sum(g.pow(2).sum() for g in grads)))


def plasticity_probe(model: nn.Module, x: torch.Tensor, steps: int = 200,
                     lr: float = 1e-3, seed: int = 0) -> float:
    """Train a *copy* of the model to regress fixed random targets; return final MSE.

    The targets are i.i.d. N(0, 1) with a fixed seed, so the probe task is the
    same at every point in training -- only the starting weights change. A
    network that has lost plasticity fits them more slowly (higher loss).
    """
    probe = copy.deepcopy(model)
    probe.train()
    with torch.no_grad():
        out_dim = probe(x[:1]).shape[1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    targets = torch.randn(x.shape[0], out_dim, generator=g).to(x.device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(steps):
        loss = nn.functional.mse_loss(probe(x), targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(nn.functional.mse_loss(probe(x), targets))


def compute_diagnostics(model: nn.Module, x: torch.Tensor, tau: float = 0.1) -> dict[str, float]:
    """All cheap representation diagnostics on one batch (no probe, no grads)."""
    was_training = model.training
    model.eval()
    acts = hidden_activations(model, x)
    model.train(was_training)
    erank, srank = effective_rank(acts[-1].flatten(1))
    return {
        "dormant_fraction": dormant_fraction(acts, tau),
        "dead_fraction": dormant_fraction(acts, 0.0),
        "effective_rank": erank,
        "srank": srank,
        "weight_l2_norm": weight_norm(model),
    }
