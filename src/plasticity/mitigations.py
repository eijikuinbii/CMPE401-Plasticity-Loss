"""
Plasticity-loss mitigations, applied in place to a model from `networks.py`.

- reset          : re-initialise the last N layers (Nikishin et al., "The Primacy
                   Bias in Deep RL", ICML 2022).
- shrink_perturb : interpolate weights toward a fresh initialisation,
                   theta <- alpha*theta + (1-alpha)*theta_fresh (Ash & Adams, 2020;
                   interpolation form as in BBF, Schwarzer et al., 2023).
- redo           : re-initialise dormant units -- fresh incoming weights, zeroed
                   outgoing weights (Sokar et al., ICML 2023).
- l2             : not here; it is the optimiser's `weight_decay`.

Each function also clears the Adam moments of the weights it touched, so stale
optimiser statistics don't immediately undo the intervention. Everything is
modified in place, so the functions are safe to use with CUDA-graph-captured
training steps (see experiments/dqn_minatar.py).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .diagnostics import hidden_activations, unit_scores

MITIGATIONS = ("none", "l2", "reset", "shrink_perturb", "redo")

_WEIGHTED = (nn.Linear, nn.Conv2d)


def _weighted_layers(model: nn.Sequential) -> list[nn.Module]:
    return [m for m in model if isinstance(m, _WEIGHTED)]


def _fresh_copy(layer: nn.Module) -> nn.Module:
    fresh = copy.deepcopy(layer)
    fresh.reset_parameters()
    return fresh


def _clear_state(optimizer: torch.optim.Optimizer | None, params) -> None:
    """Zero Adam's state (moments and step count) for params.

    Done in place rather than by deleting the state, because a CUDA-graph-captured
    optimizer step keeps pointing at the original state tensors.
    """
    if optimizer is None:
        return
    for p in params:
        for v in optimizer.state.get(p, {}).values():
            if torch.is_tensor(v):
                v.zero_()


@torch.no_grad()
def reset_layers(model: nn.Sequential, n_last: int, optimizer=None) -> None:
    """Re-initialise the last `n_last` weighted layers (all layers if n_last <= 0)."""
    layers = _weighted_layers(model)
    targets = layers if n_last <= 0 else layers[-n_last:]
    for layer in targets:
        layer.reset_parameters()
        _clear_state(optimizer, layer.parameters())


@torch.no_grad()
def shrink_perturb(model: nn.Sequential, alpha: float) -> None:
    """theta <- alpha * theta + (1 - alpha) * theta_fresh, for every layer."""
    for layer in _weighted_layers(model):
        fresh = _fresh_copy(layer)
        for p, q in zip(layer.parameters(), fresh.parameters()):
            p.mul_(alpha).add_(q, alpha=1 - alpha)


def _mask_moments(optimizer, param: torch.Tensor, index) -> None:
    """Zero Adam's moment estimates for param[index]."""
    if optimizer is None or param not in optimizer.state:
        return
    state = optimizer.state[param]
    for key in ("exp_avg", "exp_avg_sq"):
        if key in state:
            state[key][index] = 0.0


@torch.no_grad()
def redo(model: nn.Sequential, x: torch.Tensor, tau: float, optimizer=None) -> int:
    """Recycle dormant units (score <= tau on batch x). Returns how many were reset."""
    acts = hidden_activations(model, x)
    modules = list(model)
    relu_idx = [i for i, m in enumerate(modules) if isinstance(m, nn.ReLU)]
    n_reset = 0
    for i, act in zip(relu_idx, acts):
        mask = unit_scores(act) <= tau
        if not mask.any():
            continue
        n_reset += int(mask.sum())

        # Incoming layer: nearest weighted layer before the ReLU.
        inc = next(m for m in reversed(modules[:i]) if isinstance(m, _WEIGHTED))
        fresh = _fresh_copy(inc)
        inc.weight[mask] = fresh.weight[mask]
        inc.bias[mask] = fresh.bias[mask]
        _mask_moments(optimizer, inc.weight, mask)
        _mask_moments(optimizer, inc.bias, mask)

        # Outgoing layer: nearest weighted layer after the ReLU. A conv channel
        # feeding a Linear through Flatten owns a contiguous block of H*W columns.
        out = next(m for m in modules[i + 1:] if isinstance(m, _WEIGHTED))
        if isinstance(inc, nn.Conv2d) and isinstance(out, nn.Linear):
            col_mask = mask.repeat_interleave(out.in_features // mask.numel())
        else:
            col_mask = mask
        out.weight[:, col_mask] = 0.0
        _mask_moments(optimizer, out.weight, (slice(None), col_mask))
    return n_reset
