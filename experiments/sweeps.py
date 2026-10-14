"""
Named sweeps for run_sweep.py. A sweep is an ordered list of Jobs; earlier
jobs run first, so put the results you need most at the top. Within a block,
jobs are ordered seed-major (every config's seed 0, then seed 1, ...) so a
sweep cut short still has complete comparisons at fewer seeds.

`est_minutes` is the wall-clock time measured for one job running alone on an
RTX 3060 Laptop GPU (CUDA graphs on), used only for --dry-run planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class Job:
    run_id: str
    group: str
    script: str
    args: tuple = field(default_factory=tuple)
    est_minutes: float = 10.0


def _flags(**kw) -> tuple:
    out = []
    for k, v in kw.items():
        out += [f"--{k.replace('_', '-')}", str(v)]
    return tuple(out)


def _block(prefix, script, configs, est_minutes, seeds=SEEDS):
    """configs: {label: flag-dict}. One job per (seed, config), seed-major."""
    return [Job(run_id=f"{prefix}_{label}_s{seed}", group=f"{prefix}_{label}", script=script,
                args=_flags(seed=seed, **flags), est_minutes=est_minutes)
            for seed in seeds for label, flags in configs.items()]


# --- permuted MNIST: plasticity loss + all mitigations (cheap) ---------------
PMNIST = {
    "none": dict(mitigation="none"),
    "l2": dict(mitigation="l2"),
    "shrink_perturb": dict(mitigation="shrink_perturb"),
    "redo": dict(mitigation="redo"),
    # Re-initialising everything at each task boundary = a fresh network per
    # task: the "no plasticity loss" reference line, not a practical method.
    "fresh": dict(mitigation="reset", reset_layers=0),
}

# --- MinAtar Breakout with channel-permutation task shifts -------------------
SHIFT = 200_000      # env steps per task
SHIFT_STEPS = 2_000_000  # -> 10 tasks
DQN_SHIFT = {
    "none": dict(mitigation="none"),
    "fresh": dict(mitigation="reset", reset_layers=0, mit_every=SHIFT),
}
DQN_SHIFT_MITIGATIONS = {
    "l2": dict(mitigation="l2"),
    "shrink_perturb": dict(mitigation="shrink_perturb", mit_every=50_000),
    "redo": dict(mitigation="redo", mit_every=25_000),
}


def overnight():
    jobs = []
    # 1. Supervised de-risk: does plasticity loss show up, and what fixes it?
    jobs += _block("pmnist", "permuted_mnist.py", PMNIST, est_minutes=3.5)
    # 2. RL collapse under task shifts at the standard replay ratio.
    shift = dict(game="breakout", steps=SHIFT_STEPS, shift_every=SHIFT, replay_ratio=1)
    jobs += _block("breakout_shift_rr1", "dqn_minatar.py",
                   {k: {**shift, **v} for k, v in DQN_SHIFT.items()}, est_minutes=18)
    # 3. RL primacy bias: same budget of env steps, 4x the updates.
    jobs += _block("breakout_rr", "dqn_minatar.py",
                   {f"{rr}": dict(game="breakout", steps=1_000_000, replay_ratio=rr) for rr in (1, 4)},
                   est_minutes=18.5)  # ~9 min at RR 1, ~28 min at RR 4
    # 4. Mitigations in the task-shift setting (fills the rest of the night).
    jobs += _block("breakout_shift_rr1", "dqn_minatar.py",
                   {k: {**shift, **v} for k, v in DQN_SHIFT_MITIGATIONS.items()}, est_minutes=18)
    return jobs


# --- collapse pilot: which RL stressors actually cause plasticity loss? -------
CYCLE_GAMES = "breakout,space_invaders,asterix"
SWITCH = 200_000
CYCLE = dict(game=CYCLE_GAMES, switch_every=SWITCH, steps=9 * SWITCH)  # 3 passes over 3 games
SMALL = dict(conv_channels=4, hidden=32)  # 4x narrower than the reference network
FRESH = dict(mitigation="reset", reset_layers=0, mit_every=SWITCH)


def collapse_pilot():
    """2 seeds per probe; compare against the 'fresh' references and the overnight
    breakout_rr_1 / breakout_rr_4 runs."""
    configs = {
        # (label, flags, est. minutes), highest priority first
        "cycle3_none": (dict(**CYCLE), 18),
        "cycle3_fresh": (dict(**CYCLE, **FRESH), 18),
        "cycle3_small_none": (dict(**CYCLE, **SMALL), 15),
        "cycle3_small_fresh": (dict(**CYCLE, **SMALL, **FRESH), 15),
        "breakout_rr_8": (dict(game="breakout", steps=1_000_000, replay_ratio=8), 45),
        "breakout_rr_16": (dict(game="breakout", steps=500_000, replay_ratio=16), 45),
        "cycle3_rr4_none": (dict(**CYCLE, replay_ratio=4), 55),
    }
    return [Job(run_id=f"{label}_s{seed}", group=label, script="dqn_minatar.py",
                args=_flags(seed=seed, **flags), est_minutes=est)
            for seed in (0, 1) for label, (flags, est) in configs.items()]


# --- k-sweep: does learning ability degrade with the AMOUNT of prior training? --
# Target: learn Asterix for one 200k segment. Before it: k segments alternating
# Breakout / Space Invaders, always ending on Space Invaders (so the game right
# before Asterix is the same for every k > 0). Asterix never appears among the
# priors, so revisit memory can't help. k = 0 is the fresh-network reference.
# Plasticity loss predicts Asterix learning gets steadily worse as k grows;
# one-off negative transfer predicts a drop at k = 1 that then flattens.
K_PRIORS = ["breakout", "space_invaders"] * 4


def k_schedule(k: int) -> dict:
    games = K_PRIORS[len(K_PRIORS) - k:] + ["asterix"] if k else ["asterix"]
    flags = dict(game=",".join(games), steps=(k + 1) * SWITCH, obs_channels=6)  # 6 = Space Invaders
    if k:
        flags["switch_every"] = SWITCH
    return flags


def k_sweep():
    seg_min = {1: 1.5, 4: 5.6}  # measured minutes per 200k segment at replay ratio 1 / 4
    arms = [  # (label, replay ratio, ks, extra flags), highest priority first
        ("rr1_none", 1, (0, 1, 2, 4, 8), {}),
        ("rr1_l2", 1, (8,), dict(mitigation="l2")),
        ("rr1_shrink_perturb", 1, (8,), dict(mitigation="shrink_perturb", mit_every=50_000)),
        ("rr1_redo", 1, (8,), dict(mitigation="redo", mit_every=25_000)),
        ("rr4_none", 4, (0, 2, 8), {}),
    ]
    return [Job(run_id=f"kprior_{label}_k{k}_s{seed}", group=f"kprior_{label}_k{k}", script="dqn_minatar.py",
                args=_flags(seed=seed, replay_ratio=rr, **k_schedule(k), **extra),
                est_minutes=(k + 1) * seg_min[rr])
            for label, rr, ks, extra in arms for seed in SEEDS for k in ks]


def smoke():
    """~3 minute end-to-end check of every script and mitigation."""
    jobs = [Job(f"smoke_pmnist_{m}", f"smoke_pmnist_{m}", "permuted_mnist.py",
                _flags(mitigation=m, tasks=3, train_per_task=2000, probe_every=1), 0.5)
            for m in ("none", "redo", "reset")]
    jobs += [Job(f"smoke_dqn_{m}", f"smoke_dqn_{m}", "dqn_minatar.py",
                 _flags(mitigation=m, steps=20_000, learn_start=1000, shift_every=7000, replay_ratio=2,
                        mit_every=5000, log_every=5000, eval_every=10_000, probe_every=10_000), 1)
             for m in ("none", "l2", "reset", "shrink_perturb", "redo")]
    return jobs


SWEEPS = {"overnight": overnight, "collapse_pilot": collapse_pilot, "k_sweep": k_sweep, "smoke": smoke}
