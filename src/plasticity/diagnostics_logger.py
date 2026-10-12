"""
Experiment logging convention for the plasticity-loss project.

One run == one folder under `runs/<run_id>/`, containing:
  - meta.json    : the run's config + provenance (written once, at start)
  - metrics.csv  : tidy long-format metrics (appended throughout training)
  - status.json  : written by `close()`; its presence marks the run as finished
  - model.pt     : final weights (written by the training script)

metrics.csv schema (one row per measurement):
    run_id, step, metric, value

Using long format means every diagnostic -- dormant_fraction, effective_rank,
weight_l2_norm, grad_l2_norm, plasticity_probe_loss, and eval_reward -- shares
the same two columns. Metrics logged at different intervals coexist with no
schema change, and the file loads straight into pandas for seaborn CI plots:

    import pandas as pd, seaborn as sns
    df = pd.read_csv("runs/<run_id>/metrics.csv")
    sns.lineplot(data=df[df.metric == "dormant_fraction"], x="step", y="value")
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


# Canonical metric names. Log against these strings so every run is comparable.
METRICS = (
    "eval_reward",           # mean return over greedy eval episodes (RL)
    "train_return",          # mean return of training episodes finished since last log (RL)
    "task_accuracy",         # test accuracy on the current task after training on it (supervised)
    "train_accuracy",        # online accuracy on training batches during the task (supervised)
    "train_loss",            # mean training loss since last log
    "dormant_fraction",      # fraction of dormant units (score <= tau, Sokar et al. 2023)
    "dead_fraction",         # fraction of units with zero activation on the whole batch
    "effective_rank",        # effective rank of the penultimate features (Roy & Vetterli)
    "srank",                 # srank of the penultimate features (Kumar et al. 2021)
    "weight_l2_norm",        # global L2 norm of weights
    "grad_l2_norm",          # global L2 norm of gradients (mean over recent updates)
    "plasticity_probe_loss", # loss when fitting fresh targets (lower = more plastic)
    "units_recycled",        # units reset by ReDo since last log
    "steps_per_sec",         # throughput, for compute-cost comparisons
)


def _git_commit() -> str:
    """Best-effort short commit hash for provenance; empty string if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


def is_finished(root, run_id: str) -> bool:
    return (Path(root) / run_id / "status.json").exists()


class RunLogger:
    """Creates runs/<run_id>/ and appends tidy metric rows.

    Pass `run_id` to use a fixed, reproducible name (the sweep runner does, so a
    restarted sweep can skip finished runs); otherwise one is generated from
    env/algo/mitigation/seed and a timestamp.

    Example
    -------
    logger = RunLogger(
        root="runs", env="CartPole-v1", algo="PPO", mitigation="none", seed=0,
        config={"total_timesteps": 25_000, "replay_ratio": 1},
    )
    logger.log(step=1000, metric="eval_reward", value=213.4)
    logger.log(step=1000, metric="dormant_fraction", value=0.07)
    logger.close()
    """

    def __init__(self, root, env, algo, mitigation, seed, config=None, run_id=None, group=None):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = run_id or f"{env}_{algo}_{mitigation}_seed{seed}_{timestamp}"
        self.dir = Path(root) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "status.json").unlink(missing_ok=True)
        self._start = time.time()

        # Held open for the whole run (and flushed per row) so a sync client such
        # as OneDrive can't make a later re-open fail mid-run.
        self._file = (self.dir / "metrics.csv").open("w", newline="")
        self._csv = csv.writer(self._file)
        self._csv.writerow(["run_id", "step", "metric", "value"])
        self._file.flush()

        meta = {
            "run_id": self.run_id,
            "group": group or self.run_id,
            "env": env,
            "algo": algo,
            "mitigation": mitigation,
            "seed": seed,
            "git_commit": _git_commit(),
            "created": timestamp,
            "config": config or {},
        }
        with (self.dir / "meta.json").open("w") as f:
            json.dump(meta, f, indent=2)

    def log(self, step: int, metric: str, value: float) -> None:
        if metric not in METRICS:
            raise ValueError(
                f"Unknown metric {metric!r}. Use one of {METRICS} "
                f"(add new names to METRICS deliberately)."
            )
        self._csv.writerow([self.run_id, step, metric, float(value)])
        self._file.flush()

    def log_dict(self, step: int, metrics: dict) -> None:
        for name, value in metrics.items():
            self.log(step, name, value)

    def close(self, **summary) -> None:
        """Mark the run finished. Only call this when training completed normally."""
        self._file.close()
        status = {"status": "done", "wall_time_sec": round(time.time() - self._start, 1), **summary}
        with (self.dir / "status.json").open("w") as f:
            json.dump(status, f, indent=2)
