"""
Aggregate runs and plot every diagnostic: one panel per metric, one line per
run group (= config without the seed), mean +/- 95% CI across seeds.

    python analysis/plot_runs.py --filter pmnist --name pmnist_mitigations
    python analysis/plot_runs.py --filter breakout_rr --name breakout_replay_ratio --smooth 5

Also prints a summary table: each group's mean over the final 10% of training.
Figures go to figures/<name>.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

PANEL_ORDER = [
    "task_accuracy", "train_accuracy", "train_return", "eval_reward",
    "plasticity_probe_loss", "dormant_fraction", "dead_fraction",
    "effective_rank", "srank", "weight_l2_norm", "grad_l2_norm", "train_loss",
]


def load_runs(root: Path, pattern: str | None, finished_only: bool = True) -> pd.DataFrame:
    frames = []
    for meta_path in sorted(root.glob("*/meta.json")):
        run_dir = meta_path.parent
        if pattern and pattern not in run_dir.name:
            continue
        if finished_only and not (run_dir / "status.json").exists():
            continue
        meta = json.loads(meta_path.read_text())
        df = pd.read_csv(run_dir / "metrics.csv")
        df["group"], df["seed"] = meta["group"], meta["seed"]
        frames.append(df)
    if not frames:
        raise SystemExit(f"No {'finished ' if finished_only else ''}runs matching {pattern!r} under {root}")
    return pd.concat(frames, ignore_index=True)


def smooth(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 1:
        return df
    df = df.sort_values("step").copy()
    df["value"] = df.groupby(["run_id", "metric"])["value"].transform(
        lambda v: v.rolling(window, min_periods=1).mean())
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean (and 95% CI half-width across seeds) of each metric over the last 10% of steps."""
    cutoff = df.groupby("run_id")["step"].transform(lambda s: s.min() + 0.9 * (s.max() - s.min()))
    tail = df[df.step >= cutoff].groupby(["group", "seed", "metric"])["value"].mean().reset_index()
    agg = tail.groupby(["group", "metric"])["value"].agg(["mean", "std", "count"])
    agg["ci95"] = 1.96 * agg["std"] / agg["count"].pow(0.5)
    return agg[["mean", "ci95", "count"]].round(4)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(ROOT / "runs"))
    p.add_argument("--filter", default=None, help="substring a run id must contain")
    p.add_argument("--name", default="runs", help="output figure name")
    p.add_argument("--smooth", type=int, default=1, help="rolling-mean window (in log points)")
    p.add_argument("--include-unfinished", action="store_true")
    args = p.parse_args(argv)

    df = load_runs(Path(args.root), args.filter, finished_only=not args.include_unfinished)
    metrics = [m for m in PANEL_ORDER if m in set(df.metric)]
    plot_df = smooth(df, args.smooth)

    ncols = 3
    nrows = -(-len(metrics) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows), squeeze=False)
    groups = sorted(df.group.unique())
    palette = dict(zip(groups, sns.color_palette("tab10", len(groups))))
    for ax, metric in zip(axes.flat, metrics):
        sns.lineplot(data=plot_df[plot_df.metric == metric], x="step", y="value", hue="group",
                     hue_order=groups, palette=palette, errorbar=("ci", 95), ax=ax, legend=False)
        ax.set_title(metric)
        ax.set_ylabel("")
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=palette[g], label=g) for g in groups]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(groups), 4), frameon=False)
    fig.tight_layout(rect=(0, 0.04 + 0.025 * ((len(groups) - 1) // 4), 1, 1))

    out = ROOT / "figures" / f"{args.name}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    with pd.option_context("display.max_rows", 500, "display.width", 140):
        print(summary(df))


if __name__ == "__main__":
    main()
