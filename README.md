# CMPE401 — Plasticity Loss in Deep RL

**When Learners Stop Learning: Diagnosing and Mitigating Plasticity Loss in Deep Reinforcement Learning**

Course project for CMPE 401 (Deep Learning for Engineers). A controlled study of *plasticity loss* — the phenomenon where a deep RL agent gradually loses the ability to fit new targets, so continued training stops helping and can actively hurt.

## Objective

1. **Induce & observe** plasticity loss in a lightweight setting (permuted MNIST, then DQN on MinAtar).
2. **Diagnose** it: track internal signals throughout training and find which predicts collapse *before* the performance curve does.
3. **Mitigate** it: compare L2, shrink-and-perturb, ReDo neuron recycling, and resets under matched conditions (continual backprop as a stretch comparison).

Everything runs on a single laptop GPU (RTX 3060 Laptop, 6 GB).

## Repository layout

```
src/plasticity/              the library (pip install -e .)
  networks.py                MLP for permuted MNIST; MinAtar reference Q-network
  diagnostics.py             dormant/dead units, effective rank, srank, norms, plasticity probe
  mitigations.py             reset, shrink-and-perturb, ReDo (L2 = optimizer weight decay)
  graphs.py                  CUDA-graph capture of small training steps (~7x speedup)
  diagnostics_logger.py      run folders + tidy metrics.csv
experiments/
  permuted_mnist.py          continual supervised learning (500 permuted-MNIST tasks)
  dqn_minatar.py             DQN on MinAtar: replay ratio, channel-permutation task shifts
  sweeps.py                  named experiment sweeps (what runs overnight)
  run_sweep.py               unattended, resumable sweep runner
analysis/
  plot_runs.py               mean ± 95% CI figures across seeds + summary table
```

## Setup (local)

Python 3.10+ and an NVIDIA GPU with a recent driver. Keep the virtual environment **outside** any OneDrive/Dropbox folder — syncing a multi-GB PyTorch install is slow and can lock files.

```bash
python -m venv ~/venvs/plasticity
source ~/venvs/plasticity/Scripts/activate      # Windows (Git Bash); Linux/macOS: .../bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Usage

Quick end-to-end check of every script and mitigation (~3 min):

```bash
python experiments/run_sweep.py smoke
```

Single runs:

```bash
python experiments/permuted_mnist.py --mitigation none --seed 0
python experiments/dqn_minatar.py --game breakout --replay-ratio 4 --seed 0
python experiments/dqn_minatar.py --game breakout --shift-every 200000 --steps 2000000 --mitigation redo
```

All flags are documented in `--help`.

### Running a sweep unattended

```bash
python experiments/run_sweep.py overnight --dry-run      # list jobs and time estimate
python experiments/run_sweep.py overnight                # run (keeps Windows awake)
python experiments/run_sweep.py overnight --status       # progress, from any terminal
```

- Re-running the same command **resumes**: finished runs are skipped, interrupted ones restart.
- A failing job is retried once, then skipped; the sweep carries on. Each job's output is in `runs/<run_id>/stdout.log`, the sweep's timeline in `runs/_sweeps/<sweep>.log`.
- `--stop-after H` stops launching new jobs after H hours.
- One job at a time (the default) is fastest on a single laptop GPU: with CUDA graphs one DQN job saturates the GPU (~2,150 env steps/s at replay ratio 1), and two concurrent jobs are slower in total. `--workers N` exists for bigger GPUs.
- **Don't kill individual training processes (e.g. from Task Manager) while others are using the GPU.** On Windows we observed that force-killing one CUDA process can crash other running CUDA processes. Stop the whole sweep with Ctrl+C instead; re-running resumes it.

### Plotting

```bash
python analysis/plot_runs.py --filter pmnist --name pmnist
python analysis/plot_runs.py --filter breakout_shift --name breakout_shift --smooth 3
python analysis/plot_runs.py --filter breakout_rr --name breakout_replay_ratio --smooth 3
```

Writes `figures/<name>.png` (one panel per metric, mean ± 95% CI over seeds) and prints each group's end-of-training mean ± CI.

## Experiments

| Setting | How plasticity loss is induced | Reference line |
|---|---|---|
| Permuted MNIST (2×100 MLP, Adam 1e-3, 500 tasks × 10k examples, 1 epoch) | a new pixel permutation every task | `fresh`: network re-initialised at every task boundary |
| MinAtar Breakout, task shifts (2M steps) | observation channels re-permuted every 200k steps (10 tasks) | `fresh`: re-initialised at every shift |
| MinAtar Breakout, replay ratio (1M steps) | 4 updates per env step instead of 1 (primacy bias) | replay ratio 1 |

Data streams depend only on the seed, so every mitigation sees exactly the same tasks. 3 seeds per configuration.

## Diagnostics tracked

| Metric name (`METRICS`) | Meaning |
|---|---|
| `task_accuracy` / `train_return` / `eval_reward` | performance: test accuracy per task (MNIST), training-episode and greedy-eval return (RL) |
| `train_accuracy` | online accuracy while learning a task — a direct measure of learning speed |
| `dormant_fraction` | fraction of units with normalised activation ≤ τ = 0.1 (Sokar et al., 2023) |
| `dead_fraction` | fraction of units never active on a batch |
| `effective_rank`, `srank` | rank of the penultimate representation (Roy & Vetterli; Kumar et al., 2021) |
| `weight_l2_norm`, `grad_l2_norm` | global weight and gradient norms |
| `plasticity_probe_loss` | MSE after training a *copy* of the network on fixed random targets for 200 steps (lower = more plastic; Lyle et al., 2022) |
| `units_recycled`, `steps_per_sec` | ReDo activity; throughput for compute-cost comparisons |

## Mitigations

| `--mitigation` | What it does |
|---|---|
| `none` | baseline |
| `l2` | Adam weight decay (1e-4) |
| `shrink_perturb` | θ ← 0.8·θ + 0.2·θ_fresh, periodically (Ash & Adams, 2020) |
| `redo` | re-initialise dormant units' incoming weights, zero their outgoing weights (Sokar et al., 2023) |
| `reset` | re-initialise the last N layers periodically (Nikishin et al., 2022); N = all gives the `fresh` reference |

## Experiment logging convention

One run = one folder `runs/<run_id>/`, where sweep run ids look like `<setting>_<config>_s<seed>`:

- `meta.json` — config + provenance (all CLI args, seed, git commit, group)
- `metrics.csv` — tidy long format: `run_id, step, metric, value`
- `status.json` — written on successful completion (wall time, etc.)
- `model.pt` — final weights; `stdout.log` — console output

`runs/` is git-ignored (raw outputs stay local); curated figures are committed under `figures/`.

## Changes from the original project plan

- **Colab → local GPU.** Training runs on a local RTX 3060 Laptop GPU so long sweeps can run unattended overnight without session limits.
- **Stable-Baselines3 → a ~250-line DQN.** The diagnostics and mitigations need direct access to the network, its activations, and the optimizer state between updates; a self-contained DQN following the MinAtar reference implementation makes this simple and lets the training step be CUDA-graph-captured (~7× faster than eager PyTorch for networks this small).
- **Smaller permuted-MNIST network.** Pilots with a 3×256 MLP showed no accuracy loss within 50 tasks; the 2×100 MLP shows clear loss within 100 tasks at the standard Adam learning rate.
- **Task-shift RL setting.** In addition to high replay ratio, we permute MinAtar's observation channels periodically — the RL analogue of permuted MNIST — which gives a clean "fresh network" reference to measure lost plasticity against.

## Status

- [x] Local environment (PyTorch CUDA, MinAtar); diagnostics + mitigations implemented
- [x] Plasticity loss reproduced in permuted MNIST (pilot: 91.5% → 88.5% test accuracy by task 100, dormant units 11% → 66%)
- [ ] Overnight sweep: permuted MNIST (all mitigations), Breakout task-shift + replay-ratio characterisation
- [ ] Characterise the failure: which diagnostic predicts collapse, and how early
- [ ] Compare mitigations in RL
- [ ] Report, video, final repo

## References

- Dohare et al., *Loss of plasticity in deep continual learning*, Nature 2024 — `github.com/shibhansh/loss-of-plasticity`
- Nikishin et al., *The Primacy Bias in Deep RL*, ICML 2022 — `github.com/evgenii-nikishin/rl_with_resets`
- Sokar et al., *The Dormant Neuron Phenomenon in Deep RL*, ICML 2023
- Lyle et al., *Understanding and Preventing Capacity Loss in RL*, ICLR 2022
- Kumar et al., *Implicit Under-Parameterization Inhibits Data-Efficient Deep RL*, ICLR 2021
- Ash & Adams, *On Warm-Starting Neural Network Training*, NeurIPS 2020
- Young & Tian, *MinAtar*, 2019 — `github.com/kenjyoung/MinAtar`
- Klein et al., *Plasticity Loss in Deep RL: A Survey*, 2024 — `arxiv.org/abs/2411.04832` (Related Work map)
