"""
DQN on MinAtar with a configurable replay ratio, optional periodic task shifts,
plasticity diagnostics, and plug-in mitigations.

Three ways to induce plasticity loss:
  --replay-ratio R   gradient updates per environment step (MinAtar's reference
                     DQN uses 1). High R overfits early data -- the "primacy bias".
  --shift-every N    every N env steps, re-permute the observation channels, so
                     the agent must re-learn what each object channel means
                     (the RL analogue of permuted MNIST).
  --game a,b,c       cycle through several MinAtar games, switching every
     --switch-every N  N env steps (continual multi-game RL, cf. Abbas et al.,
                     2023). Observations are zero-padded to the largest channel
                     count; all games share the same 6 actions.

Architecture and defaults follow the MinAtar reference DQN (Young & Tian, 2019),
with Adam in place of RMSprop.

Speed: on CUDA the update and the greedy action are captured as CUDA graphs
(see plasticity/graphs.py; ~7x faster). Use --no-cuda-graph for plain eager
execution.

    python experiments/dqn_minatar.py --game breakout --replay-ratio 4 --seed 0
"""

from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from minatar import Environment

from plasticity import diagnostics as D
from plasticity import mitigations as M
from plasticity.diagnostics_logger import RunLogger
from plasticity.graphs import cuda_graph
from plasticity.networks import minatar_qnet


GAMES = ("breakout", "space_invaders", "asterix", "freeway", "seaquest")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="breakout",
                   help=f"one of {GAMES}, or a comma-separated list to cycle through")
    p.add_argument("--switch-every", type=int, default=0, help="env steps per game when cycling games")
    p.add_argument("--mitigation", default="none", choices=M.MITIGATIONS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000, help="environment steps")
    p.add_argument("--replay-ratio", type=float, default=1.0, help="updates per env step")
    p.add_argument("--shift-every", type=int, default=0, help="env steps between channel permutations (0 = off)")
    # DQN
    p.add_argument("--conv-channels", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--buffer", type=int, default=100_000)
    p.add_argument("--learn-start", type=int, default=5_000)
    p.add_argument("--target-every", type=int, default=1_000, help="updates between target syncs")
    p.add_argument("--eps-end", type=float, default=0.1)
    p.add_argument("--eps-steps", type=int, default=100_000)
    # mitigation hyperparameters (applied every --mit-every env steps)
    p.add_argument("--mit-every", type=int, default=100_000)
    p.add_argument("--weight-decay", type=float, default=1e-4, help="for --mitigation l2")
    p.add_argument("--reset-layers", type=int, default=2, help="for reset: last N layers (0 = all)")
    p.add_argument("--sp-alpha", type=float, default=0.8, help="for shrink_perturb")
    p.add_argument("--tau", type=float, default=0.1, help="dormancy threshold (diagnostic + ReDo)")
    # logging
    p.add_argument("--log-every", type=int, default=10_000)
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--probe-every", type=int, default=50_000)
    p.add_argument("--out", default="runs")
    p.add_argument("--run-id", default=None)
    p.add_argument("--group", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-cuda-graph", action="store_true")
    args = p.parse_args(argv)
    args.games = args.game.split(",")
    if any(g not in GAMES for g in args.games):
        p.error(f"--game must be from {GAMES}")
    if len(args.games) > 1 and not args.switch_every:
        p.error("cycling several games needs --switch-every")
    return args


class Replay:
    """Uniform replay buffer, written on the CPU and mirrored to the device in chunks.

    Updates sample from the device mirror (graph-safe: no Python-side sizes or
    randomness). Copying a chunk every `flush()` instead of one host->device
    copy per field per step keeps the per-step cost low; transitions become
    sampleable at the next flush (the training loop flushes every 100 steps).
    Flush at least once per `capacity` adds, or a full wrap-around goes unseen.
    """

    FIELDS = ("s", "a", "r", "s2", "done")

    def __init__(self, capacity: int, obs_shape, device):
        self.cpu = {
            "s": np.zeros((capacity, *obs_shape), np.uint8),
            "a": np.zeros(capacity, np.int64),
            "r": np.zeros(capacity, np.float32),
            "s2": np.zeros((capacity, *obs_shape), np.uint8),
            "done": np.zeros(capacity, np.float32),
        }
        self.dev = {k: torch.from_numpy(v).to(device) for k, v in self.cpu.items()}
        self.size_t = torch.zeros((), device=device)
        self.device = device
        self.capacity, self.ptr, self.size, self.flushed = capacity, 0, 0, 0

    def add(self, s, a, r, s2, done):
        for k, v in zip(self.FIELDS, (s, a, r, s2, done)):
            self.cpu[k][self.ptr] = v
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def flush(self):
        lo, hi = self.flushed, self.ptr
        spans = [(lo, hi)] if lo <= hi else [(lo, self.capacity), (0, hi)]
        for a, b in spans:
            if b > a:
                for k in self.FIELDS:
                    self.dev[k][a:b].copy_(torch.from_numpy(self.cpu[k][a:b]))
        self.flushed = hi
        self.size_t.fill_(float(self.size))

    def sample(self, n):
        idx = (torch.rand(n, device=self.device) * self.size_t).long()
        d = self.dev
        return d["s"][idx].float(), d["a"][idx], d["r"][idx], d["s2"][idx].float(), d["done"][idx]

    def recent_states(self, n, rng, window=10_000) -> torch.Tensor:
        """n states from the most recent `window` transitions (the current task)."""
        w = min(window, self.size)
        idx = (self.ptr - 1 - rng.integers(0, w, n)) % self.capacity
        return torch.from_numpy(self.cpu["s"][idx]).to(self.device).float()


def to_obs(state: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """MinAtar (10, 10, c) bool grid -> (C, 10, 10) uint8: zero-padded to C = len(perm)
    channels, then channels permuted."""
    obs = np.zeros((len(perm), 10, 10), dtype=np.uint8)
    obs[:state.shape[2]] = state.transpose(2, 0, 1)
    return obs[perm]


def evaluate(greedy, game, seed, perm, episodes, eps=0.01, max_steps=20_000) -> float:
    env = Environment(game)
    env.seed(seed)
    rng = np.random.default_rng(seed)
    returns, ep_ret, steps = [], 0.0, 0
    env.reset()
    while len(returns) < episodes and steps < max_steps:
        if rng.random() < eps:
            a = int(rng.integers(env.num_actions()))
        else:
            a = greedy(to_obs(env.state(), perm))
        r, done = env.act(a)
        ep_ret += r
        steps += 1
        if done:
            returns.append(ep_ret)
            ep_ret = 0.0
            env.reset()
    return float(np.mean(returns)) if returns else ep_ret


def main(argv=None):
    args = parse_args(argv)
    device = args.device
    use_graph = device.startswith("cuda") and not args.no_cuda_graph
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    shift_rng = np.random.default_rng(10_000 + args.seed)

    envs = []
    for i, g in enumerate(args.games):
        e = Environment(g)
        e.seed(args.seed + 1000 * i)
        envs.append(e)
    n_ch = max(e.state_shape()[2] for e in envs)
    n_actions = envs[0].num_actions()
    assert all(e.num_actions() == n_actions for e in envs)
    task, env = 0, envs[0]
    perm = np.arange(n_ch)

    qnet = minatar_qnet(n_ch, n_actions, args.conv_channels, args.hidden).to(device)
    target = copy.deepcopy(qnet)
    wd = args.weight_decay if args.mitigation == "l2" else 0.0
    opt = torch.optim.Adam(qnet.parameters(), lr=args.lr, weight_decay=wd, capturable=use_graph)
    buf = Replay(args.buffer, (n_ch, 10, 10), device)

    logger = RunLogger(args.out, env=f"MinAtar-{'+'.join(args.games)}", algo="DQN", mitigation=args.mitigation,
                       seed=args.seed, config=vars(args), run_id=args.run_id, group=args.group)
    print(f"[{logger.run_id}] device={device} cuda_graph={use_graph}", flush=True)

    # Static tensors: the graphs read/write these fixed buffers on every replay.
    loss_acc = torch.zeros((), device=device)
    act_in = torch.zeros((1, n_ch, 10, 10), device=device)
    act_out = torch.zeros(1, dtype=torch.long, device=device)

    def update():
        s, a, r, s2, done = buf.sample(args.batch)
        q = qnet(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            y = r + args.gamma * (1 - done) * target(s2).max(1).values
        loss = F.smooth_l1_loss(q, y)
        opt.zero_grad(set_to_none=False)
        loss.backward()
        opt.step()
        loss_acc.add_(loss.detach())

    @torch.no_grad()
    def act():
        act_out.copy_(qnet(act_in).argmax(1))

    def greedy(obs: np.ndarray) -> int:
        act_in.copy_(torch.from_numpy(obs)[None])
        act_fn()
        return int(act_out)

    act_fn = cuda_graph(act) if use_graph else act
    update_fn = None  # captured at learn_start, once the buffer holds data

    env.reset()
    obs = to_obs(env.state(), perm)
    ep_ret, episode_returns = 0.0, []
    n_updates, n_updates_window, update_debt, units_recycled = 0, 0, 0.0, 0
    grad_norms = []
    t_window = t0 = time.time()

    for step in range(1, args.steps + 1):
        # --- act (epsilon-greedy, linear decay) ---
        eps = max(args.eps_end, 1.0 - (1.0 - args.eps_end) * step / args.eps_steps)
        a = int(rng.integers(n_actions)) if rng.random() < eps else greedy(obs)
        r, done = env.act(a)
        ep_ret += r
        if done:
            episode_returns.append(ep_ret)
            ep_ret = 0.0
            env.reset()
        obs2 = to_obs(env.state(), perm)
        buf.add(obs, a, r, obs2, float(done))
        obs = obs2

        # --- learn ---
        if step % 100 == 0:
            buf.flush()
        if step > args.learn_start:
            if update_fn is None:
                buf.flush()
                update_fn = cuda_graph(update) if use_graph else update
                loss_acc.zero_()
            update_debt += args.replay_ratio
            while update_debt >= 1.0:
                update_debt -= 1.0
                update_fn()
                n_updates += 1
                n_updates_window += 1
                if n_updates % args.target_every == 0:
                    target.load_state_dict(qnet.state_dict())  # in-place copy: graph-safe
                if n_updates % 100 == 0:
                    grad_norms.append(D.grad_norm(qnet))

        # --- log ---
        if step % args.log_every == 0 and step > args.learn_start:
            now = time.time()
            metrics = {
                "train_loss": float(loss_acc) / max(n_updates_window, 1),
                "steps_per_sec": args.log_every / (now - t_window),
                **D.compute_diagnostics(qnet, buf.recent_states(1024, rng), args.tau),
            }
            if episode_returns:
                metrics["train_return"] = float(np.mean(episode_returns))
            if grad_norms:
                metrics["grad_l2_norm"] = float(np.mean(grad_norms))
            if args.mitigation == "redo":
                metrics["units_recycled"] = units_recycled
            if step % args.probe_every == 0:
                metrics["plasticity_probe_loss"] = D.plasticity_probe(qnet, buf.recent_states(512, rng))
            if step % args.eval_every == 0:
                metrics["eval_reward"] = evaluate(greedy, args.games[task % len(envs)], 100_000 + step, perm,
                                                  args.eval_episodes)
            if args.switch_every or args.shift_every:
                metrics["task_index"] = task if args.switch_every else (step - 1) // args.shift_every
            logger.log_dict(step, metrics)
            print(f"step {step:>9,}  return {metrics.get('train_return', float('nan')):7.2f}  "
                  f"dormant {metrics['dormant_fraction']:.3f}  erank {metrics['effective_rank']:6.1f}  "
                  f"{metrics['steps_per_sec']:5.0f} sps  [{(now - t0) / 60:.1f} min]", flush=True)
            episode_returns, grad_norms, units_recycled = [], [], 0
            loss_acc.zero_()
            n_updates_window, t_window = 0, time.time()

        # Boundary events come after logging, so a logged window (and its eval)
        # belongs entirely to the task it trained on.

        # --- game switch: the in-progress episode is abandoned, not recorded ---
        if args.switch_every and step % args.switch_every == 0:
            task += 1
            env = envs[task % len(envs)]
            env.reset()
            ep_ret = 0.0
            obs = to_obs(env.state(), perm)

        # --- task shift: new random meaning for each observation channel ---
        if args.shift_every and step % args.shift_every == 0:
            perm = shift_rng.permutation(n_ch)
            obs = to_obs(env.state(), perm)

        # --- mitigation ---
        if step > args.learn_start and step % args.mit_every == 0:
            if args.mitigation == "reset":
                M.reset_layers(qnet, args.reset_layers, opt)
            elif args.mitigation == "shrink_perturb":
                M.shrink_perturb(qnet, args.sp_alpha)
            elif args.mitigation == "redo":
                units_recycled += M.redo(qnet, buf.recent_states(1024, rng), args.tau, opt)

    torch.save(qnet.state_dict(), logger.dir / "model.pt")
    logger.close(updates=n_updates)


if __name__ == "__main__":
    main()
