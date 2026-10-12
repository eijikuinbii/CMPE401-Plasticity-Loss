"""
Continual supervised learning on permuted MNIST -- the cheap, controlled setting
in which plasticity loss is well documented (Dohare et al., Nature 2024).

Each task applies a new fixed random permutation to the pixels. The same network
trains on task after task; a network that keeps its plasticity reaches the same
accuracy on task 500 as on task 1, one that loses it gets steadily worse.

The task sequence (permutations + sample subsets) depends only on --seed, so all
mitigations with the same seed see exactly the same data.

    python experiments/permuted_mnist.py --mitigation none --seed 0
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from torchvision import datasets

from plasticity import diagnostics as D
from plasticity import mitigations as M
from plasticity.diagnostics_logger import RunLogger
from plasticity.graphs import cuda_graph
from plasticity.networks import mlp


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mitigation", default="none", choices=M.MITIGATIONS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks", type=int, default=500)
    p.add_argument("--train-per-task", type=int, default=10_000, help="training examples per task")
    p.add_argument("--epochs", type=int, default=1, help="passes over each task's examples")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=100)
    p.add_argument("--depth", type=int, default=2)
    # mitigation hyperparameters (all applied at task boundaries, every --mit-every tasks)
    p.add_argument("--mit-every", type=int, default=1)
    p.add_argument("--weight-decay", type=float, default=1e-4, help="for --mitigation l2")
    p.add_argument("--reset-layers", type=int, default=0, help="for reset: last N layers (0 = all)")
    p.add_argument("--sp-alpha", type=float, default=0.8, help="for shrink_perturb")
    p.add_argument("--tau", type=float, default=0.1, help="dormancy threshold (diagnostic + ReDo)")
    # logging
    p.add_argument("--probe-every", type=int, default=10, help="tasks between plasticity probes")
    p.add_argument("--out", default="runs")
    p.add_argument("--run-id", default=None)
    p.add_argument("--group", default=None)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-cuda-graph", action="store_true")
    return p.parse_args(argv)


def load_mnist(data_dir: str, device: str):
    """Whole of MNIST as flat float tensors on `device` (~220 MB)."""
    out = []
    for train in (True, False):
        ds = datasets.MNIST(data_dir, train=train, download=True)
        x = ds.data.reshape(len(ds), -1).float().div(255.0)
        out.append((x.to(device), ds.targets.to(device)))
    return out


@torch.no_grad()
def accuracy(model, x, y, chunk=5000) -> float:
    model.eval()
    correct = sum(int((model(x[i:i + chunk]).argmax(1) == y[i:i + chunk]).sum())
                  for i in range(0, len(x), chunk))
    model.train()
    return correct / len(x)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = args.device
    (x_train, y_train), (x_test, y_test) = load_mnist(args.data_dir, device)
    task_rng = torch.Generator().manual_seed(10_000 + args.seed)  # data stream only

    use_graph = device.startswith("cuda") and not args.no_cuda_graph
    model = mlp(784, args.hidden, args.depth, 10).to(device)
    wd = args.weight_decay if args.mitigation == "l2" else 0.0
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd, capturable=use_graph)

    logger = RunLogger(args.out, env="PermutedMNIST", algo="Adam-MLP", mitigation=args.mitigation,
                       seed=args.seed, config=vars(args), run_id=args.run_id, group=args.group)
    print(f"[{logger.run_id}] device={device} cuda_graph={use_graph}", flush=True)

    # Static buffers for the (optionally CUDA-graph-captured) training step: the
    # current task's data, this epoch's example order, and a step counter that
    # the step itself advances.
    n = args.train_per_task
    xt = torch.zeros(n, 784, device=device)
    yt = torch.zeros(n, dtype=torch.long, device=device)
    order = torch.zeros(n, dtype=torch.long, device=device)
    ctr = torch.zeros((), dtype=torch.long, device=device)
    offsets = torch.arange(args.batch, device=device)
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)

    def train_step():
        b = order[ctr * args.batch + offsets]
        ctr.add_(1)
        logits = model(xt[b])
        loss = F.cross_entropy(logits, yt[b])
        opt.zero_grad(set_to_none=False)
        loss.backward()
        opt.step()
        loss_sum.add_(loss.detach())
        correct.add_((logits.argmax(1) == yt[b]).sum())

    step_fn = None  # captured once task 0's data is in the buffers
    steps_per_epoch = n // args.batch
    t0 = time.time()
    for task in range(args.tasks):
        perm = torch.randperm(784, generator=task_rng).to(device)
        subset = torch.randperm(len(x_train), generator=task_rng)[:n].to(device)
        xt.copy_(x_train[subset][:, perm])
        yt.copy_(y_train[subset])

        # Mitigations act at the boundary, once the new task's inputs are known.
        units_recycled = 0
        if task > 0 and task % args.mit_every == 0:
            if args.mitigation == "reset":
                M.reset_layers(model, args.reset_layers, opt)
            elif args.mitigation == "shrink_perturb":
                M.shrink_perturb(model, args.sp_alpha)
            elif args.mitigation == "redo":
                units_recycled = M.redo(model, xt[:1024], args.tau, opt)

        if step_fn is None:
            order.copy_(torch.randperm(n, device=device))
            ctr.zero_()
            step_fn = cuda_graph(train_step) if use_graph else train_step
        loss_sum.zero_()
        correct.zero_()
        for _ in range(args.epochs):
            order.copy_(torch.randperm(n, device=device))
            ctr.zero_()
            for _ in range(steps_per_epoch):
                step_fn()
        n_steps = args.epochs * steps_per_epoch

        test_acc = accuracy(model, x_test[:, perm], y_test)
        metrics = {
            "task_accuracy": test_acc,
            "train_accuracy": float(correct) / (n_steps * args.batch),
            "train_loss": float(loss_sum) / n_steps,
            "grad_l2_norm": D.grad_norm(model),  # gradient of the task's last step
            **D.compute_diagnostics(model, xt[:1024], args.tau),
        }
        if args.mitigation == "redo":
            metrics["units_recycled"] = units_recycled
        if task % args.probe_every == 0 or task == args.tasks - 1:
            metrics["plasticity_probe_loss"] = D.plasticity_probe(model, xt[:512])
        logger.log_dict(task, metrics)

        if task % 10 == 0 or task == args.tasks - 1:
            el = time.time() - t0
            print(f"task {task:4d}  test_acc {test_acc:.4f}  dormant {metrics['dormant_fraction']:.3f}  "
                  f"erank {metrics['effective_rank']:.1f}  [{el / 60:.1f} min]", flush=True)

    torch.save(model.state_dict(), logger.dir / "model.pt")
    logger.close(final_task_accuracy=test_acc)


if __name__ == "__main__":
    main()
