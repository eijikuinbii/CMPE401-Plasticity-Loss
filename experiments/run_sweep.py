"""
Run a named sweep of training jobs (defined in sweeps.py) unattended.

- Jobs start in priority order, up to --workers at a time (default 1: with CUDA
  graphs one job saturates a laptop GPU, and two jobs are slower in total than
  one), each as its own subprocess logging to runs/<run_id>/stdout.log.
- Finished runs (runs/<run_id>/status.json exists) are skipped, so re-running
  the same command after a crash or reboot resumes where the sweep left off.
  A half-finished run is restarted from scratch.
- A failing job is retried --retries times (default 2), then skipped; the sweep
  carries on.
- While running, Windows is asked not to sleep (the display may still turn off).
- Progress is appended to runs/_sweeps/<sweep>.log.

    python experiments/run_sweep.py overnight --dry-run      # list jobs + estimates
    python experiments/run_sweep.py overnight                # run it
    python experiments/run_sweep.py overnight --status       # progress check
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from plasticity.diagnostics_logger import is_finished
from sweeps import SWEEPS

HERE = Path(__file__).resolve().parent


def keep_awake(on: bool) -> None:
    """Stop Windows from sleeping while this process runs (no-op elsewhere)."""
    if sys.platform != "win32":
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


class SweepLog:
    def __init__(self, root: Path, name: str):
        (root / "_sweeps").mkdir(parents=True, exist_ok=True)
        self.path = root / "_sweeps" / f"{name}.log"

    def __call__(self, msg: str) -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True)
        with self.path.open("a") as f:
            f.write(line + "\n")


def launch(job, root: Path) -> subprocess.Popen:
    run_dir = root / job.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "stdout.log").open("w")
    cmd = [sys.executable, str(HERE / job.script), *job.args,
           "--out", str(root), "--run-id", job.run_id, "--group", job.group]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE.parent, env=env)
    proc.log_file = log
    return proc


def fmt_minutes(m: float) -> str:
    return f"{m / 60:.1f} h" if m >= 90 else f"{m:.0f} min"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep", choices=sorted(SWEEPS))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--root", default="runs")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--stop-after", type=float, default=None,
                   help="hours; stop launching new jobs after this long (running ones finish)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--status", action="store_true")
    args = p.parse_args(argv)

    root = Path(args.root)
    if not root.is_absolute():
        root = HERE.parent / root
    jobs = SWEEPS[args.sweep]()
    todo = [j for j in jobs if not is_finished(root, j.run_id)]

    if args.dry_run or args.status:
        total = sum(j.est_minutes for j in todo)
        for j in jobs:
            mark = "done" if j not in todo else "    "
            print(f"[{mark}] {j.run_id:<45} ~{fmt_minutes(j.est_minutes)}")
        print(f"\n{len(jobs) - len(todo)}/{len(jobs)} finished. Remaining ~{fmt_minutes(total)} of job time "
              f"-> roughly {fmt_minutes(total / args.workers)} wall time with {args.workers} workers "
              f"(estimates assume the measured per-job speed under that load).")
        return

    log = SweepLog(root, args.sweep)
    log(f"sweep '{args.sweep}': {len(todo)}/{len(jobs)} jobs to run, {args.workers} workers, root={root}")
    deadline = datetime.now() + timedelta(hours=args.stop_after) if args.stop_after else None
    queue = list(todo)
    attempts = {j.run_id: 0 for j in todo}
    running: dict = {}
    failed = []
    keep_awake(True)
    try:
        while queue or running:
            while queue and len(running) < args.workers and not (deadline and datetime.now() > deadline):
                job = queue.pop(0)
                attempts[job.run_id] += 1
                running[job.run_id] = (job, launch(job, root), time.time())
                log(f"start  {job.run_id}  (attempt {attempts[job.run_id]})")
            if deadline and datetime.now() > deadline and queue:
                log(f"--stop-after reached; not starting {len(queue)} remaining jobs")
                queue.clear()
            for run_id, (job, proc, t_start) in list(running.items()):
                if proc.poll() is None:
                    continue
                proc.log_file.close()
                del running[run_id]
                mins = (time.time() - t_start) / 60
                if proc.returncode == 0 and is_finished(root, run_id):
                    log(f"done   {run_id}  ({mins:.1f} min)")
                elif attempts[run_id] <= args.retries:
                    log(f"FAILED {run_id}  (exit {proc.returncode}); will retry -- see {run_id}/stdout.log")
                    queue.insert(0, job)
                else:
                    log(f"FAILED {run_id}  (exit {proc.returncode}); giving up -- see {run_id}/stdout.log")
                    failed.append(run_id)
            time.sleep(5)
    except KeyboardInterrupt:
        log("interrupted; stopping running jobs")
        for _, proc, _ in running.values():
            proc.terminate()
    finally:
        keep_awake(False)
    log(f"sweep '{args.sweep}' finished; {len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
