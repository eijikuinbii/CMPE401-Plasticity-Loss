"""
CUDA-graph capture for small training steps.

Our networks are so small that an eager PyTorch training step is dominated by
kernel-launch overhead (~2 ms per DQN update on a Windows laptop GPU, ~0.3 ms
captured). A captured step replays the recorded kernels with one launch.

Rules for code that runs inside a captured step:
  - read and write only *static* tensors (same memory every replay): inputs
    are copied into fixed buffers, outputs accumulated into fixed tensors;
  - no Python-side randomness or data-dependent Python control flow
    (torch.rand on the device is fine: replays advance the RNG);
  - the optimizer needs `capturable=True`, and `zero_grad(set_to_none=False)`.
Anything that edits weights or optimizer state between replays must do so in
place (all of plasticity.mitigations does).
"""

from __future__ import annotations

from typing import Callable

import torch


def cuda_graph(fn: Callable[[], None], warmup: int = 3) -> Callable[[], None]:
    """Capture fn() as a CUDA graph and return its replay().

    The warm-up calls execute fn for real (e.g. `warmup` real optimizer steps).
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay
