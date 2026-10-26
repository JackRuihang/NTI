from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        # Shared GPUs can finish replay shards at substantially different times.
        dist.init_process_group(
            backend=backend, init_method="env://", timeout=timedelta(seconds=3600)
        )
    return DistributedContext(rank, local_rank, world_size, device)


def seed_everything(seed: int, rank: int, deterministic: bool = False) -> None:
    rank_seed = seed + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rank_seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def reduce_metrics(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    reduced: Dict[str, float] = {}
    for name, value in metrics.items():
        tensor = value.detach().float()
        if tensor.ndim:
            tensor = tensor.mean()
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor = tensor / dist.get_world_size()
        reduced[name] = float(tensor.item())
    return reduced


def gather_rng_states(local_state: object, context: DistributedContext):
    if not dist.is_initialized():
        return [local_state]
    gathered = [None for _ in range(context.world_size)] if context.is_main else None
    dist.gather_object(local_state, gathered, dst=0)
    return gathered


def barrier() -> None:
    if dist.is_initialized():
        device_ids = (
            [torch.cuda.current_device()] if torch.cuda.is_available() else None
        )
        dist.barrier(device_ids=device_ids)


def shutdown_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
