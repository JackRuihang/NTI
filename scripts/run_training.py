"""Train NTI in checkpointed phases, refreshing replay only between phases."""

from __future__ import annotations
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nti.config import load_experiment_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--nproc", type=int, default=8)
    p.add_argument("--bank-batch-size", type=int, default=64)
    p.add_argument(
        "--resume-step",
        type=int,
        default=0,
        help="Completed phase boundary; retain its checkpoint and replay banks",
    )
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    c = load_experiment_config(a.config)
    t = c.training
    assert a.nproc > 0 and a.bank_batch_size > 0
    assert t.history_source == "lagged_replay"
    boundaries = sorted(
        set(
            [0, t.max_steps, t.replay_end_step]
            + list(
                range(t.replay_start_step, t.replay_end_step, t.replay_refresh_steps)
            )
        )
    )
    if a.resume_step not in boundaries or a.resume_step == t.max_steps:
        p.error("resume-step must be an unfinished run phase boundary")
    run = Path(c.run.output_dir) / c.run.name
    root = Path(t.replay_bank_root)
    if a.resume_step == 0 and (run.exists() or root.exists()) and not a.dry_run:
        p.error("Fresh run requires new output and replay directories")

    def cp(step):
        return run / "checkpoints" / f"step_{step:07d}.pt"

    def call(args):
        args = [str(x) for x in args]
        print(shlex.join(args), flush=True)
        if not a.dry_run:
            subprocess.run(args, check=True)

    dist = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={a.nproc}",
    ]
    for start, end in zip(boundaries, boundaries[1:]):
        if start < a.resume_step:
            continue
        if start and not a.dry_run and not cp(start).is_file():
            raise FileNotFoundError(cp(start))
        if t.replay_start_step <= start < t.replay_end_step:
            bank = root / f"bank_step_{start:07d}"
            if not (bank / "COMPLETE.json").is_file():
                if bank.exists() and not a.dry_run:
                    raise RuntimeError(
                        f"Incomplete bank requires manual inspection: {bank}"
                    )
                call(
                    dist
                    + [
                        "scripts/build_lagged_replay_bank.py",
                        "--config",
                        a.config,
                        "--checkpoint",
                        cp(start),
                        "--output",
                        bank,
                        "--expected-step",
                        start,
                        "--batch-size",
                        a.bank_batch_size,
                        "--precision",
                        t.precision,
                    ]
                )
            call(
                [
                    sys.executable,
                    "scripts/publish_replay_pointer.py",
                    "--root",
                    root,
                    "--bank",
                    bank,
                    "--expected-step",
                    start,
                    "--expected-checkpoint",
                    cp(start),
                    "--expected-target-mode",
                    t.target_mode,
                    "--expected-target-encoding",
                    t.replay_target_encoding,
                    "--expected-top-k",
                    t.replay_top_k,
                ]
            )
        args = dist + [
            "scripts/train_nti.py",
            "--config",
            a.config,
            "--stop-at-step",
            end,
        ]
        if start:
            args += ["--resume", cp(start)]
        call(args)


if __name__ == "__main__":
    main()
