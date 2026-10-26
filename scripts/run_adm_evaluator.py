#!/usr/bin/env python3
"""Run the official ADM evaluator and persist reproducible metric records."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nti.evaluation import (  # noqa: E402
    file_record,
    inspect_adm_npz,
    validate_sample_metadata,
)


METRIC_PATTERN = re.compile(
    r"^\s*(Inception Score|FID|sFID|Precision|Recall)\s*:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
    re.MULTILINE,
)
METRIC_KEYS = {
    "Inception Score": "inception_score",
    "FID": "fid",
    "sFID": "sfid",
    "Precision": "precision",
    "Recall": "recall",
}
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--evaluator-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--samples-metadata", type=Path)
    parser.add_argument("--expected-num-samples", type=int, required=True)
    parser.add_argument("--expected-image-size", type=int, required=True)
    parser.add_argument("--expected-num-classes", type=int)
    parser.add_argument("--required-protocol")
    parser.add_argument(
        "--reference",
        action="append",
        required=True,
        metavar="LABEL=NPZ",
        help="Reference label and ADM-compatible NPZ; may be repeated.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def command_output(command: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_record(path: Path) -> dict[str, object]:
    root = path.parent.parent
    commit = command_output(["git", "-C", str(root), "rev-parse", "HEAD"])
    remote = command_output(["git", "-C", str(root), "remote", "get-url", "origin"])
    tracked_status = command_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"]
    )
    return {
        "root": str(root),
        "commit": commit,
        "origin": remote,
        "tracked_files_clean": tracked_status == ""
        if tracked_status is not None
        else None,
    }


def python_runtime_record(python: Path) -> dict[str, object]:
    freeze = command_output([str(python), "-m", "pip", "freeze"])
    return {
        "executable": str(python),
        "version": command_output([str(python), "--version"]),
        "packages": freeze.splitlines() if freeze is not None else None,
    }


def parse_reference(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"reference must be LABEL=NPZ, received {value!r}")
    label, raw_path = value.split("=", 1)
    if not LABEL_PATTERN.fullmatch(label):
        raise ValueError(f"invalid reference label {label!r}")
    if not raw_path:
        raise ValueError(f"reference {label!r} has an empty path")
    return label, Path(raw_path).expanduser().resolve()


def parse_adm_metrics(stdout: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for match in METRIC_PATTERN.finditer(stdout):
        key = METRIC_KEYS[match.group(1)]
        value = float(match.group(2))
        if not math.isfinite(value):
            raise ValueError(f"ADM evaluator returned non-finite {key}: {value}")
        if key in metrics:
            raise ValueError(f"ADM evaluator printed {key} more than once")
        metrics[key] = value
    missing = {"inception_score", "fid"} - set(metrics)
    if missing:
        raise ValueError(
            f"ADM evaluator output is missing required metrics: {sorted(missing)}"
        )
    if metrics["inception_score"] <= 0.0 or metrics["fid"] < 0.0:
        raise ValueError("ADM evaluator returned an invalid IS or FID value")
    return metrics


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def unique_references(values: Iterable[str]) -> list[tuple[str, Path]]:
    references = [parse_reference(value) for value in values]
    labels = [label for label, _ in references]
    if len(labels) != len(set(labels)):
        raise ValueError("reference labels must be unique")
    return references


def main() -> None:
    args = parse_args()
    evaluator = args.evaluator.expanduser().resolve()
    evaluator_python = args.evaluator_python.expanduser().absolute()
    samples = args.samples.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not evaluator.is_file():
        raise FileNotFoundError(evaluator)
    if not evaluator_python.is_file():
        raise FileNotFoundError(evaluator_python)
    if args.expected_num_samples <= 0 or args.expected_image_size <= 0:
        raise ValueError("expected sample count and image size must be positive")
    if output.exists():
        raise FileExistsError(
            f"metrics output already exists: {output}; use a new output path"
        )
    if args.required_protocol is not None and args.samples_metadata is None:
        raise ValueError("required-protocol needs samples-metadata")
    if args.samples_metadata is not None and args.expected_num_classes is None:
        raise ValueError("samples-metadata needs expected-num-classes")

    sample_record = file_record(samples)
    sample_schema = inspect_adm_npz(
        samples,
        num_samples=args.expected_num_samples,
        image_size=args.expected_image_size,
    )
    sample_metadata = None
    if args.samples_metadata is not None:
        metadata_path = args.samples_metadata.expanduser().resolve()
        metadata = validate_sample_metadata(
            metadata_path,
            num_samples=args.expected_num_samples,
            num_classes=args.expected_num_classes,
            required_protocol=args.required_protocol,
        )
        sample_metadata = {
            "file": file_record(metadata_path),
            "protocol": metadata["protocol"],
            "provenance": metadata["provenance"],
        }
    references = unique_references(args.reference)
    runs: list[dict[str, object]] = []
    for label, reference in references:
        reference_record = file_record(reference)
        command = [
            str(evaluator_python),
            str(evaluator),
            str(reference),
            str(samples),
        ]
        log_path = output.parent / f"{output.stem}.adm.{label}.log"
        if log_path.exists():
            raise FileExistsError(
                f"ADM stdout log already exists: {log_path}; use a new output path"
            )
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=str(evaluator.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2"},
        )
        duration = time.monotonic() - started
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"ADM evaluator failed for {label} with exit code "
                f"{completed.returncode}; see {log_path}"
            )
        metrics = parse_adm_metrics(completed.stdout)
        runs.append(
            {
                "reference_label": label,
                "reference": reference_record,
                "metrics": metrics,
                "command": command,
                "started_at_utc": started_at,
                "duration_seconds": duration,
                "working_directory": str(evaluator.parent),
                "stdout_log": str(log_path),
            }
        )

    inception_scores = [float(run["metrics"]["inception_score"]) for run in runs]
    if any(
        not math.isclose(score, inception_scores[0], rel_tol=0.0, abs_tol=1e-9)
        for score in inception_scores[1:]
    ):
        raise RuntimeError(
            "Inception Score changed across reference runs for identical samples"
        )

    payload = {
        "schema_version": 2,
        "evaluator": {
            "name": "openai-guided-diffusion-adm",
            "script": file_record(evaluator),
            "repository": git_record(evaluator),
            "runtime": python_runtime_record(evaluator_python),
            "inception_graph": (
                file_record(evaluator.parent / "classify_image_graph_def.pb")
                if (evaluator.parent / "classify_image_graph_def.pb").is_file()
                else None
            ),
        },
        "hardware": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_smi": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,driver_version,compute_cap",
                    "--format=csv,noheader",
                ]
            ),
        },
        "samples": sample_record,
        "sample_array": sample_schema,
        "sample_metadata": sample_metadata,
        "runs": runs,
    }
    atomic_json(output, payload)
    print(json.dumps({"event": "adm_evaluation_complete", "output": str(output)}))


if __name__ == "__main__":
    main()
