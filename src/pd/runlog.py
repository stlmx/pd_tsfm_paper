"""Run provenance (red line 2 of CLAUDE.md).

Every number that reaches the paper must be traceable to a row of
experiments/runs.csv, a git commit and a per-sample prediction file. A run
launched from a dirty worktree cannot be reproduced, so it is refused rather
than logged.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

__all__ = ["git_commit", "log_run", "RUNS_CSV"]

REPO = Path(__file__).resolve().parents[2]
RUNS_CSV = REPO / "experiments" / "runs.csv"


def git_commit(allow_dirty: bool = False) -> str:
    """Current commit, refusing a dirty worktree unless explicitly allowed."""
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    if dirty and not allow_dirty:
        raise RuntimeError(
            "worktree has uncommitted changes, so this run could not be reproduced:\n"
            f"{dirty}\n"
            "Commit first, or pass allow_dirty=True for a throwaway run whose "
            "numbers will never reach the paper."
        )
    return f"{head}-dirty" if dirty else head


def log_run(
    *,
    exp_id: str,
    method: str,
    metrics: dict,
    output_dir: str | Path,
    config: dict | None = None,
    backbone: str = "",
    input_interface: str = "",
    adaptation: str = "",
    shots_per_class: int | str = "",
    seed: int | str = "",
    split_file: str = "",
    gpu: str = "",
    train_time_min: float | str = "",
    status: str = "ok",
    notes: str = "",
    allow_dirty: bool = False,
) -> str:
    """Append one row to runs.csv and write config + metrics into output_dir.

    Returns the run_id. `metrics` should carry at least macro_f1 and accuracy;
    anything else is kept in the output directory's metrics.json.
    """
    commit = git_commit(allow_dirty=allow_dirty)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    run_id = f"{exp_id}_{method}_{now:%Y%m%d-%H%M%S}".replace("/", "-")

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    if config is not None:
        (out / "config.json").write_text(json.dumps(config, indent=2, default=str))

    row = {
        "run_id": run_id,
        "exp_id": exp_id,
        "date": now.strftime("%Y-%m-%d"),
        "git_commit": commit,
        "config_path": str(out / "config.json") if config is not None else "",
        "method": method,
        "backbone": backbone,
        "input_interface": input_interface,
        "adaptation": adaptation,
        "shots_per_class": shots_per_class,
        "seed": seed,
        "split_file": split_file,
        "gpu": gpu,
        "train_time_min": train_time_min,
        "macro_f1": metrics.get("macro_f1", ""),
        "accuracy": metrics.get("accuracy", ""),
        "output_dir": str(out),
        "status": status,
        "notes": notes.replace(",", "；"),  # keep the CSV single-field-safe
    }

    header = list(pd.read_csv(RUNS_CSV, nrows=0).columns)
    unknown = set(row) - set(header)
    if unknown:
        raise ValueError(f"columns not in runs.csv header: {sorted(unknown)}")

    pd.DataFrame([row])[header].to_csv(RUNS_CSV, mode="a", header=False, index=False)
    return run_id
