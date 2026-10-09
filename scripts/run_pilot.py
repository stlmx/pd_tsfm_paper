"""Pilot driver: load CSVs, render the interface, sweep the label-efficiency grid.

    # one environment at a time -- MOMENT's dependencies conflict with Chronos'
    PYTHONPATH=src .venv-main/bin/python   scripts/run_pilot.py --config configs/pilot.yaml
    PYTHONPATH=src .venv-moment/bin/python scripts/run_pilot.py --config configs/pilot.yaml \
        --backbones moment_base --out outputs/pilot_moment

Writes per-sample predictions and a results table under --out, and appends one
row per cell to experiments/runs.csv (refused if the worktree is dirty).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pd.anchoring import AnchorConfig
from pd.backbones import ENVIRONMENTS, build_embedder
from pd.data import SchemaConfig, load_dataset
from pd.pilot import PilotConfig, sweep
from pd.runlog import log_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--out", default="outputs/pilot")
    ap.add_argument("--backbones", nargs="+", help="override the config's backbone list")
    ap.add_argument("--exp-id", default="P1", help="row in experiments/registry.csv")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="throwaway run; its numbers must never reach the paper")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    schema = SchemaConfig(**cfg["data"]["schema"])
    ds = load_dataset(cfg["data"]["root"], schema)

    print(f"loaded {len(ds)} acquisitions, {len(ds.classes)} classes: {ds.classes}")
    print(ds.summary().to_string(index=False))

    pilot_kw = dict(cfg["pilot"])
    pilot_kw["anchor"] = AnchorConfig(**pilot_kw.pop("anchor"))
    if args.backbones:
        pilot_kw["backbones"] = args.backbones
    for key in ("backbones", "interfaces", "shots", "seeds"):
        pilot_kw[key] = tuple(pilot_kw[key])
    pconf = PilotConfig(**pilot_kw)

    wrong_env = [b for b in pconf.backbones if b not in ENVIRONMENTS]
    if wrong_env:
        raise SystemExit(f"unknown backbones: {wrong_env}")

    out = Path(args.out)
    rows = sweep(ds, pconf, build_embedder, out)

    table = pd.DataFrame(rows)
    table.to_csv(out / "results.csv", index=False)

    ok = table[table["status"] == "ok"]
    if ok.empty:
        print("\nno cell completed; see results.csv for the skip reasons")
        return

    print("\nmacro-F1, mean over seeds and folds:")
    print(
        ok.pivot_table(index=["backbone", "interface"], columns="shots", values="macro_f1")
        .round(3)
        .to_string()
    )

    for (backbone, interface), grp in ok.groupby(["backbone", "interface"]):
        for shots, cell in grp.groupby("shots"):
            log_run(
                exp_id=args.exp_id,
                method=f"{backbone}/{interface}",
                backbone=backbone,
                input_interface=interface,
                adaptation="linear probe",
                shots_per_class=shots,
                seed="；".join(str(s) for s in sorted(cell["seed"])),
                split_file=str(out / "results.csv"),
                metrics={
                    "macro_f1": round(cell["macro_f1"].mean(), 4),
                    "macro_f1_std": round(cell["macro_f1"].std(ddof=0), 4),
                    "accuracy": round(cell["accuracy"].mean(), 4),
                    "n_seeds": len(cell),
                },
                output_dir=out,
                config={"data": cfg["data"], "pilot": {**cfg["pilot"], "backbones": list(pconf.backbones)}},
                notes=f"split={pconf.split_scheme} group_by={pconf.group_by}",
                allow_dirty=args.allow_dirty,
            )
    print(f"\nlogged to experiments/runs.csv; predictions under {out}")


if __name__ == "__main__":
    main()
