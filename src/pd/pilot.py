"""Pilot evaluation: the label-efficiency sweep behind gate G1.

For each (backbone, interface, shots, seed) it freezes the features, fits a
linear probe on the few-shot support set and scores the held-out specimens.
Results feed the G1 decision table in docs/plan.md, which picks narrative
A, B or C.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .anchoring import AnchorConfig, anchor
from .data import Dataset
from .splits import Split, few_shot_support, grouped_k_fold, leave_one_group_out, random_window_split

__all__ = ["PilotConfig", "build_inputs", "evaluate", "sweep"]


@dataclass
class PilotConfig:
    backbones: tuple[str, ...] = ("minirocket", "mantis_v1")
    interfaces: tuple[str, ...] = ("anchored",)
    """raw: pulse record rendered without phase alignment (the ablation baseline).
    anchored: phase-binned. anchored+aux: phase-binned with amplitude re-injected."""
    shots: tuple[int, ...] = (5, 10, 20, 0)
    """0 means all training labels."""
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    split_scheme: str = "leave-one-group-out"
    """leave-one-group-out, grouped-k-fold, or random-window-LEAKY (leakage experiment only)."""
    n_folds: int = 5
    group_by: str = "specimen"
    """specimen, or batch when specimens are too few (risk R1)."""
    anchor: AnchorConfig = AnchorConfig()


def build_inputs(ds: Dataset, interface: str, cfg: AnchorConfig) -> np.ndarray:
    """Render pulse records into model input under the given interface."""
    x, aux = anchor(ds.records, cfg)

    if interface == "anchored":
        return x
    if interface == "anchored+aux":
        # Broadcast the absolute statistics as extra constant channels so a
        # frozen backbone's instance normalisation cannot remove them.
        pad = np.repeat(aux[:, :, None], x.shape[-1], axis=-1).astype(np.float32)
        return np.concatenate([x, pad], axis=1)
    if interface == "raw":
        # Phase information deliberately destroyed: each record's bins are
        # rotated by an arbitrary amount, as happens when the acquisition window
        # is not tied to the voltage zero crossing.
        rng = np.random.default_rng(0)
        out = x.copy()
        for i in range(len(out)):
            out[i] = np.roll(out[i], int(rng.integers(cfg.seq_len)), axis=-1)
        return out
    raise ValueError(f"unknown interface {interface!r}")


def make_splits(ds: Dataset, cfg: PilotConfig) -> list[Split]:
    groups = ds.specimen if cfg.group_by == "specimen" else ds.batch
    if cfg.split_scheme == "leave-one-group-out":
        return leave_one_group_out(groups, ds.y)
    if cfg.split_scheme == "grouped-k-fold":
        return grouped_k_fold(groups, n_splits=cfg.n_folds)
    if cfg.split_scheme == "random-window-LEAKY":
        return [random_window_split(len(ds), seed=s) for s in range(cfg.n_folds)]
    raise ValueError(f"unknown split_scheme {cfg.split_scheme!r}")


def evaluate(z: np.ndarray, y: np.ndarray, split: Split, support: np.ndarray) -> dict:
    """Linear probe on the support set, scored on the held-out specimens."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(z[support], y[support])
    pred = clf.predict(z[split.test])
    return {
        "macro_f1": float(f1_score(y[split.test], pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y[split.test], pred)),
        "n_support": int(len(support)),
        "n_test": int(len(split.test)),
        "y_true": y[split.test].tolist(),
        "y_pred": pred.tolist(),
        "test_specimens": sorted({str(g) for g in split.held_out_groups}),
    }


def sweep(ds: Dataset, cfg: PilotConfig, make_embedder, out_dir: str | Path) -> "list[dict]":
    """Run the grid and write per-sample predictions. Returns one row per cell.

    `make_embedder(name)` must return a fresh `Embedder`; it is called once per
    (backbone, interface) because a fitted extractor is tied to the channel count
    of the interface it saw. An extractor with `needs_fit` is refit on each
    fold's training split, so no test-fold statistics reach its features.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = make_splits(ds, cfg)
    groups = ds.specimen if cfg.group_by == "specimen" else ds.batch
    rows = []

    for interface in cfg.interfaces:
        x = build_inputs(ds, interface, cfg.anchor)
        for name in cfg.backbones:
            embedder = make_embedder(name)
            t0 = time.time()
            # A pretrained backbone is frozen, so its features are fold-agnostic
            # and computed once. A fitted extractor gets one z per fold.
            z_all = None if embedder.needs_fit else embedder(x)
            z_by_fold = {}
            if embedder.needs_fit:
                for split in splits:
                    embedder.fit(x[split.train])
                    z_by_fold[split.fold] = embedder(x)
            embed_s = time.time() - t0
            embed_dim = (z_all if z_all is not None else z_by_fold[splits[0].fold]).shape[1]

            for shots in cfg.shots:
                for seed in cfg.seeds:
                    per_fold = []
                    for split in splits:
                        try:
                            support = few_shot_support(ds.y, groups, split, shots, seed=seed)
                        except ValueError as e:  # not enough labels in this fold
                            rows.append(
                                {
                                    "backbone": name, "interface": interface, "shots": shots,
                                    "seed": seed, "fold": split.fold, "status": "skipped",
                                    "reason": str(e)[:200],
                                }
                            )
                            continue
                        z = z_all if z_all is not None else z_by_fold[split.fold]
                        per_fold.append(evaluate(z, ds.y, split, support))

                    if not per_fold:
                        continue
                    tag = f"{name}_{interface}_shots{shots}_seed{seed}"
                    (out_dir / f"pred_{tag}.json").write_text(
                        json.dumps(
                            {"config": {**asdict(cfg.anchor), "interface": interface,
                                        "backbone": name, "shots": shots, "seed": seed,
                                        "refit_per_fold": embedder.needs_fit},
                             "folds": per_fold},
                            indent=2,
                        )
                    )
                    rows.append(
                        {
                            "backbone": name,
                            "interface": interface,
                            "shots": shots,
                            "seed": seed,
                            "embed_dim": int(embed_dim),
                            "embed_s": round(embed_s, 2),
                            "refit_per_fold": embedder.needs_fit,
                            "macro_f1": float(np.mean([f["macro_f1"] for f in per_fold])),
                            "accuracy": float(np.mean([f["accuracy"] for f in per_fold])),
                            "n_folds": len(per_fold),
                            "status": "ok",
                        }
                    )
    return rows
