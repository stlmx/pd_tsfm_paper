"""Representation-property probe on synthetic one-cycle pulse trains (CPU).

Two binary tasks, each differing in one factor only:
  phase     : identical pulse statistics, clustered around 45 deg vs 225 deg of the cycle
  amplitude : identical phase distribution, pulse amplitude (and noise floor) scaled 1x vs 3x
Frozen embeddings -> logistic regression, 5-fold stratified CV accuracy.
Chance level is 0.5. This probes what a backbone's embedding keeps; it is not a
PD recognition result and must not be reported as one.

Usage:
    .venv-main/bin/python scripts/probe_phase_amplitude.py --models mantis_v1 mantis_v2 chronos_bolt chronos2 minirocket
    .venv-moment/bin/python scripts/probe_phase_amplitude.py --models moment_small moment_base moment_large
"""

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from smoke_backbones import OUT_DIR, build

L, N_PER_CLASS = 512, 100
KERNEL = np.exp(-np.arange(40) / 6.0) * np.sin(2 * np.pi * np.arange(40) / 8.0)


def pulse_train(rng, center_deg, amp_scale):
    x = 0.01 * amp_scale * rng.standard_normal(L)
    for _ in range(rng.integers(4, 10)):
        phase = (center_deg + rng.normal(0, 20)) % 360
        pos = min(int(phase / 360 * L), L - 40)
        x[pos:pos + 40] += amp_scale * rng.uniform(0.3, 1.0) * KERNEL
    return x


def make_task(task, seed=0):
    rng = np.random.default_rng(seed)
    if task == "phase":
        a = [pulse_train(rng, 45, 1.0) for _ in range(N_PER_CLASS)]
        b = [pulse_train(rng, 225, 1.0) for _ in range(N_PER_CLASS)]
    else:
        a = [pulse_train(rng, 90, 1.0) for _ in range(N_PER_CLASS)]
        b = [pulse_train(rng, 90, 3.0) for _ in range(N_PER_CLASS)]
    x = np.stack(a + b).astype(np.float32)
    y = np.array([0] * N_PER_CLASS + [1] * N_PER_CLASS)
    return x, y


def probe(z, y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    return float(np.mean(cross_val_score(clf, z.reshape(len(z), -1), y, cv=cv)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = {t: make_task(t) for t in ("phase", "amplitude")}

    for name in args.models:
        # "raw": logistic regression on the signal itself, a reference for what is linearly recoverable
        embed = (lambda x: x) if name == "raw" else build(name)[0]
        rec = {"model": name}
        for t, (x, y) in tasks.items():
            z = np.concatenate([np.asarray(embed(x[i:i + 50])) for i in range(0, len(x), 50)])
            rec[f"{t}_acc"] = round(probe(z, y), 3)
        print(json.dumps(rec))
        (OUT_DIR / f"probe_{name}.json").write_text(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
