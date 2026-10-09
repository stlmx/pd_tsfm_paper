"""Write a specimen-structured synthetic PD dataset in the pulse_list layout.

Purpose: prove the pipeline end to end before the lab CSVs arrive, and give the
tests something shaped like the real thing. The defect classes imitate published
phase behaviour qualitatively (void near the half-cycle peaks, surface discharge
broader, floating electrode positive-dominant, free particle weakly correlated
with phase) -- it is a pipeline fixture, NOT a model of PD physics, and no
number measured on it may appear in the paper.

Per-specimen jitter in phase centre and amplitude scale is what makes a grouped
split meaningfully harder than a random one.

    python scripts/make_synthetic_data.py --out /tmp/pd_synth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# (phase centres in degrees, spread, mean pulses per cycle, amplitude scale in pC)
DEFECTS = {
    "void": ([75.0, 255.0], 12.0, 6, 20.0),
    "surface": ([110.0, 290.0], 35.0, 10, 45.0),
    "floating": ([60.0, 240.0], 8.0, 4, 90.0),
    "particle": ([0.0], 180.0, 3, 12.0),
}


def one_acquisition(rng, spec, n_cycles, phase_jitter, amp_scale):
    centres, spread, rate, base_amp = spec
    phase, amp, cycle = [], [], []
    for c in range(n_cycles):
        for centre in centres:
            for _ in range(rng.poisson(rate / len(centres))):
                phase.append((centre + phase_jitter + rng.normal(0, spread)) % 360.0)
                amp.append(abs(rng.lognormal(np.log(base_amp * amp_scale), 0.4)))
                cycle.append(c)
    return pd.DataFrame({"phase_deg": phase, "q_pc": amp, "cycle": cycle})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--specimens", type=int, default=4, help="independent specimens per defect")
    ap.add_argument("--acquisitions", type=int, default=6, help="acquisitions per specimen")
    ap.add_argument("--cycles", type=int, default=4, help="power-frequency cycles per acquisition")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    rng = np.random.default_rng(args.seed)
    n = 0
    for defect, spec in DEFECTS.items():
        for s in range(args.specimens):
            # Fixed per specimen: this is the shift a grouped split must survive.
            jitter = rng.normal(0, 10.0)
            amp_scale = rng.lognormal(0, 0.25)
            for a in range(args.acquisitions):
                df = one_acquisition(rng, spec, args.cycles, jitter, amp_scale)
                path = out / defect / f"S{s}_b{a // 3}_a{a}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(path, index=False)
                n += 1

    print(f"wrote {n} acquisitions to {out}")
    print(f"{len(DEFECTS)} defects x {args.specimens} specimens x {args.acquisitions} acquisitions")


if __name__ == "__main__":
    main()
