"""Schema-driven CSV loader producing the canonical pulse representation.

The data card is not filled in yet, so the layout is declared in config rather
than hard-coded. Three layouts are supported, covering the plausible answers to
"what does one CSV correspond to":

* ``pulse_list``   -- rows are pulses, with phase and amplitude columns.
* ``raw_waveform`` -- one column is a sampled record; pulses are detected.
* ``prpd_matrix``  -- a phase x amplitude histogram, read back as binned pulses.

Every layout yields `PulseRecord`s plus aligned class / specimen / batch labels,
so `splits.py` and `anchoring.py` never see the raw file format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .anchoring import PulseRecord, detect_pulses

__all__ = ["SchemaConfig", "Dataset", "load_dataset"]


@dataclass
class SchemaConfig:
    layout: str = "pulse_list"
    """One of pulse_list, raw_waveform, prpd_matrix."""

    glob: str = "**/*.csv"
    """Pattern under the data root. One matched file is one acquisition."""

    filename_pattern: str | None = None
    """Regex with named groups over the path relative to the data root, e.g.
    r"(?P<defect>[^/]+)/(?P<specimen>S\\d+)_(?P<batch>b\\d+)". Groups named
    defect, specimen and batch become the class, group and session labels.
    Leave None when the labels live in columns instead."""

    columns: dict[str, str] = field(default_factory=dict)
    """Maps roles to column names. pulse_list needs phase and amplitude, and
    optionally cycle; raw_waveform needs signal and optionally phase_ref;
    prpd_matrix needs phase, amplitude and count. Label roles defect, specimen
    and batch may also be given here."""

    fs: float | None = None
    """Sampling rate, raw_waveform only."""

    f_line: float = 50.0

    dead_time_s: float | None = None
    """Pulse ringing duration, raw_waveform only. Required -- see detect_pulses."""

    noise_sigmas: float = 5.0

    phase_at_t0_deg: float = 0.0
    """Voltage phase at the first sample. Use the synchronous reference channel
    where one exists; a fixed value here means the phase reference is assumed,
    which must be reported separately (risk R2)."""

    qualify_specimen_with_defect: bool = False
    """Set True when specimen IDs are only unique within a defect class (e.g.
    every defect numbers its specimens from 1). The defect name is then prefixed
    so distinct specimens stay distinct groups. Leave False when IDs are globally
    unique. A collision with this flag off is an error, not a silent merge."""

    def validate(self) -> None:
        required = {
            "pulse_list": ("phase", "amplitude"),
            "raw_waveform": ("signal",),
            "prpd_matrix": ("phase", "amplitude", "count"),
        }
        if self.layout not in required:
            raise ValueError(f"unknown layout {self.layout!r}; expected one of {sorted(required)}")
        missing = [r for r in required[self.layout] if r not in self.columns]
        if missing:
            raise ValueError(f"layout {self.layout!r} needs columns for {missing}")
        if self.layout == "raw_waveform":
            if not self.fs:
                raise ValueError("raw_waveform layout needs fs")
            if not self.dead_time_s:
                raise ValueError("raw_waveform layout needs dead_time_s (see detect_pulses)")


@dataclass
class Dataset:
    records: list[PulseRecord]
    y: np.ndarray
    """Integer class labels, index into `classes`."""
    classes: list[str]
    specimen: np.ndarray
    """Grouping key for splits -- the physical defect instance."""
    batch: np.ndarray
    """Test session. Use as the grouping key when specimens are too few (R1)."""
    source: list[Path]

    def __len__(self) -> int:
        return len(self.records)

    def summary(self) -> pd.DataFrame:
        """Per-class counts for Table 1 of the paper and for gate G0."""
        df = pd.DataFrame(
            {
                "defect": [self.classes[i] for i in self.y],
                "specimen": self.specimen,
                "batch": self.batch,
                "n_pulses": [len(r.phase_deg) for r in self.records],
            }
        )
        return (
            df.groupby("defect")
            .agg(
                n_acquisitions=("specimen", "size"),
                n_specimens=("specimen", "nunique"),
                n_batches=("batch", "nunique"),
                pulses_median=("n_pulses", "median"),
                pulses_min=("n_pulses", "min"),
                pulses_max=("n_pulses", "max"),
            )
            .reset_index()
        )


def _labels_from_path(path: Path, root: Path, pattern: str) -> dict[str, str]:
    rel = path.relative_to(root).as_posix()
    m = re.search(pattern, rel)
    if not m:
        raise ValueError(f"filename_pattern did not match {rel!r}")
    return m.groupdict()


def _record_from_frame(df: pd.DataFrame, cfg: SchemaConfig) -> PulseRecord:
    col = cfg.columns
    if cfg.layout == "pulse_list":
        phase = df[col["phase"]].to_numpy(float) % 360.0
        amp = df[col["amplitude"]].to_numpy(float)
        cycle = (
            df[col["cycle"]].to_numpy(np.int64)
            if "cycle" in col
            else np.zeros(len(phase), dtype=np.int64)
        )
        return PulseRecord(
            phase,
            amp,
            cycle - cycle.min() if len(cycle) else cycle,
            int(np.ptp(cycle)) + 1 if len(cycle) else 1,  # np.ptp(): ndarray.ptp() is gone in NumPy 2
        )

    if cfg.layout == "raw_waveform":
        phase0 = cfg.phase_at_t0_deg
        if "phase_ref" in col:
            phase0 = _phase_from_reference(df[col["phase_ref"]].to_numpy(float), cfg)
        return detect_pulses(
            df[col["signal"]].to_numpy(float),
            fs=cfg.fs,
            dead_time_s=cfg.dead_time_s,
            f_line=cfg.f_line,
            noise_sigmas=cfg.noise_sigmas,
            phase_at_t0_deg=phase0,
        )

    # prpd_matrix: expand each populated cell back into `count` pulses so the
    # downstream interface sees one representation regardless of input layout.
    phase = df[col["phase"]].to_numpy(float) % 360.0
    amp = df[col["amplitude"]].to_numpy(float)
    count = df[col["count"]].to_numpy(float)
    keep = count > 0
    reps = count[keep].astype(np.int64)
    return PulseRecord(
        np.repeat(phase[keep], reps),
        np.repeat(amp[keep], reps),
        np.zeros(int(reps.sum()), dtype=np.int64),
        1,
    )


def _phase_from_reference(ref: np.ndarray, cfg: SchemaConfig) -> float:
    """Voltage phase at the first sample, from a synchronous reference channel.

    Takes the phase of the line-frequency component via a single-bin DFT, which
    is robust to the harmonics that Florkowski 2022 shows affect phase-resolved
    interpretation.
    """
    n = len(ref)
    t = np.arange(n) / cfg.fs
    comp = np.sum((ref - ref.mean()) * np.exp(-2j * np.pi * cfg.f_line * t))
    # A sine through zero rising at t=0 has PD phase 0, hence the +90 deg shift.
    return float(np.degrees(np.angle(comp)) + 90.0) % 360.0


def _resolve_specimen_keys(specimens: list[str], defects: list[str], cfg: SchemaConfig) -> np.ndarray:
    """Make the grouping key unambiguous, or refuse to guess.

    A specimen label that appears under more than one defect class is either
    (a) IDs numbered per class, so two distinct specimens collide, or (b) one
    physical specimen that genuinely carries several defect classes. Merging the
    first gives wrong folds; splitting the second leaks the same specimen across
    train and test. The two cannot be told apart from the files, so this asks.
    """
    if cfg.qualify_specimen_with_defect:
        return np.array([f"{d}/{s}" for d, s in zip(defects, specimens)])

    shared = sorted(
        {s for s in set(specimens) if len({d for d, t in zip(defects, specimens) if t == s}) > 1}
    )
    if shared:
        raise ValueError(
            f"specimen label(s) {shared[:5]} appear under more than one defect class.\n"
            "If specimen IDs are numbered per defect (so these are different physical "
            "specimens), set qualify_specimen_with_defect=True.\n"
            "If one physical specimen really carries several defect classes, rename the "
            "labels so that shared specimen explicitly stays one group -- it must never "
            "be split across train and test."
        )
    return np.array(specimens)


def load_dataset(root: str | Path, cfg: SchemaConfig) -> Dataset:
    """Read every matching CSV under `root` into one Dataset."""
    cfg.validate()
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"data root {root} does not exist")

    paths = sorted(root.glob(cfg.glob))
    if not paths:
        raise FileNotFoundError(f"no files matched {cfg.glob!r} under {root}")

    records, defects, specimens, batches, kept = [], [], [], [], []
    for path in paths:
        df = pd.read_csv(path)
        labels = (
            _labels_from_path(path, root, cfg.filename_pattern) if cfg.filename_pattern else {}
        )
        for role in ("defect", "specimen", "batch"):
            if role in cfg.columns:
                labels[role] = str(df[cfg.columns[role]].iloc[0])
        if "defect" not in labels:
            raise ValueError(
                f"no class label for {path}: give filename_pattern a 'defect' group "
                "or name a defect column"
            )

        records.append(_record_from_frame(df, cfg))
        defects.append(labels["defect"])
        # Without an explicit specimen the file itself is the finest group we can
        # defend; that is weaker than a real specimen label, so warn loudly.
        specimens.append(labels.get("specimen", path.stem))
        batches.append(labels.get("batch", specimens[-1]))
        kept.append(path)

    classes = sorted(set(defects))
    specimens = _resolve_specimen_keys(specimens, defects, cfg)

    if "specimen" not in cfg.columns and (cfg.filename_pattern or "").find("specimen") < 0:
        print(
            "WARNING: no specimen label found; falling back to one group per file. "
            "Windows from one acquisition then leak across the split -- fill in the "
            "data card and set a specimen label before trusting any number."
        )

    return Dataset(
        records=records,
        y=np.array([classes.index(d) for d in defects]),
        classes=classes,
        specimen=np.array(specimens),
        batch=np.array(batches),
        source=kept,
    )
