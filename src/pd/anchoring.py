"""The measurement-anchored input interface -- the paper's core contribution.

Two calibrated reference quantities of a PD measurement are lost when a pulse
record is fed to a general time-series foundation model:

* **Power-frequency phase.** A raw record is far longer than any backbone's
  context (one 50 Hz cycle at 10 MS/s is 2e5 points), so resampling to 512 is
  unavoidable -- and resampling a record whose window start is not tied to the
  voltage zero crossing destroys phase alignment. Binning by phase instead
  compresses the record *and* fixes each patch to a known phase window.
* **Discharge magnitude.** MOMENT and the Chronos family normalise per instance,
  so embeddings are invariant to amplitude (measured: cos(x, 10x) = 1.000, and a
  probe separating two amplitude-only classes scores at chance). Apparent charge
  is the calibrated quantity of IEC 60270, so it is re-injected alongside the
  embedding rather than left to the backbone.

Both knobs that the E8 robustness experiment sweeps live here: `n_bins` (phase
resolution, after Mas'ud et al. 2016, who showed statistical fingerprints change
with bin width) and `phase_offset_deg` (phase-reference error -- the same defect
sits in different phase windows on UHF vs HFCT, per Kong et al. 2022).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["AnchorConfig", "PulseRecord", "anchor", "detect_pulses"]


@dataclass(frozen=True)
class AnchorConfig:
    n_bins: int = 128
    """Phase bins per power-frequency cycle. 128 bins = 2.8125 deg."""

    n_cycles: int = 4
    """Cycles per model input. n_bins * n_cycles must match the backbone context
    (512 for Mantis/MOMENT; Mantis additionally needs a multiple of 32)."""

    phase_offset_deg: float = 0.0
    """Rotates the phase reference. Non-zero only in the E8 offset test."""

    log_amplitude: bool = True
    """Compress amplitude channels as log1p. PD magnitudes span decades."""

    half_cycle_symmetry: bool = False
    """Append the half-cycle-shifted amplitude channel, making the +/- half-cycle
    relation explicit (the inductive bias behind Gulski's cross-correlation
    operators)."""

    def __post_init__(self):
        if self.n_bins <= 0 or self.n_cycles <= 0:
            raise ValueError("n_bins and n_cycles must be positive")

    @property
    def seq_len(self) -> int:
        return self.n_bins * self.n_cycles

    @property
    def bin_width_deg(self) -> float:
        return 360.0 / self.n_bins


@dataclass(frozen=True)
class PulseRecord:
    """Canonical form every loader produces: one acquisition's pulses.

    phase_deg : pulse phase within the power-frequency cycle, [0, 360)
    amplitude : calibrated magnitude (pC or mV); sign carries the half cycle
    cycle     : index of the cycle each pulse falls in, 0-based
    """

    phase_deg: np.ndarray
    amplitude: np.ndarray
    cycle: np.ndarray
    n_cycles_recorded: int

    def __post_init__(self):
        n = len(self.phase_deg)
        if not (len(self.amplitude) == len(self.cycle) == n):
            raise ValueError("phase_deg, amplitude and cycle must have equal length")


def anchor(records: list[PulseRecord], cfg: AnchorConfig = AnchorConfig()) -> tuple[np.ndarray, np.ndarray]:
    """Turn pulse records into a backbone input plus the re-injected statistics.

    Returns
    -------
    x : (n_records, n_channels, seq_len) float32
        Channel 0 is the per-bin peak magnitude, channel 1 the per-bin pulse
        count; channel 2 is the half-cycle-shifted peak when enabled. The
        sequence runs cycle-major, so patch boundaries land on phase boundaries.
    aux : (n_records, 4) float32
        Absolute magnitude statistics the backbone's instance normalisation would
        otherwise discard: log peak, log mean, log total charge, pulse count.
    """
    x = np.zeros((len(records), 2, cfg.seq_len), dtype=np.float32)
    aux = np.zeros((len(records), 4), dtype=np.float32)

    for i, rec in enumerate(records):
        mag = np.abs(rec.amplitude)
        # Phase bin, with the reference rotated by the (test-time) offset.
        phase = (rec.phase_deg + cfg.phase_offset_deg) % 360.0
        b = np.minimum((phase / cfg.bin_width_deg).astype(np.int64), cfg.n_bins - 1)
        # Cycles wrap so that a record longer than n_cycles folds onto itself
        # rather than being truncated -- every pulse contributes.
        slot = (rec.cycle % cfg.n_cycles) * cfg.n_bins + b

        np.maximum.at(x[i, 0], slot, mag)
        np.add.at(x[i, 1], slot, 1.0)

        if len(mag):
            aux[i] = [mag.max(), mag.mean(), mag.sum(), len(mag)]

    if cfg.log_amplitude:
        x[:, 0] = np.log1p(x[:, 0])
        aux[:, :3] = np.log1p(aux[:, :3])
    aux[:, 3] = np.log1p(aux[:, 3])

    if cfg.half_cycle_symmetry:
        shifted = np.roll(x[:, 0].reshape(len(records), cfg.n_cycles, cfg.n_bins), cfg.n_bins // 2, axis=-1)
        x = np.concatenate([x, shifted.reshape(len(records), 1, cfg.seq_len)], axis=1)

    return x, aux


def detect_pulses(
    waveform: np.ndarray,
    fs: float,
    dead_time_s: float,
    f_line: float = 50.0,
    threshold: float | None = None,
    noise_sigmas: float = 5.0,
    phase_at_t0_deg: float = 0.0,
) -> PulseRecord:
    """Extract pulses from a raw record when no pulse list is supplied.

    Threshold defaults to `noise_sigmas` times a robust noise estimate
    (1.4826 * MAD).

    `dead_time_s` is required on purpose: it must exceed the ringing duration of
    a single discharge in *this* measurement chain, or one damped oscillation is
    counted several times and channel 1 of the interface is silently corrupted.
    Read it off a few averaged pulses from the data and record it in the data
    card -- it depends on sensor band and propagation, so there is no safe
    default.

    `phase_at_t0_deg` is the voltage phase at the first sample. It must come from
    a synchronous reference channel; estimating it from the record itself is a
    fallback that has to be reported separately (risk R2).
    """
    if dead_time_s <= 0:
        raise ValueError("dead_time_s must be positive and cover one pulse's ringing")
    if waveform.ndim != 1:
        raise ValueError(f"waveform must be 1-D, got shape {waveform.shape}")
    if threshold is None:
        mad = np.median(np.abs(waveform - np.median(waveform)))
        threshold = noise_sigmas * 1.4826 * mad
        if threshold <= 0:  # a constant or near-constant record
            threshold = np.inf

    mag = np.abs(waveform)
    dead = max(1, int(round(dead_time_s * fs)))
    candidates = np.flatnonzero(mag > threshold)

    peaks = []
    i = 0
    while i < len(candidates):
        start = candidates[i]
        # Peak within the dead-time window starting at this crossing.
        window = mag[start : start + dead]
        peaks.append(start + int(np.argmax(window)))
        i = np.searchsorted(candidates, start + dead)

    peaks = np.asarray(peaks, dtype=np.int64)
    t = peaks / fs
    phase = (phase_at_t0_deg + 360.0 * f_line * t) % 360.0
    cycle = np.floor(f_line * t + phase_at_t0_deg / 360.0).astype(np.int64)
    return PulseRecord(
        phase_deg=phase,
        amplitude=waveform[peaks],
        cycle=cycle - cycle.min() if len(cycle) else cycle,
        n_cycles_recorded=max(1, int(np.ceil(f_line * len(waveform) / fs))),
    )
