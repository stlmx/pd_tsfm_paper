"""Checks that the interface actually preserves what the backbones discard."""

import unittest

import numpy as np

from pd.anchoring import AnchorConfig, PulseRecord, anchor, detect_pulses

# The synthetic pulse below rings for 40 samples at 10 MS/s = 4 us.
PULSE_RINGING_S = 5e-6


def record(phases, amps=None, cycles=None, n_cycles=4):
    phases = np.asarray(phases, dtype=float)
    return PulseRecord(
        phase_deg=phases,
        amplitude=np.ones_like(phases) if amps is None else np.asarray(amps, dtype=float),
        cycle=np.zeros(len(phases), dtype=int) if cycles is None else np.asarray(cycles),
        n_cycles_recorded=n_cycles,
    )


class TestPhaseBinning(unittest.TestCase):
    def test_output_shape_matches_backbone_context(self):
        cfg = AnchorConfig(n_bins=128, n_cycles=4)
        self.assertEqual(cfg.seq_len, 512)
        x, aux = anchor([record([45.0]), record([225.0])], cfg)
        self.assertEqual(x.shape, (2, 2, 512))
        self.assertEqual(aux.shape, (2, 4))
        self.assertEqual(x.dtype, np.float32)

    def test_pulse_lands_in_the_bin_its_phase_implies(self):
        cfg = AnchorConfig(n_bins=360, n_cycles=1)  # 1 deg per bin
        x, _ = anchor([record([90.5])], cfg)
        self.assertEqual(int(np.argmax(x[0, 1])), 90)

    def test_phase_360_wraps_to_bin_zero(self):
        cfg = AnchorConfig(n_bins=360, n_cycles=1)
        x, _ = anchor([record([360.0])], cfg)
        self.assertEqual(int(np.argmax(x[0, 1])), 0)

    def test_counts_and_peaks_are_separate_channels(self):
        cfg = AnchorConfig(n_bins=360, n_cycles=1, log_amplitude=False)
        x, _ = anchor([record([90.1, 90.2, 90.3], amps=[1.0, 5.0, 2.0])], cfg)
        self.assertEqual(x[0, 1, 90], 3.0)  # three pulses counted
        self.assertEqual(x[0, 0, 90], 5.0)  # peak, not sum

    def test_different_phase_clusters_give_different_inputs(self):
        # The discriminative case: same statistics, different phase.
        cfg = AnchorConfig()
        x, _ = anchor([record([45.0] * 5), record([225.0] * 5)], cfg)
        self.assertFalse(np.allclose(x[0], x[1]))

    def test_cycles_beyond_n_cycles_fold_rather_than_truncate(self):
        cfg = AnchorConfig(n_bins=10, n_cycles=2, log_amplitude=False)
        x, _ = anchor([record([0.0, 0.0, 0.0], cycles=[0, 2, 4])], cfg)
        self.assertEqual(x[0, 1, 0], 3.0, "pulses from later cycles were dropped")

    def test_empty_record_is_all_zeros_not_an_error(self):
        x, aux = anchor([record([])], AnchorConfig())
        self.assertEqual(x.sum(), 0.0)
        self.assertEqual(aux.sum(), 0.0)


class TestAmplitudeReinjection(unittest.TestCase):
    def test_amplitude_scaling_changes_aux(self):
        # The gap this closes: MOMENT/Chronos embeddings are invariant to this.
        cfg = AnchorConfig()
        _, aux = anchor([record([90.0], amps=[1.0]), record([90.0], amps=[10.0])], cfg)
        self.assertFalse(np.allclose(aux[0], aux[1]))

    def test_aux_columns_are_peak_mean_total_count(self):
        cfg = AnchorConfig(log_amplitude=False)
        _, aux = anchor([record([10.0, 20.0], amps=[2.0, 4.0])], cfg)
        np.testing.assert_allclose(aux[0, :3], [4.0, 3.0, 6.0])
        np.testing.assert_allclose(aux[0, 3], np.log1p(2.0))

    def test_aux_uses_absolute_magnitude_across_half_cycles(self):
        cfg = AnchorConfig(log_amplitude=False)
        _, aux = anchor([record([10.0, 200.0], amps=[3.0, -3.0])], cfg)
        np.testing.assert_allclose(aux[0, 0], 3.0)


class TestRobustnessKnobs(unittest.TestCase):
    def test_phase_offset_rotates_the_pattern(self):
        cfg = AnchorConfig(n_bins=360, n_cycles=1)
        base, _ = anchor([record([90.0])], cfg)
        rotated, _ = anchor([record([90.0])], AnchorConfig(n_bins=360, n_cycles=1, phase_offset_deg=30.0))
        self.assertEqual(int(np.argmax(base[0, 1])), 90)
        self.assertEqual(int(np.argmax(rotated[0, 1])), 120)

    def test_offset_wraps_past_360(self):
        cfg = AnchorConfig(n_bins=360, n_cycles=1, phase_offset_deg=300.0)
        x, _ = anchor([record([90.0])], cfg)
        self.assertEqual(int(np.argmax(x[0, 1])), 30)

    def test_coarser_bins_preserve_total_count(self):
        # The cross-resolution test rebins the same pulses; none may be lost.
        pulses = record(np.linspace(0, 359, 50))
        for n_bins in (360, 128, 64, 32):
            _, _ = anchor([pulses], AnchorConfig(n_bins=n_bins, n_cycles=1))
            x, _ = anchor([pulses], AnchorConfig(n_bins=n_bins, n_cycles=1))
            self.assertEqual(x[0, 1].sum(), 50.0, f"n_bins={n_bins} lost pulses")

    def test_half_cycle_symmetry_adds_a_shifted_channel(self):
        cfg = AnchorConfig(n_bins=360, n_cycles=1, half_cycle_symmetry=True)
        x, _ = anchor([record([90.0])], cfg)
        self.assertEqual(x.shape[1], 3)
        self.assertEqual(int(np.argmax(x[0, 2])), 270)

    def test_rejects_nonpositive_geometry(self):
        with self.assertRaises(ValueError):
            AnchorConfig(n_bins=0)


class TestPulseDetection(unittest.TestCase):
    def setUp(self):
        self.fs = 10e6
        self.f_line = 50.0
        n = int(self.fs / self.f_line)  # exactly one 50 Hz cycle
        self.rng = np.random.default_rng(0)
        self.wave = 0.001 * self.rng.standard_normal(n)
        self.n = n

    def _inject(self, phase_deg, amp):
        pos = int(phase_deg / 360.0 * self.n)
        t = np.arange(40)
        self.wave[pos : pos + 40] += amp * np.exp(-t / 6.0) * np.sin(2 * np.pi * t / 8.0)
        return pos

    def test_recovers_injected_pulse_phase(self):
        self._inject(90.0, 1.0)
        rec = detect_pulses(self.wave, self.fs, dead_time_s=PULSE_RINGING_S, f_line=self.f_line)
        self.assertEqual(len(rec.phase_deg), 1)
        self.assertAlmostEqual(rec.phase_deg[0], 90.0, delta=2.0)

    def test_dead_time_counts_a_ringing_pulse_once(self):
        self._inject(90.0, 1.0)
        rec = detect_pulses(self.wave, self.fs, dead_time_s=PULSE_RINGING_S, f_line=self.f_line)
        self.assertEqual(len(rec.phase_deg), 1, "ringing tail re-triggered the detector")

    def test_separates_two_pulses_in_different_half_cycles(self):
        self._inject(45.0, 1.0)
        self._inject(225.0, 1.0)
        rec = detect_pulses(self.wave, self.fs, dead_time_s=PULSE_RINGING_S, f_line=self.f_line)
        self.assertEqual(len(rec.phase_deg), 2)
        np.testing.assert_allclose(np.sort(rec.phase_deg), [45.0, 225.0], atol=2.0)

    def test_noise_only_record_yields_no_pulses(self):
        rec = detect_pulses(self.wave, self.fs, dead_time_s=PULSE_RINGING_S, f_line=self.f_line)
        self.assertEqual(len(rec.phase_deg), 0)

    def test_rejects_nonpositive_dead_time(self):
        with self.assertRaisesRegex(ValueError, "dead_time_s"):
            detect_pulses(self.wave, self.fs, dead_time_s=0.0)

    def test_too_short_dead_time_double_counts_one_pulse(self):
        # Why dead_time_s is required: 0.1 us does not span the 4 us ringing.
        self._inject(90.0, 1.0)
        rec = detect_pulses(self.wave, self.fs, dead_time_s=1e-7, f_line=self.f_line)
        self.assertGreater(len(rec.phase_deg), 1)

    def test_constant_record_does_not_divide_by_zero(self):
        rec = detect_pulses(np.zeros(1000), self.fs, dead_time_s=PULSE_RINGING_S, f_line=self.f_line)
        self.assertEqual(len(rec.phase_deg), 0)

    def test_phase_at_t0_shifts_recovered_phase(self):
        self._inject(0.0, 1.0)
        rec = detect_pulses(self.wave, self.fs, dead_time_s=PULSE_RINGING_S, f_line=self.f_line, phase_at_t0_deg=90.0)
        self.assertAlmostEqual(rec.phase_deg[0], 90.0, delta=2.0)

    def test_detection_feeds_anchoring_end_to_end(self):
        self._inject(45.0, 1.0)
        self._inject(225.0, 2.0)
        rec = detect_pulses(self.wave, self.fs, dead_time_s=PULSE_RINGING_S, f_line=self.f_line)
        x, aux = anchor([rec], AnchorConfig(n_bins=128, n_cycles=4))
        self.assertEqual(x.shape, (1, 2, 512))
        self.assertEqual(x[0, 1].sum(), 2.0)
        self.assertGreater(aux[0, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
