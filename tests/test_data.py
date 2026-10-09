"""Loader tests: each layout must reach the same canonical representation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pd.anchoring import AnchorConfig, anchor
from pd.data import SchemaConfig, load_dataset

FS = 10e6
F_LINE = 50.0
CYCLE_N = int(FS / F_LINE)


class LayoutCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rel, frame):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)


class TestPulseListLayout(LayoutCase):
    def build(self, n_specimens=3):
        for defect in ("void", "corona"):
            for s in range(n_specimens):
                self.write(
                    f"{defect}/S{s}_b0.csv",
                    pd.DataFrame({"phase": [45.0, 230.0], "q_pc": [12.0, 8.0], "cyc": [0, 0]}),
                )

    def cfg(self, **kw):
        base = dict(
            layout="pulse_list",
            filename_pattern=r"(?P<defect>[^/]+)/(?P<specimen>S\d+)_(?P<batch>b\d+)",
            columns={"phase": "phase", "amplitude": "q_pc", "cycle": "cyc"},
            qualify_specimen_with_defect=True,
        )
        base.update(kw)
        return SchemaConfig(**base)

    def test_labels_come_from_the_path(self):
        self.build()
        ds = load_dataset(self.root, self.cfg())
        self.assertEqual(len(ds), 6)
        self.assertEqual(ds.classes, ["corona", "void"])
        self.assertEqual(len(np.unique(ds.specimen)), 6)  # S0..S2 under each of 2 defects
        self.assertEqual(sorted(np.bincount(ds.y)), [3, 3])

    def test_pulses_survive_into_the_record(self):
        self.build()
        ds = load_dataset(self.root, self.cfg())
        np.testing.assert_allclose(np.sort(ds.records[0].phase_deg), [45.0, 230.0])
        np.testing.assert_allclose(np.sort(np.abs(ds.records[0].amplitude)), [8.0, 12.0])

    def test_summary_reports_specimen_counts_for_gate_g0(self):
        self.build()
        summary = load_dataset(self.root, self.cfg()).summary()
        self.assertEqual(set(summary["defect"]), {"void", "corona"})
        self.assertTrue((summary["n_specimens"] == 3).all())

    def test_feeds_anchoring(self):
        self.build()
        ds = load_dataset(self.root, self.cfg())
        x, aux = anchor(ds.records, AnchorConfig(n_bins=128, n_cycles=4))
        self.assertEqual(x.shape, (6, 2, 512))
        self.assertTrue((aux[:, 0] > 0).all())

    def test_specimen_ids_colliding_across_classes_are_refused(self):
        # S0 exists under both defects. Merging them would silently coarsen the
        # folds; splitting a genuinely shared specimen would leak. So: ask.
        self.build()
        with self.assertRaisesRegex(ValueError, "more than one defect class"):
            load_dataset(self.root, self.cfg(qualify_specimen_with_defect=False))

    def test_globally_unique_ids_need_no_qualification(self):
        for i, defect in enumerate(("void", "corona")):
            self.write(
                f"{defect}/S{i}_b0.csv",
                pd.DataFrame({"phase": [45.0], "q_pc": [1.0], "cyc": [0]}),
            )
        ds = load_dataset(self.root, self.cfg(qualify_specimen_with_defect=False))
        self.assertEqual(sorted(np.unique(ds.specimen)), ["S0", "S1"])

    def test_phase_is_wrapped_into_range(self):
        self.write("void/S0_b0.csv", pd.DataFrame({"phase": [-45.0, 400.0], "q_pc": [1.0, 1.0], "cyc": [0, 0]}))
        ds = load_dataset(self.root, self.cfg())
        self.assertTrue(((ds.records[0].phase_deg >= 0) & (ds.records[0].phase_deg < 360)).all())


class TestRawWaveformLayout(LayoutCase):
    def build(self):
        rng = np.random.default_rng(0)
        for defect, phase_deg in (("void", 45.0), ("corona", 225.0)):
            for s in range(2):
                wave = 0.001 * rng.standard_normal(CYCLE_N)
                pos = int(phase_deg / 360.0 * CYCLE_N)
                t = np.arange(40)
                wave[pos : pos + 40] += np.exp(-t / 6.0) * np.sin(2 * np.pi * t / 8.0)
                # A synchronous reference channel: sine rising through zero at t=0.
                ref = np.sin(2 * np.pi * F_LINE * np.arange(CYCLE_N) / FS)
                self.write(f"{defect}/S{s}_b0.csv", pd.DataFrame({"v": wave, "ref": ref}))

    def cfg(self, **kw):
        base = dict(
            layout="raw_waveform",
            filename_pattern=r"(?P<defect>[^/]+)/(?P<specimen>S\d+)_(?P<batch>b\d+)",
            columns={"signal": "v"},
            fs=FS,
            f_line=F_LINE,
            dead_time_s=5e-6,
            qualify_specimen_with_defect=True,
        )
        base.update(kw)
        return SchemaConfig(**base)

    def test_detects_the_injected_pulse(self):
        self.build()
        ds = load_dataset(self.root, self.cfg())
        self.assertEqual(len(ds), 4)
        for rec in ds.records:
            self.assertEqual(len(rec.phase_deg), 1)

    def test_recovered_phase_separates_the_two_classes(self):
        self.build()
        ds = load_dataset(self.root, self.cfg())
        by_class = {c: [] for c in ds.classes}
        for yi, rec in zip(ds.y, ds.records):
            by_class[ds.classes[yi]].append(rec.phase_deg[0])
        np.testing.assert_allclose(np.mean(by_class["void"]), 45.0, atol=2.0)
        np.testing.assert_allclose(np.mean(by_class["corona"]), 225.0, atol=2.0)

    def test_reference_channel_is_used_when_named(self):
        self.build()
        ds = load_dataset(self.root, self.cfg(columns={"signal": "v", "phase_ref": "ref"}))
        phases = [r.phase_deg[0] for yi, r in zip(ds.y, ds.records) if ds.classes[yi] == "void"]
        np.testing.assert_allclose(np.mean(phases), 45.0, atol=3.0)

    def test_requires_fs_and_dead_time(self):
        self.build()
        with self.assertRaisesRegex(ValueError, "fs"):
            load_dataset(self.root, self.cfg(fs=None))
        with self.assertRaisesRegex(ValueError, "dead_time_s"):
            load_dataset(self.root, self.cfg(dead_time_s=None))


class TestPrpdMatrixLayout(LayoutCase):
    def cfg(self):
        return SchemaConfig(
            layout="prpd_matrix",
            filename_pattern=r"(?P<defect>[^/]+)/(?P<specimen>S\d+)_(?P<batch>b\d+)",
            columns={"phase": "phi", "amplitude": "q", "count": "n"},
            qualify_specimen_with_defect=True,
        )

    def test_cells_expand_to_pulses(self):
        self.write(
            "void/S0_b0.csv",
            pd.DataFrame({"phi": [45.0, 225.0, 90.0], "q": [10.0, 5.0, 3.0], "n": [3, 2, 0]}),
        )
        ds = load_dataset(self.root, self.cfg())
        rec = ds.records[0]
        self.assertEqual(len(rec.phase_deg), 5)  # the zero-count cell is dropped
        self.assertEqual(int((rec.phase_deg == 45.0).sum()), 3)
        self.assertEqual(int((rec.phase_deg == 225.0).sum()), 2)


class TestLoaderGuards(LayoutCase):
    def test_missing_root_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset(self.root / "nope", SchemaConfig(columns={"phase": "p", "amplitude": "a"}))

    def test_no_matching_files_raises(self):
        with self.assertRaisesRegex(FileNotFoundError, "matched"):
            load_dataset(self.root, SchemaConfig(columns={"phase": "p", "amplitude": "a"}))

    def test_unknown_layout_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown layout"):
            load_dataset(self.root, SchemaConfig(layout="spectrogram", columns={}))

    def test_missing_class_label_raises(self):
        self.write("x.csv", pd.DataFrame({"p": [1.0], "a": [1.0]}))
        with self.assertRaisesRegex(ValueError, "no class label"):
            load_dataset(self.root, SchemaConfig(columns={"phase": "p", "amplitude": "a"}))

    def test_labels_may_come_from_columns(self):
        self.write("x.csv", pd.DataFrame({"p": [1.0], "a": [1.0], "d": ["void"], "s": ["S1"]}))
        ds = load_dataset(
            self.root,
            SchemaConfig(columns={"phase": "p", "amplitude": "a", "defect": "d", "specimen": "s"}),
        )
        self.assertEqual(ds.classes, ["void"])
        self.assertEqual(ds.specimen[0], "S1")


if __name__ == "__main__":
    unittest.main()
