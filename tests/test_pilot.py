"""Pipeline tests. Uses a cheap deterministic embedder, not a real backbone."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pd.anchoring import AnchorConfig, PulseRecord
from pd.backbones import Embedder, build_embedder
from pd.data import Dataset
from pd.pilot import PilotConfig, build_inputs, make_splits, sweep


def toy_dataset(n_specimens=3, n_acq=4, seed=0):
    """Two classes separated by phase only, with per-specimen phase jitter."""
    rng = np.random.default_rng(seed)
    records, y, specimen, batch = [], [], [], []
    for cls, centre in enumerate((60.0, 240.0)):
        for s in range(n_specimens):
            jitter = rng.normal(0, 5.0)
            for a in range(n_acq):
                phase = (centre + jitter + rng.normal(0, 8.0, size=10)) % 360.0
                records.append(
                    PulseRecord(phase, rng.lognormal(2.0, 0.3, size=10), np.zeros(10, int), 1)
                )
                y.append(cls)
                specimen.append(f"c{cls}s{s}")
                batch.append(f"c{cls}s{s}b{a // 2}")
    return Dataset(
        records=records,
        y=np.array(y),
        classes=["a", "b"],
        specimen=np.array(specimen),
        batch=np.array(batch),
        source=[Path(f"{i}.csv") for i in range(len(records))],
    )


def mean_embedder(name=None):
    """Deterministic 4-d summary; stands in for a frozen backbone."""
    return Embedder(
        lambda x: np.stack(
            [x.mean(axis=(1, 2)), x.std(axis=(1, 2)), x.max(axis=(1, 2)), x[:, 0].argmax(axis=-1)],
            axis=1,
        )
    )


def fitted_embedder(calls):
    """Records how often fit() is called, to prove per-fold refitting."""
    state = {"scale": 1.0}

    def fit(x_train):
        calls.append(len(x_train))
        state["scale"] = float(x_train.std() + 1e-9)

    return Embedder(
        transform=lambda x: np.stack([x.mean(axis=(1, 2)) / state["scale"]], axis=1),
        needs_fit=True,
        fit=fit,
    )


class TestInterfaces(unittest.TestCase):
    def setUp(self):
        self.ds = toy_dataset()
        self.cfg = AnchorConfig(n_bins=64, n_cycles=2)

    def test_anchored_shape(self):
        x = build_inputs(self.ds, "anchored", self.cfg)
        self.assertEqual(x.shape, (24, 2, 128))

    def test_aux_adds_four_channels(self):
        x = build_inputs(self.ds, "anchored+aux", self.cfg)
        self.assertEqual(x.shape[1], 6)

    def test_aux_channels_are_constant_along_time(self):
        # Constant so instance normalisation cannot strip the magnitude.
        x = build_inputs(self.ds, "anchored+aux", self.cfg)
        np.testing.assert_allclose(x[:, 2:].std(axis=-1), 0.0, atol=1e-6)

    def test_raw_destroys_phase_alignment(self):
        anchored = build_inputs(self.ds, "anchored", self.cfg)
        raw = build_inputs(self.ds, "raw", self.cfg)
        self.assertEqual(raw.shape, anchored.shape)
        self.assertFalse(np.allclose(raw, anchored))
        # Rotation preserves total pulse count: only position is lost.
        np.testing.assert_allclose(raw[:, 1].sum(axis=-1), anchored[:, 1].sum(axis=-1))

    def test_unknown_interface_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown interface"):
            build_inputs(self.ds, "spectrogram", self.cfg)


class TestSplitSchemes(unittest.TestCase):
    def setUp(self):
        self.ds = toy_dataset()

    def test_leave_one_group_out_has_one_fold_per_specimen(self):
        splits = make_splits(self.ds, PilotConfig(split_scheme="leave-one-group-out"))
        self.assertEqual(len(splits), 6)

    def test_group_by_batch_is_available_for_risk_r1(self):
        splits = make_splits(self.ds, PilotConfig(split_scheme="leave-one-group-out", group_by="batch"))
        self.assertEqual(len(splits), 12)

    def test_leaky_scheme_is_explicitly_named(self):
        splits = make_splits(self.ds, PilotConfig(split_scheme="random-window-LEAKY", n_folds=3))
        self.assertEqual(len(splits), 3)
        self.assertTrue(all("LEAKY" in s.scheme for s in splits))

    def test_unknown_scheme_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown split_scheme"):
            make_splits(self.ds, PilotConfig(split_scheme="stratified"))


class TestSweep(unittest.TestCase):
    def setUp(self):
        self.ds = toy_dataset()
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = PilotConfig(
            backbones=("stub",),
            interfaces=("anchored",),
            shots=(2, 0),
            seeds=(0, 1),
            split_scheme="grouped-k-fold",
            n_folds=3,
            anchor=AnchorConfig(n_bins=64, n_cycles=2),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_one_row_per_cell(self):
        rows = sweep(self.ds, self.cfg, mean_embedder, self.tmp.name)
        ok = [r for r in rows if r["status"] == "ok"]
        self.assertEqual(len(ok), 4)  # 2 shots x 2 seeds

    def test_writes_per_sample_predictions(self):
        sweep(self.ds, self.cfg, mean_embedder, self.tmp.name)
        files = list(Path(self.tmp.name).glob("pred_*.json"))
        self.assertEqual(len(files), 4)
        import json

        payload = json.loads(files[0].read_text())
        self.assertIn("folds", payload)
        self.assertIn("y_true", payload["folds"][0])
        self.assertIn("test_specimens", payload["folds"][0])

    def test_frozen_embedder_is_not_fitted(self):
        rows = sweep(self.ds, self.cfg, mean_embedder, self.tmp.name)
        self.assertFalse(any(r.get("refit_per_fold") for r in rows if r["status"] == "ok"))

    def test_fitted_embedder_sees_only_training_samples(self):
        # The leakage guard: fit must be called once per fold, on train only.
        calls = []
        sweep(self.ds, self.cfg, lambda name: fitted_embedder(calls), self.tmp.name)
        self.assertEqual(len(calls), 3, "expected one fit per fold")
        n = len(self.ds)
        for n_train in calls:
            self.assertLess(n_train, n, "fit saw the whole dataset, including test folds")

    def test_insufficient_shots_are_skipped_not_crashed(self):
        cfg = PilotConfig(**{**self.cfg.__dict__, "shots": (999,)})
        rows = sweep(self.ds, cfg, mean_embedder, self.tmp.name)
        self.assertTrue(rows)
        self.assertTrue(all(r["status"] == "skipped" for r in rows))
        self.assertIn("fewer than shots", rows[0]["reason"])

    def test_metrics_are_in_range(self):
        rows = sweep(self.ds, self.cfg, mean_embedder, self.tmp.name)
        for r in (r for r in rows if r["status"] == "ok"):
            self.assertGreaterEqual(r["macro_f1"], 0.0)
            self.assertLessEqual(r["macro_f1"], 1.0)
            self.assertEqual(r["n_folds"], 3)


class TestBackboneRegistry(unittest.TestCase):
    def test_unknown_backbone_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown backbone"):
            build_embedder("gpt4ts")

    def test_raw_embedder_flattens(self):
        emb = build_embedder("raw")
        self.assertFalse(emb.needs_fit)
        self.assertEqual(emb(np.zeros((3, 2, 16), dtype=np.float32)).shape, (3, 32))


if __name__ == "__main__":
    unittest.main()
