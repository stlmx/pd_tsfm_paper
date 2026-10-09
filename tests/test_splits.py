"""Red line 1 is enforced here, not by good intentions."""

import unittest

import numpy as np

from pd.splits import (
    few_shot_support,
    grouped_k_fold,
    leave_one_group_out,
    random_window_split,
)


def toy(n_classes=3, specimens_per_class=4, windows=5):
    """Specimen-structured labels: each specimen contributes several windows."""
    y, groups = [], []
    for c in range(n_classes):
        for s in range(specimens_per_class):
            y += [c] * windows
            groups += [f"c{c}s{s}"] * windows
    return np.array(y), np.array(groups)


class TestGroupedSplits(unittest.TestCase):
    def test_no_specimen_spans_train_and_test(self):
        y, groups = toy()
        for splits in (leave_one_group_out(groups, y), grouped_k_fold(groups, n_splits=4)):
            for sp in splits:
                self.assertFalse(
                    set(groups[sp.train]) & set(groups[sp.test]),
                    f"{sp.scheme} fold {sp.fold} leaks a specimen across the split",
                )

    def test_folds_cover_every_sample_exactly_once(self):
        y, groups = toy()
        for splits in (leave_one_group_out(groups, y), grouped_k_fold(groups, n_splits=4)):
            covered = np.concatenate([sp.test for sp in splits])
            np.testing.assert_array_equal(np.sort(covered), np.arange(len(y)))

    def test_rejects_class_confounded_with_a_single_specimen(self):
        # The Raymond et al. 2017 flaw: one specimen per defect type.
        y, groups = toy(specimens_per_class=1)
        with self.assertRaisesRegex(ValueError, "confounded"):
            leave_one_group_out(groups, y)

    def test_k_fold_is_deterministic_per_seed(self):
        _, groups = toy()
        a = grouped_k_fold(groups, n_splits=4, seed=7)
        b = grouped_k_fold(groups, n_splits=4, seed=7)
        c = grouped_k_fold(groups, n_splits=4, seed=8)
        np.testing.assert_array_equal(a[0].test, b[0].test)
        self.assertFalse(np.array_equal(a[0].test, c[0].test))

    def test_k_fold_rejects_more_folds_than_groups(self):
        _, groups = toy(n_classes=1, specimens_per_class=3)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            grouped_k_fold(groups, n_splits=5)


class TestFewShotSupport(unittest.TestCase):
    def setUp(self):
        self.y, self.groups = toy()
        self.split = leave_one_group_out(self.groups, self.y)[0]

    def test_support_stays_inside_the_training_split(self):
        for shots in (1, 3, 5):
            support = few_shot_support(self.y, self.groups, self.split, shots, seed=0)
            self.assertTrue(set(support) <= set(self.split.train))
            self.assertFalse(set(support) & set(self.split.test))

    def test_exactly_k_samples_per_class(self):
        support = few_shot_support(self.y, self.groups, self.split, shots=3, seed=0)
        counts = np.bincount(self.y[support], minlength=3)
        np.testing.assert_array_equal(counts, [3, 3, 3])

    def test_draw_spreads_across_specimens(self):
        # 3 shots must not all come from one specimen when several are available.
        for seed in range(5):
            support = few_shot_support(self.y, self.groups, self.split, shots=3, seed=seed)
            for cls in np.unique(self.y):
                picked = support[self.y[support] == cls]
                self.assertGreaterEqual(
                    len(np.unique(self.groups[picked])), 2, f"seed {seed}: class {cls} drawn from one specimen"
                )

    def test_shots_zero_means_all_training_labels(self):
        np.testing.assert_array_equal(
            few_shot_support(self.y, self.groups, self.split, shots=0), self.split.train
        )

    def test_raises_when_more_shots_than_available(self):
        with self.assertRaisesRegex(ValueError, "fewer than shots"):
            few_shot_support(self.y, self.groups, self.split, shots=999)


class TestLeakyBaseline(unittest.TestCase):
    def test_random_split_is_labelled_as_leaky(self):
        sp = random_window_split(100, test_size=0.2, seed=0)
        self.assertIn("LEAKY", sp.scheme)
        self.assertEqual(len(sp.test), 20)
        self.assertEqual(len(sp.train), 80)

    def test_random_split_does_leak_specimens(self):
        # Documents why this mode is for the leakage experiment only.
        _, groups = toy()
        sp = random_window_split(len(groups), test_size=0.3, seed=0)
        self.assertTrue(set(groups[sp.train]) & set(groups[sp.test]))


if __name__ == "__main__":
    unittest.main()
