"""Leakage-free splits for PD recognition (red line 1 of CLAUDE.md).

Windows, cycles and PRPD frames cut from one acquisition are highly correlated,
so a random split over them inflates accuracy. Everything here groups by
specimen (and optionally by test session) instead, and few-shot support sets are
drawn only from training specimens.

`random_window_split` deliberately breaks that rule. It exists so the paper can
report the inflation side by side with the grouped result -- no PD paper in the
three target journals quantifies it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Split", "leave_one_group_out", "grouped_k_fold", "random_window_split", "few_shot_support"]


@dataclass(frozen=True)
class Split:
    """Index arrays into the sample axis, plus how the split was produced."""

    train: np.ndarray
    test: np.ndarray
    scheme: str
    fold: int
    held_out_groups: tuple = field(default=())

    def __post_init__(self):
        if set(self.train) & set(self.test):
            raise ValueError("train and test index sets overlap")

    def as_record(self) -> dict:
        """Flat dict for the split manifest written next to every run."""
        return {
            "scheme": self.scheme,
            "fold": self.fold,
            "held_out_groups": [str(g) for g in self.held_out_groups],
            "n_train": len(self.train),
            "n_test": len(self.test),
            "train_idx": self.train.tolist(),
            "test_idx": self.test.tolist(),
        }


def _as_groups(groups) -> np.ndarray:
    g = np.asarray(groups)
    if g.ndim != 1:
        raise ValueError(f"groups must be 1-D, got shape {g.shape}")
    return g


def leave_one_group_out(groups, y=None, min_groups_per_class: int = 2) -> list[Split]:
    """One fold per specimen: that specimen is the test set, the rest train.

    With `y` given, raises if any class has fewer than `min_groups_per_class`
    specimens -- in that case class and specimen are confounded and a grouped
    split cannot separate them (risk R1). Raymond et al. 2017 is the published
    example of this flaw: one cable joint per defect type.
    """
    groups = _as_groups(groups)
    uniq = np.unique(groups)

    if y is not None:
        y = np.asarray(y)
        if len(y) != len(groups):
            raise ValueError(f"y has {len(y)} entries but groups has {len(groups)}")
        for cls in np.unique(y):
            n = len(np.unique(groups[y == cls]))
            if n < min_groups_per_class:
                raise ValueError(
                    f"class {cls!r} spans only {n} group(s); need >= {min_groups_per_class}. "
                    "Class and specimen are confounded -- group by test session instead "
                    "and record it under risk R1."
                )

    return [
        Split(
            train=np.flatnonzero(groups != g),
            test=np.flatnonzero(groups == g),
            scheme="leave-one-group-out",
            fold=i,
            held_out_groups=(g,),
        )
        for i, g in enumerate(uniq)
    ]


def grouped_k_fold(groups, n_splits: int = 5, seed: int = 0) -> list[Split]:
    """K folds over whole specimens. Use when there are more specimens than folds."""
    groups = _as_groups(groups)
    uniq = np.unique(groups)
    if n_splits > len(uniq):
        raise ValueError(f"n_splits={n_splits} exceeds the {len(uniq)} available groups")

    shuffled = np.random.default_rng(seed).permutation(uniq)
    splits = []
    for fold, held in enumerate(np.array_split(shuffled, n_splits)):
        mask = np.isin(groups, held)
        splits.append(
            Split(
                train=np.flatnonzero(~mask),
                test=np.flatnonzero(mask),
                scheme=f"grouped-{n_splits}-fold",
                fold=fold,
                held_out_groups=tuple(held),
            )
        )
    return splits


def random_window_split(n_samples: int, test_size: float = 0.2, seed: int = 0) -> Split:
    """Leaky baseline: ignores groups, splits windows at random.

    Only for the leakage-quantification experiment. Never use it to produce a
    headline number.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    perm = np.random.default_rng(seed).permutation(n_samples)
    n_test = max(1, round(test_size * n_samples))
    return Split(
        train=np.sort(perm[n_test:]),
        test=np.sort(perm[:n_test]),
        scheme="random-window-LEAKY",
        fold=0,
    )


def few_shot_support(y, groups, split: Split, shots: int, seed: int = 0) -> np.ndarray:
    """Pick `shots` labelled samples per class, drawn only from training specimens.

    Returns indices into the sample axis, a subset of `split.train`. Spreads the
    draw across the training specimens of each class so a single specimen cannot
    supply a whole support set. `shots=0` or a negative value means all of
    `split.train` (the full-label condition).
    """
    y, groups = np.asarray(y), _as_groups(groups)
    if shots <= 0:
        return split.train

    rng = np.random.default_rng(seed)
    picked = []
    for cls in np.unique(y[split.train]):
        pool = split.train[y[split.train] == cls]
        # Round-robin over the class's training specimens, shuffled within each.
        per_group = [rng.permutation(pool[groups[pool] == g]) for g in np.unique(groups[pool])]
        rng.shuffle(per_group)
        interleaved = [i for tier in zip(*_pad(per_group)) for i in tier if i is not None]
        if len(interleaved) < shots:
            raise ValueError(
                f"class {cls!r} has {len(interleaved)} training samples, fewer than shots={shots}"
            )
        picked.append(np.array(interleaved[:shots]))

    support = np.sort(np.concatenate(picked))
    assert set(support) <= set(split.train), "support set escaped the training split"
    return support


def _pad(arrays):
    """Right-pad ragged arrays with None so zip() keeps every element."""
    width = max(len(a) for a in arrays)
    return [list(a) + [None] * (width - len(a)) for a in arrays]
