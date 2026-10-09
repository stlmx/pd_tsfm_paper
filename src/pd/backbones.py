"""One embed() interface over every backbone and baseline feature extractor.

Each entry returns a callable mapping (n, c, seq_len) -> (n, d) frozen features,
so the pilot driver treats a pretrained TSFM and MiniRocket identically. Models
that cannot take multiple channels are applied per channel and concatenated.

Measured properties of each backbone (parameters, context, amplitude
invariance) are in docs/backbone_validation.md. Note which environment each one
needs: MOMENT lives in .venv-moment, everything else in .venv-main.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = ["ENVIRONMENTS", "Embedder", "available", "build_embedder"]


@dataclass
class Embedder:
    """A frozen feature extractor, plus whether it has to be fitted first.

    A pretrained backbone needs no fitting, so its features can be computed once
    for the whole dataset. MiniRocket does need fitting -- it estimates quantiles
    of its convolution outputs from data -- and fitting it on the full dataset
    would leak test-fold statistics. `needs_fit` makes the pilot refit it on each
    fold's training split instead.
    """

    transform: Callable[[np.ndarray], np.ndarray]
    needs_fit: bool = False
    fit: Callable[[np.ndarray], None] | None = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.transform(x))

ENVIRONMENTS = {
    "mantis_v1": "main",
    "mantis_v2": "main",
    "chronos_bolt": "main",
    "chronos2": "main",
    "minirocket": "main",
    "moment_small": "moment",
    "moment_base": "moment",
    "moment_large": "moment",
    "raw": "any",
}


def available(env: str) -> list[str]:
    """Backbone names runnable in the given environment ("main" or "moment")."""
    return [k for k, v in ENVIRONMENTS.items() if v in (env, "any")]


def _per_channel(fn, x: np.ndarray) -> np.ndarray:
    """Apply a univariate embedder to each channel and concatenate."""
    return np.concatenate([np.asarray(fn(x[:, c, :])) for c in range(x.shape[1])], axis=1)


def build_embedder(name: str, batch_size: int = 64) -> Embedder:
    """Return an Embedder for x of shape (n, c, seq_len).

    Build a fresh one per (backbone, interface): a fitted extractor is tied to
    the channel count of the interface it saw.
    """
    if name == "raw":
        return Embedder(lambda x: x.reshape(len(x), -1))

    if name.startswith("mantis"):
        from mantis.architecture import Mantis8M, MantisV2
        from mantis.trainer import MantisTrainer

        repo = "paris-noah/Mantis-8M" if name == "mantis_v1" else "paris-noah/MantisV2"
        net = (Mantis8M if name == "mantis_v1" else MantisV2)(device="cpu").from_pretrained(repo)
        trainer = MantisTrainer(device="cpu", network=net)
        # Mantis already handles the channel axis and concatenates internally.
        return Embedder(lambda x: np.asarray(trainer.transform(x, batch_size=batch_size)))

    if name in ("chronos_bolt", "chronos2"):
        import torch
        from chronos import BaseChronosPipeline

        repo = "amazon/chronos-bolt-base" if name == "chronos_bolt" else "amazon/chronos-2"
        pipe = BaseChronosPipeline.from_pretrained(repo, device_map="cpu", dtype=torch.float32)

        def embed_1d(u: np.ndarray) -> np.ndarray:
            if name == "chronos_bolt":
                emb, _ = pipe.embed(torch.from_numpy(np.ascontiguousarray(u)))
                return emb.mean(1).numpy()
            embs, _ = pipe.embed(u[:, None, :], batch_size=batch_size)
            return np.stack([e[0].mean(0).numpy() for e in embs])

        return Embedder(lambda x: _per_channel(embed_1d, x))

    if name.startswith("moment"):
        import torch
        from momentfm import MOMENTPipeline

        size = name.split("_")[1]
        model = MOMENTPipeline.from_pretrained(
            f"AutonLab/MOMENT-1-{size}", model_kwargs={"task_name": "embedding"}
        )
        model.init()
        model.eval()

        def embed_1d(u: np.ndarray) -> np.ndarray:
            out = []
            with torch.no_grad():
                for i in range(0, len(u), batch_size):
                    chunk = torch.from_numpy(np.ascontiguousarray(u[i : i + batch_size]))
                    out.append(model(x_enc=chunk[:, None, :]).embeddings.numpy())
            return np.concatenate(out)

        return Embedder(lambda x: _per_channel(embed_1d, x))

    if name == "minirocket":
        from aeon.transformations.collection.convolution_based import MiniRocket

        tr = MiniRocket(random_state=0)
        return Embedder(
            transform=lambda x: np.asarray(tr.transform(x)),
            needs_fit=True,
            fit=lambda x_train: tr.fit(x_train),
        )

    raise ValueError(f"unknown backbone {name!r}; known: {sorted(ENVIRONMENTS)}")
