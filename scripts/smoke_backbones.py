"""Smoke test for candidate TSFM backbones (CPU, synthetic input).

Checks per backbone: loads from Hugging Face, embedding shape, parameter count,
CPU latency, and amplitude invariance (cosine similarity between embed(x) and
embed(10x)). Inputs are random pulse trains used only to exercise the APIs;
nothing here is an accuracy result.

Usage:
    .venv-main/bin/python scripts/smoke_backbones.py --models mantis_v1 mantis_v2 chronos_bolt chronos2 minirocket
    .venv-moment/bin/python scripts/smoke_backbones.py --models moment_small moment_base moment_large
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

N, L = 32, 512
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "smoke"


def synthetic_pulses(n, length, seed=0):
    """Sparse damped pulses on a noise floor, loosely PD-like; API exercise only."""
    rng = np.random.default_rng(seed)
    x = 0.01 * rng.standard_normal((n, length))
    t = np.arange(40)
    kernel = np.exp(-t / 6.0) * np.sin(2 * np.pi * t / 8.0)
    for i in range(n):
        for pos in rng.choice(length - 40, size=rng.integers(3, 12), replace=False):
            x[i, pos:pos + 40] += rng.uniform(0.2, 1.0) * kernel
    return x.astype(np.float32)


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def cosine(a, b):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return float(np.mean(num / den))


def build(name):
    """Return (embed_fn, param_count, notes) for a backbone name."""
    if name.startswith("mantis"):
        from mantis.architecture import Mantis8M, MantisV2
        from mantis.trainer import MantisTrainer
        if name == "mantis_v1":
            net = Mantis8M(device="cpu").from_pretrained("paris-noah/Mantis-8M")
        else:
            net = MantisV2(device="cpu").from_pretrained("paris-noah/MantisV2")
        trainer = MantisTrainer(device="cpu", network=net)
        return (lambda x: trainer.transform(x[:, None, :])), n_params(net), "input length must be a multiple of num_patches; pretrained at 512"

    if name in ("chronos_bolt", "chronos2"):
        from chronos import BaseChronosPipeline
        repo = "amazon/chronos-bolt-base" if name == "chronos_bolt" else "amazon/chronos-2"
        pipe = BaseChronosPipeline.from_pretrained(repo, device_map="cpu", dtype=torch.float32)

        def embed(x):
            if name == "chronos_bolt":
                emb, _ = pipe.embed(torch.from_numpy(x))
                return emb.mean(1).numpy()
            embs, _ = pipe.embed(x[:, None, :])
            return np.stack([e[0].mean(0).numpy() for e in embs])

        ctx = pipe.model.config.chronos_config.get("context_length")
        return embed, n_params(pipe.model), f"model context_length={ctx}; mean-pooled encoder tokens"

    if name.startswith("moment"):
        from momentfm import MOMENTPipeline
        size = name.split("_")[1]
        model = MOMENTPipeline.from_pretrained(f"AutonLab/MOMENT-1-{size}", model_kwargs={"task_name": "embedding"})
        model.init()
        model.eval()

        def embed(x):
            with torch.no_grad():
                return model(x_enc=torch.from_numpy(x)[:, None, :]).embeddings.numpy()

        return embed, n_params(model), "fixed input length 512, patch 8"

    if name == "minirocket":
        from aeon.transformations.collection.convolution_based import MiniRocket
        tr = MiniRocket(random_state=0)
        tr.fit(synthetic_pulses(N, L, seed=1)[:, None, :])
        return (lambda x: np.asarray(tr.transform(x[:, None, :]))), 0, "random convolution features, no pretraining"

    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x = synthetic_pulses(N, L)
    torch.set_num_threads(4)

    for name in args.models:
        rec = {"model": name}
        try:
            t0 = time.time()
            embed, params, notes = build(name)
            rec["load_s"] = round(time.time() - t0, 1)
            embed(x[:2])  # warm-up
            t0 = time.time()
            z = np.asarray(embed(x))
            rec["cpu_ms_per_sample"] = round((time.time() - t0) / N * 1000, 1)
            rec["embedding_shape"] = list(z.shape)
            rec["params_M"] = round(params / 1e6, 2)
            rec["cos_x_vs_10x"] = round(cosine(z, np.asarray(embed(10 * x))), 4)
            rec["notes"] = notes
            rec["status"] = "ok"
        except Exception as e:  # record the failure and continue with the next model
            rec["status"] = "failed"
            rec["error"] = f"{type(e).__name__}: {e}"[:500]
        print(json.dumps(rec, ensure_ascii=False))
        (OUT_DIR / f"{name}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
