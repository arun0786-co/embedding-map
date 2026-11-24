"""The GPU layout against umap-learn on MNIST (70,000 digits, 784 dims):
time, and how well each 2D layout keeps the structure.

- neighbours kept: of each point's 15 nearest neighbours in 784 dims, the
  fraction that are among its 15 nearest in 2D;
- 2D kNN accuracy: the digit predicted by a vote of the 15 nearest 2D
  neighbours (leaving the point out);
- trustworthiness (Venna and Kaski) on 5,000 random points, k = 15.

    python bench/quality.py
"""

import gzip
import json
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atlas.knn import default_device, knn  # noqa: E402
from atlas.layout import umap  # noqa: E402

MNIST = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "mnist")


def load_mnist():
    xs, ys = [], []
    for split in ("train", "t10k"):
        with gzip.open(os.path.join(MNIST, f"{split}-images-idx3-ubyte.gz")) as f:
            xs.append(np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784))
        with gzip.open(os.path.join(MNIST, f"{split}-labels-idx1-ubyte.gz")) as f:
            ys.append(np.frombuffer(f.read(), np.uint8, offset=8))
    return np.concatenate(xs).astype(np.float32) / 255.0, np.concatenate(ys)


def scores(y2, hi_idx, labels, seed=0):
    from sklearn.manifold import trustworthiness

    lo_idx, _ = knn(y2, 15, "euclidean")
    lo_idx = lo_idx.cpu().numpy()
    kept = np.mean([len(set(a) & set(b)) / 15 for a, b in zip(hi_idx, lo_idx)])
    votes = labels[lo_idx]
    pred = np.array([np.bincount(v, minlength=10).argmax() for v in votes])
    acc = float((pred == labels).mean())
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(y2), 5000, replace=False)
    return {"neighbours kept": round(float(kept), 3), "2D kNN accuracy": round(acc, 4),
            "trustworthiness": round(float(trustworthiness(X[sub], y2[sub], n_neighbors=15)), 4)}


if __name__ == "__main__":
    X, labels = load_mnist()
    dev = default_device()
    t = time.perf_counter()
    hi_idx, hi_d = knn(X, 15, "euclidean")
    torch.mps.synchronize() if dev == "mps" else None
    knn_s = time.perf_counter() - t
    hi = hi_idx.cpu().numpy()
    rows = []

    timings = {}
    t = time.perf_counter()
    y = umap(X, metric="euclidean", timings=timings, seed=0)
    rows.append({"method": "this repo, GPU (" + dev + ")", "seconds": round(time.perf_counter() - t, 1), **timings,
                 **scores(y, hi, labels)})
    print(rows[-1], flush=True)
    np.save(os.path.join(os.path.dirname(__file__), "..", "results", "mnist_layout.npy"), y.astype(np.float32))

    import umap as umap_learn
    for init in ("spectral", "pca"):
        t = time.perf_counter()
        y = umap_learn.UMAP(n_neighbors=15, min_dist=0.1, init=init, random_state=None).fit_transform(X)
        rows.append({"method": f"umap-learn {umap_learn.__version__}, init={init}",
                     "seconds": round(time.perf_counter() - t, 1), **scores(y, hi, labels)})
        print(rows[-1], flush=True)

    out = os.path.join(os.path.dirname(__file__), "..", "results", "quality_mnist.json")
    with open(out, "w") as f:
        json.dump({"machine": {"chip": platform.machine(), "torch": torch.__version__, "cpus": os.cpu_count(),
                               "load": os.getloadavg()[0]},
                   "exact knn s": round(knn_s, 2), "rows": rows}, f, indent=1)
