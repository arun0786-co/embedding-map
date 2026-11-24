import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atlas.knn import knn  # noqa: E402
from atlas.layout import fuzzy_graph, umap  # noqa: E402

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_knn_is_exact(dev, metric):
    from sklearn.neighbors import NearestNeighbors

    x = np.random.default_rng(0).normal(size=(3000, 32)).astype(np.float32)
    i, d = knn(x, 10, metric, device=dev, qblock=700, chunk=1100)  # blocks that do not divide n
    dd, ii = NearestNeighbors(n_neighbors=11, metric=metric).fit(x).kneighbors(x)
    assert (i.cpu().numpy() == ii[:, 1:]).mean() > 0.999  # ties may order differently
    assert np.abs(d.cpu().numpy() - dd[:, 1:]).max() < 1e-4


def test_fuzzy_graph_is_symmetric_union():
    x = np.random.default_rng(1).normal(size=(500, 8)).astype(np.float32)
    i, d = knn(x, 10, "euclidean", device="cpu")
    r, c, w = fuzzy_graph(i, d)
    m = torch.zeros(500, 500)
    m[r, c] = w
    assert torch.allclose(m, m.T)
    assert w.max() <= 1 + 1e-6 and w.min() > 0
    # each row's directed weights sum to log2(k) before the union
    assert (m > 0).sum(1).min() >= 10


def test_layout_separates_clusters():
    rng = np.random.default_rng(2)
    centers = rng.normal(scale=10, size=(5, 50))
    labels = rng.integers(0, 5, 2000)
    x = (centers[labels] + rng.normal(size=(2000, 50))).astype(np.float32)
    y = umap(x, metric="euclidean", device="cpu", n_epochs=200)
    i, _ = knn(y, 10, "euclidean", device="cpu")
    same = (labels[i.numpy()] == labels[:, None]).mean()
    assert same > 0.99
