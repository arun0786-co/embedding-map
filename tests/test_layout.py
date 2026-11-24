import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atlas.knn import ivf_knn, knn, top_rows  # noqa: E402
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


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_ivf_matches_exact_when_probing_every_cell(dev, half, metric):
    if half and dev == "cpu":
        pytest.skip("float16 matmul on cpu is slow")
    rng = np.random.default_rng(1)
    x = (rng.normal(size=(20, 24)) * 4)[rng.integers(0, 20, 4000)] + rng.normal(size=(4000, 24))
    x = x.astype(np.float16 if half else np.float32)  # the arxiv build passes float16
    ei, ed = knn(x.astype(np.float32), 10, metric, device=dev)
    i, d = ivf_knn(x, 10, metric, device=dev, lists=16, probe=16, qblock=300, cblock=700, half=half)
    overlap = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(i.cpu().numpy(), ei.cpu().numpy())])
    assert overlap > (0.97 if half else 0.999)  # float16 swaps near ties, so compare sets
    assert np.abs(d.cpu().numpy() - ed.cpu().numpy()).max() < (1e-3 if half else 1e-4)
    i4, _ = ivf_knn(x, 10, metric, device=dev, lists=16, probe=4, half=half)
    hit = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(i4.cpu().numpy(), ei.cpu().numpy())])
    assert hit > 0.9  # clustered data: four of sixteen cells hold nearly all neighbours


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


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_two_stage_topk_is_exact(dev, dtype):
    s = torch.randn(300, 4096 + 77, generator=torch.Generator().manual_seed(0)).to(dev, dtype)  # a partial group
    s[:, 1000:1040] = float("-inf")  # padding, as ivf_knn masks it
    v, i = top_rows(s, 16)
    rv, _ = torch.topk(s, 16, dim=1)
    assert torch.equal(v, rv)
    assert torch.equal(torch.gather(s, 1, i), v)  # the indices point at those values
