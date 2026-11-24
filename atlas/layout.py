"""A UMAP layout computed on the GPU.

The steps are those of McInnes, Healy and Melville (2018):

1. a fuzzy graph from the k nearest neighbours: each point gets a distance
   offset rho (to its nearest neighbour) and a scale sigma chosen so its
   membership weights sum to log2(k);
2. the graph made symmetric with the fuzzy union w + w' - w w';
3. stochastic gradient descent on a 2D layout: every edge pulls its two ends
   together, with a frequency proportional to its weight, and every pull
   comes with a few random points pushed away (negative sampling).

umap-learn runs step 3 one edge at a time on the CPU. Here one epoch is a
handful of tensor operations over all the edges due in it, with updates
summed by index_add_ (as GPU implementations do, races and all).
"""

import math

import numpy as np
import torch

from .knn import default_device, sync


def find_ab(spread=1.0, min_dist=0.1):
    """a, b of the low-dimensional similarity 1 / (1 + a d^2b), fitted to a
    curve that is 1 below min_dist and decays exponentially after it."""
    from scipy.optimize import curve_fit

    xv = np.linspace(0, spread * 3, 300)
    yv = np.where(xv < min_dist, 1.0, np.exp(-(xv - min_dist) / spread))
    (a, b), _ = curve_fit(lambda x, a, b: 1.0 / (1.0 + a * x ** (2 * b)), xv, yv)
    return float(a), float(b)


def fuzzy_graph(idx, dist, iters=64):
    """idx, dist: [n, k] neighbours (self excluded). Returns symmetric edges
    (rows, cols, weights) with each pair once per direction."""
    n, k = idx.shape
    dev = idx.device
    # rho: distance to the nearest neighbour that is not at distance 0
    pos = torch.where(dist > 0, dist, torch.full_like(dist, float("inf")))
    rho = pos.min(1).values
    rho = torch.where(torch.isinf(rho), torch.zeros_like(rho), rho)
    target = math.log2(k)
    lo = torch.zeros(n, device=dev)
    hi = torch.full((n,), float("inf"), device=dev)
    sigma = torch.ones(n, device=dev)
    excess = (dist - rho[:, None]).clamp_min(0)
    for _ in range(iters):
        psum = torch.exp(-excess / sigma[:, None]).sum(1)
        over = psum > target
        hi = torch.where(over, sigma, hi)
        lo = torch.where(over, lo, sigma)
        sigma = torch.where(torch.isinf(hi), sigma * 2, (lo + hi) / 2)
    mean_d = dist.mean()
    sigma = torch.maximum(sigma, 1e-3 * torch.where(rho > 0, dist.mean(1), mean_d))
    w = torch.exp(-excess / sigma[:, None])

    rows = torch.arange(n, device=dev).repeat_interleave(k)
    cols = idx.reshape(-1)
    w = w.reshape(-1)
    # fuzzy union over both directions: pairs found from both ends combine as a + b - ab
    r2 = torch.cat([rows, cols])
    c2 = torch.cat([cols, rows])
    w2 = torch.cat([w, w])
    key = r2 * n + c2
    key, order = torch.sort(key)
    w2 = w2[order]
    uniq, inv, counts = torch.unique_consecutive(key, return_inverse=True, return_counts=True)
    s = torch.zeros(len(uniq), device=dev).index_add_(0, inv, w2)
    first = torch.cumsum(counts, 0) - counts  # a key appears at most twice, consecutively
    p = w2[first] * w2[first + counts - 1]
    ws = torch.where(counts > 1, s - p, s)
    return uniq // n, uniq % n, ws


def pca_init(x, dims=2, scale=10.0, device=None):
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    x = x - x.mean(0)
    _, _, v = torch.pca_lowrank(x, q=dims, center=False)
    y = x @ v[:, :dims]
    return y / y.abs().max() * scale


def optimize(rows, cols, w, init, n_epochs=500, a=1.577, b=0.895, negative=5, lr=0.25, gamma=1.0, seed=0,
             log=None):
    """SGD with negative sampling. Edges whose weight is below max / n_epochs
    are never sampled, as in umap-learn.

    All the edges due in an epoch are applied at once, from the same
    positions, and a point's updates add up. umap-learn applies them one at
    a time, each from the latest positions, with a learning rate of 1. The
    summed version overshoots at that rate: on MNIST it loses 2 points of
    2D kNN accuracy. A rate of 0.25 over 500 epochs matches umap-learn's
    quality (bench/quality.py)."""
    dev = init.device
    y = init.clone()
    n = y.shape[0]
    keep = w >= w.max() / n_epochs
    rows, cols, w = rows[keep], cols[keep], w[keep]
    eps = w.max() / w  # epochs between samples of each edge
    nxt = eps.clone()
    g = torch.Generator(device="cpu").manual_seed(seed)
    for e in range(n_epochs):
        alpha = lr * (1.0 - e / n_epochs)
        act = (nxt <= e + 1).nonzero().squeeze(1)
        if len(act) == 0:
            continue
        nxt[act] += eps[act]
        i, j = rows[act], cols[act]
        diff = y[i] - y[j]
        d2 = (diff * diff).sum(1, keepdim=True)
        coef = torch.where(d2 > 0, -2 * a * b * d2.pow(b - 1) / (a * d2.pow(b) + 1), torch.zeros_like(d2))
        grad = (coef * diff).clamp(-4, 4) * alpha
        y.index_add_(0, i, grad)
        y.index_add_(0, j, -grad)
        # negative samples: pushed away from random points
        ii = i.repeat(negative)
        kk = torch.randint(0, n, (len(ii),), generator=g).to(dev)
        diff = y[ii] - y[kk]
        d2 = (diff * diff).sum(1, keepdim=True)
        coef = 2 * gamma * b / ((0.001 + d2) * (a * d2.pow(b) + 1))
        grad = torch.where(d2 > 0, (coef * diff).clamp(-4, 4), torch.full_like(diff, 4.0)) * alpha
        grad = torch.where((ii == kk)[:, None], torch.zeros_like(grad), grad)
        y.index_add_(0, ii, grad)
        if log and (e % 50 == 0 or e == n_epochs - 1):
            log(e, len(act))
    return y


def umap(x, k=15, n_epochs=None, min_dist=0.1, spread=1.0, metric="cosine", seed=0, device=None, knn_result=None,
         timings=None, half=False):
    """The whole pipeline. Returns the [n, 2] layout as a numpy array."""
    import time

    from .knn import knn

    dev = device or default_device()
    t = {}
    s = time.perf_counter()
    idx, dist = knn_result if knn_result is not None else knn(x, k, metric, dev, half=half)
    idx, dist = torch.as_tensor(idx, device=dev), torch.as_tensor(dist, device=dev, dtype=torch.float32)
    sync(dev)
    t["knn s"] = time.perf_counter() - s
    s = time.perf_counter()
    rows, cols, w = fuzzy_graph(idx, dist)
    sync(dev)
    t["graph s"] = time.perf_counter() - s
    s = time.perf_counter()
    init = pca_init(x, device=dev)
    a, b = find_ab(spread, min_dist)
    n_epochs = n_epochs or 500
    y = optimize(rows, cols, w, init, n_epochs, a, b, seed=seed)
    sync(dev)
    t["optimize s"] = time.perf_counter() - s
    if timings is not None:
        timings.update({key: round(v, 2) for key, v in t.items()})
        timings["edges"] = int(len(w))
    return y.cpu().numpy()
