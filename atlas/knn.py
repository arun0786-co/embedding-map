"""Exact k nearest neighbours on the GPU by blocked brute force: one block of
queries against one chunk of the points at a time, a running top-k merged
across chunks. For n = 1M and d = 384 that is about 8e14 flops: minutes on a
laptop GPU, and exact, so the layout's input graph has no approximation
error of its own."""

import torch
import torch.nn.functional as F


def default_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def knn(x, k, metric="cosine", device=None, qblock=2048, chunk=131072, half=False):
    """Returns (indices [n, k] int64, distances [n, k] float32), nearest
    first, each point excluding itself. metric: "cosine" (1 - cos) or
    "euclidean". half: do the matmuls in float16 (faster, distances to
    about 1e-3)."""
    dev = device or default_device()
    x = torch.as_tensor(x, dtype=torch.float32, device=dev)
    n = x.shape[0]
    if metric == "cosine":
        x = F.normalize(x, dim=1)
    xm = x.half() if half else x
    sq = (x * x).sum(1)
    out_i = torch.empty(n, k, dtype=torch.int64, device=dev)
    out_d = torch.empty(n, k, dtype=torch.float32, device=dev)
    for q0 in range(0, n, qblock):
        q1 = min(q0 + qblock, n)
        q = xm[q0:q1]
        best_d = torch.full((q1 - q0, k), float("inf"), device=dev)
        best_i = torch.zeros((q1 - q0, k), dtype=torch.int64, device=dev)
        for c0 in range(0, n, chunk):
            c1 = min(c0 + chunk, n)
            dot = (q @ xm[c0:c1].T).float()
            if metric == "cosine":
                d = 1.0 - dot
            else:
                d = (sq[q0:q1, None] + sq[None, c0:c1] - 2 * dot).clamp_min_(0)
            lo, hi = max(q0, c0), min(q1, c1)
            if lo < hi:  # the queries themselves are in this chunk
                r = torch.arange(lo, hi, device=dev)
                d[r - q0, r - c0] = float("inf")
            kk = min(k, c1 - c0)
            cd, ci = torch.topk(d, kk, dim=1, largest=False)
            best_d, pos = torch.topk(torch.cat([best_d, cd], 1), k, dim=1, largest=False)
            best_i = torch.gather(torch.cat([best_i, ci + c0], 1), 1, pos)
        out_i[q0:q1], out_d[q0:q1] = best_i, best_d
    if metric == "euclidean":
        out_d = out_d.sqrt()
    return out_i, out_d
