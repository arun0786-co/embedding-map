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


def kmeans(x, k, iters=12, seed=0, block=65536):
    """Lloyd's algorithm on the GPU (euclidean). Returns centroids [k, d]."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g)[:k].to(x.device)].clone()
    for _ in range(iters):
        a = assign(x, c, block)
        s = torch.zeros_like(c).index_add_(0, a, x)
        cnt = torch.bincount(a, minlength=k).float()
        empty = cnt == 0
        c = torch.where(empty[:, None], c, s / cnt.clamp_min(1)[:, None])
    return c


def assign(x, c, block=65536):
    cc = (c * c).sum(1)
    out = torch.empty(len(x), dtype=torch.int64, device=x.device)
    for i in range(0, len(x), block):
        out[i:i + block] = (cc[None] - 2 * x[i:i + block] @ c.T).argmin(1)
    return out


def ivf_knn(x, k, metric="cosine", device=None, lists=None, probe=16, train=50, qblock=1024, cblock=65536,
            seed=0, stats=None, half=True):
    """Approximate k nearest neighbours: k-means splits the points into
    `lists` cells; the points of a cell search the points of the `probe`
    cells whose centroids are nearest to that cell's centroid. About
    n * probe * n / lists distances instead of n^2.

    Returns the same as knn(). Points whose true neighbours sit in cells
    that were not searched get worse neighbours: measure recall against
    knn() on a sample (atlas.build_arxiv does). half: keep the points in
    float16 for the matrix products (half the memory: 3.3M x 384 is 2.5 GB
    instead of 5); squared norms and distances stay float32."""
    dev = device or default_device()
    x = torch.as_tensor(x, dtype=torch.float32, device=dev)
    if metric == "cosine":
        x = F.normalize(x, dim=1)  # then |a - b|^2 = 2 (1 - cos)
    n = len(x)
    lists = lists or max(1, int(n ** 0.5))
    g = torch.Generator(device="cpu").manual_seed(seed)
    sample = x[torch.randperm(n, generator=g)[:min(n, lists * train)].to(dev)]
    cent = kmeans(sample, lists, seed=seed)
    a = assign(x, cent)
    order = torch.argsort(a)
    counts = torch.bincount(a, minlength=lists)
    starts = torch.cumsum(counts, 0) - counts
    cc = (cent * cent).sum(1)
    near = torch.topk(-(cc[:, None] + cc[None] - 2 * cent @ cent.T), min(probe, lists), dim=1).indices
    sq = (x * x).sum(1)
    if half:
        x = x.half()
        if dev == "mps":
            torch.mps.empty_cache()
    out_i = torch.empty(n, k, dtype=torch.int64, device=dev)
    out_d = torch.empty(n, k, dtype=torch.float32, device=dev)
    starts_l, counts_l, near_l = starts.tolist(), counts.tolist(), near.tolist()
    searched = 0
    for c in range(lists):
        if counts_l[c] == 0:
            continue
        qi = order[starts_l[c]:starts_l[c] + counts_l[c]]
        ci = torch.cat([order[starts_l[p]:starts_l[p] + counts_l[p]] for p in near_l[c]])
        searched += len(qi) * len(ci)
        for q0 in range(0, len(qi), qblock):
            q = qi[q0:q0 + qblock]
            best_d = torch.full((len(q), k), float("inf"), device=dev)
            best_i = torch.zeros((len(q), k), dtype=torch.int64, device=dev)
            for c0 in range(0, len(ci), cblock):
                cand = ci[c0:c0 + cblock]
                d = (x[q] @ x[cand].T).float().mul_(-2).add_(sq[q, None]).add_(sq[None, cand]).clamp_min_(0)
                d.masked_fill_(q[:, None] == cand[None], float("inf"))
                kk = min(k, len(cand))
                cd, pos = torch.topk(d, kk, dim=1, largest=False)
                best_d, p2 = torch.topk(torch.cat([best_d, cd], 1), k, dim=1, largest=False)
                best_i = torch.gather(torch.cat([best_i, cand[pos]], 1), 1, p2)
            out_i[q], out_d[q] = best_i, best_d
    if stats is not None:
        stats.update({"lists": lists, "probe": probe, "distances": searched, "fraction of n^2": searched / n / n})
    out_d = out_d / 2 if metric == "cosine" else out_d.sqrt()
    return out_i, out_d
