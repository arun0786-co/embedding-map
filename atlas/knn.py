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
        out[i:i + block] = (cc[None] - 2 * x[i:i + block].float() @ c.T).argmin(1)
    return out


def top_rows(s, k, group=256):
    """torch.topk(s, k, dim=1), largest first, in two stages: the top k of
    each group of `group` columns, then the top k of those. Exact, since a
    row's k largest are each among the k largest of their own group. On MPS,
    1024 x 32768 in float16 takes 3.7 ms against 12.2 ms in one call."""
    rows, cols = s.shape
    if cols % group or cols < 2 * group or k > group:
        return torch.topk(s, k, dim=1)
    v, i = torch.topk(s.view(rows, -1, group), k, dim=2)
    i = i + torch.arange(0, cols, group, device=s.device)[None, :, None]
    v, j = torch.topk(v.view(rows, -1), k, dim=1)
    return v, torch.gather(i.view(rows, -1), 1, j)


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
    instead of 5); candidates are ranked in float16 too, so near ties can
    swap order, and the merged results are float32."""
    dev = device or default_device()
    n = len(x)
    # Move the points over in slices, so the full matrix never exists in
    # float32 on the device (5 GB at 3.3M points on top of the 2.5 GB copy).
    x = x if torch.is_tensor(x) else torch.from_numpy(x)
    dt = torch.float16 if half else torch.float32
    xd = torch.empty(n, x.shape[1], dtype=dt, device=dev)
    sq = torch.empty(n, dtype=torch.float32, device=dev)
    for i in range(0, n, 262144):
        b = x[i:i + 262144].to(dev).float()
        if metric == "cosine":
            b = F.normalize(b, dim=1)  # then |a - b|^2 = 2 (1 - cos)
        b = b.to(dt)
        xd[i:i + len(b)] = b
        sq[i:i + len(b)] = b.float().pow(2).sum(1)
    x = xd
    del xd
    lists = lists or max(1, int(n ** 0.5))
    g = torch.Generator(device="cpu").manual_seed(seed)
    sample = x[torch.randperm(n, generator=g)[:min(n, lists * train)].to(dev)].float()
    cent = kmeans(sample, lists, seed=seed)
    del sample
    a = assign(x, cent)
    order = torch.argsort(a)
    counts = torch.bincount(a, minlength=lists)
    starts = torch.cumsum(counts, 0) - counts
    cc = (cent * cent).sum(1)
    near = torch.topk(-(cc[:, None] + cc[None] - 2 * cent @ cent.T), min(probe, lists), dim=1).indices
    if dev == "mps":
        torch.mps.empty_cache()
    out_i = torch.empty(n, k, dtype=torch.int64, device=dev)
    out_d = torch.empty(n, k, dtype=torch.float32, device=dev)
    starts_l, counts_l, near_l = starts.tolist(), counts.tolist(), near.tolist()
    searched = 0

    def padded(idx, g):
        # Every cell has a different size, and a caching allocator (MPS's
        # especially) keeps a buffer for every distinct shape it has seen:
        # unpadded, 3.3M points grew the process past 27 GB. Rounding up to a
        # multiple of g leaves a few dozen shapes; the padding is masked.
        m = -(-len(idx) // g) * g
        return torch.cat([idx, idx.new_zeros(m - len(idx))]) if m > len(idx) else idx

    qg, cg = min(256, qblock), min(8192, cblock)
    for c in range(lists):
        if counts_l[c] == 0:
            continue
        # Nothing in this loop waits for the GPU, so without a sync the CPU
        # queues hundreds of cells ahead, each holding its temporaries.
        sync(dev)
        if dev == "mps" and c % 64 == 63:
            torch.mps.empty_cache()  # the small index tensors still vary in size
        qi = order[starts_l[c]:starts_l[c] + counts_l[c]]
        ci = torch.cat([order[starts_l[p]:starts_l[p] + counts_l[p]] for p in near_l[c]])
        searched += len(qi) * len(ci)
        for q0 in range(0, len(qi), qblock):
            q = qi[q0:q0 + qblock]
            nq = len(q)
            q = padded(q, qg)
            # Rank by a score, larger is nearer, so the block needs no
            # float32 copy: for unit vectors the dot product itself; else
            # q.c - |c|^2 / 2, which orders like -|q - c|^2 / 2. One extra
            # slot holds the query itself, removed after the merge, instead
            # of comparing every pair of indices.
            best_s = torch.full((len(q), k + 1), float("-inf"), device=dev)
            best_i = torch.zeros((len(q), k + 1), dtype=torch.int64, device=dev)
            for c0 in range(0, len(ci), cblock):
                cand = ci[c0:c0 + cblock]
                nc = len(cand)
                cand = padded(cand, cg)
                if metric == "cosine":
                    s = x[q] @ x[cand].T  # unit vectors: |s| <= 1, fine in float16
                else:
                    # float16 has a step of 0.25 near 400, so products of
                    # unnormalised points are taken in float32
                    s = (x[q].float() @ x[cand].float().T).sub_(sq[None, cand] / 2)
                if nc < len(cand):
                    s[:, nc:] = float("-inf")
                cs, pos = top_rows(s, min(k + 1, nc))
                best_s, p2 = torch.topk(torch.cat([best_s, cs.float()], 1), k + 1, dim=1)
                best_i = torch.gather(torch.cat([best_i, cand[pos]], 1), 1, p2)
            best_s.masked_fill_(best_i == q[:, None], float("-inf"))
            best_s, p2 = torch.topk(best_s, k, dim=1)
            best_i = torch.gather(best_i, 1, p2)
            if metric == "cosine":
                d = (1 - best_s).clamp_min_(0)
            else:
                d = (sq[q, None] - 2 * best_s).clamp_min_(0).sqrt_()
            out_i[q[:nq]], out_d[q[:nq]] = best_i[:nq], d[:nq]
    if stats is not None:
        stats.update({"lists": lists, "probe": probe, "distances": searched, "fraction of n^2": searched / n / n})
    return out_i, out_d
