"""The GPU pipeline (IVF kNN + layout) against umap-learn on growing random
subsets of the arXiv embeddings: time, and how much structure each 2D map
keeps. Needs the embeddings from atlas.build_arxiv.

- neighbours kept: for 2,000 random papers, the fraction of their 15 exact
  nearest neighbours (cosine, full dims) that are among their 15 nearest
  in 2D;
- trustworthiness on 5,000 random papers, k = 15.

    python bench/scale.py 100000 300000 1000000
"""

import glob
import json
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atlas.knn import default_device, knn, sync  # noqa: E402
from atlas.layout import umap  # noqa: E402


def kept(x, y, dev, queries=2000, k=15, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.choice(len(x), queries, replace=False)
    xt = torch.nn.functional.normalize(torch.as_tensor(x, dtype=torch.float32, device=dev), dim=1)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    hits = 0
    for s in range(0, queries, 500):
        qq = torch.as_tensor(q[s:s + 500], device=dev)
        r = torch.arange(len(qq), device=dev)
        sim = xt[qq] @ xt.T
        sim[r, qq] = -2
        hi = torch.topk(sim, k, dim=1).indices
        d2 = ((yt[qq][:, None] - yt[None]) ** 2).sum(-1)
        d2[r, qq] = float("inf")
        lo = torch.topk(d2, k, dim=1, largest=False).indices
        for a, b in zip(hi.cpu().numpy(), lo.cpu().numpy()):
            hits += len(set(a) & set(b))
    return hits / queries / k


def trust(x, y, seed=0):
    from sklearn.manifold import trustworthiness

    sub = np.random.default_rng(seed).choice(len(x), 5000, replace=False)
    return trustworthiness(x[sub], y[sub], n_neighbors=15, metric="cosine")


def main():
    sizes = [int(s) for s in sys.argv[1:]] or [100_000, 300_000, 1_000_000]
    emb = np.concatenate([np.load(p) for p in sorted(glob.glob("data/arxiv/emb/emb_*.npy"))]).astype(np.float32)
    dev = default_device()
    rows = []
    for n in sizes:
        sub = emb[np.random.default_rng(1).choice(len(emb), n, replace=False)]
        tm = {}
        t = time.perf_counter()
        y = umap(sub, metric="cosine", device=dev, timings=tm, knn_result=None)
        sync(dev)
        rows.append({"n": n, "method": f"this repo ({dev})", "seconds": round(time.perf_counter() - t, 1), **tm,
                     "neighbours kept": round(kept(sub, y, dev), 3), "trustworthiness": round(trust(sub, y), 4)})
        print(rows[-1], flush=True)
        import umap as umap_learn
        t = time.perf_counter()
        y = umap_learn.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", init="pca", low_memory=True).fit_transform(sub)
        rows.append({"n": n, "method": f"umap-learn {umap_learn.__version__} ({os.cpu_count()} cores)",
                     "seconds": round(time.perf_counter() - t, 1),
                     "neighbours kept": round(kept(sub, y, dev), 3), "trustworthiness": round(trust(sub, y), 4)})
        print(rows[-1], flush=True)
        with open("results/scale_arxiv.json", "w") as f:
            json.dump({"machine": {"chip": platform.machine(), "torch": torch.__version__, "load": os.getloadavg()[0]},
                       "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
