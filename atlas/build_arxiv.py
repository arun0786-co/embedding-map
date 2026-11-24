"""The arXiv map, end to end: parse, embed, nearest neighbours, layout,
labels, export. Every stage is cached under data/arxiv/, and the timings go
to results/arxiv_build.json.

    python -m atlas.build_arxiv --zip ../datasets/arxiv/arxiv.zip [--limit 200000]
"""

import argparse
import json
import os
import pickle
import platform
import time

import numpy as np
import torch

from . import arxiv
from .embed import embed
from .export import export
from .knn import default_device, ivf_knn, knn, sync
from .labels import make_labels
from .layout import find_ab, fuzzy_graph, optimize, pca_init


def recall(x, idx, k=15, queries=2000, dev=None, seed=0):
    """Fraction of the exact k nearest neighbours of a random sample of
    points that the approximate search found."""
    rng = np.random.default_rng(seed)
    q = rng.choice(len(x), queries, replace=False)
    xt = torch.as_tensor(x, dtype=torch.float32, device=dev)
    xt = torch.nn.functional.normalize(xt, dim=1)
    hits = 0
    for s in range(0, queries, 500):
        qq = torch.as_tensor(q[s:s + 500], device=dev)
        sim = xt[qq] @ xt.T
        sim[torch.arange(len(qq), device=dev), qq] = -2
        exact = torch.topk(sim, k, dim=1).indices.cpu().numpy()
        for a, b in zip(exact, idx[q[s:s + 500]]):
            hits += len(set(a) & set(b))
    return hits / queries / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="../datasets/arxiv/arxiv.zip")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--probe", type=int, default=32)
    ap.add_argument("--out", default="web/data/arxiv")
    a = ap.parse_args()
    dev = default_device()
    tag = f"arxiv{'_' + str(a.limit) if a.limit else ''}"
    cache = os.path.join("data", tag)
    os.makedirs(cache, exist_ok=True)
    t = {}

    def stage(name, fn):
        s = time.perf_counter()
        out = fn()
        sync(dev)
        t[name] = round(time.perf_counter() - s, 1)
        print(f"{name}: {t[name]} s", flush=True)
        return out

    def parse():
        p = os.path.join(cache, "papers.pkl")
        if os.path.exists(p):
            return pickle.load(open(p, "rb"))
        rows = list(arxiv.read(a.zip, a.limit))
        pickle.dump(rows, open(p, "wb"))
        return rows

    rows = stage("parse s", parse)
    ids, titles, abstracts, cat, year, primary = map(list, zip(*rows))
    cat, year = np.array(cat), np.array(year)
    print(f"{len(rows):,} papers, {year.min()} to {year.max()}", flush=True)
    texts = [f"{ti}. {ab}" for ti, ab in zip(titles, abstracts)]
    emb = stage("embed s", lambda: embed(texts, os.path.join(cache, "emb")))

    def neighbours():
        p = os.path.join(cache, f"knn_p{a.probe}.npz")
        if os.path.exists(p):
            z = np.load(p)
            return z["i"], z["d"], json.loads(str(z["stats"]))
        st = {}
        if len(emb) <= 200_000:
            i, d = knn(emb, 15, "cosine", dev)
            st = {"exact": True}
        else:
            i, d = ivf_knn(emb, 15, "cosine", dev, probe=a.probe, stats=st)
        i, d = i.cpu().numpy(), d.cpu().numpy()
        np.savez(p, i=i, d=d, stats=json.dumps(st))
        return i, d, st

    idx, dist, kst = stage("knn s", neighbours)
    rec = stage("recall check s", lambda: recall(emb, idx, dev=dev))
    print(f"recall@15 on 2,000 random papers: {rec:.4f}", flush=True)

    def layout():
        p = os.path.join(cache, "layout.npy")
        if os.path.exists(p):
            return np.load(p)
        it, dt = torch.as_tensor(idx, device=dev), torch.as_tensor(dist, device=dev)
        r, c, w = fuzzy_graph(it, dt)
        aa, bb = find_ab()
        y = optimize(r, c, w, pca_init(emb, device=dev), a=aa, b=bb,
                     log=lambda e, m: print(f"  epoch {e}, {m:,} edges", flush=True))
        y = y.cpu().numpy()
        np.save(p, y)
        return y

    xy = stage("layout s", layout)
    labels = stage("labels s", lambda: make_labels(xy, titles, device=dev))
    shown = [f"{ti} ({pid})" for ti, pid in zip(titles, ids)]
    stage("export s", lambda: export(a.out, xy, cat, year, shown, arxiv.NAMES, labels,
                                     name=f"arXiv, {len(rows):,} papers"))
    os.makedirs("results", exist_ok=True)
    with open(f"results/{tag}_build.json", "w") as f:
        json.dump({"papers": len(rows), "years": [int(year.min()), int(year.max())], "seconds": t,
                   "knn": {**kst, "recall@15": round(rec, 4)},
                   "machine": {"chip": platform.machine(), "torch": torch.__version__, "device": dev,
                               "load": os.getloadavg()[0]}}, f, indent=1)


if __name__ == "__main__":
    main()
