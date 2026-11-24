"""A synthetic map for developing and measuring the renderer: n points in
nested Gaussian clusters, with categories, years and placeholder titles.

    python -m atlas.synthetic 3000000 web/data/synthetic
"""

import sys

import numpy as np

from .export import export


def make(n, seed=0):
    rng = np.random.default_rng(seed)
    k1, k2 = 12, 25
    top = rng.normal(scale=40, size=(k1, 2))
    sub = top[:, None] + rng.normal(scale=8, size=(k1, k2, 2))
    a = rng.integers(0, k1, n)
    b = rng.integers(0, k2, n)
    xy = sub[a, b] + rng.standard_t(4, size=(n, 2)) * rng.uniform(0.3, 1.5, size=(k1, k2, 1))[a, b]
    year = np.clip(1995 + rng.gamma(6, 3.5, n) * (0.6 + a / k1 * 0.5), 1995, 2025).astype(int)
    labels = [{"text": f"group {i}", "x": float(top[i, 0]), "y": float(top[i, 1]), "level": 0, "size": int((a == i).sum())}
              for i in range(k1)]
    labels += [{"text": f"{i}.{j}", "x": float(sub[i, j, 0]), "y": float(sub[i, j, 1]), "level": 1,
                "size": int(((a == i) & (b == j)).sum())} for i in range(k1) for j in range(k2)]
    titles = [f"point {i} in group {a[i]}.{b[i]}" for i in range(n)]
    return xy.astype(np.float32), a, year, titles, [f"group {i}" for i in range(k1)], labels


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    out = sys.argv[2] if len(sys.argv) > 2 else "web/data/synthetic"
    xy, cat, year, titles, names, labels = make(n)
    export(out, xy, cat, year, titles, names, labels, name=f"synthetic, {n:,} points")
    print("wrote", out)
