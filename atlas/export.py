"""Writes a layout in the format the browser reads (web/data/<name>/):

- points.bin: 6 bytes per point, x and y as uint16 over the bounds, a
  category byte and a year byte (years since meta.year0). Points are sorted
  by tile of a G x G grid, so the points of a tile are one contiguous range.
- tiles/<t>.json: the titles of tile t's points, in order, fetched only when
  the pointer comes near them.
- meta.json: counts, bounds, the grid's tile offsets, categories with
  colours, cluster labels at several zoom levels.

2.5 million points come to 15 MB of positions, small enough to load at
once; the titles (about 200 MB) never load all together.
"""

import json
import os

import numpy as np

PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f",
           "#bab0ac", "#86bcb6", "#d37295", "#a0cbe8", "#ffbe7d", "#8cd17d", "#f1ce63"]


def export(out, xy, category, year, titles, category_names, labels=(), grid=64, year0=None, name="", ids=None,
           link=None):
    """xy: [n, 2] float; category: [n] ints into category_names; year: [n]
    ints; titles: list of n strings; labels: dicts {text, x, y, level, size}
    in layout coordinates. With ids and a link prefix, a click on a point
    opens link + id."""
    os.makedirs(os.path.join(out, "tiles"), exist_ok=True)
    n = len(xy)
    lo, hi = xy.min(0), xy.max(0)
    span = float(max(hi - lo)) * 1.02
    lo = (lo + hi) / 2 - span / 2
    q = np.clip(np.round((xy - lo) / span * 65535), 0, 65535).astype(np.uint16)
    tx = np.minimum(q[:, 0].astype(np.int64) * grid // 65536, grid - 1)
    ty = np.minimum(q[:, 1].astype(np.int64) * grid // 65536, grid - 1)
    tile = ty * grid + tx
    order = np.argsort(tile, kind="stable")
    q, tile = q[order], tile[order]
    year0 = int(year.min()) if year0 is None else year0
    rec = np.zeros(n, dtype=[("x", "<u2"), ("y", "<u2"), ("c", "u1"), ("t", "u1")])
    rec["x"], rec["y"] = q[:, 0], q[:, 1]
    rec["c"] = np.asarray(category)[order]
    rec["t"] = np.clip(np.asarray(year)[order] - year0, 0, 255)
    rec.tofile(os.path.join(out, "points.bin"))
    offsets = np.searchsorted(tile, np.arange(grid * grid + 1)).tolist()
    titles = [titles[i] if ids is None else [titles[i], ids[i]] for i in order]
    for t in range(grid * grid):
        a, b = offsets[t], offsets[t + 1]
        if b > a:
            with open(os.path.join(out, "tiles", f"{t}.json"), "w") as f:
                json.dump(titles[a:b], f, ensure_ascii=False, separators=(",", ":"))
    counts = np.bincount(np.asarray(category), minlength=len(category_names))
    meta = {
        "name": name, "n": n, "grid": grid, "offsets": offsets, "link": link,
        "bounds": [float(lo[0]), float(lo[1]), span],
        "year0": year0, "year1": int(np.asarray(year).max()),
        "categories": [{"name": c, "color": PALETTE[i % len(PALETTE)], "count": int(counts[i])}
                       for i, c in enumerate(category_names)],
        "labels": [{**lb, "x": (lb["x"] - float(lo[0])) / span, "y": (lb["y"] - float(lo[1])) / span}
                   for lb in labels],
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    return order
