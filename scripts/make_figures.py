"""Plot the layout benchmark from results/scale_arxiv.json (written by bench/scale.py). Nothing is re-run.

    uv run --no-project --with matplotlib python scripts/make_figures.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "images")

BLUE, LIGHT, GRAY = "#257de0", "#9cc3ef", "#8a8a8a"
TEXT, DIM = "#191919", "#666666"

plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 10,
    "axes.edgecolor": "#cccccc", "axes.labelcolor": TEXT,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": DIM, "ytick.color": DIM,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.dpi": 150,
})


def main():
    with open(os.path.join(ROOT, "results", "scale_arxiv.json")) as f:
        rows = json.load(f)["rows"]
    series = [("this repo (mps), 200 epochs", "this repo, 200 epochs", BLUE),
              ("this repo (mps), 500 epochs", "this repo, 500 epochs", LIGHT),
              ("umap-learn", "umap-learn 0.6.0, 12 cores", GRAY)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for key, label, c in series:
        rs = sorted([r for r in rows if r["method"].startswith(key)], key=lambda r: r["n"])
        n = [r["n"] for r in rs]
        for ax, m in [(axes[0], "seconds"), (axes[1], "neighbours kept")]:
            ax.plot(n, [r[m] for r in rs], color=c, lw=2, marker="o", ms=7, label=label)
    axes[0].set_title("layout time (s), kNN included")
    axes[1].set_title("share of 15 nearest neighbours kept in 2D")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([1e5, 3e5, 1e6], ["100k", "300k", "1M"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("arXiv papers (random subset)")
        ax.grid(color="#eeeeee")
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "scale_vs_umap.png")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    print(path, os.path.getsize(path) // 1024, "KB")


if __name__ == "__main__":
    main()
