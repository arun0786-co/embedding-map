# How it works

The map is built in five steps. Each paper's title and abstract become a 384-number vector. Papers with nearby vectors should land near each other in 2D. The browser has to draw millions of them, smoothly, and say what each region is about.

```
arXiv metadata (3.26M papers)
  -> embeddings      all-MiniLM-L6-v2, 384 dims          atlas/embed.py
  -> neighbours      15 nearest per paper, on the GPU    atlas/knn.py
  -> 2D layout       UMAP, on the GPU                    atlas/layout.py
  -> labels          clusters named by their titles      atlas/labels.py
  -> browser files   6 bytes per point + title tiles     atlas/export.py
  -> WebGL2          density rendering, hover, filters   web/atlas.js
```

## 1. Nearest neighbours (`knn.py`)

UMAP needs each point's k nearest neighbours (k = 15).

**Exact search.** `knn()` does blocked brute force. It takes a block of 2,048 queries against a chunk of 131,072 points: one matrix product, then a top-k, merged into a running top-k across the chunks. It agrees with scikit-learn on every index (tests). Its cost is quadratic, and the top-k selection, not the matrix product, dominates. At 200,000 points of 384 dims it takes 49 s on the M4 Pro, and float16 does not help. At 3.26M points it would take about 3.6 hours.

**IVF search.** `ivf_knn()` uses an inverted file:
1. k-means splits the points into about √n cells.
2. Each cell finds the `probe` cells whose centroids are nearest its own.
3. All of the cell's points are compared only with the points of those cells.

Grouping by cell keeps the work in large matrix products, which suits a GPU. With probe = 32 of √n cells, about 32/√n of all pairs get compared: 1.8% at 3.26M (1,806 cells).

The cost of grouping is that a point near the edge of its cell searches the cells near its *cell's* centre, not near itself, and may miss a neighbour just across a border. Recall is measured, never assumed. The build compares the found neighbours of 2,000 random papers with an exact search and records the fraction in `results/`. On MNIST, probe 16 gives 0.935 and probe 32 gives 0.972, in about 1 s.

**Memory, the hard part at 3.26M.** The first full runs died: one with an MPS out-of-memory error, one with the Mac swapping 11 GB until it had to be restarted. The live tensors were small (1.4 GB at 1.2M points); what the GPU driver held was not. Three causes, each measured with `torch.mps.driver_allocated_memory()`:

1. **A new shape every cell.** Each cell has a different number of points and candidates, so every distance block had a new size, and the caching allocator kept a buffer for each. At 400k points the driver held 14.1 GB. Rounding blocks up to multiples of 256 queries and 8,192 candidates, and masking the padding with infinity, leaves a few dozen shapes: 3.3 GB.
2. **The CPU running ahead.** Nothing in the loop reads a result back, so the CPU queued hundreds of cells of work, each holding its own temporaries. One `synchronize()` per cell bounds that, at no measurable cost (66.9 s against 66.0 s at 1.2M). Peak driver memory at 1.2M fell from 7.5 GB to 4.3 GB.
3. **Full float32 copies.** The points are moved to the GPU and normalised in slices straight into float16 (2.5 GB for 3.26M × 384, instead of 5 GB plus a 5 GB temporary). PCA for the layout's starting positions, which also copied everything to float32 twice, now accumulates the 384 × 384 covariance over slices and takes its top eigenvectors; it matches scikit-learn's PCA to a correlation of 1.0000.

On a unified-memory Mac, GPU memory is the system's memory, so an allocator holding 20 GB of buffers pushes everything else into swap.

## 2. Layout (`layout.py`)

UMAP (McInnes, Healy, Melville, 2018) in three steps:

**A fuzzy graph.** For point i with neighbour distances d_ij, take rho_i, the distance to the nearest neighbour. Then find sigma_i by binary search so that

    sum_j exp(-(d_ij - rho_i) / sigma_i) = log2(k).

The edge weight is w_ij = exp(-(d_ij - rho_i) / sigma_i). Every point thus has at least one weight-1 edge, and the scale adapts to local density: a point in a sparse region gets a large sigma.

**Symmetric.** The directed graph becomes undirected by fuzzy union: w = a + b - ab when both directions exist. The implementation sorts the edge keys i·n + j of both directions, then combines duplicates.

**SGD.** The 2D positions start from PCA and are optimised for 500 epochs:
- **Edges.** Every edge is sampled every max(w)/w epochs, and pulls its two ends together. The gradient comes from the low-dimensional similarity 1/(1 + a d^2b); a and b are fitted to min_dist = 0.1.
- **Negative samples.** Each pull comes with 5 random points pushed away from the edge's head.
- **Clipping.** Gradients are clipped to ±4. The learning rate falls linearly to 0.

umap-learn applies these updates one edge at a time, each from the latest positions. Here an epoch is a few tensor operations over every edge due in it, and the updates are summed with `index_add_`.

**The difference matters.** A point with 30 edges due in an epoch gets 30 pulls computed from the same stale positions, and they add up. At umap-learn's learning rate of 1 that overshoots: on MNIST the 2D kNN accuracy was 0.939 against umap-learn's 0.970. A rate of 0.25 over 500 epochs brought it to 0.970, with trustworthiness 0.950 against 0.951 (`bench/quality.py`, `results/quality_mnist.json`). Batched GPU implementations face the same race; the fix is a smaller step and more epochs.

**Memory.** Each epoch's edges are processed in chunks of 4M, so tens of millions of edges fit.

## 3. Labels (`labels.py`)

k-means on the 2D layout at three sizes (30, 200 and 1,200 clusters), one per zoom level. Each cluster is named by class-based TF-IDF over its titles: a term's share of the cluster's words, times log(1 + mean words per cluster / the term's total count).

Two rules made the names readable:
- **Two-word terms get a 1.6x boost.** "black holes" beats "black".
- **No shared words.** A term sharing a word (ignoring a trailing s) with an already chosen one is skipped, so a cluster is not called "galaxies · galaxy".

TeX accents in titles are stripped first; otherwise "Schr\\\"odinger" splits into "schr" and "odinger".

## 4. What the browser gets (`export.py`)

- **points.bin**, 6 bytes per point: x and y as uint16 over the layout's bounding square, a category byte and a year byte. 3.26M papers come to 20 MB, which loads in one fetch.
- **The order.** Points are sorted by tile of a 64 × 64 grid, so each tile's points form one contiguous range of ids.
- **tiles/t.json** holds titles and arXiv ids for tile t, in the same order. The browser fetches a tile the first time the pointer comes near one of its points. All titles together are hundreds of MB; a session typically needs a few tiles.
- **meta.json** holds the bounds, the tile offsets, the categories and colours, and the labels.

## 5. Drawing millions of points (`web/atlas.js`)

**Why not just draw the points.** Drawing millions of points with ordinary alpha blending shows mostly overdraw. The last point drawn on a pixel wins, and a pixel with 1 point looks like a pixel with 1,000.

**Two passes instead:**
1. **Accumulate.** Every point is drawn as a soft disc into a half-float framebuffer with additive blending. A pixel collects (sum of colour × weight, sum of weight). The point's position comes straight from the uint16 attributes (`vertexAttribIPointer`); its colour comes from a 256 × 1 texture indexed by the category byte. Hidden categories and years after the slider are moved out of view in the vertex shader, so filtering costs nothing on the CPU.
2. **Tone-map.** A full-screen triangle turns each pixel's sums into the mean colour, faded in by 1 - exp(-density × exposure) and pushed towards white in the densest cores.

**Keeping the brightness steady.** The exposure is set from the mean coverage of the data's footprint, so the picture keeps its brightness as zoom changes the point size (which grows with zoom^0.45).

**Hover.** A 1,024 × 1,024 grid over the quantised positions is built once, by counting sort into two flat arrays, in about 100 ms for 3M points. Each mouse move checks the few cells within 8 pixels.

**Labels.** Labels are drawn on a 2D canvas over the map, largest clusters first. One is skipped if its box would overlap one already placed. The zoom picks the level.

**Redrawing.** The page draws only when something changes. `?bench=1` animates zoom and pan for 8 s and reports frame times, which is how `results/render_*.json` was measured.

## References

- L. McInnes, J. Healy, J. Melville. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. 2018. https://arxiv.org/abs/1802.03426
- J. Venna, S. Kaski. Neighborhood preservation in nonlinear projection methods: an experimental study. ICANN 2001. (Trustworthiness.)
- H. Jégou, M. Douze, C. Schmid. Product quantization for nearest neighbor search. IEEE TPAMI 2011. (The inverted file.)
- N. Reimers, I. Gurevych. Sentence-BERT. EMNLP 2019. https://arxiv.org/abs/1908.10084
- M. Grootendorst. BERTopic: neural topic modeling with a class-based TF-IDF procedure. 2022. https://arxiv.org/abs/2203.05794
- arXiv metadata: Cornell University, Kaggle dataset Cornell-University/arxiv.
