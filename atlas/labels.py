"""Names for regions of the map: k-means on the 2D layout at a few sizes
(coarse to fine, one per zoom level), each cluster named by the title terms
most particular to it (class-based TF-IDF: a term's frequency in the
cluster, weighted by how rare it is across clusters)."""

import numpy as np
import torch

from .knn import assign, kmeans

STOP = set("""a an and are as at be by for from in into is of on or the to with without via using based new study
towards toward its their this that these we our can not non some two one three over under between through than
analysis approach method methods model models results problem problems case cases effect effects application
applications theory general paper note remarks properties some certain large small high low
""".split())


def clusters(xy, k, device=None, seed=0):
    x = torch.as_tensor(xy, dtype=torch.float32, device=device)
    c = kmeans(x, k, iters=25, seed=seed)
    return assign(x, c).cpu().numpy()


def stem(w):
    """Crude singular: galaxies -> galaxy, fields -> field, glass stays."""
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def name_clusters(titles, cl, k, terms=2, min_df=30):
    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b[a-z][a-z0-9\-]{2,}\b", ngram_range=(1, 2),
                         min_df=min_df, max_features=300_000,
                         stop_words=list(STOP))
    m = cv.fit_transform(titles)  # [n, vocab]
    vocab = np.array(cv.get_feature_names_out())
    import scipy.sparse as sp

    onehot = sp.csr_matrix((np.ones(len(cl)), (cl, np.arange(len(cl)))), shape=(k, len(cl)))
    tf = (onehot @ m).toarray().astype(np.float64)  # [k, vocab]
    words = tf.sum(1, keepdims=True)
    idf = np.log(1 + words.mean() / np.maximum(tf.sum(0), 1))
    score = tf / np.maximum(words, 1) * idf
    # two-word terms name things better ("black holes" over "black"), so they
    # get a boost; a term sharing a word (plural or not) with a chosen one is skipped
    score = score * np.where(np.char.count(vocab.astype(str), " ") > 0, 1.6, 1.0)[None]
    stems = [{stem(w) for w in t.split()} for t in vocab]
    names = []
    for c in range(k):
        best, used = [], set()
        for t in np.argsort(-score[c])[:30]:
            if stems[t] & used:
                continue
            best.append(vocab[t])
            used |= stems[t]
            if len(best) == terms:
                break
        names.append(" · ".join(best))
    return names


def make_labels(xy, titles, levels=(30, 200, 1200), device=None):
    out = []
    for level, k in enumerate(levels):
        cl = clusters(xy, k, device)
        names = name_clusters(titles, cl, k)
        for c in range(k):
            sel = cl == c
            if sel.sum() < 20:
                continue
            mx, my = np.median(xy[sel], axis=0)
            out.append({"text": names[c], "x": float(mx), "y": float(my), "level": level, "size": int(sel.sum())})
    return out
