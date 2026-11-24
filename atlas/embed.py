"""Sentence embeddings with all-MiniLM-L6-v2 (384 dims), written in shards
so a long run can resume."""

import os

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embed(texts, out_dir, device=None, shard=200_000, batch=256, max_len=128, log=print):
    from sentence_transformers import SentenceTransformer

    from .knn import default_device

    os.makedirs(out_dir, exist_ok=True)
    model = SentenceTransformer(MODEL, device=device or default_device())
    model.max_seq_length = max_len
    parts = []
    for s0 in range(0, len(texts), shard):
        p = os.path.join(out_dir, f"emb_{s0 // shard:04d}.npy")
        if not os.path.exists(p):
            e = model.encode(texts[s0:s0 + shard], batch_size=batch, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)
            np.save(p, e.astype(np.float16))
            log(f"embedded {min(s0 + shard, len(texts)):,} / {len(texts):,}")
        parts.append(np.load(p))
    return np.concatenate(parts)
