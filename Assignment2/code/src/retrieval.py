"""Part 5 — open-ended image retrieval challenge.

Target category: motorbike. No classifier is trained: every method turns the
query and the corpus images into vectors and ranks the corpus by similarity.

Features are read from data/encodings/<feature>/<class>_<subset>.npz, the same
cache used by Parts 1-4 (BoVW/VLAD/FV from src.extract, deep features from
src.extract_deep).
"""
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score

from .evaluate import CLASSES, DATA, ROOT, image_path, l2norm, load_splits

TARGET = "motorbike"
KS = (5, 10, 20)
RESULTS = ROOT / "results" / "part5"
FIGS = RESULTS / "figs"


# =============================================================================
# Data: raw features (no normalisation) of one subset
# =============================================================================
@dataclass
class Corpus:
    X: np.ndarray        # (N, D) raw features, rows sorted by image id
    names: np.ndarray    # (N,) image ids
    labels: list         # (N,) list of classes of each image


@lru_cache(maxsize=None)
def load_raw(feature, subset):
    """All images of a subset (each once, sorted by id), features NOT normalised.

    The Part 4 loader (evaluate.load_split) L2-normalises every vector; here
    the raw vectors are needed so that each method can choose its own
    normalisation (L1, Hellinger, power, ...).
    """
    labels = load_splits()[subset]
    vectors = {}
    for cat in CLASSES + ["background"]:
        with np.load(DATA / "encodings" / feature / f"{cat}_{subset}.npz") as z:
            vectors.update(zip(z["file_names"].astype(str), z["histograms"]))
    names = np.array(sorted(labels))
    X = np.stack([vectors[n] for n in names]).astype(np.float32)
    return Corpus(X, names, [labels[n] for n in names])


def has_feature(feature):
    folder = DATA / "encodings" / feature
    return all((folder / f"background_{s}.npz").exists() for s in ("train", "val", "test"))


def relevance(corpus, target=TARGET):
    """Ground truth: True where the corpus image contains the target class."""
    return np.array([target in l for l in corpus.labels])


def corpus_summary(corpus, target=TARGET):
    """Composition of a corpus: relevant vs irrelevant images per label set."""
    rel = relevance(corpus, target)
    rows = {"images": len(rel), "relevant": int(rel.sum()),
            "relevant, single label": int(sum(r and len(l) == 1 for r, l in zip(rel, corpus.labels))),
            "relevant, multi-label": int(sum(r and len(l) > 1 for r, l in zip(rel, corpus.labels)))}
    for c in CLASSES + ["background"]:
        if c != target:
            rows[f"irrelevant with {c}"] = int(sum((not r) and c in l
                                                   for r, l in zip(rel, corpus.labels)))
    return pd.Series(rows)


# =============================================================================
# Queries
# =============================================================================
def query_pool(target=TARGET, subset="train"):
    """Ids of all images of `subset` that contain the target class."""
    c = load_raw("dinov2_vits14", subset)
    return c.names[relevance(c, target)]


def near_duplicates(query_names, corpus_subset, feature="dinov2_vits14", thr=0.95):
    """(query, corpus image, cosine) pairs whose embeddings are almost equal.

    DINOv2 is used because it is very sensitive to near-copies (same photo,
    re-encoded or slightly cropped) and much less to images that only share
    the category.
    """
    q = embed(feature, "train", query_names)
    c = load_raw(feature, corpus_subset)
    S = l2norm(q) @ l2norm(c.X).T
    i, j = np.where(S > thr)
    return pd.DataFrame({"query": np.asarray(query_names)[i], "corpus": c.names[j],
                         "cosine": S[i, j]}).sort_values("cosine", ascending=False)


def embed(feature, subset, names):
    """Raw feature rows of the given image ids."""
    c = load_raw(feature, subset)
    index = {n: k for k, n in enumerate(c.names)}
    return c.X[[index[n] for n in names]]


def diverse_queries(pool, n=12, feature="dinov2_vits14", seed=0):
    """n diverse queries: k-means with n clusters on the embeddings of the pool,
    then the image closest to each cluster centre (medoid).

    Reproducible and covers different looks of the category (side/front view,
    with rider, small object, cluttered scene, ...).
    """
    X = l2norm(embed(feature, "train", pool))
    km = KMeans(n_clusters=n, n_init=10, random_state=seed).fit(X)
    picks = [np.argmax(X @ l2norm(km.cluster_centers_[k:k + 1]).ravel()
                       - 10 * (km.labels_ != k)) for k in range(n)]
    return np.asarray(pool)[picks]


def typical_queries(pool, n=12, feature="dinov2_vits14"):
    """The n images of the pool closest to the mean embedding (prototypical
    motorbikes) — used to contrast with the diverse set."""
    X = l2norm(embed(feature, "train", pool))
    sim = X @ l2norm(X.mean(0, keepdims=True)).ravel()
    return np.asarray(pool)[np.argsort(-sim)[:n]]


def typicality(names, pool, feature="dinov2_vits14"):
    """Cosine of each query to the mean embedding of the pool (1 = very typical)."""
    P = l2norm(embed(feature, "train", pool))
    centre = l2norm(P.mean(0, keepdims=True)).ravel()
    return l2norm(embed(feature, "train", names)) @ centre


# =============================================================================
# Methods: normalisation, optional PCA and similarity
# =============================================================================
def signed_sqrt(X):
    return np.sign(X) * np.sqrt(np.abs(X))


PREP = {
    "raw": lambda X: X,
    "l1": lambda X: X / np.maximum(np.abs(X).sum(1, keepdims=True), 1e-12),
    "l2": l2norm,
    # Hellinger: square root of the L1-normalised histogram, then L2
    "hellinger": lambda X: l2norm(np.sqrt(np.maximum(PREP["l1"](X), 0))),
    # power normalisation used for VLAD / Fisher vectors
    "power": lambda X: l2norm(signed_sqrt(X)),
}


@dataclass
class Method:
    name: str                 # label used in tables and plots
    family: str               # BoVW, VLAD, FV, ResNet-18, DINOv2, CLIP image
    feature: str              # folder in data/encodings
    prep: str = "l2"          # key of PREP
    sim: str = "cos"          # cos | euclid | hik (histogram intersection)
    pca: int = 0              # 0 = no PCA; PCA is fitted on TRAIN features only
    _pca: object = field(default=None, repr=False)

    def fit(self):
        """Fit PCA on the train split (never on val/test, which hold the corpora)."""
        if self.pca:
            Xtr = PREP[self.prep](load_raw(self.feature, "train").X)
            self._pca = PCA(self.pca, random_state=0).fit(Xtr)
        return self

    def transform(self, X):
        Z = PREP[self.prep](X)
        if self._pca is not None:
            Z = l2norm(self._pca.transform(Z))
        return Z.astype(np.float32)

    @property
    def dim(self):
        return self.pca or load_raw(self.feature, "train").X.shape[1]


def similarity(Q, C, sim):
    """Scores (n_queries, n_corpus); larger = more similar."""
    if sim == "cos":                     # vectors are already L2-normalised
        return Q @ C.T
    if sim == "euclid":                  # minus squared Euclidean distance
        return -(np.sum(Q ** 2, 1)[:, None] - 2 * Q @ C.T + np.sum(C ** 2, 1)[None])
    if sim == "hik":                     # histogram intersection
        return np.stack([np.minimum(q[None], C).sum(1) for q in Q])
    raise ValueError(sim)


def score_queries(method, queries, corpus_subset, query_subset="train"):
    """Similarity of every query (image ids) to every image of the corpus."""
    Q = method.transform(embed(method.feature, query_subset, queries))
    C = method.transform(load_raw(method.feature, corpus_subset).X)
    return similarity(Q, C, method.sim)


# =============================================================================
# Metrics
# =============================================================================
def ranking_metrics(scores, rel, ks=KS):
    """AP, P@K and R@K of one ranking (scores over the corpus, rel = ground truth)."""
    order = np.argsort(-scores, kind="stable")
    hits = rel[order]
    out = {"AP": average_precision_score(rel, scores)}
    for k in ks:
        out[f"P@{k}"] = hits[:k].mean()
    for k in ks:
        out[f"R@{k}"] = hits[:k].sum() / rel.sum()
    return out


def evaluate(scores, rel, queries, ks=KS):
    """One row per query."""
    return pd.DataFrame([ranking_metrics(s, rel, ks) for s in scores],
                        index=pd.Index(queries, name="query"))


def summarise(per_query):
    """mAP ± std over queries and mean of the other metrics."""
    m = per_query.mean()
    out = {"mAP": m["AP"], "AP std": per_query["AP"].std(ddof=0)}
    out.update({c: m[c] for c in per_query.columns if c != "AP"})
    return out


def pr_curve(scores, rel, levels=np.linspace(0, 1, 101)):
    """Interpolated precision at fixed recall levels, averaged over queries."""
    curves = []
    for s in np.atleast_2d(scores):
        hits = rel[np.argsort(-s, kind="stable")]
        tp = np.cumsum(hits)
        precision = tp / np.arange(1, len(hits) + 1)
        recall = tp / rel.sum()
        # interpolated precision: best precision at any recall >= level
        best = np.maximum.accumulate(precision[::-1])[::-1]
        idx = np.searchsorted(recall, levels, side="left")
        curves.append(np.where(idx < len(best), best[np.minimum(idx, len(best) - 1)], 0))
    return levels, np.mean(curves, 0)


def run(methods, queries, corpus_subset, target=TARGET):
    """Evaluate a list of methods on one corpus.

    Returns (summary table, per-query AP table, {method name: score matrix}).
    """
    corpus = load_raw(methods[0].feature, corpus_subset)
    rel = relevance(corpus, target)
    rows, ap_rows, all_scores = [], {}, {}
    for m in methods:
        S = score_queries(m, queries, corpus_subset)
        pq = evaluate(S, rel, queries)
        rows.append({"method": m.name, "family": m.family, "dim": m.dim, **summarise(pq)})
        ap_rows[m.name] = pq["AP"]
        all_scores[m.name] = S
    return pd.DataFrame(rows).set_index("method"), pd.DataFrame(ap_rows), all_scores


def pick_best(val_table, by="mAP"):
    """Best variant of every family on the VALIDATION corpus."""
    return val_table.reset_index().sort_values(by, ascending=False) \
        .drop_duplicates("family").set_index("family")["method"]


# =============================================================================
# CLIP text -> image
# =============================================================================
TEXT_QUERIES = [
    "motorbike",
    "a photo of a motorbike.",
    "a photo of a motorcycle.",
    "a photo of a person riding a motorbike.",
    "a photo of a motorbike parked on the street.",
    "a close-up photo of a motorbike with two wheels, an engine, a seat and handlebars.",
    "a photo of a motorbike, a motorcycle, a scooter or a dirt bike.",
    # control: a visually related but different concept
    "a photo of a bicycle.",
]


def text_scores(model, prompts, corpus_subset, encode_text):
    """Cosine similarity of each text prompt to every corpus image (CLIP)."""
    T = encode_text(model, prompts)
    C = l2norm(load_raw("clip_vit_b_32", corpus_subset).X)
    return T @ C.T


# =============================================================================
# Costs
# =============================================================================
def query_time_ms(method, queries, corpus_subset, repeats=3):
    """Mean time (ms) to score and sort the corpus for one query, with the
    corpus already encoded and normalised (what a search server would do)."""
    C = method.transform(load_raw(method.feature, corpus_subset).X)
    Q = method.transform(embed(method.feature, "train", queries))
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        for q in Q:
            s = similarity(q[None], C, method.sim)[0]
            np.argsort(-s)
        best = min(best, (time.perf_counter() - t0) / len(Q))
    return 1000 * best


def handcrafted_extraction_ms(encoder_file, names, cache_dir=None):
    """Mean ms/image to compute dense SIFT and encode it (CPU, no cache)."""
    import os
    from .encoders import ImageEncoder
    enc = ImageEncoder(DATA / "encoders" / encoder_file)
    # relative paths: cv2.imread fails on absolute paths with non-ASCII
    # characters ("2º Ano"), which is also why the README uses ../data/...
    paths = [Path(os.path.relpath(image_path(n))) for n in names]
    enc.encode(paths[0], cache_dir=cache_dir)                      # warm-up
    t0 = time.perf_counter()
    for p in paths:
        enc.encode(p, cache_dir=cache_dir)
    return 1000 * (time.perf_counter() - t0) / len(names)


# =============================================================================
# Visualisation
# =============================================================================
def _framed(ax, name, colour, title, fontsize=7):
    ax.imshow(Image.open(image_path(name)).convert("RGB"))
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(colour), s.set_linewidth(4)
    ax.set_title(title, fontsize=fontsize)


def show_results(rows, corpus_subset, top=10, target=TARGET, title=""):
    """One row per (label, query image or text, scores): the query followed by
    its top results, green frame = relevant, red frame = irrelevant."""
    corpus = load_raw("dinov2_vits14", corpus_subset)
    rel = relevance(corpus, target)
    fig, axes = plt.subplots(len(rows), top + 1, figsize=(1.6 * (top + 1), 1.9 * len(rows)),
                             squeeze=False)
    for ax_row, (label, query, s) in zip(axes, rows):
        order = np.argsort(-s, kind="stable")[:top]
        q_ax = ax_row[0]
        if str(query).isdigit():                     # image id
            _framed(q_ax, query, "royalblue", f"{label}\nquery {query}")
        else:                                        # text query
            q_ax.axis("off")
            q_ax.text(0.5, 0.5, "\n".join(_wrap(query, 18)), ha="center", va="center",
                      fontsize=7, transform=q_ax.transAxes)
            q_ax.set_title(label, fontsize=7)
        for r, (ax, i) in enumerate(zip(ax_row[1:], order), 1):
            other = ",".join(corpus.labels[i])
            _framed(ax, corpus.names[i], "limegreen" if rel[i] else "red",
                    f"#{r} {s[i]:.2f}\n{other}", fontsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _wrap(text, width):
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > width and line:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    return lines + [line]


def show_queries(names, titles, ncols=6, title=""):
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows + .4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, n, t in zip(axes.ravel(), names, titles):
        ax.imshow(Image.open(image_path(n)).convert("RGB"))
        ax.set_title(t, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def frequent_false_positives(scores, corpus_subset, top=20, n=12, target=TARGET):
    """Irrelevant images that appear most often in the top-`top` of the queries."""
    corpus = load_raw("dinov2_vits14", corpus_subset)
    rel = relevance(corpus, target)
    counts = {}
    for s in scores:
        for i in np.argsort(-s, kind="stable")[:top]:
            if not rel[i]:
                counts[i] = counts.get(i, 0) + 1
    best = sorted(counts, key=counts.get, reverse=True)[:n]
    return [(corpus.names[i], counts[i], ",".join(corpus.labels[i])) for i in best]
