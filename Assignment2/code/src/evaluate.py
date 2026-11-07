"""Evaluation helpers for Part 4 (pretrained representations).

The binary SVMs cannot be compared across classes (each one has its own C and
never saw the other classes), so the confusion matrix uses a separate
multi-class SVM with 5 classes + background (evaluate_multiclass).

Features are read from data/encodings/<feature>/<class>_<subset>.npz
(written by src.extract_deep, or by src.extract for BoVW/VLAD/FV).
"""
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.svm import LinearSVC
from torchvision import transforms

from .utils import train_svm

ROOT = Path(__file__).resolve().parents[2]   # Assignment2/
DATA = ROOT / "data"
CLASSES = ["aeroplane", "car", "horse", "motorbike", "person"]
BACKGROUND = len(CLASSES)                    # label index of background (5)
C_GRID = [0.01, 0.1, 1, 10, 100, 1000]
MAX_ITER = 10000   # utils.train_svm defaults to 100 iterations, too few here


# =============================================================================
# Data
# =============================================================================
def l2norm(X):
    """Divide every row of X by its L2 norm."""
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


@dataclass
class Split:
    """All images of one subset (each image once, sorted by name)."""
    X: np.ndarray       # (N, D) L2-normalised features
    names: np.ndarray   # (N,) image ids
    Y: np.ndarray       # (N, 5) True where the image contains CLASSES[c]
    bg: np.ndarray      # (N,) True for background images

    def task(self, c):
        """Binary task of class c: images with the class (+1) or background (-1).

        Returns the features, the labels and the mask of the selected images.
        """
        mask = self.Y[:, c] | self.bg
        y = np.where(self.Y[mask, c], 1, -1)
        return self.X[mask], y, mask

    def multiclass(self, background=True):
        """Images with exactly one class (+ background images if requested).

        Returns the mask of the selected images and their labels:
        0..4 = index in CLASSES, 5 = background.
        """
        keep = self.Y.sum(1) == 1
        if background:
            keep = keep | self.bg
        y = np.where(self.bg, BACKGROUND, self.Y.argmax(1))
        return keep, y[keep]


@lru_cache(maxsize=None)
def load_splits():
    """splits.json: {subset: {image id: [classes]}}."""
    with open(DATA / "splits" / "splits.json", encoding="utf-8") as f:
        return json.load(f)


HELLINGER = "_hellinger"   # suffix: Hellinger map of a BoVW feature (Part 1 final setup)


def has_feature(feature):
    """True if the feature was extracted for train, val and test."""
    folder = DATA / "encodings" / feature.removesuffix(HELLINGER)
    return all((folder / f"background_{s}.npz").exists() for s in ("train", "val", "test"))


@lru_cache(maxsize=None)
def load_split(feature, subset):
    """Read all <class>_<subset>.npz files of a feature into one Split.

    "<bovw feature>_hellinger" gives the Hellinger map used in Part 1:
    L1 normalisation, square root, then L2.
    """
    hellinger = feature.endswith(HELLINGER)
    feature = feature.removesuffix(HELLINGER)
    labels = load_splits()[subset]                 # image id -> list of classes
    vectors = {}                                   # image id -> feature vector
    for cat in CLASSES + ["background"]:
        with np.load(DATA / "encodings" / feature / f"{cat}_{subset}.npz") as z:
            vectors.update(zip(z["file_names"].astype(str), z["histograms"]))

    names = np.array(sorted(labels))
    X = np.stack([vectors[n] for n in names]).astype(np.float32)
    Y = np.array([[c in labels[n] for c in CLASSES] for n in names])
    bg = np.array(["background" in labels[n] for n in names])
    if hellinger:
        X = np.sqrt(np.maximum(X, 0) / np.maximum(X.sum(1, keepdims=True), 1e-12))
    return Split(l2norm(X), names, Y, bg)


def load_all(feature):
    return {s: load_split(feature, s) for s in ("train", "val", "test")}


def image_path(name):
    return DATA / "images" / f"{name}.jpg"


# =============================================================================
# Binary SVMs (AP per class) — Stages A, B, D
# =============================================================================
def fit(X, y, C, seed=42):
    """Linear SVM (hinge loss). Returns weights W and bias b."""
    return train_svm(X, y, C=C, max_iter=MAX_ITER, seed=seed)


def ap(y, scores):
    """Average precision of a ranking (positive class = 1 / True)."""
    return average_precision_score(y, scores)


def pick_C(sweep, metric):
    """C with the best validation score; ties go to the C closest to 1."""
    best = sweep[np.isclose(sweep[metric], sweep[metric].max())]
    return best.loc[np.abs(np.log10(best.C)).idxmin(), "C"]


def evaluate_feature(feature, verbose=True):
    """Binary SVM per class: choose C on val, train on train, evaluate on test.

    Returns a dict with
        per_class : DataFrame (class, C, val_ap, test_ap, train_s)
        sweep     : DataFrame with train/val AP for every C (Stage D)
        W, b      : final weights (5, D) and biases (5,)
        scores    : (N_test, 5) scores of every test image
        test      : the test Split
        infer_us  : time to score one image for the 5 classes (microseconds)
        dim       : feature dimension
    """
    d = load_all(feature)
    rows, sweep, W, b = [], [], [], []
    for c, cls in enumerate(CLASSES):
        Xtr, ytr, _ = d["train"].task(c)
        Xva, yva, _ = d["val"].task(c)
        Xte, yte, _ = d["test"].task(c)

        # 1) sweep C: train on train, measure AP on train and val
        cls_sweep = []
        for C in C_GRID:
            w, b0 = fit(Xtr, ytr, C)
            cls_sweep.append({"class": cls, "C": C,
                              "train_ap": ap(ytr, Xtr @ w + b0),
                              "val_ap": ap(yva, Xva @ w + b0)})
        C = pick_C(pd.DataFrame(cls_sweep), "val_ap")
        sweep += cls_sweep

        # 2) final model with the chosen C, evaluated once on test
        t0 = time.perf_counter()
        w, b0 = fit(Xtr, ytr, C)
        train_s = time.perf_counter() - t0
        rows.append({"class": cls, "C": C, "val_ap": max(r["val_ap"] for r in cls_sweep),
                     "test_ap": ap(yte, Xte @ w + b0), "train_s": train_s})
        W.append(w)
        b.append(b0)

    # 3) scores of all test images for the 5 classes (timed: SVM inference)
    W, b = np.stack(W), np.array(b)
    t0 = time.perf_counter()
    scores = d["test"].X @ W.T + b
    infer_us = 1e6 * (time.perf_counter() - t0) / len(scores)

    per_class = pd.DataFrame(rows)
    if verbose:
        aps = "  ".join(f"{r['class'][:5]}={r['test_ap']:.3f}" for r in rows)
        print(f"{feature:26s} dim={W.shape[1]:4d}  {aps}  mAP={per_class.test_ap.mean():.3f}")
    return {"feature": feature, "per_class": per_class, "sweep": pd.DataFrame(sweep),
            "W": W, "b": b, "scores": scores, "test": d["test"],
            "infer_us": infer_us, "dim": W.shape[1]}


def ap_table(results, value="test_ap"):
    """Table with one row per feature, one column per class, plus mAP."""
    table = pd.DataFrame({name: r["per_class"].set_index("class")[value]
                          for name, r in results.items()}).T
    table["mAP"] = table[CLASSES].mean(1)
    return table


# =============================================================================
# Multi-class SVM and confusion matrix
# =============================================================================
def evaluate_multiclass(feature, verbose=True):
    """One-vs-rest linear SVM over 6 classes (5 objects + background).

    Trained on single-label + background images of train, C chosen on the
    val macro-F1. Returns the (N_test, 6) scores of every test image and the
    AP of each class against ALL other test images (harder than vs background).
    """
    d = load_all(feature)
    keep_tr, ytr = d["train"].multiclass()
    keep_va, yva = d["val"].multiclass()
    Xtr, Xva = d["train"].X[keep_tr], d["val"].X[keep_va]

    def train(C):
        return LinearSVC(C=C, max_iter=MAX_ITER, random_state=42).fit(Xtr, ytr)

    sweep = []
    for C in C_GRID:
        clf = train(C)
        sweep.append({"C": C,
                      "train_f1": f1_score(ytr, clf.predict(Xtr), average="macro"),
                      "val_f1": f1_score(yva, clf.predict(Xva), average="macro")})
    sweep = pd.DataFrame(sweep)
    C = pick_C(sweep, "val_f1")
    clf = train(C)

    test = d["test"]
    scores = clf.decision_function(test.X)                  # (N_test, 6)
    ap_all = {cls: ap(test.Y[:, c], scores[:, c]) for c, cls in enumerate(CLASSES)}
    if verbose:
        aps = "  ".join(f"{k[:5]}={v:.3f}" for k, v in ap_all.items())
        print(f"{feature:26s} C={C:<6} {aps}  mAP(vs all)={np.mean(list(ap_all.values())):.3f}")
    return {"feature": feature, "C": C, "sweep": sweep, "scores": scores,
            "test": test, "ap_vs_all": ap_all}


def _predictions(scores, split):
    """True and predicted labels (argmax) of the images used in the confusion
    matrix. Scores with 6 columns -> background images are a 6th class."""
    background = scores.shape[1] == 6
    keep, true = split.multiclass(background)
    pred = scores[keep].argmax(1)
    labels = CLASSES + ["background"] * background
    return keep, true, pred, labels


def confusion(scores, split, title="", ax=None):
    """Plot the confusion matrix; returns it and the per-class report."""
    _, true, pred, labels = _predictions(scores, split)
    idx = list(range(len(labels)))
    cm = confusion_matrix(true, pred, labels=idx)
    report = classification_report(true, pred, labels=idx, target_names=labels,
                                   output_dict=True, zero_division=0)
    ax = ax or plt.subplots(figsize=(5.5, 4.5))[1]
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, colorbar=False, xticks_rotation=45, values_format="d")
    ax.set_title(f"{title}\nacc={report['accuracy']:.3f}  "
                 f"macro-F1={report['macro avg']['f1-score']:.3f}")
    return cm, pd.DataFrame(report).T


def misclassified(scores, split, n=12):
    """The n most confident mistakes: (image id, true label, predicted label)."""
    keep, true, pred, labels = _predictions(scores, split)
    s, names = scores[keep], split.names[keep]
    wrong = np.where(pred != true)[0]
    # confidence = how much the wrong class beats the true class
    gap = s[wrong, pred[wrong]] - s[wrong, true[wrong]]
    wrong = wrong[np.argsort(-gap)][:n]
    return [(names[i], labels[true[i]], labels[pred[i]]) for i in wrong]


# =============================================================================
# Few-shot learning, flip augmentation and C with few positives — Stages C, D
# =============================================================================
def sample_positives(pos, k, seed):
    """k random positive indices (all of them when k is None)."""
    if k is None or k >= len(pos):
        return pos
    return np.random.default_rng(seed).choice(pos, size=k, replace=False)


def few_shot(feature, ks=(1, 5, 10, 20, None), seeds=range(5), C=1.0,
             flip=(False, True)):
    """Val AP vs number of positive training images (k=None means all).

    Negatives are always all background images of train. With flip=True the
    mirrored features of the selected images are added to the training set
    (train only; val is never augmented).
    """
    train, val = load_split(feature, "train"), load_split(feature, "val")
    if True in flip:
        train_flip = load_split(feature + "_flip", "train")   # same image order

    rows = []
    for c, cls in enumerate(CLASSES):
        pos, neg = np.where(train.Y[:, c])[0], np.where(train.bg)[0]
        Xva, yva, _ = val.task(c)
        for k in ks:
            # with all positives every seed gives the same sample -> one run
            for seed in (seeds if k is not None else [0]):
                idx = np.concatenate([sample_positives(pos, k, seed), neg])
                X, y = train.X[idx], np.where(train.Y[idx, c], 1, -1)
                for use_flip in flip:
                    if use_flip:
                        Xf, yf = np.vstack([X, train_flip.X[idx]]), np.concatenate([y, y])
                    else:
                        Xf, yf = X, y
                    w, b = fit(Xf, yf, C, seed=seed)
                    rows.append({"class": cls, "k": "all" if k is None else str(k),
                                 "seed": seed, "flip": use_flip,
                                 "val_ap": ap(yva, Xva @ w + b)})
    return pd.DataFrame(rows)


def few_shot_c_sweep(feature, k=5, seeds=range(5)):
    """Train and val AP for every C when only k positives are available."""
    train, val = load_split(feature, "train"), load_split(feature, "val")
    rows = []
    for c, cls in enumerate(CLASSES):
        pos, neg = np.where(train.Y[:, c])[0], np.where(train.bg)[0]
        Xva, yva, _ = val.task(c)
        for seed in seeds:
            idx = np.concatenate([sample_positives(pos, k, seed), neg])
            X, y = train.X[idx], np.where(train.Y[idx, c], 1, -1)
            for C in C_GRID:
                w, b = fit(X, y, C, seed=seed)
                rows.append({"class": cls, "seed": seed, "C": C,
                             "train_ap": ap(y, X @ w + b),
                             "val_ap": ap(yva, Xva @ w + b)})
    return pd.DataFrame(rows)


def tta_check(feature, W, b, subset="val"):
    """Test-time flip with a linear SVM: averaging the features of the image
    and its mirror gives the same score as averaging their scores
    (w.(a+f)/2 + b = ((w.a + b) + (w.f + b)) / 2). Only an extra L2
    normalisation of the averaged feature changes the result."""
    split = load_split(feature, subset)
    split_flip = load_split(feature + "_flip", subset)
    rows = []
    for c, cls in enumerate(CLASSES):
        X, y, mask = split.task(c)
        Xf = split_flip.X[mask]

        def score(Z):
            return Z @ W[c] + b[c]

        mean_scores = (score(X) + score(Xf)) / 2
        mean_features = score((X + Xf) / 2)
        rows.append({"class": cls,
                     "AP no TTA": ap(y, score(X)),
                     "AP mean of scores": ap(y, mean_scores),
                     "AP mean of features": ap(y, mean_features),
                     "AP mean of features + L2": ap(y, score(l2norm((X + Xf) / 2))),
                     "max |score difference|": np.abs(mean_scores - mean_features).max()})
    return pd.DataFrame(rows).set_index("class")


def plot_curve(df, x, y="val_ap", hue="flip", ax=None, title="", xlabel=None):
    """Mean ± std over seeds of y averaged over classes, one line per hue.

    x is treated as categorical: its values are placed at equal spacing in the
    order they first appear in df (e.g. k = 1, 5, 10, 20, all or the C grid).
    """
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    order = list(dict.fromkeys(df[x]))
    # average over classes -> one value per (hue, x, seed)
    per_seed = df.groupby([hue, x, "seed"])[y].mean().reset_index()
    for h, g in per_seed.groupby(hue):
        stats = g.groupby(x)[y].agg(["mean", "std"]).reindex(order)
        ax.errorbar(range(len(order)), stats["mean"], yerr=stats["std"].fillna(0),
                    marker="o", capsize=3, label=f"{hue}={h}")
    ax.set_xticks(range(len(order)), order)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(f"{y} (mean over classes)")
    ax.set_title(title)
    ax.grid(alpha=.3)
    ax.legend()
    return ax


# =============================================================================
# CLIP zero-shot — Stage E
# =============================================================================
TEMPLATES = {
    "{}": ["{}"],
    "a photo of a {}.": ["a photo of a {}."],
    "a picture of a {}.": ["a picture of a {}."],
    "a blurry photo of a {}.": ["a blurry photo of a {}."],
    "ensemble (7 templates)": [
        "a photo of a {}.", "a picture of a {}.", "a blurry photo of a {}.",
        "a close-up photo of a {}.", "a photo of the {}.",
        "a low resolution photo of a {}.", "a photo of a {} in a scene.",
    ],
}
CLASS_SYNONYMS = {"aeroplane": ["aeroplane", "airplane", "plane", "aircraft"],
                  "car": ["car", "automobile"],
                  "horse": ["horse"],
                  "motorbike": ["motorbike", "motorcycle"],
                  "person": ["person", "people", "man", "woman"]}
BACKGROUND_PROMPTS = ["a photo of a room.", "a photo of a street.",
                      "a photo of a landscape.", "a photo of an object.",
                      "a photo of an animal."]


def load_clip(device):
    """CLIP ViT-B/32 (the image features are already cached; used for text)."""
    import clip
    model, _ = clip.load("ViT-B/32", device=device)
    return model


@torch.inference_mode()
def encode_text(model, prompts):
    """L2-normalised CLIP text embeddings, one row per prompt."""
    import clip
    device = next(model.parameters()).device
    tokens = clip.tokenize(prompts).to(device)
    return l2norm(model.encode_text(tokens).float().cpu().numpy())


def class_text_embeddings(model, templates, synonyms=False, background=False):
    """One embedding per class = mean over all its prompts (template x name).

    synonyms=True uses CLASS_SYNONYMS as names; background=True adds a 6th row
    built from BACKGROUND_PROMPTS.
    """
    rows = []
    for cls in CLASSES:
        names = CLASS_SYNONYMS[cls] if synonyms else [cls]
        prompts = [t.format(n) for t in templates for n in names]
        rows.append(encode_text(model, prompts).mean(0))
    if background:
        rows.append(encode_text(model, BACKGROUND_PROMPTS).mean(0))
    return l2norm(np.stack(rows))


def zero_shot_eval(split, text_emb):
    """Cosine similarities, per-class AP (class vs background) and accuracy
    on single-label images (argmax over the 5 class prompts)."""
    S = split.X @ text_emb.T            # both sides are L2-normalised
    aps = {}
    for c, cls in enumerate(CLASSES):
        _, y, mask = split.task(c)
        aps[cls] = ap(y, S[mask, c])
    keep, true = split.multiclass(background=False)
    acc = (S[keep, :5].argmax(1) == true).mean()
    return S, aps, acc


def prompt_selection(model, val, test):
    """val (used for the choice) and test (only reported) results of every
    prompt set, with and without synonyms."""
    rows = []
    for name, templates in TEMPLATES.items():
        for synonyms in (False, True):
            emb = class_text_embeddings(model, templates, synonyms)
            _, val_aps, val_acc = zero_shot_eval(val, emb)
            _, test_aps, test_acc = zero_shot_eval(test, emb)
            rows.append({"templates": name, "synonyms": synonyms,
                         "val_mAP": np.mean(list(val_aps.values())), "val_acc": val_acc,
                         "test_mAP": np.mean(list(test_aps.values())), "test_acc": test_acc})
    return pd.DataFrame(rows)


# =============================================================================
# Costs
# =============================================================================
def extraction_costs():
    """Extraction log of src.extract_deep (one row per feature, no flips)."""
    df = pd.read_csv(ROOT / "results" / "deep_extraction.csv")
    df = df[~df.feature.str.endswith("_flip")]
    return df.drop_duplicates("feature", keep="last").set_index("feature")


# =============================================================================
# Visual analysis — Stage F
# =============================================================================
def show_grid(names, titles, ncols=6, title=""):
    """Grid of images with one title each."""
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows + .4))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, name, t in zip(np.ravel(axes), names, titles):
        ax.imshow(Image.open(image_path(name)).convert("RGB"))
        ax.set_title(t, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def ranking_cases(scores, split, c, n=6):
    """For class c, the (names, scores) of: the best-ranked positives, the
    best-ranked background images (false positives) and the worst-ranked
    positives (missed)."""
    _, y, mask = split.task(c)
    s, names = scores[mask, c], split.names[mask]
    order = np.argsort(-s)                      # best score first
    pos = order[y[order] == 1]
    neg = order[y[order] == -1]
    missed = pos[::-1]
    return {"top positives": (names[pos[:n]], s[pos[:n]]),
            "top false positives": (names[neg[:n]], s[neg[:n]]),
            "missed positives": (names[missed[:n]], s[missed[:n]])}


# Images as the networks see them: Resize 256 + CenterCrop 224
CROP_224 = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])


@torch.inference_mode()
def local_features(kind, model, preprocess, name, device):
    """Grid of local features (H, W, D) of one image + the 224x224 crop.

    kind = "resnet18_layer4" -> 7x7 positions of block layer4.1 (512-D)
    kind = "dinov2_vits14_patchmean" -> 16x16 DINOv2 patch tokens (384-D)
    """
    img = Image.open(image_path(name)).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    if kind == "resnet18_layer4":
        _, blocks = model(x)                                  # IntermediateFeatureExtractor
        f = blocks["layer4.1"][0].permute(1, 2, 0)            # (7, 7, 512)
    else:
        patches = model.forward_features(x)["x_norm_patchtokens"][0]   # (256, 384)
        f = patches.reshape(16, 16, -1)
    return CROP_224(img), f.float().cpu().numpy()


def show_evidence(kind, names, w, b, cls, titles, device, ncols=4):
    """Where does the SVM of `kind` find evidence for class `cls`?

    The SVM was trained on g/|g|, where g = mean of the local features f_p.
    Its score therefore splits over the positions p:
        w.g/|g| + b = mean_p (w.f_p / |g|) + b
    so w.f_p/|g| is the exact contribution of position p (a CAM that uses our
    SVM instead of the ImageNet head). Red = for the class, blue = against.
    """
    from .deep_models import IntermediateFeatureExtractor, getModel
    if kind == "resnet18_layer4":
        resnet, preprocess = getModel("resnet18", device=device)
        model = IntermediateFeatureExtractor(resnet)
    else:
        model, preprocess = getModel("dinov2_vits14", device=device)

    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows + .4))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, name, t in zip(np.ravel(axes), names, titles):
        crop, f = local_features(kind, model, preprocess, name, device)
        g = f.reshape(-1, f.shape[-1]).mean(0)               # pooled feature
        contrib = f @ w / np.linalg.norm(g)                  # (H, W) map
        score = contrib.mean() + b
        v = np.abs(contrib).max()
        ax.imshow(crop)
        ax.imshow(contrib, cmap="bwr", vmin=-v, vmax=v, alpha=.5,
                  extent=(0, 224, 224, 0), interpolation="bilinear")
        ax.set_title(f"{t}\n{name}  score={score:+.2f}", fontsize=8)
    fig.suptitle(f"{kind}: SVM evidence for '{cls}' (red = for, blue = against)")
    fig.tight_layout()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fig
