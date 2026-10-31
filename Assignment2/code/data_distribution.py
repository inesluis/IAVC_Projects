"""Análise simples da distribuição dos dados (splits em data/splits).

Uso (a partir de code/):
    python data_distribution.py
Gera um resumo no terminal e figuras em ../docs/data_distribution/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
IMG_DIR = ROOT / "data" / "images"
OUT_DIR = ROOT / "docs" / "data_distribution"


def load_splits():
    """Uma linha por (imagem, classe, subset)."""
    rows = []
    for f in sorted(SPLITS_DIR.glob("*.txt")):
        cls, subset = f.stem.rsplit("_", 1)
        for name in f.read_text().split():
            rows.append({"image": name, "class": cls, "subset": subset})
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_splits()
    classes = sorted(df["class"].unique())

    # 1) Nº de imagens por classe e subset
    counts = df.pivot_table(index="class", columns="subset", values="image",
                            aggfunc="count", fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts["% train"] = (100 * counts["train"] / counts["total"]).round(1)
    print("\n=== Imagens por classe ===")
    print(counts)

    # 2) Imagens únicas e sobreposição train/val
    train_imgs = set(df.loc[df.subset == "train", "image"])
    val_imgs = set(df.loc[df.subset == "val", "image"])
    print("\n=== Imagens únicas ===")
    print(f"train: {len(train_imgs)} | val: {len(val_imgs)} | "
          f"em ambos (fuga de dados?): {len(train_imgs & val_imgs)}")
    n_files = len(list(IMG_DIR.glob("*.jpg")))
    print(f"ficheiros em data/images: {n_files} | "
          f"sem split: {n_files - len(train_imgs | val_imgs)}")

    # 3) Multi-label: nº de classes por imagem
    labels_per_img = df.groupby(["subset", "image"])["class"].nunique()
    print("\n=== Nº de classes por imagem ===")
    print(labels_per_img.groupby("subset").value_counts().unstack(fill_value=0))

    # 4) Co-ocorrência entre classes (train + val)
    onehot = pd.crosstab(df["image"], df["class"]).clip(upper=1)
    cooc = onehot.T @ onehot
    print("\n=== Co-ocorrência entre classes ===")
    print(cooc)

    # 5) Tamanhos das imagens
    sizes = []
    for name in sorted(train_imgs | val_imgs):
        p = IMG_DIR / f"{name}.jpg"
        if p.exists():
            with Image.open(p) as im:
                sizes.append(im.size)
    sizes = np.array(sizes)
    print("\n=== Tamanho das imagens (largura x altura) ===")
    print(pd.DataFrame(sizes, columns=["width", "height"]).describe().round(1))

    # --- Figuras ---
    ax = counts[["train", "val"]].plot.bar(figsize=(8, 4), rot=0)
    ax.set_ylabel("Nº de imagens")
    ax.set_title("Imagens por classe e subset")
    for c in ax.containers:
        ax.bar_label(c, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "class_counts.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cooc.values, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cooc.values[i, j], ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title("Co-ocorrência de classes")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cooccurrence.png", dpi=150)
    plt.close()

    if len(sizes):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(sizes[:, 0], sizes[:, 1], s=5, alpha=0.3)
        ax.set_xlabel("Largura (px)")
        ax.set_ylabel("Altura (px)")
        ax.set_title("Tamanho das imagens")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "image_sizes.png", dpi=150)
        plt.close()

    print(f"\nFiguras guardadas em {OUT_DIR}")


if __name__ == "__main__":
    main()
