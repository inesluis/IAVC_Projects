"""Cria o split de validação a partir do treino e guarda-o em data/splits/splits.json.

- Os ficheiros *_train.txt são divididos em train / val.
- Os ficheiros *_val.txt passam a ser o conjunto de teste (não são tocados).
- Estratificação pela combinação de classes de cada imagem (o dataset é
  multi-label); combinações com menos de MIN_GROUP imagens são agrupadas.

Uso (a partir da raiz do projeto ou de code/):
    python code/make_splits.py
"""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
OUT_FILE = SPLITS_DIR / "splits.json"

VAL_FRACTION = 0.20
SEED = 42
MIN_GROUP = 5


def load_labels(subset):
    """Devolve {imagem: [classes]} para o subset 'train' ou 'val' dos .txt."""
    labels = defaultdict(set)
    for f in sorted(SPLITS_DIR.glob(f"*_{subset}.txt")):
        cls = f.stem.rsplit("_", 1)[0]
        for name in f.read_text().split():
            labels[name].add(cls)
    return {k: sorted(v) for k, v in sorted(labels.items())}


def class_counts(images, labels):
    counts = defaultdict(int)
    for img in images:
        for c in labels[img]:
            counts[c] += 1
    return dict(sorted(counts.items()))


def main():
    train_labels = load_labels("train")
    test_labels = load_labels("val")  # o "_val" original é o teste

    images = list(train_labels)
    strata = pd.Series(["+".join(train_labels[i]) for i in images])
    sizes = strata.map(strata.value_counts())
    strata[sizes < MIN_GROUP] = "rare"

    train_imgs, val_imgs = train_test_split(
        images, test_size=VAL_FRACTION, random_state=SEED, stratify=strata
    )
    train_imgs, val_imgs = sorted(train_imgs), sorted(val_imgs)

    assert not set(train_imgs) & set(val_imgs)
    assert not (set(train_imgs) | set(val_imgs)) & set(test_labels)

    splits = {
        "meta": {
            "description": "Validação criada a partir de *_train.txt; "
                           "teste = *_val.txt originais.",
            "val_fraction": VAL_FRACTION,
            "seed": SEED,
            "stratify": "combinação de classes por imagem "
                        f"(combinações < {MIN_GROUP} imagens agrupadas)",
            "sizes": {"train": len(train_imgs), "val": len(val_imgs),
                      "test": len(test_labels)},
            "class_counts": {
                "train": class_counts(train_imgs, train_labels),
                "val": class_counts(val_imgs, train_labels),
                "test": class_counts(test_labels, test_labels),
            },
        },
        # imagem -> lista de classes
        "train": {i: train_labels[i] for i in train_imgs},
        "val": {i: train_labels[i] for i in val_imgs},
        "test": test_labels,
    }

    OUT_FILE.write_text(json.dumps(splits, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    table = pd.DataFrame(splits["meta"]["class_counts"]).fillna(0).astype(int)
    table["% val"] = (100 * table["val"] / (table["train"] + table["val"])).round(1)
    print(table)
    print(splits["meta"]["sizes"])
    print(f"Guardado em {OUT_FILE}")


if __name__ == "__main__":
    main()
