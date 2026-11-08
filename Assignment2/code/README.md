# IACV Assignment #2 — Object Recognition and Image Retrieval

**Group:** _Carolina Gonçalves (2022221124)_, _Inês Luís (2022216107)_

This folder contains the code for all five parts of the assignment:

| Part | Topic | Where |
|------|-------|-------|
| 1 | BoVW + linear SVM | `1_handcraft_final.ipynb` |
| 2 | VLAD and Fisher vectors | `1_handcraft_final.ipynb` |
| 3 | Pretrained CNN/ViT features and CAM | `2_deep_models.ipynb` |
| 4 | Pretrained representations (ResNet, ViT, DINOv2, CLIP) | `3_part4_pretrained.ipynb` |
| 5 | Retrieval challenge (motorbike) | `4_part5_retrieval.ipynb` |

All commands below are run **from this directory (`code/`)**. The dataset lives
one level up, in `../data/`.

---

## 1. Environment

Tested with **Python 3.11** (conda), on Windows 11 and Linux, with an NVIDIA GPU
(CUDA 13.0). Everything also runs on CPU, but deep-feature extraction is much
slower.

On Linux:
```bash
conda env create -f env.yml
```

On Windows, `cyvlfeat` has no conda build, so a prebuilt wheel is used. Copy the
`wheel/` folder (course OneDrive materials) to `code/`, so that
`wheel/cyvlfeat-0.7.1-cp311-cp311-win_amd64.whl` exists, then:
```bash
conda env create -f env-win.yml
```

Then, on both systems:
```bash
conda activate cvc-a2
python -m pip install -r requirements.txt
```

`requirements.txt` installs the **CUDA 13.0** builds of PyTorch. If CUDA does
not work on your machine, comment out the CUDA lines, uncomment the CPU-only
`torch`/`torchvision` lines and run the install again.

Dependency files:

- `env.yml` / `env-win.yml`: Python 3.11, pip, `cyvlfeat` 0.7.1.
- `requirements.txt`: PyTorch, torchvision, OpenCV, scikit-learn, scikit-image,
  pandas, matplotlib, tqdm, CLIP, etc. (pinned versions).

The first run of Parts 3–5 downloads pretrained weights (TorchVision, DINOv2
through `torch.hub`, CLIP), so an internet connection is needed.

> **Windows note.** `cv2.imread` fails on absolute paths with non-ASCII
> characters (e.g. `2º Ano`). The scripts therefore use relative paths
> (`../data/...`); keep running them from `code/`.

## 2. Dataset

Extract `data.zip` (course materials) into `../data/`, giving:

```text
data/
├── images/      # all images (.jpg)
└── splits/      # original <class>_train.txt / <class>_val.txt files
```

Classes: aeroplane, car, horse, motorbike, person, plus `background` (negatives).

## 3. Full pipeline (in order)

```bash
# 3.1 Train/val/test split  -> ../data/splits/splits.json
python make_splits.py

# 3.2 Hand-crafted encoders (trained on the train subset only) -> ../data/encoders/
python -m src.train -e bovw -k 512 -c 0 --splits-file ../data/splits/splits.json --img-dir ../data/images --output-dir ../data/encoders
python -m src.train -e vlad -k 32 -c 80 --splits-file ../data/splits/splits.json --img-dir ../data/images --output-dir ../data/encoders
python -m src.train -e fv   -k 32 -c 80 --splits-file ../data/splits/splits.json --img-dir ../data/images --output-dir ../data/encoders

# 3.3 Extra BoVW vocabularies (K x seeds) for the vocabulary-size study (Parts 1 and 5)
python -m handcrafted.build_encoders --seeds 0 1 2 --ks 32 64 128 256 512 1024

# 3.4 Extract hand-crafted embeddings -> ../data/encodings/<encoder>/
python -m src.extract -e ../data/encoders/bovw_K512_C0.npz --splits-file ../data/splits/splits.json --img-dir ../data/images --output-dir ../data/encodings/ --cache-dir cache/ -f
python -m src.extract -e ../data/encoders/vlad_K32_C80.npz --splits-file ../data/splits/splits.json --img-dir ../data/images --output-dir ../data/encodings/ --cache-dir cache/ -f
python -m src.extract -e ../data/encoders/fv_K32_C80.npz   --splits-file ../data/splits/splits.json --img-dir ../data/images --output-dir ../data/encodings/ --cache-dir cache/ -f
# repeat for bovw_K256_C0.npz and bovw_K1024_C0.npz (needed by Part 5)

# 3.5 Frozen deep features (ResNet-18 layers 2-4, ViT-B/32, DINOv2, CLIP; with and without flip)
python -m src.extract_deep

# 3.6 Notebooks, in order
#   1_handcraft_final.ipynb -> 2_deep_models.ipynb -> 3_part4_pretrained.ipynb -> 4_part5_retrieval.ipynb
```

Useful options:

- `src.train`: `-i` images per class (default 150), `-n` descriptors per image
  (default 400), `--cache-dir`.
- `src.extract`: `-f` overwrite, `-j` threads, `--catg` subset of classes
  (always include `background`).
- `src.extract_deep`: `-m` models (`resnet18 vit_b_32 dinov2_vits14
  clip_vit_b_32`), `--flip both|none|only`, `-b` batch size, `-f` overwrite.

Outputs:

- Encoders: `../data/encoders/<ENCODER>_K<K>_C<C>.npz` (`C0` = no PCA).
- Embeddings: `../data/encodings/<feature>/<class>_{train,val,test}.npz`, each
  with `file_names` and `histograms` (one row per image).
- Tables and figures: `../results/` (Part 4 in `results/part4/`, Part 5 in
  `results/part5/`).

## 4. Evaluation protocol

`make_splits.py` builds a fixed split (seed 42) from the original files:

| Subset | Source | Images |
|--------|--------|--------|
| `train` | 80% of the original `*_train.txt` | 1977 |
| `val`   | 20% of the original `*_train.txt` | 495  |
| `test`  | the original `*_val.txt` (unchanged) | 2474 |

The split is stratified by each image's combination of classes (the dataset is
multi-label). Encoders and SVMs are trained on `train`. Hyperparameters (SVM
`C`, K, PCA dimension, layer, etc.) are chosen on `val`. `test` is used only
once, for the final numbers. Small-data experiments use at least 3 seeds and
report mean ± std.

`data_distribution.py` prints dataset statistics (class counts, multi-label
overlap, co-occurrence, image sizes) and saves plots to
`../docs/data_distribution/`.

## 5. Project structure

```text
code/
├── make_splits.py              # Builds data/splits/splits.json.
├── data_distribution.py        # Dataset statistics and plots.
├── handcrafted/
│   └── build_encoders.py       # Extra BoVW vocabularies (K x seeds) + fit times.
├── src/
│   ├── train.py                # Train BoVW / VLAD / Fisher encoders.
│   ├── extract.py              # Extract embeddings with a trained encoder.
│   ├── encoders.py             # Encoder implementations and aggregation.
│   ├── features.py             # Image preparation, dense SIFT/PHOW.
│   ├── utils.py                # Dataset tables, splits, SVM helpers.
│   ├── visualization.py        # Rankings, PR curves, embedding plots.
│   ├── deep_models.py          # Pretrained models and intermediate features.
│   ├── extract_deep.py         # Cache frozen ResNet/ViT/DINOv2/CLIP features.
│   ├── metrics.py              # AP, mAP, P@K, R@K, PR curves.
│   ├── evaluate.py             # Part 4: C selection, AP, few-shot, CLIP zero-shot, CAM.
│   └── retrieval.py            # Part 5: queries, similarity ranking, P@K/R@K/AP.
├── 1_handcraft_final.ipynb     # Parts 1-2.
├── 2_deep_models.ipynb         # Part 3.
├── 3_part4_pretrained.ipynb    # Part 4.
├── 4_part5_retrieval.ipynb     # Part 5.
├── env.yml / env-win.yml       # Conda environments.
└── requirements.txt            # Pip dependencies.
```

## 6. Reused code and references

**Starter code provided by the course (IACV 2026/27).** The following files come
from the assignment's base code. They were adapted to read the train/val/test
split from `splits.json`; the algorithms are unchanged:

- `src/train.py`, `src/extract.py`, `src/encoders.py`, `src/features.py`,
  `src/utils.py`, `src/visualization.py`, `src/deep_models.py`, `env.yml`,
  `env-win.yml`, `requirements.txt`.

**Written by the group:** `make_splits.py`, `data_distribution.py`,
`handcrafted/build_encoders.py` (reuses the sampling and k-means code from
`src/train.py`), `src/extract_deep.py`, `src/metrics.py`, `src/evaluate.py`,
`src/retrieval.py`, and the four notebooks.

**External libraries and pretrained models:**

- [VLFeat](https://www.vlfeat.org/) through
  [cyvlfeat](https://github.com/menpo/cyvlfeat): dense SIFT (PHOW), k-means,
  GMM, VLAD and Fisher encoding.
- [scikit-learn](https://scikit-learn.org/): linear SVMs, PCA, AP metrics.
- [TorchVision](https://github.com/pytorch/vision)
  ([models](https://pytorch.org/vision/stable/models.html)): ResNet-18 and
  ViT-B/32 ImageNet weights and preprocessing.
- [DINOv2](https://github.com/facebookresearch/dinov2): `dinov2_vits14`, loaded
  with `torch.hub`.
- [OpenAI CLIP](https://github.com/openai/CLIP): `ViT-B/32` image and text
  encoders.
- [OpenCV](https://opencv.org/), [scikit-image](https://scikit-image.org/),
  [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/).
