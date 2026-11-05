# Image Encoding Assignment

This project trains visual encoders from local image descriptors and extracts
one global embedding per image. The supported encoders are:

- **BoVW**: a histogram of visual-word assignments.
- **VLAD**: aggregated residuals to visual-word centers.
- **Fisher vector**: first- and second-order statistics from a GMM.

Run the commands below from the project root, the directory containing
`README.md`.

## Install Anaconda Environment

Create the project environment from the included `env.yml` file:

On Linux, run:
```bash
conda env create -f env.yml
```

On Windows, download the `wheel` folder from the OneDrive materials and copy it
to the project root. It should contain:

```text
wheel/cyvlfeat-0.7.1-cp311-cp311-win_amd64.whl
```

Then run:
```bash
conda env create -f env-win.yml
```

Then
```
conda activate cvc-a2
```

Install the Python packages listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

The requirements file is configured for CUDA 12.6 PyTorch by default. If CUDA
installation or execution does not work on your machine, edit
`requirements.txt`: comment out the CUDA lines and uncomment the CPU-only
`torch` and `torchvision` lines, then rerun the installation command.


## Project Structure

```text
.
├── data/
│   ├── images/                 # Dataset images.
│   ├── splits/                 # Original *_train.txt / *_val.txt files
│   │   └── splits.json         #   + train/val/test split (make_splits.py).
│   ├── encoders/               # Trained encoder configurations (*.npz).
│   └── encodings/              # Per-image embeddings grouped by encoder.
├── src/
│   ├── train.py                # Train BoVW, VLAD, or Fisher encoders.
│   ├── extract.py              # Extract embeddings using a trained encoder.
│   ├── encoders.py             # Encoder implementations and image aggregation.
│   ├── deep_models.py          # Pretrained models and intermediate features.
│   ├── features.py             # Image preparation and dense SIFT/PHOW features.
│   ├── utils.py                # Dataset tables, splits, and SVM helpers.
│   └── visualization.py        # Ranking, precision-recall, and embedding plots.
├── data_distribution.py        # Dataset statistics and plots.
├── make_splits.py              # Builds data/splits/splits.json.
├── 1_handcraft.ipynb           # Hand-crafted feature experiments.
├── 2_deep_models.ipynb         # Deep-model features and CAM visualization.
```

Encoder files are named `<ENCODER>_K<K>_C<C>.npz`, where `K<K>` is the number of
visual words or GMM components and `C<C>` is the PCA dimension. `C0` means PCA
is disabled. Extracted files use the same encoder name as a directory, for
example:

```text
data/encodings/bovw_K512_C0/
├── aeroplane_train.npz
├── aeroplane_val.npz
├── aeroplane_test.npz
├── background_train.npz
└── ...
```

Each encoding archive contains `file_names` and `histograms`. The latter is a
matrix with one row per image.

> **Paths.** `python -m src.*` must be run from this directory (`code/`),
> while the dataset lives one level up in `../data/`. The commands below
> therefore pass the `../data/...` paths explicitly.

## Data Analysis and Splits

### Data distribution (`data_distribution.py`)

```bash
python data_distribution.py
```

Reads the original split files in `data/splits/` and prints:

- number of images per class in the original `train` and `val` files;
- number of unique images per subset and the overlap between them (should be 0);
- number of classes per image (the dataset is **multi-label**: the same image
  can appear in, e.g., `person_train.txt` and `horse_train.txt`);
- class co-occurrence matrix;
- image size statistics.

Plots are saved to `../docs/data_distribution/` (`class_counts.png`,
`cooccurrence.png`, `image_sizes.png`).

### Train / validation / test split (`make_splits.py`)

The original files are split into `*_train.txt` and `*_val.txt`. The protocol
requires a validation set taken from the training data, so:

| New subset | Source                            | Images |
|------------|-----------------------------------|--------|
| `train`    | 80% of the original `*_train.txt` | 1977   |
| `val`      | 20% of the original `*_train.txt` | 495    |
| `test`     | the original `*_val.txt` (unchanged) | 2474 |

```bash
python make_splits.py
```

- **Stratified by class:** images are stratified by their *combination* of
  classes (e.g. `horse+person`), so both per-class proportions and the
  multi-label cases are preserved. Every class has 19.6–20.1% of its training
  images in `val`. Combinations with fewer than 5 images are grouped together.
- **Fixed seed (42):** running the script again produces exactly the same file.
- **Output:** `data/splits/splits.json`, with a `meta` block (seed, fraction,
  per-class counts) and, for each subset, a mapping `image id -> [classes]`.
  The original `.txt` files are only read, never modified.

**The whole team should use the same `splits.json`.** Do not regenerate it
with a different seed or fraction.

To load a subset in Python (same table format as `loadSamplesTable`):

```python
from src.utils import loadSplitsTable, loadSplitsDataTables

df_train = loadSplitsTable("../data/splits/splits.json", "../data/images", subset="train")
df_all = loadSplitsDataTables("../data/splits/splits.json", "../data/images")  # adds a 'subset' column
```

## Train Encoders

Encoders are trained **only on the `train` subset of `splits.json`**, so they
never see validation or test images.

> Encoders trained before `splits.json` existed used the full original
> training set (including the images that are now `val`) and must be
> **retrained**.

Train a BoVW encoder with 512 visual words and no PCA, and VLAD and Fisher
encoders with 32 components and an 80-dimensional PCA projection:

```bash
python -m src.train -e bovw -k 512 -c 0 \
  --splits-file ../data/splits/splits.json --img-dir ../data/images --output-dir ../data/encoders

python -m src.train -e vlad -k 32 -c 80 \
  --splits-file ../data/splits/splits.json --img-dir ../data/images --output-dir ../data/encoders

python -m src.train -e fv -k 32 -c 80 \
  --splits-file ../data/splits/splits.json --img-dir ../data/images --output-dir ../data/encoders
```

The resulting configurations are written to `data/encoders/` using names such
as `vlad_K32_C80.npz`. Useful options include:

```text
--splits-file    splits.json from make_splits.py (default: data/splits/splits.json)
--img-dir        Image directory (default: data/images)
--output-dir     Encoder output directory (default: data/encoders)
-i, --per-class  Images sampled per class; -1 uses all (default: 150)
-n, --per-image  Local descriptors sampled per image (default: 400)
-j, --jobs       Reserved parallel-job option (default: 4)
--cache-dir      Optional feature-cache directory (default: disabled)
```

## Extract Embeddings

Extraction writes one file per class and subset, `<class>_train.npz`,
`<class>_val.npz` and `<class>_test.npz`, using `splits.json`.

> Encodings extracted before `splits.json` existed have a `*_val.npz` that
> actually contains the **test** images. After retraining the encoders,
> re-extract **with `-f`** so the old files are overwritten.

```bash
python -m src.extract -e ../data/encoders/bovw_K512_C0.npz \
  --splits-file ../data/splits/splits.json --img-dir ../data/images \
  --output-dir ../data/encodings/ --cache-dir cache/ -f

python -m src.extract -e ../data/encoders/vlad_K32_C80.npz \
  --splits-file ../data/splits/splits.json --img-dir ../data/images \
  --output-dir ../data/encodings/ --cache-dir cache/ -f

python -m src.extract -e ../data/encoders/fv_K32_C80.npz \
  --splits-file ../data/splits/splits.json --img-dir ../data/images \
  --output-dir ../data/encodings/ --cache-dir cache/ -f
```

Extraction writes files under a directory named after the encoder, for
example `data/encodings/vlad_K32_C80/`. By default, existing files are kept;
add `-f` to overwrite them. Use `-j` to control the number of worker threads
and `--cache-dir cache/` to reuse cached local descriptors. Add `--catg` to
extract only some classes (always include `background`, which is used as the
negative class):

```bash
python -m src.extract \
  -e ../data/encoders/vlad_K32_C80.npz \
  --splits-file ../data/splits/splits.json --img-dir ../data/images \
  --output-dir ../data/encodings/ \
  --catg background motorbike aeroplane person \
  --cache-dir cache/ \
  -j 4 \
  -f
```

## Required Changes in `1_handcraft.ipynb`

After re-extracting the encodings, the notebook must be updated. It currently
loads `subset="val"` as its **test** data; after re-extraction that file is
the new **validation** set. The intended protocol is: train on `train`, choose
hyperparameters on `val`, evaluate **once** on `test`.

1. **Data loading cell**: load the three subsets.

   ```python
   histograms, labels, names = loadEncodingSubset(category, encoding, subset="train")
   valHistograms, valLabels, valNames = loadEncodingSubset(category, encoding, subset="val")
   testHistograms, testLabels, testNames = loadEncodingSubset(category, encoding, subset="test")
   ```

   Also add `printDatasetStats(valHistograms, valNames, valLabels, subset="val")`.

2. **SVM training cell**: choose `C` on the validation set instead of fixing
   `C=10`.

   ```python
   from sklearn.metrics import average_precision_score

   val_ap = {}
   for C in [0.01, 0.1, 1, 10, 100]:
       W, b = utils.train_svm(histograms, labels, C=C)
       val_ap[C] = average_precision_score(valLabels, W @ valHistograms.T + b)
       print(f"C={C:<6} val AP={val_ap[C]:.4f}")

   best_C = max(val_ap, key=val_ap.get)
   W, b = utils.train_svm(histograms, labels, C=best_C)
   print("Best C:", best_C)
   ```

   The same applies to other choices (encoder type, `K`, PCA dimension,
   training fraction in `get_subset`): compare them on `val`, not on `test`.

3. **Stage C (test) cells**: keep using `testHistograms`, `testLabels` and
   `testNames` (now loaded from `subset="test"`), and run them only with the
   final model chosen in step 2. Nothing else changes in these cells.

## Deep Models and References

The deep-model notebook uses pretrained backbones and utilities from these
projects:

- [TorchVision models documentation](https://pytorch.org/vision/stable/models.html)
  and [TorchVision GitHub repository](https://github.com/pytorch/vision) for
  ResNet-18 and ViT-B/32 weights and preprocessing.
- [DINOv2 GitHub repository](https://github.com/facebookresearch/dinov2) for
  the self-supervised `dinov2_vits14` model loaded through Torch Hub.
- [OpenAI CLIP GitHub repository](https://github.com/openai/CLIP) for the
  `ViT-B/32` image-text model and preprocessing pipeline.

These models may download pretrained weights the first time they are loaded.
The notebook automatically selects CUDA when available and otherwise uses CPU.
