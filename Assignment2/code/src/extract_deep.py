"""Extract frozen deep features (Part 4) for every image in splits.json.

Each image is passed through the network once; the features are then written
in the same format as the handcrafted encodings, so `utils.loadEncodingSubset`
and the SVM/AP code work unchanged:

    data/encodings/<feature>/<category>_<train|val|test>.npz
        file_names : (N,)   image names without extension
        histograms : (N, D) float32 features (not normalised)

Features written (D):
    resnet18_layer2 (128), resnet18_layer3 (256), resnet18_layer4 (512)
        global average pooling of blocks layer2.1 / layer3.1 / layer4.1
    vit_b_32 (768)                 CLS token, supervised ImageNet ViT-B/32
    dinov2_vits14 (384)            normalised CLS token
    dinov2_vits14_patchmean (384)  mean of the normalised patch tokens
    clip_vit_b_32 (512)            CLIP image embedding

With flips enabled, the same features are also computed on the horizontally
mirrored image and saved under "<feature>_flip" (for Part 4, Stage C).

Each output folder gets a meta.json with the model, preprocessing, dimension,
hardware and time per image. A summary row per feature is appended to
results/deep_extraction.csv.

Usage (from code/):
    python -m src.extract_deep                      # all models, original + flip
    python -m src.extract_deep -m dinov2_vits14 --limit 64   # quick test
"""
import argparse
import csv
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .deep_models import IntermediateFeatureExtractor, getModel

ROOT = Path(__file__).resolve().parents[2]  # Assignment2/

MODEL_NAMES = ["resnet18", "vit_b_32", "dinov2_vits14", "clip_vit_b_32"]
RESNET_LAYERS = {"layer2.1": "resnet18_layer2",
                 "layer3.1": "resnet18_layer3",
                 "layer4.1": "resnet18_layer4"}


class ImageList(Dataset):
    """Loads images by name and applies the model's official preprocessing."""

    def __init__(self, names, img_dir, preprocess, flip=False):
        self.names = names
        self.img_dir = Path(img_dir)
        self.preprocess = preprocess
        self.flip = flip

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(self.img_dir / f"{self.names[i]}.jpg").convert("RGB")
        if self.flip:
            img = ImageOps.mirror(img)
        return self.preprocess(img)


def build_forward(model_name, model):
    """Return a function batch -> {feature_name: (B, D) tensor}."""
    if model_name == "resnet18":
        extractor = IntermediateFeatureExtractor(model)

        def forward(x):
            _, feats = extractor(x)
            return {out: feats[layer].mean(dim=(2, 3))
                    for layer, out in RESNET_LAYERS.items()}
    elif model_name == "vit_b_32":
        def forward(x):
            return {"vit_b_32": model(x)}  # heads replaced by Identity -> CLS
    elif model_name == "dinov2_vits14":
        def forward(x):
            out = model.forward_features(x)
            return {"dinov2_vits14": out["x_norm_clstoken"],
                    "dinov2_vits14_patchmean": out["x_norm_patchtokens"].mean(dim=1)}
    elif model_name == "clip_vit_b_32":
        def forward(x):
            return {"clip_vit_b_32": model.encode_image(x)}
    else:
        raise ValueError(model_name)
    return forward


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def extract(names, img_dir, model_name, model, preprocess, device, flip,
            batch_size, workers):
    """Run the model over all images; returns features and timings."""
    loader = DataLoader(ImageList(names, img_dir, preprocess, flip),
                        batch_size=batch_size, num_workers=workers,
                        pin_memory=device.type == "cuda",
                        persistent_workers=False)
    forward = build_forward(model_name, model)
    model_dtype = next(model.parameters()).dtype  # CLIP runs in fp16 on GPU

    # warm-up pass (cuDNN autotuning, lazy CUDA init) so it is not timed
    x0 = torch.stack([loader.dataset[0]] * min(batch_size, 8)).to(device).to(model_dtype)
    forward(x0)
    sync(device)

    chunks, t_forward = {}, 0.0
    sync(device)
    t_start = time.perf_counter()
    desc = f"{model_name}{' (flip)' if flip else ''}"
    for x in tqdm(loader, desc=desc, ncols=100):
        x = x.to(device, non_blocking=True).to(model_dtype)
        sync(device)
        t0 = time.perf_counter()
        out = forward(x)
        sync(device)
        t_forward += time.perf_counter() - t0
        for k, v in out.items():
            chunks.setdefault(k, []).append(v.float().cpu().numpy())
    t_total = time.perf_counter() - t_start

    feats = {k: np.concatenate(v).astype(np.float32) for k, v in chunks.items()}
    n = len(names)
    timings = {"ms_per_img_total": 1000 * t_total / n,      # decode + preprocess + forward
               "ms_per_img_forward": 1000 * t_forward / n}  # network only
    return feats, timings


def save_feature(name, feats, names, index, splits, out_root, meta):
    """Write <category>_<subset>.npz files for one feature + meta.json."""
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for subset in ("train", "val", "test"):
        per_cat = {}
        for img, cats in splits[subset].items():
            if img in index:  # may be missing with --limit
                for c in cats:
                    per_cat.setdefault(c, []).append(img)
        for cat, imgs in per_cat.items():
            np.savez_compressed(
                out_dir / f"{cat}_{subset}.npz",
                file_names=np.asarray(imgs),
                histograms=feats[[index[i] for i in imgs]],
            )
    meta = {**meta, "feature": name, "dim": int(feats.shape[1]),
            "n_images": len(names),
            "bytes_float32_total": int(feats.shape[0] * feats.shape[1] * 4),
            "bytes_float32_per_image": int(feats.shape[1] * 4)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def log_csv(path, meta):
    fields = ["date", "feature", "model", "flip", "dim", "n_images", "device",
              "ms_per_img_total", "ms_per_img_forward", "model_load_s",
              "bytes_float32_per_image", "bytes_float32_total", "batch_size"]
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(meta)


def parse_arguments():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    p.add_argument("--flip", choices=["both", "none", "only"], default="both",
                   help="Extract original images, mirrored images, or both.")
    p.add_argument("--splits-file", default=ROOT / "data/splits/splits.json", type=Path)
    p.add_argument("--img-dir", default=ROOT / "data/images", type=Path)
    p.add_argument("--output-dir", default=ROOT / "data/encodings", type=Path)
    p.add_argument("--results-csv", default=ROOT / "results/deep_extraction.csv", type=Path)
    p.add_argument("-b", "--batch-size", type=int, default=64)
    p.add_argument("-j", "--workers", type=int, default=4,
                   help="DataLoader workers (use 0 if multiprocessing fails).")
    p.add_argument("--limit", type=int, default=None,
                   help="Use only the first N images (quick test; writes to a _test folder).")
    p.add_argument("-f", "--force", action="store_true", help="Overwrite existing features.")
    return p.parse_args()


def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = json.loads(args.splits_file.read_text(encoding="utf-8"))
    names = sorted(set(splits["train"]) | set(splits["val"]) | set(splits["test"]))
    out_root = args.output_dir
    if args.limit:
        names = names[:args.limit]
        out_root = out_root.parent / (out_root.name + "_test")
    index = {n: i for i, n in enumerate(names)}
    print(f"{len(names)} images | device: {device} | output: {out_root}")

    flips = {"both": [False, True], "none": [False], "only": [True]}[args.flip]
    hardware = (torch.cuda.get_device_name(0) if device.type == "cuda"
                else platform.processor())

    for model_name in args.models:
        todo = []
        for flip in flips:
            # skip if every feature of this model/flip already has a meta.json
            probe = list(RESNET_LAYERS.values()) if model_name == "resnet18" else \
                ["dinov2_vits14", "dinov2_vits14_patchmean"] if model_name == "dinov2_vits14" \
                else [model_name]
            done = all((out_root / (p + ("_flip" if flip else "")) / "meta.json").exists()
                       for p in probe)
            if done and not args.force:
                print(f"Skipping {model_name} flip={flip} (exists; use -f to overwrite)")
            else:
                todo.append(flip)
        if not todo:
            continue

        t0 = time.perf_counter()
        model, preprocess = getModel(model_name, device=device, nofc=True)
        sync(device)
        load_s = time.perf_counter() - t0

        for flip in todo:
            feats, timings = extract(names, args.img_dir, model_name, model, preprocess,
                                     device, flip, args.batch_size, args.workers)
            for feat_name, X in feats.items():
                name = feat_name + ("_flip" if flip else "")
                meta = save_feature(name, X, names, index, splits, out_root, {
                    "date": datetime.now().isoformat(timespec="seconds"),
                    "model": model_name,
                    "flip": flip,
                    "device": hardware,
                    "torch": torch.__version__,
                    "preprocess": repr(preprocess),
                    "batch_size": args.batch_size,
                    "model_load_s": round(load_s, 2),
                    # timings cover the whole forward pass of this model, shared
                    # by the features it produces (e.g. the 3 ResNet layers)
                    **{k: round(v, 3) for k, v in timings.items()},
                })
                if not args.limit:
                    log_csv(args.results_csv, meta)
                print(f"  {name:28s} dim={X.shape[1]:4d}  "
                      f"{timings['ms_per_img_total']:.2f} ms/img total, "
                      f"{timings['ms_per_img_forward']:.2f} ms/img forward")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
