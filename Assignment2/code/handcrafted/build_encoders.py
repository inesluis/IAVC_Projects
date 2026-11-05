"""Part 1/3 (Person A): train the extra BoVW vocabularies (K x seeds).

Reuses the professor's sampling / k-means code (src.train) unchanged. Each seed
draws its own training sample (150 imgs/class, 400 descriptors/img, train split
only) and fits k-means with that seed, so seeds measure vocabulary variance.
Fit times are logged to results/codebook_times.csv.

    cd code && python -m handcrafted.build_encoders --seeds 0 1 2 --ks 32 64 128 256 512 1024
"""
import argparse, csv, time
from pathlib import Path

import numpy as np

from src.train import sampleLocalFeatures, train_encoder
from src.utils import loadSamplesTable

# Relative paths on purpose: cv2.imread (used by src.features) fails on non-ASCII absolute paths on Windows.
DATA, RESULTS = Path("../data"), Path("../results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ks", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024])
    ap.add_argument("-i", "--per-class", type=int, default=150)
    ap.add_argument("-n", "--per-image", type=int, default=400)
    args = ap.parse_args()

    (DATA / "encoders").mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    log = RESULTS / "codebook_times.csv"
    new = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["seed", "K", "n_images", "n_descriptors", "t_sample_s", "t_kmeans_s"])
        for seed in args.seeds:
            np.random.seed(seed)
            df = loadSamplesTable(str(DATA / "splits"), str(DATA / "images"), subset="train",
                                  per_class=args.per_class, random_state=seed)
            t0 = time.perf_counter()
            descs = sampleLocalFeatures(df, num_samples=args.per_image, seed=seed)
            t_sample = time.perf_counter() - t0
            print(f"seed {seed}: {len(df)} images -> {descs.shape} descriptors in {t_sample:.1f}s")
            for k in args.ks:
                out = DATA / "encoders" / f"bovw_K{k}_C0_s{seed}.npz"
                if out.exists():
                    print("skip", out.name)
                    continue
                t0 = time.perf_counter()
                enc = train_encoder(descs, "bovw", num_words=k, pca_dim=0, seed=seed)
                t_fit = time.perf_counter() - t0
                enc.update(num_samples=descs.shape[0], num_samples_per_image=args.per_image,
                           num_images=len(df), seed=seed)
                np.savez(out, **enc)
                w.writerow([seed, k, len(df), descs.shape[0], f"{t_sample:.2f}", f"{t_fit:.2f}"])
                f.flush()
                print(f"  K={k}: k-means {t_fit:.1f}s -> {out.name}")


if __name__ == "__main__":
    main()
