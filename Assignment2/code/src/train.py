
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from skimage.feature import learn_gmm
from pathlib import Path

from .features import extractFeatures
from .utils import loadSamplesTable

#-------------------------------------------------------------------------------
def sampleLocalFeatures(df, num_samples, seed=None, cache_dir=""):
   
    def extract_descriptors(item):
        idx, row = item
        image_path = row.filepath
        # print(image_path)
        _, desc, _ = extractFeatures(image_path, cache_dir=cache_dir)
        if desc.shape[0] > num_samples:
            rng = np.random.default_rng(None if seed is None else seed + idx)
            indices = rng.choice(desc.shape[0], size=num_samples, replace=False)
            desc = desc.take(indices, axis=0)
        return desc

    with ThreadPoolExecutor() as executor:
        descs = list(tqdm(
            executor.map(extract_descriptors, list(df.iterrows())),
            total=len(df), desc="Sampling local features", ncols=150, leave=False
        ))
    descs = np.vstack(descs)
    # shuffle the descriptors to avoid any ordering bias
    descs = descs[np.random.permutation(descs.shape[0]), :]
    return descs

#-------------------------------------------------------------------------------
# Training routine for a global image encoder: sample descriptors, optionally
# reduce their dimension with PCA, and fit the codebook / GMM parameters.
def train_encoder(descs, encoder_type="fv", num_words=32, pca_dim=0, seed=0, cache_dir=""):

    start_time = time.time()
    # Store the common encoder configuration and feature extractor name in a
    # dictionary that can later be reloaded by ImageEncoder.
    encoder = {
        "type": encoder_type,
        "num_words": num_words,
        "pca_dim": pca_dim,
        "projection": None,
    }

    # Apply PCA to reduce the dimensionality of the descriptors while
    # preserving the most informative directions.
    if pca_dim > 0:
        print(f"Applying PCA to reduce descriptor dimensionality from {descs.shape[1]} to {pca_dim}.")
        pca = PCA(n_components=pca_dim, random_state=seed)
        pca.fit(descs)
        descs = pca.transform(descs)
        encoder["projection"] = pca.components_
        encoder["projection_mean"] = pca.mean_

    # For Fisher vectors, fit a Gaussian mixture model to estimate means,
    # variances, and mixture weights over the descriptor distribution.
    if encoder_type == "fv":
        gmm = learn_gmm(descs.astype(np.float64), n_modes=num_words)
        encoder["means"] = gmm.means_
        encoder["covariances"] = gmm.covariances_
        encoder["priors"] = gmm.weights_

    # For VLAD and BoVW, learn a codebook with K-means.    
    elif encoder_type in ["vlad", "bovw"]:
        kmeans = KMeans(n_clusters=num_words, random_state=seed)
        kmeans.fit(descs)
        encoder["words"] = kmeans.cluster_centers_
    
    print(f"Training encoder of type '{encoder_type}' completed in {time.time() - start_time:.2f} seconds.")

    return encoder


################################################################################
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Train encoders for different encoding types.")
    parser.add_argument("--data-dir", default="data/splits",
                        help="Directory containing dataset files.")
    parser.add_argument("--img-dir", default="data/images",
                        help="Directory containing images.")
    parser.add_argument("--output-dir", default="data/encoders",
                        help="Directory for trained encoders.")
    parser.add_argument("-e", "--encoder-type", default="bovw",
                        choices=["bovw", "vlad", "fv"],
                        help="Encoder type (default: bovw).")
    parser.add_argument("-k", "--num-words", type=int, default=32,
                        help="Number of visual words (default: 32).")
    parser.add_argument("-c", "--pca-dim", type=int, default=0,
                        help="PCA dimension; 0 disables PCA (default: 0).")
    parser.add_argument("-i", "--per-class", type=int, default=150,
                        help="Images per class; -1 uses all (default: 150).")
    parser.add_argument("-n", "--per-image", type=int, default=400,
                        help="Local features per image (default: 400).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--cache-dir", default="",
                        help="Directory for cached features (default: disabled).")
    parser.add_argument("-j", "--jobs", type=int, default=4,
                        help="Number of parallel jobs (default: 4).")
    args = parser.parse_args()

    ############################################################################

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    df = loadSamplesTable(
        txt_dir = args.data_dir, 
        img_dir = args.img_dir, 
        subset="train", 
        per_class=args.per_class, # Number of images per class; -1 uses all
        random_state=args.seed # Random seed for reproducibility
    )

    print("Total number of images for encoder training:", len(df))
    print("Total number of classes for encoder training:", df["category"].nunique())
    print("Total number of local features for encoder training:", len(df) * args.per_image)

    # start_time = time.time()

    descs = sampleLocalFeatures(
        df,
        num_samples=args.per_image,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )

    # extract features from the dataset same for all
    output_path = OUTPUT_DIR / f"{args.encoder_type}_K{args.num_words}_C{args.pca_dim}.npz"

    encoder = train_encoder(
        descs, 
        encoder_type=args.encoder_type,
        num_words=args.num_words,
        pca_dim=args.pca_dim,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )

    encoder.update(dict(
        num_samples=descs.shape[0],
        num_samples_per_image=args.per_image,
        num_images=len(df),
        seed=args.seed,
        cache_dir=args.cache_dir,
    ))

    np.savez(output_path, **encoder)
    