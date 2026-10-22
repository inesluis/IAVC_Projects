import os
from IAVC_Projects.Assignment2.code.src import utils
from IAVC_Projects.Assignment2.code.src.encoders import ImageEncoder
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from .utils import loadDataTables


################################################################################
def encodeFeatures(df, encoder_path, output_dir, cache_dir, overwrite=False, jobs=1):
    print("Encoder path:", encoder_path)
    print(f"Output directory for encodings: {output_dir}")
    print(f"Using encoder from {encoder_path}")
    t_start = time.time()

    total = df[["category", "subset"]].drop_duplicates().shape[0]
    main_bar = tqdm(total=total, desc="Encoding features", ncols=150, leave=True)
    encoder = ImageEncoder(encoder_path)

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        for (cat, subset), rows in df.groupby(["category", "subset"]):

            output_file = f"{output_dir}/{cat}_{subset}.npz"
            if os.path.exists(output_file) and not overwrite:
                main_bar.write(f"Output file {output_file} already exists. Skipping...")
                continue

            rows = list(rows.itertuples())

            def encode_row(row):
                fpath = row.filepath
                assert fpath.exists(), f"File {fpath} does not exist"
                return fpath.stem, encoder.encode(fpath, cache_dir=cache_dir)

            encoded = list(tqdm(
                executor.map(encode_row, rows),
                total=len(rows),
                desc=f"Extracting {len(rows)} {cat:15}/{subset:5s} features",
                ncols=150,
                leave=False,
                # bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
            ))

            file_names = [name for name, _ in encoded]
            histograms = [feature for _, feature in encoded]

            np.savez_compressed(
                output_file,
                file_names=np.asarray(file_names),
                histograms=np.stack(histograms),
            )
            main_bar.update(1)
    main_bar.close()
    print(f"Finished encoding features using {encoder_path} in {time.time() - t_start:.2f} seconds.")

################################################################################
def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract features from images using a pre-trained encoder."
    )
    parser.add_argument("--data-dir", default="data/splits", help="Dataset directory.")
    parser.add_argument("--img-dir", default="data/images", help="Image directory.")
    parser.add_argument("-e","--encoder-path", default="data/encoders/bovw_K512_C0.npz",
                        help="Path to the pre-trained encoder.")
    parser.add_argument("--catg", default=None, nargs="+", type=str, 
        choices=["background", "aeroplane","motorbike","person","car","horse"],
        help="Category to filter (default: None).")
    parser.add_argument("--output-dir", default="data/encodings/", help="Feature output directory.")
    parser.add_argument(
        "-f", "--force", action="store_true",
        help="Re-extract features when output files exist."
    )
    parser.add_argument("--cache-dir", default="cache/", help="Computed Features Cache directory.")
    parser.add_argument(
        "-j", "--jobs", type=int, default=os.cpu_count() // 2,
        help="Number of parallel jobs (default: half of CPU cores)."
    )

    return parser.parse_args()


if __name__ == "__main__":
   
    args = parse_arguments()

    df = loadDataTables(args.data_dir, args.img_dir)
    print(df)

    assert args.encoder_path is not None, "Please provide a valid encoder path using --encoder-path."
    assert Path(args.encoder_path).exists(), f"Encoder path {args.encoder_path} does not exist."

    output_dir = Path(args.output_dir)
    encoder_path = Path(args.encoder_path)

    output_dir = output_dir / encoder_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.catg is not None:
        unique_categories = df["category"].unique()
        df = df[df["category"].isin(args.catg)]
        print(f"Filtered dataset to categories: {args.catg}. Remaining samples: {len(df)}")
        assert len(df) > 0, f"No samples found for the specified categories: {args.catg}. Available categories: {unique_categories}"

    encodeFeatures(
        df, 
        encoder_path, 
        output_dir=output_dir,
        cache_dir=args.cache_dir,
        overwrite=args.force,
        jobs=args.jobs,           
        )

