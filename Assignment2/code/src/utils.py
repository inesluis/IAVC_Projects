import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC
from pathlib import Path
import pandas as pd

def load_npz(file_path):
    """
    Load an encoder configuration from a .npz file and return it as a dictionary.
    
    Args:
        file_path (str or Path): Path to the .npz file containing the encoder configuration.
        
    Returns:
        dict: A dictionary containing the encoder configuration.
    """
    with np.load(file_path, allow_pickle=True) as data:
        encoder_cfg = {key: data[key] for key in data.files}
    return encoder_cfg

def train_svm(x, y, C, max_iter=100, tol=0.001, seed=42):
    clf = LinearSVC(C=C, max_iter=max_iter, dual=True, loss="hinge", tol=tol, random_state=seed)
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x, y)
    W = clf.coef_.copy().ravel()
    b = clf.intercept_.copy()[0]
    return W, b

def selectTrainSamples(x, y, names, fraction=np.inf, random_state=42):
    """
    Split the dataset into training and validation sets.
    
    Args:
        x (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        names (list): List of image names.
        fraction (float): Fraction of the dataset to use for training (default: 0.8).
        random_state (int): Random seed for reproducibility (default: 42).
    Returns:
        x_train, y_train, names_train: Training set.
    """
    num_images = len(y)
    if np.isinf(fraction):
        num_selected = num_images
    elif fraction < 1:
        num_selected = int(np.ceil(fraction * num_images))
    else:
        num_selected = int(fraction)

    rng = np.random.default_rng(seed=random_state)
    sel = rng.choice(num_images, size=num_selected, replace=False)
    names = [names[i] for i in sel]
    x = x[sel,:]
    y = y[sel]
    return x, y, names

def loadSamplesTable(txt_dir, img_dir, subset="train", per_class=-1, random_state=42):
    assert subset in ["train", "val"], "subset must be 'train' or 'val'"
    assert Path(txt_dir).exists(), f"txt_dir {txt_dir} does not exist"
    
    # per-class must be -1 (use all) or a positive integer
    assert per_class == -1 or per_class > 0, "per_class must be -1 (use all) or a positive integer"
    
    paths = list(Path(txt_dir).glob("*.txt"))
    # split into train and test files
    train_paths = [p for p in paths if "train" in p.name]
    test_paths = [p for p in paths if "val" in p.name]

    if subset == "train":
        subset_paths = train_paths
    elif subset == "val":
        subset_paths = test_paths
    else:
        raise ValueError(f"Invalid subset: {subset}. Must be 'train' or 'val'.")

    # load train samples into a dataframe
    file_data = []
    for p in subset_paths:
        with open(p, "r") as f:
            file_data.extend([(p.name, p.name.split("_")[0], l.strip()) for l in f.readlines()])

    df = pd.DataFrame(file_data, columns=["imageset","category", "filename"])
    df["filepath"] = df.filename.apply(lambda x: Path(f"{img_dir}/{x}.jpg"))
    # check if paths exist
    assert df.filepath.apply(lambda x: x.exists()).all(), "Some file paths do not exist"

    if per_class > 0:
        df["category"] = df["category"].apply(lambda x: "_background" if x == "background" else x)
        df = df.sort_values(by="category", ascending=True)
        df = df.drop_duplicates(subset=["filename"], keep="first")
    
        # randomly sample n_samples images from each category
        df = df.groupby("category").apply(
            lambda x: x.sample(n=min(per_class, len(x)), random_state=random_state))
        df = df.reset_index(drop=False).drop(columns=["level_1"])

    return df

def loadDataTables(dataDir, imgDir):
    df_train = loadSamplesTable(dataDir, imgDir, subset="train")
    df_train["subset"] = "train"
    print(df_train.head())
    print(f"Loaded {len(df_train)} samples from train subset.")

    df_valid = loadSamplesTable(dataDir, imgDir, subset="val")
    df_valid["subset"] = "val"
    print(f"Loaded {len(df_valid)} samples from val subset.")
    
    # concatenate and sort by category and subset
    df = pd.concat([df_train, df_valid], ignore_index=True)
    df = df.sort_values(by=["category", "subset","filename"]).reset_index(drop=True)
    print(f"Loaded {len(df)} samples from train and val subsets.")
    return df

def loadEncodingSubset(category, encoding, subset="train", root_dir="data"):
    base_dir = f"{root_dir}/encodings/{encoding}/"
    pos_file = f"{base_dir}/{category}_{subset}.npz"
    neg_file = f"{base_dir}/background_{subset}.npz"
            
    p_data = np.load(pos_file)
    p_hist, p_name = p_data['histograms'], p_data['file_names']
    p_label = np.ones(len(p_name), dtype=np.int32)

    n_data = np.load(neg_file)
    n_hist, n_name = n_data['histograms'], n_data['file_names']
    n_label = -np.ones(len(n_name), dtype=np.int32)

    hist = np.concatenate([p_hist, n_hist], axis=0)
    names = np.concatenate([p_name, n_name], axis=0)
    labels = np.concatenate([p_label, n_label], axis=0)
    
    return hist, labels, names


# To Verify
################################################################################
def getDataSplitSummary(df):
    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    l = pd.DataFrame(df[["category", "subset"]].value_counts()).reset_index()
    l = l.pivot(index="category", columns="subset", values="count").fillna(0).astype(int).T
    print("\nSplit summary:\n")
    print(l.to_markdown(tablefmt="fancy_grid", stralign="center", numalign="center"))
    return l