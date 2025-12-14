from ppca import PPCA
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import urllib.request

# ensure data/ exists at repo root
# REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# filenames and urls
FILES = {
    "binarized_mnist_train.amat": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat",
    "binarized_mnist_valid.amat": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat",
    "binarized_mnist_test.amat":  "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat",
}

# download (once) and load into numpy arrays
def load_or_download(fname, url):
    npy_path = DATA_DIR / (fname + ".npy")
    if npy_path.exists():
        return np.load(npy_path)
    # download & load, then cache as .npy
    print(f"Downloading {fname} ...")
    arr = np.loadtxt(url, dtype=np.float32).reshape(-1, 28, 28, order="C")
    np.save(npy_path, arr)
    return arr