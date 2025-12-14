from ppca import PPCA
import matplotlib.pyplot as plt
import numpy as np
from ppca import load_or_download, FILES
import torch

# Load datasets
xtrain = load_or_download("binarized_mnist_train.amat", FILES["binarized_mnist_train.amat"])
xvalid = load_or_download("binarized_mnist_valid.amat", FILES["binarized_mnist_valid.amat"])
xtest =  load_or_download("binarized_mnist_test.amat",  FILES["binarized_mnist_test.amat"])

# flatten datasets for PPCA fitting
p = 784
X = xtrain.reshape((-1, p))
xvalid = xvalid.reshape((-1, p))
xtest = xtest.reshape((-1, p))


# PPCA parameters
n_components = 10 # ten digits in MNIST
epoch_list = [10, 100, 1000]
# epoch_list = [1, 2, 3]
n_samples_per_epoch = 4
method = 'em'
torch.manual_seed(42) # for reproducibility of sampling (to be checked)
 
models = {}
samples = {}

for ep in epoch_list:
    print(f"Training PPCA (method={method}) for {ep} epochs...")
    pca = PPCA(n_components=n_components, method=method, max_iter=ep)
    pca.fit(X)
    models[ep] = pca
    # sample n images from the fitted model
    X_gen = pca.sample(n_samples=n_samples_per_epoch)
    # clip/binarize for visualization (keep float for smoothing option)
    samples[ep] = X_gen.reshape((-1, 28, 28))


# Plot 3 rows (one per epoch) x 4 columns (samples)
nrows = n_samples_per_epoch
ncols = len(epoch_list)
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.0))

for c, ep in enumerate(epoch_list):          # column = epoch
    imgs = samples[ep]
    for r in range(nrows):                   # row = sample index
        # handle different shapes of axes returned by plt.subplots
        if nrows == 1 and ncols == 1:
            ax = axes
        elif nrows == 1:
            ax = axes[c]
        elif ncols == 1:
            ax = axes[r]
        else:
            ax = axes[r, c]
        ax.imshow(imgs[r], cmap='gray', interpolation='bicubic', vmin=0, vmax=1)
        ax.axis('off')
# label column headers with epoch values
for c, ep in enumerate(epoch_list):
    if nrows > 0:
        if nrows == 1 and ncols == 1:
            axes.set_title(f'epoch={ep}')
        elif nrows == 1:
            axes[c].set_title(f'epoch={ep}')
        else:
            axes[0, c].set_title(f'epoch={ep}')

plt.suptitle(f'PPCA samples (cols = epochs {epoch_list})', fontsize=14)
plt.tight_layout()
plt.show()

