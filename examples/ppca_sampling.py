from ppca import PPCA
import matplotlib.pyplot as plt
import numpy as np
from ppca import load_or_download, FILES

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
epoch_list = [10, 20, 50]
n_samples_per_epoch = 4
method = 'em'

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
nrows = len(epoch_list)
ncols = n_samples_per_epoch
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.0))
for r, ep in enumerate(epoch_list):
    imgs = samples[ep]
    for c in range(ncols):
        ax = axes[r, c] if nrows > 1 else axes[c]
        ax.imshow(imgs[c], cmap='gray', interpolation='bicubic', vmin=0, vmax=1)

plt.suptitle('PPCA samples at different training epochs (rows)', fontsize=14)
plt.tight_layout()
plt.show()

