# Example of PPCA on Iris dataset with missing values

# # Percentage of variance explained for each components
# print(
#     "explained variance ratio (first two components): %s"
#     % str(pca.explained_variance_ratio_)
# )


import matplotlib.pyplot as plt
from sklearn import datasets
from ppca import PPCA
# from sklearn.decomposition import PCA
import numpy as np

def make_missing(X, missing_rate, seed=0):
    """Return a copy of X with given fraction of entries set to np.nan."""
    rng = np.random.default_rng(seed)
    Xm = X.copy()
    n_samples, n_features = Xm.shape
    n_missing = int(np.floor(n_samples * n_features * missing_rate))
    idx = rng.choice(n_samples * n_features, size=n_missing, replace=False)
    Xm_flat = Xm.flatten()
    Xm_flat[idx] = np.nan
    return Xm_flat.reshape(n_samples, n_features)

# Load data
iris = datasets.load_iris()
X_orig = iris.data.astype(np.float32)
y = iris.target
target_names = iris.target_names

# Decide missing rates to plot
missing_rates = [0.0, 0.15, 0.30]
seed = 0

# Prepare figure: 1 row x 3 columns
fig, axes = plt.subplots(1, len(missing_rates), figsize=(15, 4))

cmap = plt.get_cmap("tab20")
class_colors = [cmap(i) for i in range(len(target_names))]
lw = 1.0

# TODO: fix missing values first in PPCA implementation
# for ax, mr in zip(axes, missing_rates):
#     Xm = make_missing(X_orig, mr, seed=seed)
#     pca = PPCA(n_components=2, method='em', max_iter=100)
#     # fit_transform should handle missing values in PPCA implementation
#     X_r = pca.fit_transform(Xm)

#     for color, i, target_name in zip(class_colors, [0, 1, 2], target_names):
#         ax.scatter(
#             X_r[y == i, 0], X_r[y == i, 1],
#             color=color, alpha=0.8, lw=lw, label=target_name, s=40
#         )

#     ax.set_xlabel('PC 1')
#     ax.set_ylabel('PC 2')
#     ax.legend(loc='lower left', fontsize='small')

# plt.suptitle('PPCA on Iris: no missing, 15% missing, 30% missing', fontsize=14)
# plt.tight_layout()
# plt.show()