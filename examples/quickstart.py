from ppca import PPCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris,load_breast_cancer,load_wine
from sklearn.decomposition import PCA
import argparse

# Loading data
# data = load_iris()
data = load_breast_cancer()
# data = load_wine()

X, y = data['data'], data['target']
# Parse CLI args for n_components and method
parser = argparse.ArgumentParser(description='Quickstart PPCA example')
parser.add_argument('--n_components', '-n', type=int, default=2,
                    help='Number of principal components (default: 2)')
parser.add_argument('--method', '-m', type=str, default='svd',
                    help='PPCA method to use, e.g. "svd" (default: svd)')
parser.add_argument('--epochs', '-e', type=int, default=200,
                    help='Number of epochs for EM method (default: 200)')
args = parser.parse_args()

# Use parsed arguments
pca = PPCA(n_components=args.n_components, method=args.method, max_iter=args.epochs)
Xt = pca.fit_transform(X)

print("y shape:", y.shape)
print("n_components:", args.n_components)

pca_sklearn = PCA(n_components=args.n_components)
Xt_sklearn = pca_sklearn.fit_transform(X)

# Grouped plots in one figure with two subplots
fig, axs = plt.subplots(1, 3, figsize=(14, 6))

# Left: sklearn PCA projection
scatter1 = axs[0].scatter(Xt_sklearn[:, 0], Xt_sklearn[:, 1], c=y, cmap='viridis', alpha=0.8)
axs[0].set_title(f'Sklearn PCA Projection (2D) {args.n_components} components')
axs[0].set_xlabel('Principal Component 1')
axs[0].set_ylabel('Principal Component 2')

# Middle: original PPCA projection
scatter2 = axs[1].scatter(Xt[:, 0], Xt[:, 1], c=y, cmap='viridis', alpha=0.8)
axs[1].set_title(f'PPCA Projection (2D) {args.n_components} components')
axs[1].set_xlabel('Principal Component 1')
axs[1].set_ylabel('Principal Component 2')

# Right: overlay generated samples on a faded original projection
X_gen_t = pca.sample_transform(n_samples=200)
axs[2].scatter(Xt[:, 0], Xt[:, 1], c=y, cmap='viridis', alpha=0.25)
axs[2].scatter(X_gen_t[:, 0], X_gen_t[:, 1], c='red', marker='x', label='Generated Samples')
axs[2].set_title(f'PPCA Generated Samples {args.n_components} components')
axs[2].set_xlabel('Principal Component 1')
axs[2].set_ylabel('Principal Component 2')
axs[2].legend()

plt.tight_layout()
plt.show()