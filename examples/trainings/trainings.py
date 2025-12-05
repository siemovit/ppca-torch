from ppca import PPCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris,load_breast_cancer,load_wine
from sklearn.decomposition import PCA

# Loading data

# data = load_iris()
data = load_breast_cancer()
# data = load_wine()

X, y = data['data'], data['target']

# PPCA parameters
n_components = 3
epochs = 10

# PPCA
Xts = []
for method in ['baseline', 'svd', 'em', 'sgd']:
    print(f"Running PPCA with method: {method}")
    pca = PPCA(n_components=n_components, method=method, max_iter=epochs)
    Xt = pca.fit_transform(X)
    Xts.append(Xt)
    
    
# Grouped plots in one figure with two subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for i, method in enumerate(['baseline', 'svd', 'em', 'sgd']):
    Xt = Xts[i]
    scatter = axs[i // 2, i % 2].scatter(Xt[:, 0], Xt[:, 1], c=y, cmap='viridis', alpha=0.8)
    axs[i // 2, i % 2].set_title(f'PPCA Projection (2D) {n_components} components - Method: {method}')
    axs[i // 2, i % 2].set_xlabel('Principal Component 1')
    axs[i // 2, i % 2].set_ylabel('Principal Component 2')
    
plt.tight_layout()
plt.show()