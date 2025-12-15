from ppca import PPCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

# Loading data
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# PPCA
n_components = 2
epochs = 50

Xts = []
for method in ['baseline', 'svd', 'em', 'gd']:
    print(f"Running PPCA with method: {method}")
    pca = PPCA(n_components=n_components, method=method, max_iter=epochs)
    Xt = pca.fit_transform(X)
    Xts.append(Xt)
    
    
# Plot projections for each method
methods = ['baseline', 'svd', 'em', 'gd']
lw = 2

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
cmap = plt.get_cmap("tab20")
class_colors = [cmap(i) for i in range(len(target_names))]

for idx, method in enumerate(methods):
    Xt = Xts[idx]
    ax = axs[idx // 2, idx % 2]

    # plot each class separately so legend and colors match target_names
    for cls in np.unique(y):
        mask = (y == cls)
        ax.scatter(Xt[mask, 0], Xt[mask, 1],
                   color=class_colors[int(cls)],
                   label=target_names[int(cls)],
                   alpha=0.8, edgecolor='k', linewidths=0.1, s=40)

    ax.set_title(f'{method} - iterations={epochs}')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.legend(loc='best', fontsize='small')

plt.tight_layout()
plt.show()