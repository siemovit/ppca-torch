from ppca import PPCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris,load_breast_cancer,load_wine
import argparse

# Loading data
# data = load_iris()
# data = load_breast_cancer()
data = load_wine()

X, y = data['data'], data['target']

# Parse CLI args for n_components and method
parser = argparse.ArgumentParser(description='Quickstart PPCA example')
parser.add_argument('--n_components', '-n', type=int, default=2,
                    help='Number of principal components (default: 2)')
parser.add_argument('--method', '-m', type=str, default='svd',
                    help='PPCA method to use, e.g. "svd" (default: svd)')
args = parser.parse_args()

# Use parsed arguments
pca = PPCA(n_components=args.n_components, method=args.method)
Xt = pca.fit_transform(X)

# Plot
plt.figure(figsize=(8,6))
scatter = plt.scatter(Xt[:,0], Xt[:,1], c=y, cmap='viridis', alpha=0.7)
plt.title('PPCA Projection (2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2') 
plt.colorbar(scatter, label='Classes')
plt.show()