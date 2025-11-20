from ppca import PPCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris,load_breast_cancer,load_wine

# Loading data
# data = load_iris()
# data = load_breast_cancer()
data = load_wine()

X, y = data['data'], data['target']
pca = PPCA(n_components=2, method='svd')
Xt = pca.fit_transform(X)

# Plot
plt.figure(figsize=(8,6))
scatter = plt.scatter(Xt[:,0], Xt[:,1], c=y, cmap='viridis', alpha=0.7)
plt.title('PPCA Projection (2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2') 
plt.colorbar(scatter, label='Classes')
plt.show()