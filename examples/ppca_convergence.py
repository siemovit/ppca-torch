# ...existing code...
from ppca import PPCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import numpy as np
from ppca import load_or_download, FILES

# Loading data
# data = load_breast_cancer()
# X, y = data['data'], data['target']

xtrain = load_or_download("binarized_mnist_train.amat", FILES["binarized_mnist_train.amat"])
xvalid = load_or_download("binarized_mnist_valid.amat", FILES["binarized_mnist_valid.amat"])
xtest =  load_or_download("binarized_mnist_test.amat",  FILES["binarized_mnist_test.amat"])
# flatten datasets for PPCA fitting
p = 784
X = xtrain.reshape((-1, p))
xvalid = xvalid.reshape((-1, p))
xtest = xtest.reshape((-1, p))

# PPCA parameters
n_components = 10
epochs = 10

# Run PPCA for each method and keep the fitted models
methods = ['svd', 'em', 'sgd']
models = {}
thetas = {}

for method in methods:
    print(f"Running PPCA with method: {method}")
    pca = PPCA(n_components=n_components, method=method, max_iter=epochs)
    # fit may return (W, mu, sigma2) or None depending on implementation, we keep the model nonetheless
    try:
        ret = pca.fit(X)
    except Exception as e:
        print(f"fit failed for method={method}: {e}")
        ret = None
    models[method] = pca
    thetas[method] = ret

# Plot convergence diagnostics: pick attributes if available
plt.figure(figsize=(8, 6))

# SVD pseudo-loss (constant)
svd_model = models.get('svd')
svd_losses = getattr(svd_model, 'losses', []) or []
plt.title('Convergence Plot - Comparison of Methods')
plt.xlabel('Iterations')
plt.ylabel('Value')
# plt.xlim(100, epochs)
# plt.ylim(min(40000, min(svd_losses)-1), 200000)
plt.plot(range(len(svd_losses)), svd_losses, label='SVD Pseudo-loss', color='black')
plt.legend()
# SGD losses 
sgd_model = models.get('sgd')
sgd_losses = getattr(sgd_model, 'losses', []) or []
plt.plot(range(len(sgd_losses)), sgd_losses, label='SGD Loss', color='green')
plt.legend()

# EM log-likelihoods (if present)
em_model = models.get('em')
em_losses = getattr(em_model, 'losses', []) or []
plt.plot(range(len(em_losses)), em_losses, label='EM Loss', color='orange')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()