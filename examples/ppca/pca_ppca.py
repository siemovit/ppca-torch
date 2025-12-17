import numpy as np
from ppca import PCA, PPCA
import torch
torch.random.manual_seed(0)
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt


# Parameters
mean = [0, 0]
covariance = [[23, 0], [0, 5]]
num_points = 20

# sample from a 2D Gaussian (torch)
dist = MultivariateNormal(torch.tensor(mean, dtype=torch.float32),
                          torch.tensor(covariance, dtype=torch.float32))
X_t = dist.sample((num_points,))            # (N, 2) torch tensor
X = X_t.numpy()                             # use numpy for plotting / simple ops

# ----- PCA -----
# Fit torch-based PCA (nb_components=1)
pca = PCA(nb_components=1)
proj_pca = pca.fit_transform(X)             # torch tensor (N,1)
eigenvector = pca.components.cpu().numpy().squeeze()   # (2,)

# Reconstruct data projected onto PCA axis
mu_pca = pca.mu.cpu().numpy()
Xc = X - mu_pca
scores_pca = (Xc @ eigenvector)             # (N,)
Xrec_pca = scores_pca[:, None] * eigenvector[None, :] + mu_pca  # (N,2)

# ----- PPCA -----
# Fit PPCA (use 'svd' to get analytic solution comparable to PCA)
ppca = PPCA(n_components=1, method='svd')
Z_ppca = ppca.fit_transform(X)              # torch tensor (N,1)
W_ppca = ppca.W.cpu().numpy().squeeze()     # (2,)
mu_ppca = ppca.mu.cpu().numpy()
sigma2 = float(ppca.sigma2)

# compute posterior mean-based reconstruction for PPCA
W_mat = W_ppca.reshape(2, 1)                # (2,1)
M = (W_mat.T @ W_mat) + sigma2 * np.eye(1)  # (1,1)
M_inv = np.linalg.inv(M)
z_mean = (M_inv @ W_mat.T @ (X - mu_ppca).T).T   # (N,1)
Xrec_ppca = (z_mean @ W_mat.T) + mu_ppca         # (N,2)


# ----- PPCA sigma2=5-----
# Fit PPCA (use 'svd' to get analytic solution comparable to PCA)
ppca2 = PPCA(n_components=1, method='svd')
ppca2.sigma2 = 10.0  # manually set sigma2
Z_ppca2 = ppca2.fit_transform(X)              # torch tensor (N,1)
W_ppca2 = ppca2.W.cpu().numpy().squeeze()     # (2,)
mu_ppca2 = ppca2.mu.cpu().numpy()
sigma2 = 15.0

# compute posterior mean-based reconstruction for PPCA
W_mat = W_ppca2.reshape(2, 1)                # (2,1)
M = (W_mat.T @ W_mat) + sigma2 * np.eye(1)  # (1,1)
M_inv = np.linalg.inv(M)
z_mean = (M_inv @ W_mat.T @ (X - mu_ppca2).T).T   # (N,1)
Xrec_ppca2 = (z_mean @ W_mat.T) + mu_ppca2         # (N,2)

# Plot data and projections
# golden ratio for nice figure
phi = (1 + 5 ** 0.5) / 2
width = 6
height = width / phi

def plot_projections(X, Xrec, mu, component, color, fig_name):
    plt.figure(figsize=(width, height))
    plt.scatter(X[:,0], X[:,1], label='data', alpha=0.7)
    # draw segments from each data point to its projection
    for i in range(X.shape[0]):
        plt.plot([X[i,0], Xrec[i,0]],
                 [X[i,1], Xrec[i,1]],
                 c=color, alpha=0.25, linewidth=0.8)
    # draw component axis line for visualization
    center = mu
    vec = component / np.linalg.norm(component)
    line_pts = np.vstack([center - 8*vec, center + 8*vec])
    plt.plot(line_pts[:,0], line_pts[:,1], c=color, alpha=0.5, linewidth=1)
    plt.xlabel('x1'); plt.ylabel('x2')
    plt.axis('equal')
    plt.savefig(fig_name)
    # plt.show()
    
# Plot PCA projections
plot_projections(X, Xrec_pca, mu_pca, eigenvector, 'C3', "figures/pca_projection.png")
# Plot PPCA projections
plot_projections(X, Xrec_ppca, mu_ppca, W_ppca, 'C3', "figures/ppca_projection.png")
# Plot PPCA projections (sigma2=15)
plot_projections(X, Xrec_ppca2, mu_ppca2, W_ppca2, 'C3', "figures/ppca_sigma2_15_projection.png")

# Print diagnostics
print("PCA eigenvector:", eigenvector)
print("PPCA W (direction):", W_ppca)
print("PPCA sigma2:", ppca.sigma2)
print("PPCA sigma2:", sigma2)
