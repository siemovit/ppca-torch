import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

class PPCA(nn.Module):
    def __init__(self, n_components=3, method='svd'):
        super(PPCA, self).__init__()
        self.method = method
        self.n_components = n_components
        
    def fit(self, X):
        # Ensure X is a tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
            
        # SVD
        if self.method == 'svd':
            self._fit_eig_decomp(X)
        else:
            raise NotImplementedError(f"Method {self.method} not implemented.")
    
    def _fit_eig_decomp(self, X):
        # Define N and D
        N = X.shape[0] # number of samples
        D = X.shape[1] # data dimensionality
        
        # 1. Estimate data mean (mu)
        self.mu = torch.mean(X, dim=0) # shape (D,)
        
        # Sample covariance matrix (Bishop, Tipping, 1999 - Eq. 5)
        # Use D x D covariance: S = (1/N) * (X - mu)^T (X - mu)
        S = (1.0 / N) * (X - self.mu).transpose(0, 1) @ (X - self.mu)  # shape (D, D)

        # Eigen decomposition for symmetric matrix -> real eigenvalues/eigenvectors
        eig_val, eig_vec = torch.linalg.eigh(S)  # eig_val shape (D,), eig_vec shape (D, D)
        # Sort eigenvalues/vectors in descending order
        eig_sort = torch.argsort(eig_val, descending=True)
        U_q = eig_vec[:, eig_sort[:self.n_components]]  # D x q matrix of principal eigenvectors
        Lambda_q = torch.diag(eig_val[eig_sort[:self.n_components]])  # q x q diagonal matrix
          
        # 2. Estimate sigma2 as variance 'lost' in the projection, averaged over the lost dimensions (Bishop, Tipping, 1999 - Eq. 8)
        if D > self.n_components:
            # sum of the discarded eigenvalues (from q to D-1)
            discarded = eig_val[eig_sort[self.n_components:]]
            self.sigma2 = torch.sum(discarded) / (D - self.n_components)
        else:
            self.sigma2 = torch.tensor(0.0, device=X.device)
        
        # 3. Estimate W by constructing W = U_k @ sqrt(lambda_k - sigma2*I_k)) (Bishop, Tipping, 1999 - Eq. 7)
        # Ensure numerical stability inside sqrt
        adjust = Lambda_q - self.sigma2 * torch.eye(self.n_components, device=X.device)
        adjust = torch.clamp(adjust, min=0.0)
        self.W = U_q @ torch.sqrt(adjust) 
        
    def transform(self, X):
        # Ensure X is a tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
            
        # We sample X from it's conditional distribution (observation model)
        # For that, we sample from a Gaussian N(x|W*z+mu, sigma2*I) (Bishop, Tipping, 1999 - Eq. 6)
        
        N = X.shape[0]
        k = self.n_components
        Xt = torch.zeros((N, k), device=X.device)  # shape (N, k)

        # M is k x k
        M = self.W.transpose(0, 1) @ self.W + self.sigma2 * torch.eye(k, device=X.device)
        M_inv = torch.linalg.inv(M)

        # Gaussian posterior parameters for latent z: mean (k x N), cov (k x k)
        transform_means = M_inv @ self.W.transpose(0, 1) @ (X - self.mu).transpose(0, 1)  # k x N
        transform_covariances = self.sigma2 * M_inv  # k x k

        for i in range(N):
            mean_i = transform_means[:, i]
            sample = MultivariateNormal(mean_i, transform_covariances).sample()
            Xt[i] = sample

        return Xt
           
    def inverse_transform(self, X_transformed):
        # Implement the inverse transformation procedure for PPCA
        pass
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


