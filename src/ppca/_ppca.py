import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

class PPCA(nn.Module):
    def __init__(self, n_components=3, method='svd', max_iter=20, stopping_criterion=1e-6):
        super(PPCA, self).__init__()
        self.n_components = n_components
        self.method = method
        self.max_iter = max_iter
        self.stopping_criterion = stopping_criterion
        
    def fit(self, X):
        # Ensure X is a tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
            
        # SVD
        if self.method == 'svd':
            self._fit_eig_decomp(X)
        # EM
        elif self.method == 'em':
            self._fit_em(X)
        # Other methods can be added here
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
        # TODO: understand why (X - mu)^T (X - mu) and not (X - mu) (X - mu)^T like in the paper
        # NB: I guess the paper data as columns (D×N), so their (X - mu)(X - mu)^T corresponds to our (X - mu)^T (X - mu)
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
        
    def _e_step(self, X):
        # E-step: compute expected values of latent variables given current parameters 
        # (Bishop, Tipping, 1999 - Eq. 25, 26)
        self.M = self.W.transpose(0, 1) @ self.W + self.sigma2 * torch.eye(self.n_components, device=X.device)
        self.M_inv = torch.linalg.inv(self.M)
        
        # the paper uses "x" to denote "z" (latent variables)
        self.x = self.M_inv @ torch.t(self.W) @ torch.t((X - self.mu))  # k x N
        self.xxT = self.sigma2 * self.M_inv + self.x @ self.x.transpose(0, 1)  # k x k
        
    def _m_step(self, X):
        # M-step: update parameters W, mu, sigma2 based on expected values from E-step
        N = X.shape[0]
        d = X.shape[1]
        # (Bishop, Tipping, 1999 - Eq. 27, 28)
        # self.W = torch.sum((X - self.mu).transpose(0, 1) @ self.x.transpose(0, 1), dim=0) @ torch.sum(torch.linalg.inv(self.xxT), dim=0)
        # self.sigma2 = (1.0 / (N * d)) * torch.sum(torch.linalg.norm(X - self.mu)**2 - 2*self.x.transpose(0, 1) @ self.W.transpose(0, 1) @ (X - self.mu).transpose(0, 1) + torch.trace(self.xxT @ (self.W.transpose(0, 1) @ self.W)))
        
        # another way to look at it from Bishop, Tipping, 1999 - Eq. 29, 30, 31
        # Compute centered data once
        Xc = X - self.mu  # shape (N, d)

        # Efficient computation of SW = S @ W without forming S explicitly:
        # SW = (1/N) * Xc^T @ (Xc @ W) -> cost O(N d q)
        SW = (1.0 / N) * Xc.transpose(0, 1) @ (Xc @ self.W)  # shape (d, q)

        # Compute trace(S) cheaply: tr(S) = (1/N) * sum_i ||x_i - mu||^2
        trS = torch.sum(Xc * Xc) / N

        self.Ik = torch.eye(self.n_components, device=X.device)

        # Use SW and trS in the updates (avoid explicit S)
        WT_SW = self.W.transpose(0, 1) @ SW  # q x q
        den = self.sigma2 * self.Ik + self.M_inv @ WT_SW  # q x q
        W_hat = SW @ torch.linalg.inv(den)  # d x q

        # sigma2 update using trace identity (avoid forming S explicitly)
        # old: sigma2 = (1/d) * trace(S - S W M_inv W_hat^T)
        # note: S W = SW, so second term = trace(M_inv W_hat^T SW)
        self.sigma2 = (1.0 / d) * (trS - torch.trace(self.M_inv @ torch.t(W_hat) @ SW))

        self.W = W_hat
        
    def _fit_em(self, X):
        # initialize parameters
        N, d = X.shape
        self.mu = torch.mean(X, dim=0)  # shape (d,)
        self.W = torch.randn(d, self.n_components, device=X.device)  # random initialization
        self.sigma2 = torch.tensor(1.0, device=X.device)
        print("Starting EM fitting...")
        for iter in range(self.max_iter):
            W = self.W
            # E-step: compute expected values of latent variables given current parameters
            self._e_step(X)

            # M-step: update parameters W, mu, sigma2 based on expected values from E-step
            self._m_step(X)
            
            # Stopping criterion based on change in W
            if torch.norm(self.W - W) < self.stopping_criterion:
                break
            
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
        # TODO: understand why (X - mu)^T and not (X - mu) like in the paper
        # NB: in the original paper data are columns (D×N), so their (X - mu) corresponds to our (X - mu)^T
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


