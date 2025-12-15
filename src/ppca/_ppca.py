import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import tqdm
from sklearn.decomposition import PCA

class PPCA(nn.Module):
    def __init__(self, n_components=3, method='svd', max_iter=200, stopping_criterion=1e-6):
        super(PPCA, self).__init__()
        self.n_components = n_components
        self.method = method
        self.max_iter = max_iter
        self.stopping_criterion = stopping_criterion
        print(f"PPCA initialized with n_components={n_components}, method={method}")
        self.losses = []

    def fit(self, X):
        # Ensure X is a tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
            
        self.N = X.shape[0]  # number of samples
        self.d = X.shape[1]  # data dimensionality
            
        # SVD
        if self.method == 'svd':
            W, mu, sigma2 = self._fit_eig_decomp(X)
            print("cost_svd", self._cost_function(X))
        # EM
        elif self.method == 'em':
            W, mu, sigma2 = self._fit_em(X)
        # GD
        elif self.method == 'gd':
            W, mu, sigma2 = self._fit_gd(X)
        # BASELINE
        elif self.method == 'baseline':
            self._fit_baseline(X)

        else:
            raise NotImplementedError(f"Method {self.method} not implemented.")
        
        return W, mu, sigma2
        
    def _fit_baseline(self, X):
        self.pca = PCA(n_components=self.n_components).fit(X.numpy())
        self.W = torch.tensor(self.pca.components_.T, dtype=torch.float32)
        self.mu = torch.tensor(self.pca.mean_, dtype=torch.float32)
        self.sigma2 = torch.tensor(0.0, dtype=torch.float32)
        
        return self.W, self.mu, self.sigma2
    
    def _fit_eig_decomp(self, X):
        # Define N and D
        N = X.shape[0] # number of samples
        D = X.shape[1] # data dimensionality
        
        # 1. Estimate data mean (mu)
        self.mu = torch.mean(X, dim=0) # shape (d,)
        
        # Sample covariance matrix (Bishop, Tipping, 1999 - Eq. 5)
        # Use D x D covariance: S = (1/N) * (X - mu)^T (X - mu)
        # TODO: understand why (X - mu)^T (X - mu) and not (X - mu) (X - mu)^T like in the paper
        # NB: I guess the paper data as columns (D×N), so their (X - mu)(X - mu)^T corresponds to our (X - mu)^T (X - mu)
        S = (1.0 / self.N) * (X - self.mu).transpose(0, 1) @ (X - self.mu)  # shape (D, D)

        # Eigen decomposition for symmetric matrix -> real eigenvalues/eigenvectors
        eig_val, eig_vec = torch.linalg.eigh(S)  # eig_val shape (D,), eig_vec shape (D, D)
        # Sort eigenvalues/vectors in descending order
        eig_sort = torch.argsort(eig_val, descending=True)
        U_q = eig_vec[:, eig_sort[:self.n_components]]  # D x q matrix of principal eigenvectors
        Lambda_q = torch.diag(eig_val[eig_sort[:self.n_components]])  # q x q diagonal matrix
        
        # enforce deterministic sign: make the largest-abs element positive for each eigenvector
        max_abs_idx = torch.argmax(torch.abs(U_q), dim=0)                     # (q,)
        signs = torch.sign(U_q[max_abs_idx, torch.arange(U_q.shape[1])])      # (q,)
        U_q = U_q * signs.unsqueeze(0) 
          
        # 2. Estimate sigma2 as variance 'lost' in the projection, averaged over the lost dimensions (Bishop, Tipping, 1999 - Eq. 8)
        if self.d > self.n_components:
            # sum of the discarded eigenvalues (from q to D-1)
            discarded = eig_val[eig_sort[self.n_components:]]
            self.sigma2 = torch.sum(discarded) / (self.d - self.n_components)
        else:
            self.sigma2 = torch.tensor(0.0, device=X.device)
        
        # 3. Estimate W by constructing W = U_k @ sqrt(lambda_k - sigma2*I_k)) (Bishop, Tipping, 1999 - Eq. 7)
        # Ensure numerical stability inside sqrt
        adjust = Lambda_q - self.sigma2 * torch.eye(self.n_components, device=X.device)
        adjust = torch.clamp(adjust, min=0.0)
        self.W = U_q @ torch.sqrt(adjust) 
        
        self.losses = [self._cost_function(X) for _ in range(self.max_iter)]  # no losses for SVD method
        
        return self.W, self.mu, self.sigma2 
        
    def _e_step(self, X):
        """E-step: compute expected values of latent variables given current parameters 
        Bishop, Tipping, 1999 - Eq. 25, 26

        Args:
            X (tensor): data of shape (N, d)
        """
        self.M = self.W.transpose(0, 1) @ self.W + self.sigma2 * torch.eye(self.n_components, device=X.device)
        self.M_inv = torch.linalg.inv(self.M)
        
        # the paper uses "x" to denote "z" (latent variables)
        self.x = self.M_inv @ torch.t(self.W) @ torch.t((X - self.mu))  # k x N
        self.xxT = self.sigma2 * self.M_inv + self.x @ self.x.transpose(0, 1)  # k x k
        
    def _m_step(self, X, compute_SW=True):
        """M-step: update parameters W, mu, sigma2 based on expected values from E-step

        Args:
            X (tensor): data of shape (N, d)
            compute_SW (bool, optional): compute S @ W directly or not.
                « When q << d, considerable computational savings might be obtained by not explicitly evaluating S, even though this need only be done once at initialization. 
                The computation of S requires O(Nd²) operations, but an inspection of equations (27) and (28) indicates that the complexity is only O(Ndq). 
                This is reflected by the fact that equations (29) and (30) only require terms of the form SW and tr(S). 
                
                For the former, computing SW as Σ,, x,(xW) is O(Ndq) and so more
                efficient than (Σ, x,x)W, which is equivalent to finding S explicitly. 
                
                The trade-off between the cost of initially computing S directly and that of computing SW more cheaply at each iteration will clearly
                depend on the number of iterations needed to obtain the accuracy of solution required and the ratio of d to q.»
                
                Defaults to True.
        """
        # M-step: update parameters W, mu, sigma2 based on expected values from E-step
        N = X.shape[0]
        d = X.shape[1]
        
        # Using Bishop, Tipping, 1999 - Eq. 29, 30, 31
        self.Ik = torch.eye(self.n_components, device=X.device)
        
        if compute_SW is False:
            self.S = (1.0 / self.N) * torch.t(X - self.mu) @ (X - self.mu)  # shape (d, d)
            W_hat = self.S @ self.W @ torch.linalg.inv(self.sigma2 * self.Ik + self.M_inv @ torch.t(self.W) @ self.S @ self.W)
            self.sigma2 = (1.0 / self.d) * torch.trace(self.S - self.S @ self.W @ self.M_inv @ torch.t(W_hat))
            
        else: 
            # Efficient branch: avoid forming full d x d covariance S.
            # Centered data
            Xc = X - self.mu  # shape (N, d)
            
            # SW = (1/N) * Xc^T @ (Xc @ W)  -> shape (d, q), cost O(N d q)
            SW = (1.0 / self.N) * Xc.transpose(0, 1) @ (Xc @ self.W)  # d x q
            
            # trace(S) = (1/N) * sum_i ||x_i - mu||^2
            trS = torch.sum(Xc * Xc) / self.N  # scalar
            
            # Precompute W^T @ SW (q x q)
            WT_SW = self.W.transpose(0, 1) @ SW  # q x q
            
            # Denominator for W update: sigma2 * I + M_inv @ (W^T SW)
            den = self.sigma2 * self.Ik + self.M_inv @ WT_SW  # q x q
            
            # Compute W_hat efficiently
            W_hat = SW @ torch.linalg.inv(den)  # d x q
            
            # sigma2 update using trace identity: avoid explicit S
            # sigma2 = (1/d) * (tr(S) - trace(M_inv @ W_hat^T @ SW))
            # TODO: verify correctness of this expression
            self.sigma2 = (1.0 / self.d) * (trS - torch.trace(self.M_inv @ torch.t(W_hat) @ SW))
		
        self.W = W_hat
        
    def _fit_em(self, X):
        # initialize parameters
        self.mu = torch.mean(X, dim=0)  # shape (d,)
        self.W = torch.randn(self.d, self.n_components, device=X.device)  # random initialization
        self.sigma2 = torch.tensor(1.0, device=X.device)
        print("Starting EM fitting on {} epochs...".format(self.max_iter))
        pbar = tqdm.tqdm(range(self.max_iter), desc="EM", unit="iter")
        for iter in pbar:
            W = self.W.clone()
            self._e_step(X)
            self._m_step(X)

            # Stopping criterion based on change in W
            W_norm_diff = torch.norm(self.W - W)
            
            losses = self._cost_function(X)  # compute log-likelihood
            self.losses.append(float(losses))

            # Update progress bar postfix with the latest change norm
            try:
                pbar.set_postfix({"W_change": float(W_norm_diff.item()), "cost": float(losses.item())})
                
            except Exception:
                # In case of any issue converting tensor to float, ignore
                pass

            if W_norm_diff < self.stopping_criterion:
                pbar.close()
                break
            
        return self.W, self.mu, self.sigma2
            
    def _fit_gd(self, X):
        # initialize parameters for optimization
        self.W = torch.nn.Parameter(torch.randn(X.shape[1], self.n_components))  # random initialization
        self.mu = torch.nn.Parameter(torch.mean(X, dim=0))
        self.sigma2 = torch.tensor(0.5)  # positive scalar 
        self.log_sigma2 = torch.nn.Parameter(torch.log(self.sigma2.detach() + 1e-6))

        # Define optimizer
        optimizer = torch.optim.Adam(
            [self.W, self.log_sigma2],
            lr=1e-1, weight_decay=1e-2
        )        
        print("Starting gd fitting on {} epochs...".format(self.max_iter))
        pbar = tqdm.tqdm(range(self.max_iter), desc="gd", unit="iter")
        
        for iter in pbar:
            optimizer.zero_grad()
            loss = self._cost_function_gd(X)
            loss.backward()
            optimizer.step()
            self.losses.append(float(loss.item()))

            # Stopping criterion based on change in cost
            try:
                pbar.set_postfix({"loss": float(loss.item())})
            except Exception:
                # In case of any issue converting tensor to float, ignore
                pass
        
        return self.W, self.mu, self.sigma2
    
    def _cost_function_gd(self, X):
        Xc = X - self.mu
        sigma2 = torch.exp(self.log_sigma2)  # > 0
        C = self.W @ self.W.T + sigma2 * torch.eye(self.d, device=X.device) + 1e-6*torch.eye(self.d, device=X.device)

        S = (1.0 / self.N) * Xc.T @ Xc
        sign, logabsdet = torch.linalg.slogdet(C)
        C_inv = torch.linalg.inv(C)
        trace_term = torch.trace(C_inv @ S)

        const = self.d * torch.log(torch.tensor(2.0 * torch.pi, device=X.device))
        return self.N / 2.0 * (const + logabsdet + trace_term)
        
    def _cost_function(self, X):
        """Compute the negative log-likelihood cost function for PPCA (to be minimized)
            Bishop, Tipping, 1999 - Eq. 4

        Args:
            X (tensor): data of shape (N, d)
        """

        # Centered data
        Xc = X - self.mu

        # Build model covariance C = W W^T + sigma2 * I (d x d)
        C = self.W @ self.W.T + torch.abs(self.sigma2) * torch.eye(self.d, device=self.W.device)
        
        # Compute S = (1/N) Xc^T Xc (sample covariance)
        S = (1.0 / self.N) * Xc.T @ Xc

        # Compute log|C| and C^{-1}
        sign, logabsdet = torch.linalg.slogdet(C)
        if torch.any(sign <= 0):
            # If determinant sign is non-positive, add tiny jitter for numerical stability
            jitter = 1e-6
            C = C + jitter * torch.eye(self.d, device=C.device)
            sign, logabsdet = torch.linalg.slogdet(C)

        C_inv = torch.linalg.inv(C)
        
        # Compute trace(C^{-1} S)
        trace_term = torch.trace(C_inv @ S)
        
        # Constant term
        const = self.d * torch.log(torch.tensor(2.0 * torch.pi, device=X.device))
        
        cost = self.N / 2.0 * (const + logabsdet + trace_term )
        
        return cost

    def transform(self, X):
        # Ensure X is a tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
            
        if self.method == 'baseline':
            return self.pca.transform(X.numpy())
        
        else:
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
    
    def sample(self, n_samples):
        """Sample n_samples from the PPCA model
        
        With the marginal distribution over the latent variables also Gaussian and conventionally
        defined by x ~ N(0, I), the marginal distribution for the observed data t is readily obtained
        by integrating out the latent variables and is likewise Gaussian: t ~ N(μ, C) where C = W*W^T + sigma2*I

        Args:
            N (int): Number of samples to generate
            
        """
        self.C = self.W @ self.W.transpose(0, 1) + self.sigma2 * torch.eye(self.W.shape[0], device=self.W.device)
        if self.mu is None:
            raise ValueError("Model must be fitted before generating samples.")
        generated_samples = MultivariateNormal(self.mu, self.C).sample((n_samples,))
        
        return generated_samples
    
    def sample_transform(self, n_samples):
        """Sample N points from the PPCA model and transform them to the latent space

        Args:
            n_samples (int): Number of samples to generate and transform
        Returns:
            torch.Tensor: Transformed samples in the latent space
        """
        return self.transform(self.sample(n_samples))   
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
