import torch
import torch.nn as nn
import math
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt

import numpy as np
import tqdm

class MPPCA(nn.Module):
    def __init__(self,k_mixtures,n_components,method='em',max_iter=200, stopping_criterion=1e-6):
        super().__init__()
        self.n_components = n_components
        self.k_mixtures = k_mixtures
        self.method = method
        self.max_iter = max_iter
        self.stopping_criterion = stopping_criterion
        print(f"MPPCA initialized with k_mixtures={k_mixtures}, n_components={n_components}, method={method}")
    
    def fit(self,X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
            
        self.N = X.shape[0]  # number of samples
        self.d = X.shape[1]  # data dimensionality

        if(self.method=="em"):
            self._fit_em(X)
        
        if(self.method =="missing_data"):
            self._fit_em_missingdata(X)

    def _update_responsabilities(self,X):
        N = X.shape[0]

        llh = torch.zeros(N,self.k_mixtures)
        for k in range(self.k_mixtures):

            if self.sigma2[k] <= 1e-2: # Gives singular expression. Reset of sigma2
                self.sigma2[k] = 0.01 + np.random.rand()

            # TODO Cinv_k can be expressed with Minv (save calculations)
            Ck = torch.mm(self.W[k], torch.transpose(self.W[k],dim0=0,dim1=1)) + self.sigma2[k] * torch.eye(self.d)

            try:
                dist = MultivariateNormal(self.mu[k], Ck)            
            except:
                raise ValueError("Unable to convergence. Check dims.")
            
            llh[:,k] = torch.log(self.pi[k] + 1e-12) + dist.log_prob(X)

        self.Rni = torch.clamp(torch.softmax(llh,dim=1),min=1e-8) # shape (N,k)
        self.Rni = self.Rni / self.Rni.sum(dim=1, keepdim=True) # re-normalization after clamp

        observed_loglikelihood = torch.logsumexp(llh,dim=1).sum(dim=0) # observed loglikelihood
        return observed_loglikelihood
    
    def _e_step(self,X):
        """
        E-Step of EM Algorithm.
        Compute responsabilities according to X entries.
        """

        observed_loglikelihood = self._update_responsabilities(X)
        return observed_loglikelihood
    
    def _m_step(self,X):
        """
        M-Step of EM Algorithm

        Update Wk,muk,sigmak and pi according to update formulas in litterature.
        """
        
        self.pi = torch.mean(self.Rni,dim=0)
        I = torch.eye(self.n_components, device=X.device)

        Nk = torch.sum(self.Rni, dim=0) # (k,)
        self.mu = (self.Rni.T @ X) / Nk.unsqueeze(1) # (k,)

        for k in range(self.k_mixtures):
            
            Mk = self.W[k].transpose(0, 1) @ self.W[k] + self.sigma2[k] * torch.eye(self.n_components, device=X.device)
            Minv_k = torch.inverse(Mk + 1e-6 * torch.eye(Mk.shape[0]))
            Xc = X - self.mu[k]  # shape (N, d)
            
            S = (1/(self.pi[k]*self.N)) * torch.transpose(self.Rni[:,k].unsqueeze(1) * Xc, dim0=0, dim1=1) @ (Xc)
            SW = S @ self.W[k]  # d x q
            trS = torch.trace(S)
            
            # Precompute W^T @ SW (q x q)
            WT_SW = self.W[k].transpose(0, 1) @ SW  # q x q
            
            # Denominator for W update: sigma2 * I + M_inv @ (W^T SW)
            den = self.sigma2[k] * I + Minv_k @ WT_SW  # q x q
            
            # Compute W_hat efficiently
            W_hat = SW @ torch.linalg.inv(den)  # d x q
            
            # sigma2 update using trace identity: avoid explicit S
            self.sigma2[k] = (1.0 / self.d) * (trS - torch.trace(Minv_k @ torch.t(W_hat) @ SW))
            self.W[k] = W_hat

    def _fit_em(self,X):
        """
        We follow the EM algorithm derived by Tipping and Bishop
        in Mixtures of Probabilistic Principal Component
        Analysers (1999).
        """
        
        self.W = torch.randn(self.k_mixtures, self.d, self.n_components, device=X.device) # shape (k,d,n)
        self.mu = torch.mean(X, dim=0).repeat(self.k_mixtures,1)  # shape (k,d)
        self.sigma2 = torch.ones(self.k_mixtures,device=X.device) # shape (k,)

        alpha = torch.ones(self.k_mixtures)
        self.pi = torch.distributions.Dirichlet(alpha).sample() # shape (k,)

        print("Starting EM fitting on {} epochs...".format(self.max_iter))
        pbar = tqdm.tqdm(range(self.max_iter), desc="EM - MPPCA", unit="iter")

        llh_history = [-torch.inf]
        for iter in pbar:
            obs_llh = self._e_step(X)
            pbar.set_postfix({
                "loglik": f"{obs_llh:.2f}",
            })
            self._m_step(X)
            if(abs(obs_llh - llh_history[-1]) < self.stopping_criterion or torch.isnan(obs_llh)):
                pbar.close()
                break
            
            llh_history.append(obs_llh.float())

        self.llh_history = llh_history[20:]

    def showstats(self):
        """
        Show model'statistics evolution (e.g loglikelihood)
        """
        fig,axs = plt.subplots()
        axs.plot(range(1,len(self.llh_history)+1), self.llh_history)
        axs.set_title("Loglikelihood")
        axs.set_xlabel("epochs")
        axs.set_ylabel("llh")

        plt.show()

    
    def transform(self, X):
        # Ensure X is a tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
            
        if self.method == 'baseline':
            return self.pca.transform(X.numpy())
        
        else:
            N = X.shape[0]

            # Hierarchical modelling : we determine clusters of each line
            ks = torch.zeros(N)
            Xt = torch.zeros((N, self.n_components), device=X.device)  # shape (N, n)

            # We sample X from it's conditional distribution given its cluster (observation model)
            # For that, we sample from a Gaussian N(x|Wk*z+mu, sigma2k*I) (Bishop, Tipping, 1999 - Eq. 6)
            self._update_responsabilities(X)
            
            for i in range(N):
                k = torch.argmax(self.Rni[i,:],dim=0)

                M = self.W[k].transpose(0, 1) @ self.W[k] + self.sigma2[k] * torch.eye(self.n_components, device=X.device)

                try:
                    M_inv = torch.linalg.inv(M)          
                except:
                    raise ValueError("Unable to convergence. Check dims.")
                

                transform_mean = M_inv @ self.W[k].transpose(0, 1) @ (X[i,:] - self.mu[k])  # n x 1
                transform_covariance = self.sigma2[k] * M_inv  # n x n

                mean_i = transform_mean
                sample = MultivariateNormal(mean_i, transform_covariance).sample()
                Xt[i] = sample
                ks[i] = k

            return Xt, ks
        
    def _cost_function(self, X):
        """Compute the negative log-likelihood cost function for PPCA (to be minimized)
            Bishop, Tipping, 1999 - Eq. 4

        Args:
            X (tensor): data of shape (N, d)
        """

        cost = 0
        N = X.shape[0]
        for k in range(self.k_mixtures):
            # Centered data
            Xc = X - self.mu

            # Build model covariance C = W W^T + sigma2 * I (d x d)
            C = self.W[k] @ self.W[k].T + torch.abs(self.sigma2[k]) * torch.eye(self.d, device=self.W.device)
            
            # Compute S = (1/N) Xc^T Xc (sample covariance)
            S = (1.0 / N) * Xc.T @ Xc

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
            
            cost += self.pi[k] * (N / 2.0 * (const + logabsdet + trace_term / N))
        
        return cost
    
    def sample(self, n_samples):
        """Sample n_samples from the PPCA model
        
        With the marginal distribution over the latent variables also Gaussian and conventionally
        defined by x ~ N(0, I), the marginal distribution for the observed data t is readily obtained
        by integrating out the latent variables and is likewise Gaussian: t ~ N(μ, C) where C = W*W^T + sigma2*I

        Args:
            N (int): Number of samples to generate
            
        """
        ks = torch.multinomial(self.pi, num_samples=n_samples, replacement=True)
        counts = torch.bincount(ks, minlength=len(self.pi))
        generated_samples = []
        label_samples = []
        for k in range(len(counts)):
            if counts[k]==0: # no sample for this value
                continue

            Ck = self.W[k] @ self.W[k].transpose(0, 1) + self.sigma2[k] * torch.eye(self.W[k].shape[0], device=self.W[k].device)
            if self.mu[k] is None:
                raise ValueError("Model must be fitted before generating samples.")
            
            samples_k = MultivariateNormal(self.mu[k], Ck).sample((counts[k],))
            generated_samples.append(samples_k)
            label_samples += [k]*counts[k]

        return torch.vstack(generated_samples),label_samples
    
    def infer_missingdata(self,X,obs_mask):
        """
        Infer missing entries in a data vector using a trained MPPCA model.

        This method estimates the missing values in `X` given a boolean mask `obs_mask`
        that indicates observed entries.

        Args:
        X (torch.Tensor): Input data vector of shape (d,) with missing entries.
        obs_mask (torch.BoolTensor): Boolean mask of shape (d,) where True indicates
                                     observed entries and False indicates missing entries.

        Returns:
            Xcorr (torch.Tensor): Complete data vector with missing entries imputed.
        """
        Xobs = X[obs_mask]

        llh = torch.zeros(1,self.k_mixtures)
        Cs = []  
        for k in range(self.k_mixtures):
            Ck = torch.mm(self.W[k], torch.transpose(self.W[k],dim0=0,dim1=1)) + self.sigma2[k] * torch.eye(self.d)
            Cs.append(Ck)
            
            dist = MultivariateNormal(self.mu[k][obs_mask], Ck[obs_mask,:][:,obs_mask])            
            llh[:,k] = torch.log(self.pi[k] + 1e-12) + dist.log_prob(Xobs)

        self.Rni = torch.clamp(torch.softmax(llh,dim=1),min=1e-8) # shape (N,k)
        self.Rni = self.Rni / self.Rni.sum(dim=1, keepdim=True) # re-normalization after clamp

        Xmiss = torch.zeros(torch.sum((~obs_mask)))
        for k in range(self.k_mixtures):
            Xmiss += torch.sum(self.Rni[:,k]) * (self.mu[k][~obs_mask] +  Cs[k][:,obs_mask][~obs_mask,:] @ torch.inverse(Cs[k][:,obs_mask][obs_mask,:]) @ (Xobs - self.mu[k][obs_mask]))

        Xcorr = torch.zeros(len(obs_mask), device=Xobs.device)
        Xcorr[obs_mask] = Xobs
        Xcorr[~obs_mask] = Xmiss
        return Xcorr
    
    def sample_transform(self, n_samples):
        """Sample N points from the MPPCA model and transform them to the latent space

        Args:
            n_samples (int): Number of samples to generate and transform
        Returns:
            torch.Tensor: Transformed samples in the latent space
        """
        return self.transform(self.sample(n_samples))   
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)