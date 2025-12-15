import torch

class PCA:
    """Torch-based PCA compatible with the PPCA implementation style.

    Public API mirrors the simple numpy PCA: `fit`, `transform`, `fit_transform`.
    Accepts numpy arrays or torch tensors; internal computations use torch and
    returned values are torch tensors on the selected device.
    """

    def __init__(self, nb_components=1, dtype=torch.float32):
        self.nb_components = nb_components
        self.dtype = dtype
        self.r2 = None
        self.mu = None
        self.components = None

    def fit(self, X):
        # Accept numpy array or torch tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self.dtype)

        N, d = X.shape
        # compute mean (d,)
        self.mu = torch.mean(X, dim=0)

        # center data
        Xc = X - self.mu

        # covariance matrix (d x d): (1/N) Xc^T Xc
        S = (1.0 / N) * Xc.T @ Xc

        # eigen decomposition (symmetric) -> eigh returns ascending eigenvalues
        eigvals, eigvecs = torch.linalg.eigh(S)
        # sort descending
        idx = torch.argsort(eigvals, descending=True)
        eigvals_desc = eigvals[idx]
        eigvecs_desc = eigvecs[:, idx]

        # keep top components
        self.components = eigvecs_desc[:, :self.nb_components]

        # explained variance ratio
        total = torch.sum(eigvals_desc)
        self.r2 = float(torch.sum(eigvals_desc[:self.nb_components]) / (total + 1e-12))

        return self

    def transform(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self.dtype)

        return (X - self.mu) @ self.components

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)