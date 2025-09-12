""" Neighborhood preserving embedding using k-nearest neighborhood graphs. 
The original Paper: Neighborhood preserving embedding, (He et al. 2005),
Link to the paper: https://ieeexplore.ieee.org/document/1544858

The code is adapted from https://lvdmaaten.github.io/drtoolbox/ MATLAB Code
"""

import numpy as np
from scipy import linalg
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import kneighbors_graph


class NeighborhoodPreservingEmbedding(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    n_components : int
        Number of dimensions for the projection.
    n_neighbors : int
        Number of neighbors to build the graph.
    reg_tol : float
        Tolerance used while optimizing the weights
    """
    def __init__(self, n_components=2, n_neighbors=10, reg_tol=1e-5):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.reg_tol = reg_tol

    def fit(self, X):
        X = check_array(X)
        n_samples, n_features = X.shape

        G = kneighbors_graph(X, n_neighbors=self.n_neighbors,
                             mode='distance', include_self=False)
        G = G.maximum(G.T)
        neighborhoods = [G[i].nonzero()[1] for i in range(n_samples)]

        W = np.zeros((n_samples, n_samples))
        for i, neighbors in enumerate(neighborhoods):
            if len(neighbors) == 0:
                continue
            Z = X[neighbors] - X[i]
            C = Z @ Z.T
            if len(neighbors) > n_features:
                C += np.eye(len(neighbors)) * self.reg_tol * np.trace(C)
            w = np.linalg.solve(C, np.ones(len(neighbors)))
            w /= np.sum(w)
            W[i, neighbors] = w

        I_minus_W = np.eye(n_samples) - W
        M = I_minus_W.T @ I_minus_W

        XMX = X.T @ M @ X
        XDX = X.T @ X
        XMX = (XMX + XMX.T) / 2
        XDX = (XDX + XDX.T) / 2

        evals, evecs = linalg.eigh(XMX, XDX, subset_by_index=(0, self.n_components - 1))

        idx = np.argsort(evals)
        self.components_ = evecs[:, idx[:self.n_components]]
        return self

    def transform(self, X):
        X = check_array(X)
        return X @ self.components_