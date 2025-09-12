""" Locality Preserving Projection using k-nearest neighborhood graphs. 
The original Paper: Locality Preserving Projections, (He and Niyogi,2003),
Link to the paper: https://proceedings.neurips.cc/paper_files/paper/2003/file/d69116f8b0140cdeb1f99a4d5096ffe4-Paper.pdf
"""

import numpy as np
from scipy import linalg
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import kneighbors_graph


class LocalityPreservingProjection(BaseEstimator, TransformerMixin):
    """

    Parameters
    ----------
    n_components : int
        Number of dimensions for the projection.
    n_neighbors : int
    Number of neighbors to build the graph.
    weight : str, optional ['adjacency'|'heat']
        Type of edge weighting. 'adjacency' = binary, 'heat' = Gaussian kernel.
    weight_width : float or None
        Width (sigma) for the heat kernel. If None, uses median pairwise distance.
    """
    def __init__(self, n_components=2, n_neighbors=10,
                 weight='heat', weight_width=None):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.weight = weight
        self.weight_width = weight_width

    def fit(self, X, y=None):
        X = check_array(X)
        W = self._compute_weights(X)
        self.components_ = self._compute_projection(X, W)
        return self

    def transform(self, X):
        X = check_array(X)
        return X @ self.components_

    def _compute_projection(self, X, W):
        X = check_array(X)
        D = np.diag(W.sum(1))
        L = D - W

        XLX = X.T @ L @ X
        XDX = X.T @ D @ X

        _ , evecs = linalg.eigh(XLX, XDX, subset_by_index=(0, self.n_components - 1))
        return evecs

    def _compute_weights(self, X):
        X = check_array(X)
        W = kneighbors_graph(X, n_neighbors=self.n_neighbors,
                             mode='distance' if self.weight == 'heat' else 'connectivity',
                             include_self=False)

        if self.weight == 'heat':
            if self.weight_width is None:
                dists = pairwise_distances(X)
                self.weight_width = np.median(dists[dists > 0])
            W.data = np.exp(-W.data ** 2 / self.weight_width ** 2)

        W = W.maximum(W.T)
        return W.toarray()
