"""
Adapted from: "Manifold Fitting" (Yao et al., 2023)
Original Paper: https://arxiv.org/abs/2304.07680
Original repository: https://github.com/zhigang-yao/manifold-fitting
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree

@dataclass
class YSL23:
    """Manifold-fitting projection and noise-level estimation.
    Parameters
    ----------
    data : np.ndarray, shape (N, D)
        Reference point cloud (the manifold samples). The KD-tree is built on this.
    c0, c1, c2 : float
        Fixed constants controlling the radii r0, r1, r2.
    sigma : Optional[float]
        Known noise level. If not provided, it can be estimated with estimate_sigma function.
    estimate_sig : Optional[bool]
        If True, sigma will be estimated.
    d : Optional[int] = None
        Intrinsic dimensionality used for sigma estimation. If None, it will be set to 0
    k : int, default=5
        Minimum neighbor count for the initial Euclidean ball (also used for `query`).
    """

    data: np.ndarray
    c0: Optional[float] = None
    c1: Optional[float] = None
    c2: Optional[float] = None
    sigma: Optional[float] = None
    estimate_sig: Optional[bool] = None
    d : Optional[int] = None
    verbose : Optional[bool] = False
    k: Optional[int]  = 5

    def __post_init__(self) -> None:
        
        self.c0 = self.c0 if self.c0 is not None else 2.0
        self.c1 = self.c1 if self.c0 is not None else 3.0
        self.c2 = self.c2 if self.c0 is not None else 4.0
        self.d = self.d if self.d is not None else 0.0
        self.estimate_sig = self.estimate_sig if self.estimate_sig is not None else False
        self.sigma = self.sigma if self.sigma is not None else 0.05

        self.tree = cKDTree(self.data)
        self.D = self.data.shape[1]
        self.N = self.data.shape[0]
        self.r0 = None
        self.r1 = None
        self.r2 = None
        self.data = None

        if self.estimate_sig:
            _ = self.estimate_sigma(sigma_init=self.sigma)

    def _set_radii(self, sigma: float) -> None:
        self.r0 = self.c0 * sigma
        self.r1 = self.c1 * sigma
        self.r2 = self.c2* sigma * np.sqrt(np.log(1 / sigma))

    def _project(
            self,
            x,
            *,
            idx: Optional[int] = None,
            # Use idx only if you don't want to consider the point during the projection
            verbose: bool = False,
        )-> Tuple[np.array, int]:
        """projects a noisy point x"""
        
        # Estimate contraction direction
        IDX1 = self.tree.query_ball_point(x, r= self.r0)
        _, IDX2 = self.tree.query(x, k=self.k+1 if idx else self.k)
        IDX = np.union1d(IDX1, IDX2).astype(np.int64)
        BNbr = self.tree.data[IDX[1 if idx else 0:], :]

        if verbose and idx is None:
            print("BNbr:", len(IDX))

        xbar = np.mean(BNbr, axis=0)
        dx = x - xbar
        dx = dx / np.linalg.norm(dx)

        # Construct the clynder
        e = np.zeros_like(dx)
        e[0] = 1  
        v = dx - e
        v = v / np.linalg.norm(v) 
        sample_s = self.tree.data - x
        v_dot_sample = sample_s @ v 
        sample_s -= 2 * np.outer(v_dot_sample, v)
        CNbr = (np.abs(sample_s[:, 0]) < self.r2/2.0) & (np.sum(sample_s[:, 1:] ** 2, axis=1) < self.r1 ** 2)

        if idx: # Exclude the point x
            CNbr[idx]=False

        if np.sum(CNbr) > 3:
            return np.mean(self.tree.data[CNbr, :], axis=0), int(CNbr.sum())
        else:
            return xbar, int(CNbr.sum())

    # Sigma estimation
    def estimate_sigma(
        self,
        *,
        c0: Optional[float] = None,
        c1: Optional[float] = None,
        c2: Optional[float] = None,
        d: Optional[int] = None,
        sigma_init: Optional[float] = 0.1,
        max_iter: int = 10,
        tol: float = 1e-5,
        verbose: bool = False,
        ) -> float:
        """Estimates the noise level sigma via the iterative scheme."""
        if c0 is not None:
            self.c0 = c0
        if c1 is not None:
            self.c1 = c1
        if c2 is not None:
            self.c2 = c2
        if d is not None:
            self.d = d
        if verbose is not None:
            verbose = self.verbose

        N, D = self.N, self.D
        # print("check")
        denom = (D - self.d) * N
        sig = sigma_init

        for it in range(max_iter):
            Mout = self.tree.data.copy()
            # Update radii from current sigma
            self._set_radii(sig)

            for ii in range(N):
                x = self.tree.data[ii]
                Mout[ii], _ = self._project(x,idx=ii)

            sig_new = np.sqrt(np.sum((Mout - self.tree.data) ** 2) / denom)
            if verbose:
                print(f"Iter {it+1}: sigma={sig_new:.6f}")

            if abs(sig_new - sig) < tol:
                if verbose:
                    print(f"Estimated sigma converged to {sig_new:.6f}")
                self.sigma = sig_new
                return sig_new
            sig = sig_new

        if verbose:
            print(f"Max iterations reached. Final sigma = {sig:.6f}")
        self.sigma = sig
        return sig

    # Projection
    def project(
        self,
        points: np.ndarray,
        *,
        sigma: Optional[float] = None,
        c0: Optional[float] = None,
        c1: Optional[float] = None,
        c2: Optional[float] = None,
        d: Optional[int] = None,
        sigma_init: Optional[float] = None,
        verbose: bool = False,
        ) -> Tuple[np.ndarray, float]:
        """Project given points onto the manifold.
        Parameters:
        ----------
        points: np.ndarray, shape (m, D)
        Returns:
        ----------
        Mout : np.ndarray, shape (m, D), Projected points.
        avg_cnbr : float, Average count of neighbors used per point projection. 
        """
        if not isinstance(points, np.ndarray):
            points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != self.D:
            raise ValueError("points must be (m, D) with D matching the reference data.")

        # Update constants if provided
        if c0 is not None:
            self.c0 = c0
        if c1 is not None:
            self.c1 = c1
        if c2 is not None:
            self.c2 = c2

        # Determine sigma
        if sigma is not None:
            sig = float(sigma)
        elif self.sigma is not None:
            sig = float(self.sigma)
        else:
            sig = self.estimate_sigma(
                c0=self.c0,
                c1=self.c1,
                c2=self.c2,
                d=(0 if d is None else d),
                sigma_init=sigma_init,
                verbose=verbose,
            )
        self._set_radii(sig)

        m = points.shape[0]

        Mout = points.copy()
        CNBr_counts: List[int] = []

        for ii in range(m):
            # print(ii)
            x = points[ii]
            Mout[ii], nghbr_count = self._project(x,verbose=verbose)
            CNBr_counts.append(nghbr_count)

        avg_cnbr = float(np.mean(CNBr_counts))
        return Mout, avg_cnbr
    
    def deviations(self,
        points: np.ndarray,
        *,
        sigma: Optional[float] = None,
        c0: Optional[float] = None,
        c1: Optional[float] = None,
        c2: Optional[float] = None,
        d: Optional[int] = None,
        sigma_init: Optional[float] = None,
        verbose: bool = False,
        ) -> Tuple[np.ndarray, float]:
        """
        Returns deviations only
        """
        if not isinstance(points, np.ndarray):
            points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != self.D:
            raise ValueError("points must be (m, D) with D matching the reference data.")

        # Update constants if provided
        if c0 is not None:
            self.c0 = c0
        if c1 is not None:
            self.c1 = c1
        if c2 is not None:
            self.c2 = c2

        # Determine sigma
        if sigma is not None:
            sig = float(sigma)
        elif self.sigma is not None:
            sig = float(self.sigma)
        else:
            sig = self.estimate_sigma(
                c0=self.c0,
                c1=self.c1,
                c2=self.c2,
                d=(0 if d is None else d),
                sigma_init=sigma_init,
                verbose=verbose,
            )
        
        self._set_radii(sig)
        projected_points, _ = self.project(points=points,verbose=verbose)

        return np.linalg.norm(projected_points-points,axis=1).reshape(-1,1)

