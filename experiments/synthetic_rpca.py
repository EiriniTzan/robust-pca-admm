import os
import numpy as np
import matplotlib.pyplot as plt
from robust_pca_admm.admm_rpca import rpca_admm

def generate_low_rank(m: int, n: int, r: int, seed: int = 0) -> np.ndarray:
    """
    Generate a low-rank matrix of size (m, n) with rank r.

    Parameters
    ----------
    m : int
        Number of rows in the matrix.
    n : int
        Number of columns in the matrix.
    r : int
        Rank of the matrix.
    seed : int, optional
        Seed for the random number generator. Defaults to 0.

    Returns
    -------
    np.ndarray
        A low-rank matrix of size (m, n) with rank r.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, r))
    B = rng.standard_normal((r, n))
    return A @ B


def generate_sparse(
    m: int,
    n: int,
    p: float,
    scale: float = 10.0,
    seed: int = 1,
) -> np.ndarray:
    """
    Generate a sparse matrix of size (m, n) with a given proportion p of
    non-zero elements.

    Parameters
    ----------
    m : int
        Number of rows in the matrix.
    n : int
        Number of columns in the matrix.
    p : float
        Proportion of non-zero elements in the matrix.
    scale : float, optional
        Scaling factor for the non-zero elements. Defaults to 10.0.
    seed : int, optional
        Seed for the random number generator. Defaults to 1.

    Returns
    -------
    np.ndarray
        A sparse matrix of size (m, n) with a given proportion p of non-zero
        elements.
    """
    rng = np.random.default_rng(seed)
    S = np.zeros((m, n))
    k = int(p * m * n)
    i = rng.choice(m * n, size=k, replace=False)
    S.flat[i] = scale * rng.standard_normal(k)
    return S