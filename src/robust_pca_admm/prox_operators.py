import numpy as np

def l1_prox(X: np.ndarray, tau: float) -> np.ndarray:

    """
    Compute the proximal operator for the L1 norm, given a matrix X and a regularization
    parameter tau.

    Parameters
    ----------
    X : np.ndarray
        The input matrix.
    tau : float
        The regularization parameter.

    Returns
    -------
    np.ndarray
        The proximal operator of X with respect to the L1 norm and regularization
        parameter tau.
    """
    if tau < 0:
        raise ValueError("tau must be non-negative")
    X = np.asarray(X) #ensure we have a Numpy array
    
    #Soft-thresholding: shrinking magnitudes by tau
    S = np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)
    return S


def nuclear_prox(X: np.ndarray, tau: float) -> np.ndarray:
    
    """
    Compute the proximal operator for the nuclear norm, given a matrix X and a
    regularization parameter tau.

    Parameters
    ----------
    X : np.ndarray
        The input matrix.
    tau : float
        The regularization parameter.

    Returns
    -------
    np.ndarray
        The proximal operator of X with respect to the nuclear norm and regularization
        parameter tau.
    """
    if tau < 0:
        raise ValueError("tau must be non-negative")
    X = np.asarray(X)
    
    U, s, Vt = np.linalg.svd(X, full_matrices=False) #thin SVD(smaller matrices, faster)
    
    s_shrunk = np.maximum(s - tau, 0.0) #SVT: soft-thresholding singular values
    #s_shrunk[i] corresponds to s_i' in the mathematical formulation
    
    L = (U * s_shrunk) @ Vt #reconstruct the matrix after SVT:
#multiplying U by s_shrunk scales each left singular vector u_i by s_shrunk[i]
#This is equivalent to U @ diag(s_shrunk) @ Vt, but avoids creating the diagonal matrix.
    
    return L