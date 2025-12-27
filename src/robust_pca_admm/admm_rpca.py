from dataclasses import dataclass

import numpy as np

from .prox_operators import l1_prox, nuclear_prox


@dataclass
class RPCAReport:
    """
    Contains diagnostic information from ADMM (Alternating Direction Method of Multipliers)
    iterations for the Robust PCA problem.

    """
    residuals: list[float] #relative primal residual at each iteration
    iters: int #number of ADMM iterations executed
    converged: bool #True if ADMM reached convergence tolerance
    lam: float #sparsity weight used in PCP(Principal Component Pursuit) objective
    rho: float #ADMM penalty parameter (augmented Lagrangian)
    

def rpca_admm(
    M: np.ndarray,
    lam: float | None = None,
    rho: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-7,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, RPCAReport]:
    """
    Solve the Robust PCA problem using ADMM (Alternating Direction Method of Multipliers)
    to separate a low-rank matrix from a sparse matrix.

    Parameters
    ----------
    M : np.ndarray
        The input matrix to be decomposed.
    lam : float | None, optional
        Sparsity weight used in PCP (Principal Component Pursuit) objective.
        If None, defaults to 1 / sqrt(max(m, n)), a standard choice in RPCA.
    rho : float, optional
        ADMM penalty parameter (augmented Lagrangian). Defaults to 1.0.
    max_iter : int, optional
        Maximum number of ADMM iterations. Defaults to 1000.
    tol : float, optional
        Convergence tolerance of the relative primal residual. Defaults to 1e-7.
    verbose : bool, optional
        If True, prints progress occasionally (first iteration, every 50 iterations
        and at convergence). Defaults to False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, RPCAReport]
        A tuple containing the low-rank matrix L, the sparse matrix S, and
        a RPCAReport object containing diagnostic information from the ADMM
        iterations."""
    
    #check for valid input
    if rho <= 0:
        raise ValueError("rho must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")
    
    #use float NumPy array for more reliable SVD/thresholding 
    M = np.asarray(M, dtype=float)
    m, n = M.shape #matrix dimensions (rows, cols)
    
    #in case lambda is not provided, use a common RPCA default
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n)) #scales with matrix size
        
    #initialize with zeros of the same shape/dtype as M (avoids shape/dtype mismatches):
    #L: low-rank estimate
    #S: sparse estimate
    #Y: scaled dual variable (keeps track of the mismatch between M and L + S)
    L = np.zeros_like(M)
    S = np.zeros_like(M)
    Y = np.zeros_like(M)
    
    #compute Frobenius norm to normalize the primal residual
    #this makes the stopping rule scale-invariant (independent of units/scaling)
    norm_M = np.linalg.norm(M, ord="fro")

    if norm_M == 0:
        #special case: M is the zero matrix
        #the decomposition is trivial since M = L + S holds for L = 0 and S = 0
        report = RPCAReport(residuals=[0.0], iters=0, converged=True, lam=lam, rho=rho)
        return L, S, report

    #store convergence diagnostics
    residuals: list[float] = []
    converged = False
    
    #ADMM loop
    for i in range(1, max_iter + 1):

        #low-rank update: S fixed, compute a low-rank L that fits M - S (SVT on singular
        #values)
        L = nuclear_prox(M - S + Y, tau=1.0 / rho)

        #sparse update: L fixed, compute a sparse S that fits M - L (soft-thresholding 
        #on entries)
        S = l1_prox(M - L + Y, tau=lam / rho)

        #dual update: R is the constraint residual (how far L + S is from M)
        #Y stores info about how far L + S is from M, helps correct it over iterations
        R = M - L - S
        Y = Y + R

        #convergence check (scale-invariant): relative primal residual
        relative_res = np.linalg.norm(R, ord="fro") / norm_M
        residuals.append(relative_res)
        
        #print progress occasionally if verbose is enabled
        if verbose and (i == 1 or i % 50 == 0 or relative_res < tol):
            print(f"rpca_admm: iter={i:4d}  rel_res={relative_res:.3e}")

        if relative_res < tol:
            converged = True
            break

    report = RPCAReport(
        residuals=residuals,
        iters=len(residuals),
        converged=converged,
        lam=lam,
        rho=rho,
    )
    return L, S, report


        
