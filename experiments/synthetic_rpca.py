import os
import numpy as np
import matplotlib.pyplot as plt
from robust_pca_admm.admm_rpca import rpca_admm

def generate_low_rank(m: int, n: int, r: int, seed: int = 0) -> np.ndarray:
    """
    Generate a low-rank matrix of size (m, n) with rank at most r.

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
        A low-rank matrix of size (m, n) with rank at most r.
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
    #p must be in [0, 1], it represents a fraction
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")
    
    rng = np.random.default_rng(seed)
    S = np.zeros((m, n))
    k = int(p * m * n) #number of non-zero elements
    i = rng.choice(m * n, size=k, replace=False)
    S.flat[i] = scale * rng.standard_normal(k)
    return S


def save_heatmap(A: np.ndarray, title: str, outpath: str) -> None:
    
    """
    Save a heatmap of a given matrix A to a file.

    Parameters
    ----------
    A : np.ndarray
        The input matrix.
    title : str
        The title of the heatmap.
    outpath : str
        The path to save the heatmap to.

    Returns
    -------
    None
    """
    plt.figure()
    plt.imshow(A, aspect="auto")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    
def save_residual_curve(residuals: list[float], outpath: str) -> None:
    
    """
    Save a plot of the relative primal residual against the iteration number
    to a file.

    Parameters
    ----------
    residuals : list[float]
        A list of relative primal residuals from each iteration of the ADMM
        algorithm.
    outpath : str
        The path to save the plot to.

    Returns
    -------
    None
    """
    plt.figure()
    plt.plot(residuals)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Relative primal residual (log scale)")
    plt.title("ADMM convergence")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    
def main() -> None:
    #Synthetic problem parameters
    m, n = 60, 60
    r_true = 3
    p_true = 0.10
    scale = 10.0

    #Output directory
    out_dir = os.path.join("results", "synthetic")
    os.makedirs(out_dir, exist_ok=True)

    #Generate synthetic data
    L0 = generate_low_rank(m, n, r_true, seed=0)
    S0 = generate_sparse(m, n, p_true, scale=scale, seed=1)
    M = L0 + S0
    
    #Run Robust PCA
    L, S, info = rpca_admm(
        M,
        rho=1.0,
        max_iter=1000,
        tol=1e-6,
        verbose=True,
    )

    #Basic diagnostics
    reconstruction_error = (
        np.linalg.norm(M - L - S, ord="fro") / np.linalg.norm(M, ord="fro")
    )
    estimated_rank = np.linalg.matrix_rank(L)
    sparsity_fraction = float(np.mean(np.abs(S) > 1e-12))
    
    #Save figures
    save_heatmap(
        M,
        "Observed matrix M = L0 + S0",
        os.path.join(out_dir, "M_heatmap.png"),
    )
    save_heatmap(
        L,
        "Recovered low-rank matrix L",
        os.path.join(out_dir, "L_heatmap.png"),
    )
    save_heatmap(
        S,
        "Recovered sparse matrix S",
        os.path.join(out_dir, "S_heatmap.png"),
    )
    save_residual_curve(
        info.residuals,
        os.path.join(out_dir, "residual_curve.png"),
    )

    #Save log
    log_path = os.path.join(out_dir, "log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("--- Synthetic RPCA experiment ---\n")
        f.write(f"shape: ({m}, {n})\n")
        f.write(f"true rank: {r_true}\n")
        f.write(f"true sparsity p: {p_true}\n\n")
        f.write(f"lambda: {info.lam}\n")
        f.write(f"rho: {info.rho}\n")
        f.write(f"converged: {info.converged}\n")
        f.write(f"iterations: {info.iters}\n")
        f.write(f"final residual: {info.residuals[-1]:.3e}\n")
        f.write(f"reconstruction error: {reconstruction_error:.3e}\n")
        f.write(f"estimated rank(L): {estimated_rank}\n")
        f.write(f"sparsity fraction(S): {sparsity_fraction:.3f}\n")

    #Console summary
    print("\nSynthetic RPCA experiment completed.")
    print("Results saved to:", out_dir)
    print("Final residual:", f"{info.residuals[-1]:.3e}")
    print("Reconstruction error:", f"{reconstruction_error:.3e}")
    print("Estimated rank(L):", estimated_rank)
    print("Sparsity fraction(S):", f"{sparsity_fraction:.3f}")


if __name__ == "__main__":
    main()









    