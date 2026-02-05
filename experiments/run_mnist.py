from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from robust_pca_admm.admm_rpca import rpca_admm


def load_mnist(digit: int = 3, n_samples: int = 80, root: str = "data/mnist") -> np.ndarray:
    """
    Returns images as numpy array shape (N, 28, 28) in [0,1].
    Uses torchvision and will auto-download if not present.
    """
    from torchvision import datasets, transforms

    ds = datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())

    imgs = []
    for x, y in ds:
        if int(y) == digit:
            imgs.append(x.squeeze(0).numpy())  # (28,28)
            if len(imgs) >= n_samples:
                break

    if len(imgs) < n_samples:
        raise RuntimeError(f"Could not collect {n_samples} samples for digit={digit}. Got {len(imgs)}.")

    return np.stack(imgs, axis=0)  # (N,28,28)


def corrupt_salt_pepper(images: np.ndarray, p: float = 0.10, seed: int = 0) -> np.ndarray:
    """
    Salt & pepper corruption on images in [0,1].
    """
    rng = np.random.default_rng(seed)
    corrupted = images.copy()
    mask = rng.random(images.shape) < p
    salt = rng.random(images.shape) < 0.5
    corrupted[mask & salt] = 1.0
    corrupted[mask & ~salt] = 0.0
    return corrupted


def stack_as_matrix(images: np.ndarray) -> np.ndarray:
    """
    images: (N,28,28) -> M: (784, N)
    """
    N = images.shape[0]
    return images.reshape(N, -1).T


def save_triplet_grid(corrupted: np.ndarray, Limgs: np.ndarray, Simgs: np.ndarray, out_path: Path, k: int = 10):
    """
    Save grid: rows = corrupted / L / S, cols = k samples
    """
    k = min(k, corrupted.shape[0])
    plt.figure(figsize=(1.6 * k, 5))

    for i in range(k):
        for row, arr, title in [
            (0, corrupted, "Corrupted"),
            (1, Limgs, "Low-rank L"),
            (2, Simgs, "Sparse S"),
        ]:
            ax = plt.subplot(3, k, row * k + i + 1)
            ax.imshow(arr[i], cmap="gray")
            ax.axis("off")
            if i == 0:
                ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    out_dir = Path("results/mnist")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Settings we can modify ----
    digit = 3
    N = 80
    p_corrupt = 0.10    
    rho = 1.0
    tol = 1e-6
    max_iter = 3000
    # --------------------------------

    images = load_mnist(digit=digit, n_samples=N, root="data/mnist")  # (N,28,28) in [0,1]
    corrupted = corrupt_salt_pepper(images, p=p_corrupt, seed=0)

    M = stack_as_matrix(corrupted)  # (784, N)

    L, S, report = rpca_admm(M, rho=rho, tol=tol, max_iter=max_iter, verbose=True)

    # back to images
    Limgs = L.T.reshape(N, 28, 28)
    Simgs = S.T.reshape(N, 28, 28)

    # visuals
    save_triplet_grid(corrupted, Limgs, Simgs, out_dir / "mnist_triplets.png", k=10)

    # convergence plot
    if report.residuals:
        plt.figure()
        plt.plot(report.residuals)
        plt.yscale("log")
        plt.title("Convergence (relative primal residual)")
        plt.xlabel("iteration")
        plt.ylabel("residual")
        plt.tight_layout()
        plt.savefig(out_dir / "convergence.png", dpi=200)
        plt.close()

    # log
    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"digit: {digit}\n")
        f.write(f"N: {N}\n")
        f.write(f"p_corrupt: {p_corrupt}\n")
        f.write(f"lam(auto): {report.lam}\n")
        f.write(f"rho: {report.rho}\n")
        f.write(f"iters: {report.iters}\n")
        f.write(f"converged: {report.converged}\n")
        f.write(f"final_residual: {report.residuals[-1] if report.residuals else None}\n")

    print(f"Saved MNIST results to: {out_dir}")


if __name__ == "__main__":
    main()
