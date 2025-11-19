import numpy as np
import scipy.sparse.linalg as sla
from numba import njit
from numpy.typing import NDArray


class DiffusionMaps:
    """Computes an embedding via diffusion maps.

    After the computation, the following extra information is available as class attributes:
    - `bandwidth_`: Bandwidth that was used in the Gaussian kernel.
    - `eigenvalues_`: Computed eigenvalues of the diffusion matrix.
    - `eigenvectors_`: Computed eigenvectors of the diffusion matrix.
    - `dimension_estimate_`: Estimation of the dimension of the embedding.

    Attributes:
        alpha: Diffusion maps parameter for controlling the influence of data point density.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = 0.5

    def __call__(self, distance_matrix: NDArray, num_coordinates: int) -> NDArray:
        opt_bandwidth, dimension_estimate = compute_optimal_bandwidth(distance_matrix)
        similarity_matrix = np.exp(-(distance_matrix**2) / opt_bandwidth)
        apply_diffusion_normalization_to_similarity_matrix(
            similarity_matrix, self.alpha
        )
        eigenvalues, eigenvectors = compute_spectrum(
            similarity_matrix, num_coordinates + 1
        )
        diffusion_coordinates = (
            eigenvectors[:, 1:] * eigenvalues[np.newaxis, 1:]  # type: ignore
        )

        # store extra information
        self.bandwidth_ = opt_bandwidth**0.5
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors
        self.dimension_estimate_ = dimension_estimate

        return diffusion_coordinates


def apply_diffusion_normalization_to_similarity_matrix(
    mat: NDArray, alpha: float
) -> None:
    """Apply the diffusion normalization in place.

    The similarity matrix should be symmetric with ones on the diagonal.
    `alpha` is a parameter related to the influence of the data point density
    and is typically chosen between 0 and 1.
    """
    # D⁻ᵅ L D⁻ᵅ
    row_sum = np.sum(mat, axis=0)
    mat /= np.outer(row_sum**alpha, row_sum**alpha)

    # (Dᵅ)⁻¹ L
    mat /= np.sum(mat, axis=1, keepdims=True)


def compute_spectrum(
    diffusion_matrix: NDArray, num_eigenvalues: int
) -> tuple[NDArray, NDArray]:
    """Computes the dominant spectrum of matrix.

    The eigenvalues and eigenvectors are converted to real numbers
    and sorted in descending order.
    """
    eigenvalues, eigenvectors = sla.eigs(diffusion_matrix, num_eigenvalues)  # type: ignore
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    idxs = np.argsort(eigenvalues)
    idxs = np.flip(idxs)
    eigenvalues = eigenvalues[idxs]
    eigenvectors = eigenvectors[:, idxs]
    return eigenvalues, eigenvectors


def compute_optimal_bandwidth(
    distance_matrix: NDArray,
) -> tuple[float, float]:
    """Optimizes the diffusion bandwidth and computes an estimate for the intrinsic dimension.

    Returns:
        (optimal diffusion bandwidth, dimension estimate)
    """
    d_squared = distance_matrix**2
    epsilons = np.logspace(-6, 2, 101)
    elasts = [_elasticity(ep, d_squared) for ep in epsilons]
    optim_idx = np.argmax(elasts)
    optim_epsilon = epsilons[optim_idx]
    dimension_estimate = 2 * _elasticity(optim_epsilon, d_squared)
    return optim_epsilon, dimension_estimate


# helper function that computes the two sums in a single pass over d_squared:
# `sum_exp = mean(exp(-D_squared / ϵ)) = S(ϵ)`
# `sum_d2_exp = mean(exp(-D_squared / ϵ) * D_squared) = dS(ϵ) * ϵ^2`
# The elasticity is then `ϵ * dS(ϵ) / S(ϵ) = sum_d2_exp / sum_exp / ϵ`
@njit(cache=True)
def _compute_means(epsilon: float, d_squared: NDArray) -> tuple[float, float]:
    n = d_squared.shape[0]
    sum_exp = n  # contribution from the n diagonal elements
    sum_d2_exp = 0

    for i in range(n):
        for j in range(i):
            d = d_squared[i, j]
            exp_val = np.exp(-d / epsilon)
            sum_exp += 2 * exp_val
            sum_d2_exp += 2 * d * exp_val
    return sum_exp / n**2, sum_d2_exp / n**2


@njit(cache=True)
def _elasticity(epsilon: float, d_squared: NDArray) -> float:
    sum_exp, sum_d2_exp = _compute_means(epsilon, d_squared)
    return sum_d2_exp / sum_exp / epsilon
