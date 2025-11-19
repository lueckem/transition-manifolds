from typing import Literal

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


class DistanceMatrixGaussianMMD:
    """Computes distance matrix via MMD.

    The distance between transition densities is estimated via maximum mean discrepancy (MMD)
    using a Gaussian kernel.

    More precisely, the distance matrix is given by
    ``D_ij = K_ii + K_jj - 2 K_ij``
    where the kernel matrix is defined via
    ``K_ij = E[k(x[i], x[j])]``
    and `k` is the Gaussian kernel
    ``k(x_i, y_j) = exp(-||x_i - y_j||^2 / sigma^2)``.

    Attributes:
        bandwidth: Bandwidth of the Gaussian kernel.
        mode: Either "u-statistic" for quadratic complexity but more precise estimation or "standard" for standard sample mean with linear complexity but less accuracy.
    """

    def __init__(
        self, bandwidth: float, mode: Literal["standard", "u-stat"] = "standard"
    ):
        self.bandwidth = bandwidth
        self.mode = mode

    def __call__(self, data: NDArray) -> NDArray:
        d = (
            compute_kernel_matrix_standard(data, self.bandwidth)
            if self.mode == "standard"
            else compute_kernel_matrix_u(data, self.bandwidth)
        )
        convert_kernel_to_distance(d)
        return d


@njit(cache=True)
def convert_kernel_to_distance(kernel_matrix: NDArray) -> None:
    """Convert kernel matrix K to distance matrix D in place.

    ``D_ij = K_ii + K_jj - 2 K_ij``

    The kernel matrix may be 0 above the diagonal due to symmetry.
    The distance matrix will have all elements filled in
    (above and below diagonal) and it will be 0 on the diagonal.

    Args:
        kernel_matrix: `shape = (num_anchors, num_anchors)`
    """
    num_anchors = kernel_matrix.shape[0]

    # convert
    for i in range(num_anchors):
        for j in range(i):
            value = kernel_matrix[i, i] + kernel_matrix[j, j] - 2 * kernel_matrix[i, j]
            kernel_matrix[i, j] = value
            kernel_matrix[j, i] = value

    # set diagonal to 0
    for i in range(num_anchors):
        kernel_matrix[i, i] = 0


@njit(cache=True, parallel=True)
def compute_kernel_matrix_standard(x_samples: NDArray, sigma: float) -> NDArray:
    """Compute matrix K with `K_ij = E[k(x[i], x[j])]`.

    K_ij is estimated via standard sample mean.
    Since K is symmetric, the entries above the diagonal are not filled in and left to be 0.

    Args:
        x_samples: `shape = (num_anchors, num_samples, d)`
        sigma: bandwidth

    Returns: kernel matrix with `shape = (num_anchors, num_anchors)`.
    """
    num_anchors = x_samples.shape[0]

    kernel_matrix = np.zeros((num_anchors, num_anchors))
    for i in prange(num_anchors):
        for j in range(i + 1):
            kernel_matrix[i, j] = gaussian_kernel_eval_standard(
                x_samples[i], x_samples[j], sigma
            )

    return kernel_matrix


@njit(cache=True, parallel=True)
def compute_kernel_matrix_u(x_samples: NDArray, sigma: float) -> NDArray:
    """Compute matrix K with `K_ij = E[k(x[i], x[j])]`.

    `K_ij` is estimated via u-statistic.
    Since K is symmetric, the entries above the diagonal are not filled in and left to be 0.

    Args:
        x_samples: `shape = (num_anchors, num_samples, d)`
        sigma: bandwidth

    Returns: kernel matrix with `shape = (num_anchors, num_anchors)`.
    """
    num_anchors = x_samples.shape[0]

    kernel_matrix = np.zeros((num_anchors, num_anchors))
    for i in prange(num_anchors):
        for j in range(i + 1):
            kernel_matrix[i, j] = gaussian_kernel_eval_u(
                x_samples[i], x_samples[j], sigma
            )

    return kernel_matrix


@njit(cache=True)
def gaussian_kernel_eval_standard(x: NDArray, y: NDArray, sigma: float) -> float:
    """Estimate ``E[k(X,Y)]`` from samples x and y using the standard sample mean.

    Calculates `1/n Sum_i k(x_i, y_i)`,
    where `k` is the gaussian kernel with bandwidth `sigma`, i.e.,
    ``k(x_i, y_j) = exp(-||x_i - y_j||^2 / sigma^2)``.

    Args:
        x: `shape = (n, d)`
        y: `shape = (n, d)`
        sigma: bandwidth
    """
    out = np.sum((x - y) ** 2, axis=1)
    out /= -(sigma**2)
    np.exp(out, out)
    return np.mean(out)


@njit(cache=True)
def gaussian_kernel_eval_u(x: NDArray, y: NDArray, sigma: float) -> float:
    """Estimate `E[k(X,Y)]` from samples x and y using the u-statistic.

    Calculates ``1/(mn) Sum_{i,j} k(x_i, y_j)``,
    where `k` is the gaussian kernel with bandwidth `sigma`, i.e.,
    ``k(x_i, y_j) = exp(-||x_i - y_j||^2 / sigma^2)``.

    Args:
        x: `shape = (m, d)`
        y: `shape = (n, d)`
        sigma: bandwidth
    """
    nx = x.shape[0]
    ny = y.shape[0]

    X = np.sum(x * x, axis=1).reshape((nx, 1)) * np.ones((1, ny))
    Y = np.sum(y * y, axis=1) * np.ones((nx, 1))
    out = X + Y - 2 * np.dot(x, y.T)
    out /= -(sigma**2)
    np.exp(out, out)
    return np.mean(out)
