from typing import Literal

import numpy as np
from numba import njit, prange
from numpy.random import default_rng
from numpy.typing import NDArray
from scipy.spatial.distance import pdist


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

    After the computation, the following extra information is available as class attributes:
    - `bandwidth_`: Bandwidth that was used in the Gaussian kernel.

    Attributes:
        bandwidth: Bandwidth of the Gaussian kernel. If `None`, a reasonable bandwidth is estimated from the data.
        mode: Either "u-statistic" for quadratic complexity but more precise estimation or "standard" for standard sample mean with linear complexity but less accuracy.
    """

    def __init__(
        self,
        bandwidth: float | None = None,
        mode: Literal["standard", "u-stat"] = "u-stat",
    ):
        if mode == "standard":
            raise NotImplementedError("mode=standard is broken!")
        self.bandwidth = bandwidth
        self.mode = mode

    def __call__(self, data: NDArray) -> NDArray:
        self.bandwidth_ = (
            self.bandwidth
            if self.bandwidth is not None
            else subsample_and_tune_bandwidth(data)
        )

        d = (
            compute_kernel_matrix_standard(data.astype(np.float32), self.bandwidth_)
            if self.mode == "standard"
            else compute_kernel_matrix_u(data.astype(np.float32), self.bandwidth_)
        )
        convert_kernel_to_distance(d)
        return d


def subsample_and_tune_bandwidth(data: NDArray, num_points: int = 100) -> float:
    """Choose random points from data and estimate bandwidth for Gaussian kernel.

    Args:
        num_points: How many points to use for bandwidth estimation.
        data: `shape = (num_anchors, num_samples, d)`

    Returns:
        bandwidth
    """
    d = data.shape[2]
    rng = default_rng(123)
    points = rng.choice(np.reshape(data, (-1, d)), num_points, replace=False)
    return tune_bandwidth_to_data(points)


def tune_bandwidth_to_data(data: NDArray) -> float:
    """Estimate reasonable bandwidth of Gaussian kernel for data.

    The bandwidth is chosen such that the Gaussian kernel has the value 0.01
    at the 95% largest distance in the data.

    Args:
        data: `shape = (num_points, d)`

    Returns:
        bandwidth
    """
    quant = 0.95
    val_at_quant = 0.01

    pairwise_sqeucl = pdist(data, metric="sqeuclidean")
    q = np.quantile(pairwise_sqeucl, quant)
    bandwidth = np.sqrt(-q / np.log(val_at_quant))
    return bandwidth


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
            value = abs(value)
            kernel_matrix[i, j] = value
            kernel_matrix[j, i] = value

    # set diagonal to 0
    for i in range(num_anchors):
        kernel_matrix[i, i] = 0


@njit(fastmath=True)
def compute_kernel_matrix_standard(x_samples: NDArray, sigma: float) -> NDArray:
    """Compute matrix K with `K_ij = E[k(x[i], x[j])]`.

    `K_ij` is estimated via standard sample mean.
    Since K is symmetric, the entries above the diagonal are not filled in and left to be 0.

    Args:
        x_samples: `shape = (num_anchors, num_samples, d)`
        sigma: bandwidth

    Returns: kernel matrix with `shape = (num_anchors, num_anchors)`.
    """
    num_anchors = x_samples.shape[0]

    kernel_matrix = np.zeros((num_anchors, num_anchors))
    for i in range(num_anchors):
        kernel_matrix[i, i] = gaussian_kernel_eval_diag_standard(x_samples[i], sigma)

    for i in range(num_anchors):
        for j in range(i):
            kernel_matrix[i, j] = gaussian_kernel_eval_standard(
                x_samples[i], x_samples[j], sigma
            )

    return kernel_matrix


@njit(fastmath=True)
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

    for i in range(num_anchors):
        kernel_matrix[i, i] = gaussian_kernel_eval_diag_u(x_samples[i], sigma)

    for i in range(num_anchors):
        for j in range(i):
            kernel_matrix[i, j] = gaussian_kernel_eval_u(
                x_samples[i], x_samples[j], sigma
            )

    return kernel_matrix


@njit(fastmath=True, parallel=True)
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
    nx, dx = x.shape
    ny, dy = y.shape
    d = min(dx, dy)
    n = min(nx, ny)

    out = np.float32(0.0)
    inv_sigma_sq = np.float32(-1.0 / (sigma * sigma))

    for i in prange(n):
        dist_sq = np.float32(0.0)
        for k in range(d):
            diff = np.float32(x[i, k]) - np.float32(y[i, k])
            dist_sq += diff * diff
        out += np.exp(dist_sq * inv_sigma_sq)
    return out / n


@njit(fastmath=True, parallel=True)
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
    nx, dx = x.shape
    ny, dy = y.shape
    d = min(dx, dy)

    out = np.float32(0.0)
    inv_sigma_sq = np.float32(-1.0 / (sigma * sigma))

    for i in prange(nx):
        out_i = np.float32(0.0)
        for j in range(ny):
            dist_sq = np.float32(0.0)
            for k in range(d):
                diff = np.float32(x[i, k]) - np.float32(y[j, k])
                dist_sq += diff * diff
            out_i += np.exp(dist_sq * inv_sigma_sq)
        out += out_i

    return out / (nx * ny)


@njit(parallel=True, fastmath=True)
def gaussian_kernel_eval_diag_standard(x: NDArray, sigma: float) -> float:
    """Estimate `E[k(X,X)]` from samples x using standard sample mean.

    Calculates `1/(n-1) Sum_i k(x_i, x_{i+1})`,
    where `k` is the gaussian kernel with bandwidth `sigma`, i.e.,
    ``k(x_i, y_j) = exp(-||x_i - y_j||^2 / sigma^2)``.

    Args:
        x: `shape = (m, d)`
        sigma: bandwidth
    """
    nx, d = x.shape
    out = np.float32(0.0)
    inv_sigma_sq = np.float32(-1.0 / (sigma * sigma))

    for i in prange(nx - 1):
        dist_sq = np.float32(0.0)
        for k in range(d):
            diff = np.float32(x[i, k]) - np.float32(x[i + 1, k])
            dist_sq += diff * diff
        out += np.exp(dist_sq * inv_sigma_sq)

    return out / (nx - 1)


@njit(fastmath=True, parallel=True)
def gaussian_kernel_eval_diag_u(x: NDArray, sigma: float) -> float:
    """Estimate `E[k(X,X)]` from samples x using the u-statistic.

    Args:
        x: `shape = (m, d)`
        sigma: bandwidth
    """
    nx, d = x.shape

    out = np.float32(0.0)
    inv_sigma_sq = np.float32(-1.0 / (sigma * sigma))

    for i in prange(nx):
        out_i = np.float32(0.0)
        for j in range(i):  # skip the diagonal entries
            dist_sq = np.float32(0.0)
            for k in range(d):
                diff = np.float32(x[i, k]) - np.float32(x[j, k])
                dist_sq += diff * diff
            out_i += np.float32(2.0) * np.exp(dist_sq * inv_sigma_sq)
        out += out_i

    out += np.float32(nx)  # diagonal entries are all 1
    return out / (nx * nx)
