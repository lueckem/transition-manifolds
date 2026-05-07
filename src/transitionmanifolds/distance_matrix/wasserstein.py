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
        mode: Either "v-stat" for quadratic complexity V-statistic with more precise estimation, or "d-stat" for linear complexity sample mean (D-statistic) with less precise estimation.
    """

    def __init__(
        self,
        bandwidth: float | None = None,
        mode: Literal["v-stat", "d-stat"] = "v-stat",
    ):
        self.bandwidth = bandwidth
        self.mode = mode

    def __call__(self, data: NDArray) -> NDArray:
        self.bandwidth_ = (
            self.bandwidth
            if self.bandwidth is not None
            else subsample_and_tune_bandwidth(data)
        )

        d = (
            compute_kernel_matrix_d(data, self.bandwidth_)
            if self.mode == "d-stat"
            else compute_kernel_matrix_v(data, self.bandwidth_)
        )
        convert_kernel_to_distance(d)
        return d
