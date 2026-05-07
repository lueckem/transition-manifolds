# from typing import Literal

import numpy as np
# from numba import njit, prange
# from numpy.random import default_rng
from numpy.typing import NDArray
# from scipy.spatial.distance import pdist

import ot
from joblib import Parallel, delayed


class DistanceMatrixWasserstein:
    """Computes distance matrix via Wasserstein.

    The distance between transition densities is estimated via the Wasserstein distance.

    # TODO: Berechnungsprozess beschreiben


    After the computation, the following extra information is available as class attributes:
    # TODO: class attributes beschreiben

    Attributes:
        # TODO: Attributes beschreiben (attr: [Explanation])
    """

    def __init__(
        self,
        regularize : bool | None = None,
        reg_factor: float | None = None,
        reg_type: str | None = None,
        n_jobs: int | None = None
    ):
        self.regularize = regularize
        self.reg_factor = reg_factor
        self.reg_type = reg_type
        self.n_jobs = n_jobs

    def __call__(
            self, 
            data: NDArray
        ) -> NDArray:

        distance_matrix = _dist_matrix_Wasserstein(
            x_samples=data, 
            regularize=self.regularize, 
            reg_factor=self.reg_factor, 
            reg_type=self.reg_type, 
            n_jobs=self.n_jobs
            )
        
        return distance_matrix
    

### JBL: Add _dist_matrix_Wasserstein (numba?)
def _dist_matrix_Wasserstein(
    x_samples: np.ndarray,
    regularize : bool | None = True,
    reg_factor: float | None = 0.1,
    reg_type: str | None = 'KL',
    n_jobs: int | None = -1
) -> np.ndarray:
    """
    Parameters
    ----------
    x_samples : np.ndarray
        Shape = (num_anchor_points, num_samples, dimension).
    reg_factor : float = 0.1,
    reg_type : str = 'KL',
    n_jobs: int = -1

    Returns
    -------
    np.ndarray
        1) Distance matrix, shape = (num_anchor_points, num_anchor_points).
    """
    num_anchor, num_samples, dimension = x_samples.shape

    if not regularize:
        reg_factor = None

    # marginals
    a = np.ones(shape=(num_samples,)) / num_samples
    b = a

    metric = "sqeuclidean"

    # Build all (i, j) pairs
    pairs = [(i, j) for i in range(num_anchor) for j in range(i)]

    # Parallel computation
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_wasserstein_pair)(i, j, x_samples, a, b, reg_factor, reg_type, metric)
        for (i, j) in pairs
    )

    distance_matrix = np.zeros(shape=(num_anchor,num_anchor))

    for i, j, val in results:
        distance_matrix[i, j] = val

    distance_matrix = distance_matrix + np.transpose(distance_matrix)

    return distance_matrix

    ###################################################################################
    ### Alter Berechnungsblock: Jetzt in _wasserstein_pair() 
    # for i in range(num_anchor):     # TODO: Hier mit prange arbeiten falls numba verwendet werden soll
    #     for j in range(i):
    #         # Calculate "Sqeuclidean" distance between sampling points
    #         M = ot.dist(x1=x_samples[i], x2=x_samples[j], metric=metric)

    #         # Calculate Wasserstein distance between x_samples[i] and x_samples[j]
    #         res = ot.solve(M=M, a=a, b=b, reg=reg_factor, reg_type=reg_type)
    #         distance_matrix[i,j] = res.value
    ###################################################################################


def _wasserstein_pair(i, j, x_samples, a, b, reg_factor, reg_type, metric):
    M = ot.dist(x1=x_samples[i], x2=x_samples[j], metric=metric)
    res = ot.solve(M=M, a=a, b=b, reg=reg_factor, reg_type=reg_type)
    return i, j, res.value
