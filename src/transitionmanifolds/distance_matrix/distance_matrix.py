from typing import Protocol

from numpy.typing import NDArray


class DistanceMatrixAlgorithm(Protocol):
    """A `DistanceMatrixAlgorithm` computes the distance matrix from data.

    Given burst simulation data of the shape `(num_anchors, num_runs, d)`,
    the distance matrix of shape `(num_anchors, num_anchors)` should contain the pairwise distances between the transition densities.
    The distances between transition densities `p_x` and `p_y` for two anchor points `x` and `y` should be estimated from the data.
    Hence, the distance matrix should always be symmetric with zeros on the diagonal.
    """

    def __call__(self, data: NDArray) -> NDArray: ...
