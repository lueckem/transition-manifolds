from typing import Protocol

from numpy.typing import NDArray


class EmbeddingAlgorithm(Protocol):
    """An `EmbeddingAlgorithm` computes an embedding from a distance matrix.

    Given a distance matrix of shape `(num_anchors, num_anchors)`,
    which contains the pairwise distances between anchor points,
    a `num_coordinates`-dimensional embedding should be computed.
    Hence, the output should have the shape `(num_anchors, num_coordinates)`.
    """

    def __call__(self, distance_matrix: NDArray, num_coordinates: int) -> NDArray: ...
