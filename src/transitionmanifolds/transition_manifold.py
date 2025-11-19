from numpy.typing import NDArray

from transitionmanifolds.distance_matrix.distance_matrix import DistanceMatrixAlgorithm
from transitionmanifolds.embedding.embedding import EmbeddingAlgorithm


def compute_transition_manifold(
    data: NDArray,
    num_coordinates: int,
    distance_matrix_algorithm: DistanceMatrixAlgorithm,
    embedding_algorithm: EmbeddingAlgorithm,
) -> NDArray:
    """Compute the transition manifold for the given data.

    The data should contain the end states of `num_runs` burst simulations for each anchor point.

    Args:
        data: `shape = (num_anchors, num_runs, d)`.
        num_coordinates: Number of coordinates returned in the transition manifold.
        distance_matrix_algorithm: Algorithm for computing the distances between transition densities.
        embedding_algorithm: Algorithm for computing the embedding from the distance matrix.

    Returns:
        embedding coordinates, `shape = (num_anchors, num_coordinates)`
    """
    distance_mat = distance_matrix_algorithm(data)
    return embedding_algorithm(distance_mat, num_coordinates)
