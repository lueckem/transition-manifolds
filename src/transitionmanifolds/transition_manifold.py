from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class TMResult:
    distance_matrix: np.ndarray
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    diffusion_coordinates: np.ndarray
    dimension_estimate: np.ndarray
    bandwidth_diffusion_maps: np.ndarray


def compute_transition_manifold(
    data: NDArray,
    num_coordinates: int,
) -> TMResult:
    """Compute the transition manifold for the given data.

    The data should contain the end states of `num_runs` burst simulations for each anchor point.

    Args:
        data: `shape = (num_anchors, num_runs, d)`.
        num_coordinates: Number of coordinates returned in the transition manifold.

    Returns:
        `TMResult`:
        ...
    """
    pass
