import numpy as np
import pytest
from numpy.random import default_rng
# from scipy.spatial.distance import pdist

# from transitionmanifolds import DistanceMatrixGaussianMMD
# from transitionmanifolds.distance_matrix.mmd import (
#     convert_kernel_to_distance,
#     gaussian_kernel_eval_d,
#     gaussian_kernel_eval_v,
#     tune_bandwidth_to_data,
# )

from transitionmanifolds import DistanceMatrixWasserstein
# from transitionmanifolds.distance_matrix.wasserstein import(
#     # TODO: Sonstige Funktionen die in Zukunft getestet werden sollen
# )


@pytest.fixture
def samples():
    rng = default_rng(123)
    num_anchors = 6
    num_runs = 20
    d = 3
    samples = np.zeros((num_anchors, num_runs, d))
    for i in range(num_anchors):
        samples[i] = rng.normal(i, 1, size=(num_runs, d))
    return samples

@pytest.fixture
def alg():
    return DistanceMatrixWasserstein()

def test_distance_matrix_algorithms(alg, samples):
    distance_matrix = alg(samples)

    assert distance_matrix.shape == (6, 6)  # Correct shape
    assert np.all(distance_matrix >= 0)  # Non-negative entries
    assert np.all(np.diag(distance_matrix) == 0)  # Diagonal 0
    assert np.all(distance_matrix == distance_matrix.T)  # Symmetric

# TODO: Weitere Tests einbauen