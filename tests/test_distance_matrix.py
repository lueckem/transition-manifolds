import numpy as np
import pytest
from numpy.random import default_rng
from scipy.spatial.distance import pdist

from transitionmanifolds import DistanceMatrixGaussianMMD
from transitionmanifolds.distance_matrix.mmd import (
    convert_kernel_to_distance,
    gaussian_kernel_eval_standard,
    gaussian_kernel_eval_u,
    tune_bandwidth_to_data,
)


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


@pytest.mark.parametrize(
    "alg",
    [
        (DistanceMatrixGaussianMMD(1, "standard")),
        (DistanceMatrixGaussianMMD(1, "u-stat")),
        (DistanceMatrixGaussianMMD()),
    ],
)
def test_distance_matrix_algorithms(alg, samples):
    distance_matrix = alg(samples)

    assert distance_matrix.shape == (6, 6)  # Correct shape
    assert np.all(np.diag(distance_matrix) == 0)  # Diagonal 0
    assert np.all(distance_matrix == distance_matrix.T)  # Symmetric


###################### DistanceMatrixGaussianMMD ##############################


def test_convert_kernel_to_distance():
    kernel_mat = np.array([[3, 0, 0], [2, 4, 0], [1, 2, 5]])
    distance_mat = np.array([[0, 3, 6], [3, 0, 5], [6, 5, 0]])
    convert_kernel_to_distance(kernel_mat)
    assert np.all(kernel_mat == distance_mat)


def test_gaussian_kernel_eval_u():
    x = np.array([[1, 1], [2, 3], [0, -1]], dtype=np.float64)
    y = np.array([[1, 1], [2, 0]], dtype=np.float64)
    sigma = 2.0**0.5

    # sqeucl = np.array([[0, 2], [5, 9], [5, 5]])
    kernel = np.array(
        [
            [1, np.exp(-1)],
            [np.exp(-2.5), np.exp(-4.5)],
            [np.exp(-2.5), np.exp(-2.5)],
        ]
    )
    assert np.allclose(np.mean(kernel), gaussian_kernel_eval_u(x, y, sigma))


def test_gaussian_kernel_eval_standard():
    x = np.array([[1, 1], [2, 3], [0, -1]], dtype=np.float64)
    y = np.array([[1, 1], [2, 0], [1, -1]], dtype=np.float64)
    sigma = 2.0**0.5

    # sqeucl = np.array([0, 9, 1])
    kernel = np.array(
        [
            1,
            np.exp(-4.5),
            np.exp(-0.5),
        ]
    )
    assert np.allclose(np.mean(kernel), gaussian_kernel_eval_standard(x, y, sigma))


def test_tune_bandwidth():
    points = np.random.default_rng(123).random((10, 3))
    bandwidth = tune_bandwidth_to_data(points)
    kernel_evals = np.exp(-pdist(points, metric="sqeuclidean") / bandwidth**2)
    assert np.isclose(np.quantile(kernel_evals, 0.05), 0.01, atol=0.001)


def test_convergence_to_0_standard():
    points = np.random.default_rng(123).random((2, 2000, 2))
    distance_mat = DistanceMatrixGaussianMMD(0.3, "standard")(points)
    assert distance_mat[1, 0] < 1e-2


def test_convergence_to_0_ustat():
    points = np.random.default_rng(123).random((2, 1000, 2))
    distance_mat = DistanceMatrixGaussianMMD(0.3, "u-stat")(points)
    assert distance_mat[1, 0] < 1e-3
