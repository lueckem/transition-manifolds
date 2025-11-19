import numpy as np
import pytest

from transitionmanifolds import DiffusionMaps
from transitionmanifolds.embedding.diffusion_maps import (
    apply_diffusion_normalization_to_similarity_matrix,
    compute_spectrum,
)


@pytest.fixture
def distance_matrix():
    return np.array(
        [
            [0, 0.7, 0.3],
            [0.7, 0, 0.3],
            [0.3, 0.3, 0],
        ]
    )


@pytest.mark.parametrize(
    "alg",
    [
        (DiffusionMaps()),
    ],
)
def test_embeddings(alg, distance_matrix):
    embed_coords = alg(distance_matrix, 2)
    assert embed_coords.shape == (3, 2)


######################### Diffusion maps #############################
@pytest.fixture
def similarity_matrix():
    return np.array(
        [
            [1, 0.7, 0.3],
            [0.7, 1, 0.3],
            [0.3, 0.3, 1],
        ]
    )


@pytest.fixture
def diffusion_matrix():
    # alpha = 0.5
    diffusion_matrix = np.array(
        [
            [0.5, 0.35, 0.167705],
            [0.35, 0.5, 0.167705],
            [0.167705, 0.167705, 0.625],
        ]
    )
    D_inv = np.diag([1.0 / 1.017705, 1.0 / 1.017705, 1.0 / 0.96041])
    diffusion_matrix = D_inv @ diffusion_matrix
    return diffusion_matrix


def test_diffusion_normalization(similarity_matrix, diffusion_matrix):
    alpha = 0.5
    apply_diffusion_normalization_to_similarity_matrix(similarity_matrix, alpha)
    assert np.allclose(diffusion_matrix, similarity_matrix)


def test_compute_spectrum(diffusion_matrix):
    eigvals = [1.0, 0.4859763, 0.14739045]
    eigvecs = [
        [-5.77350269e-01, -3.92489276e-01, -7.07106781e-01],
        [-5.77350269e-01, -3.92489276e-01, 7.07106781e-01],
        [-5.77350269e-01, 8.31807872e-01, 1.26497716e-16],
    ]
    eigenvalues, eigenvectors = compute_spectrum(diffusion_matrix, 3)
    assert np.allclose(eigenvalues, eigvals)
    assert np.allclose(eigenvectors, eigvecs)
