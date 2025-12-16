# Transition manifolds

This package provides methods related to the transition manifold approach for finding collective variables for dynamical systems.
For further information about transition manifolds see for instance:

[1] A. Bittracher, S. Klus, B. Hamzi, P. Koltai, and C. Schütte. Dimensionality reduction of complex
metastable systems via kernel embeddings of transition manifolds. Journal of Nonlinear Science,
31(1), 2020.

[2] M. Lücke, S. Winkelmann, J. Heitzig, N. Molkenthin, and P. Koltai. Learning interpretable collective
variables for spreading processes on networks. Physical Review E, 109(2):l022301, 2024.


## Basic Usage

The transition manifold approach requires short burst simulations for a set of initial system states, which are called *anchor states* here.
The data array provided to the method should contain the final states of these simulations and have the shape ```(num_anchors, num_runs, d)```, where

- ```num_anchors``` is the number of anchor states.
- ```num_runs``` is the number of simulations executed per anchor state.
- ```d``` is the dimension of the system states.

Computing the transition manifold is done in two steps.

In the first step, the pairwise distances between the transition densities $p_x^\tau$ for all anchor points $x$ have to be estimated.
The transition density $p_x^\tau$ describes the probability distribution of the system at time $\tau$ when starting in state $x$ at time $0$.
The matrix of these pairwise distances is called *distance matrix* here.

The second step utilizes the computed distance matrix to compute a low-dimensional embedding of the anchor points.

### Step 1: Distance matrix
Given the data array ```samples```, the distance matrix can be computed using a `DistanceMatrixAlgorithm`, for example `DistanceMatrixGaussianMMD`, which estimates the distances via _maximum mean discrepancy_ (MMD):

```python
import transitionmanifolds as tm

mmd = tm.DistanceMatrixGaussianMMD()
distance_matrix = mmd(samples)
```

### Step 2: Embedding
An `EmbeddingAlgorithm` computes the low-dimensional coordinates of the embedding, for example `DiffusionMaps`:
```python
diffusion_maps = tm.DiffusionMaps()
num_coords = 3  # 3-dimensional embedding
coords = diffusion_maps(distance_matrix, num_coords)  # shape = (num_anchors, num_coords)
```

Note that additional information about the computations is stored as class fields, see the documentation for each algorithm.
For example, the bandwidth used in the diffusion maps computation is accessible via `diffusion_maps.bandwidth_`.

Both steps can be executed at once using the convenience function `compute_transition_manifold`:
```python
mmd = tm.DistanceMatrixGaussianMMD()
diffusion_maps = tm.DiffusionMaps()
num_coords = 3  # 3-dimensional embedding
coords = tm.compute_transition_manifold(
    samples,
    num_coords,
    distance_matrix_algorithm=mmd,
    embedding_algorithm=diffusion_maps,
)
```
