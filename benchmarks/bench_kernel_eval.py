from time import perf_counter

import numpy as np
from numpy.random import default_rng

import transitionmanifolds as tm

nx = 1000
d = 1000
sigma = (d / 2.0) ** 0.5
num_samples = 100
rng = default_rng(123)
x = np.random.random((nx, d))

print(tm.distance_matrix.mmd.gaussian_kernel_eval_diag(x, sigma))

start = perf_counter()
for _ in range(num_samples):
    tm.distance_matrix.mmd.gaussian_kernel_eval_diag(x, sigma)
end = perf_counter()
duration = (end - start) / num_samples
print(f"Took {duration:.4f} seconds")
