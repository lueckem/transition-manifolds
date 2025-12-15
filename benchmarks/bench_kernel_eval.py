from time import perf_counter

import numpy as np
from numpy.random import default_rng

import transitionmanifolds as tm


def print_size(num_anchors, num_samples, d, bytes_per):
    size_per_anchor = bytes_per * d * num_samples
    print(f"Total size: {size_per_anchor * num_anchors / 10**6} mb")
    print(f"Size per anchor: {size_per_anchor / 10**3} kb")
    print(f"Size per sample: {bytes_per * d / 10**3} kb")


def bench_u_f32():
    num_anchors = 200
    num_samples = 400
    d = 1000
    sigma = (d / 2.0) ** 0.5

    print_size(num_anchors, num_samples, d, 4)

    rng = default_rng(123)
    x = rng.random((num_anchors, num_samples, d), dtype=np.float32)

    # compilation
    tm.distance_matrix.mmd.compute_kernel_matrix_u(x[:2, :10, :10], sigma)

    start = perf_counter()
    k = tm.distance_matrix.mmd.compute_kernel_matrix_u(x, sigma)
    end = perf_counter()
    print(k[1, 0])
    duration = end - start
    print(f"Took {duration:.4f} seconds")
    # numpy: Took 57.4142 seconds
    # loop: Took 33.9371 seconds


def bench_u_i8():
    num_anchors = 200
    num_samples = 400
    d = 1000
    sigma = (d / 2.0) ** 0.5

    print_size(num_anchors, num_samples, d, 4)

    rng = default_rng(123)
    x = rng.integers(0, 2, (num_anchors, num_samples, d), dtype=np.uint8)

    # compilation
    tm.distance_matrix.mmd.compute_kernel_matrix_u(x[:2, :10, :10], sigma)

    start = perf_counter()
    k = tm.distance_matrix.mmd.compute_kernel_matrix_u(x, sigma)
    end = perf_counter()
    print(k[1, 0])
    duration = end - start
    print(f"Took {duration:.4f} seconds")


if __name__ == "__main__":
    # bench_u_f32()
    bench_u_i8()
