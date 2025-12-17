from time import perf_counter

import numpy as np
from numpy.random import default_rng

import transitionmanifolds as tm


def print_size(num_anchors, num_samples, d, bytes_per):
    size_per_anchor = bytes_per * d * num_samples
    print(f"Total size: {size_per_anchor * num_anchors / 10**6} mb")
    print(f"Size per anchor: {size_per_anchor / 10**3} kb")
    print(f"Size per sample: {bytes_per * d / 10**3} kb")


def bench_v_f32():
    num_anchors = 200
    num_samples = 400
    d = 1000
    sigma = (d / 2.0) ** 0.5

    print_size(num_anchors, num_samples, d, 4)

    rng = default_rng(123)
    x = rng.random((num_anchors, num_samples, d), dtype=np.float32)

    # compilation
    tm.distance_matrix.mmd.compute_kernel_matrix_v(x[:2, :10, :10], sigma)

    start = perf_counter()
    k = tm.distance_matrix.mmd.compute_kernel_matrix_v(x, sigma)
    end = perf_counter()
    print(k[1, 0])
    duration = end - start
    print(f"Took {duration:.4f} seconds")
    # numpy: Took 57.4142 seconds
    # loop: Took 33.9371 seconds


def bench_v_i8():
    num_anchors = 200
    num_samples = 400
    d = 1000
    sigma = (d / 2.0) ** 0.5

    print_size(num_anchors, num_samples, d, 1)

    rng = default_rng(123)
    x = rng.integers(0, 2, (num_anchors, num_samples, d), dtype=np.int8)

    # compilation
    tm.distance_matrix.mmd.compute_kernel_matrix_v(x[:2, :10, :10], sigma)

    start = perf_counter()
    k = tm.distance_matrix.mmd.compute_kernel_matrix_v(x, sigma)
    end = perf_counter()
    print(k[1, 0])
    duration = end - start
    print(f"Took {duration:.4f} seconds")


def bench_d_f32():
    num_anchors = 400
    num_samples = 400
    d = 2000
    sigma = (d / 2.0) ** 0.5

    print_size(num_anchors, num_samples, d, 4)

    rng = default_rng(123)
    x = rng.random((num_anchors, num_samples, d), dtype=np.float32)

    # compilation
    tm.distance_matrix.mmd.compute_kernel_matrix_d(x[:2, :10, :10], sigma)

    start = perf_counter()
    k = tm.distance_matrix.mmd.compute_kernel_matrix_d(x, sigma)
    end = perf_counter()
    print(k[1, 0])
    duration = end - start
    print(f"Took {duration:.4f} seconds")
    # numpy: Took 28.0687 seconds
    # loop: Took 8.8436 seconds


def bench_d_i8():
    num_anchors = 400
    num_samples = 400
    d = 2000
    sigma = (d / 2.0) ** 0.5

    print_size(num_anchors, num_samples, d, 1)

    rng = default_rng(123)
    x = rng.integers(0, 2, (num_anchors, num_samples, d), dtype=np.int8)

    # compilation
    tm.distance_matrix.mmd.compute_kernel_matrix_d(x[:2, :10, :10], sigma)

    start = perf_counter()
    k = tm.distance_matrix.mmd.compute_kernel_matrix_d(x, sigma)
    end = perf_counter()
    print(k[1, 0])
    duration = end - start
    print(f"Took {duration:.4f} seconds")


if __name__ == "__main__":
    print("--- V-Statistic f32 ---")
    bench_v_f32()
    print("")
    print("--- V-Statistic i8 ---")
    bench_v_i8()
    print("")
    print("--- D-Statistic f32 ---")
    bench_d_f32()
    print("")
    print("--- D-Statistic i8 ---")
    bench_d_i8()
