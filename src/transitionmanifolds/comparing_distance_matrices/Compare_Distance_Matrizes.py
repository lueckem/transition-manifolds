import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial import procrustes
from sklearn.manifold import MDS
from pathlib import Path
from numpy.random import default_rng


import time

from transitionmanifolds import DistanceMatrixGaussianMMD, DistanceMatrixWasserstein


import warnings


def main(Wasserstein_exakt, reduce_sample_size_bool, num_x_ancherpoints, samples_per_anchor_reduction, skip_plotting):
    # Hyperparameters
    path_samples = "data/comparing_distance_matrices/x_data_Check.npz"
    path_distance_matrix = f'data/comparing_distance_matrices/distance_matrices_sample_numAnchorPoints_{num_x_ancherpoints}.npz'

    # Reduction parameters for amount of anchor points and reduction
    num_anchor_points_reduction = num_x_ancherpoints
    samples_per_anchor_reduction = samples_per_anchor_reduction # TODO: Überflüssig

    k = 50   # For kkn_overlap

    # Generate distance matrices, if not already done
    generate_distance_matrices(path_distance_matrix, path_samples, num_anchor_points_reduction, samples_per_anchor_reduction, Wasserstein_exakt, reduce_sample_size_bool)

    # Load distance matrices
    MMD_matrix, W_matrix = load_matrices(path_distance_matrix)


    # Compare distance matrices
    results = compare_distance_matrices(MMD_matrix, W_matrix, k)

    # Print infos of comparison values
    print_results(results)

    # Plot comparison of distance matrices
    if not skip_plotting:
        plot_results(MMD_matrix, W_matrix, results)




def flatten_upper_triangle(D: np.ndarray):
    """Läuft Zeilenweise über die strikt obere Dreiecksmatrix und gibt content in flattend array zurück

    Args:
        D (np.ndarray): Distance matrix

    Returns:
        np.ndarray: ndim = 1
    """
    n = D.shape[0]
    return D[np.triu_indices(n, k=1)]

def spearman_distance_correlation(D1: np.ndarray, D2: np.ndarray):
    """Calculate spearman correlation coefficient of two distance matrizes. Further information: "https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient"

    Args:
        D1 (np.ndarray): distance matrix
        D2 (np.ndarray): distance matrix

    Returns:
        float: Spearman coefficient
    """
    d1 = flatten_upper_triangle(D1)
    d2 = flatten_upper_triangle(D2)
    corr, _ = spearmanr(d1, d2)
    return corr

def knn_overlap(D1: np.ndarray, D2: np.ndarray, k: int = 5, knn_adaptive=True):
    """Consider for each point the relative overlap of the k nearest neighbours considering the two different distance matrizes. Returns the mean value over all relative overlaps.

    Args:
        D1 (np.ndarray): distance matrix
        D2 (np.ndarray): distance matrix
        k (int, optional): method considers the k nearest neighbours. Defaults to 5.

    Returns:
        float: The mean value over all relative overlaps. 
    """

    n = D1.shape[0]
    k = min(n-1,k)
    overlaps = []

    if knn_adaptive:
        k = (int)(0.2*n)//1 

    assert 1 <= k < n

    for i in range(n):
        nn1 = np.argsort(D1[i])[1:k+1]  # skip self (index 0)
        nn2 = np.argsort(D2[i])[1:k+1]

        overlap = len(set(nn1) & set(nn2)) / k
        overlaps.append(overlap)

    return np.mean(overlaps)

def normalized_matrix_difference(D1, D2):
    return np.linalg.norm(D1 - D2) / np.linalg.norm(D1)

def embedding_procrustes_error(D1, D2, dim=2):
    mds = MDS(
        n_components=dim,
        metric=True,              # statt dissimilarity
        n_init=4,                 # Warning fix
        init="classical_mds",     # zukunftssicher
        random_state=0
    )


    X1 = mds.fit_transform(D1)
    X2 = mds.fit_transform(D2)

    _, _, disparity = procrustes(X1, X2)
    return disparity

def compare_distance_matrices(D1, D2, k=5):
    results = {}

    results["spearman_corr"] = spearman_distance_correlation(D1, D2)
    results["knn_overlap"] = knn_overlap(D1, D2, k=k, knn_adaptive=False)
    results["knn_overlap_adaptive"] = knn_overlap(D1, D2, k=k, knn_adaptive=True)
    results["normalized_diff"] = normalized_matrix_difference(D1, D2)
    results["embedding_disparity"] = embedding_procrustes_error(D1, D2)

    return results

def _random_upper_triangular(n: int):
    """Random positive values on upper triangular

    Args:
        n (int): dim

    Returns:
        np.ndarray: random matrix, shape = (n,n)
    """
    M = np.zeros(shape=(n, n))
    
    # indices of upper triangle (excluding diagonal)
    i, j = np.triu_indices(n, k=1)
    
    # fill with positive random numbers
    M[i, j] = np.random.rand(len(i))
    
    return M

def load_matrix_old(distance: str, dim: int):
    """Aktuell wird noch zum testen eine zufällige Distanzmatrix verwendet

    Args:
        distance (str): Welche Distanz verwendet werden soll
        dim (int): Dimension der Matrix (dim x dim)

    Returns:
        np.ndarray: loaded/generated distance matrix, shape = (dim, dim)
    """

    # Dummy data
    n = dim
    distance_matrix = np.zeros(shape=(n,n))

    # if distance == 'MMD':
    d_upper = _random_upper_triangular(n)

    if distance == 'Wasserstein':
        # Kleine Störung
        noise = 0.01 * _random_upper_triangular(d_upper.shape[0])  # kleine Störung
        d_upper = np.maximum(d_upper + noise, 0.0)

    distance_matrix = distance_matrix + d_upper + np.transpose(d_upper)

    return distance_matrix

def print_results(results: dict):
    descriptions = {
        "spearman_corr": {
            "range": "[-1, 1]",
            "info": "1=gleiche Rangordnung, 0=kein Zusammenhang"
        },
        "knn_overlap": {
            "range": "[0, 1]",
            "info": "1=gleiche Nachbarn"
        },
        "knn_overlap_adaptive": {
            "range": "[0, 1]",
            "info": "1=gleiche Nachbarn"
        },
        "normalized_diff": {
            "range": "[0, ∞)",
            "info": "0=identisch (skalenabhängig!)"
        },
        "embedding_disparity": {
            "range": "[0, ∞)",
            "info": "0=gleiche Geometrie"
        },
    }

    for key, value in results.items():
        desc = descriptions.get(key, {})
        rng = desc.get("range", "?")
        info = desc.get("info", "")
        print(f"{key:20s}: {value:10.4f}   range={rng:8s}   {info}")

def plot_results(MMD_matrix: np.ndarray, W_matrix: np.ndarray, results: list[float]):
    """Plot comparison between two distance matrices.

    Args:
        MMD_matrix (np.ndarray): distance matrix
        W_matrix (np.ndarray): distance matrix
        results (list[float]): list of summary statistics
    """

    # obere Dreieckseinträge (ohne Diagonale)
    i, j = np.triu_indices(MMD_matrix.shape[0], k=1)
    mmd_vals = MMD_matrix[i, j]
    w_vals = W_matrix[i, j]

    # Scatterplot
    plt.figure()
    plt.scatter(mmd_vals, w_vals, alpha=0.6)

    plt.xlabel("MMD distances")
    plt.ylabel("Wasserstein distances")
    plt.title("MMD vs Wasserstein Distance Comparison")

    # # optionale Referenzlinie y = x
    # min_val = min(mmd_vals.min(), w_vals.min())
    # max_val = max(mmd_vals.max(), w_vals.max())
    # plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    # Skalen anpassen
    max_mmd = np.max(mmd_vals)
    max_w = np.max(w_vals)
    plt.xlim(0, max_mmd)
    plt.ylim(0, max_w)

    # Referenzlinie y = x im gleichen Bereich
    plt.plot([0, max_mmd], [0, max_w], linestyle="--")

    # Ergebnisse als Text in den Plot
    text_str = "\n".join([
        f"metric_{i}: {val}"
        for i, val in enumerate(results)
    ])
    plt.text(
        0.05, 0.95,
        text_str,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.1)
    )

    plt.tight_layout()
    plt.show()

def generate_distance_matrices(path_distance_matrix, sample_path, num_anchor_points= 1, samples_per_anchor= 1, Wasserstein_exakt= True, reduce_sample_size_bool= False):
    # Path with maybe already calculated test distance matrices
    path_distance_matrix_Path = Path(path_distance_matrix)

    # Calc if it doesn't exist
    if not path_distance_matrix_Path.exists():
        # Load samples to calculate the Wasserstein and MMD Distances
        x_samples = np.load(sample_path)["x_samples"]
        x_samples.shape

        # Reduce the sample size
        if reduce_sample_size_bool:
            x_samples = reduce_sample_size(x_samples, num_anchor_points, samples_per_anchor)


        # Calculate distance matrices with time measurement
        dim = x_samples.shape[2]
        reg_factor = 0.1 * dim
        sigma = np.sqrt(dim/2)

        print('Starte MMD Computation:')
        time_start = time.perf_counter()
        distance_matrix_MMD = calculate_distance_MMD(x_samples, sigma=sigma)
        print(f'Zeit: {time.perf_counter()-time_start}')

        print('Starte Wasserstein Computation:')
        time_start = time.perf_counter()
        distance_matrix_Wasserstein = calculate_distance_Wasserstein(x_samples, Wasserstein_exakt, reg_factor)
        print(f'Zeit: {time.perf_counter()-time_start}')

        # Save distance matrices
        np.savez(file=path_distance_matrix, MMD=distance_matrix_MMD, Wasserstein=distance_matrix_Wasserstein)

def calculate_distance_MMD(x_samples, sigma):
    algo = DistanceMatrixGaussianMMD(bandwidth=sigma)   # Future: apply different settings
    return algo(data=x_samples)

def calculate_distance_Wasserstein(x_samples, Wasserstein_exakt, reg_factor):
    algo = DistanceMatrixWasserstein(regularize=not Wasserstein_exakt, reg_factor=reg_factor)  # Future: apply different settings
    return algo(data=x_samples)

def load_matrices(path_distance_matrix):
    data = np.load(file=path_distance_matrix)
    return data['MMD'], data['Wasserstein']

def reduce_sample_size_deprecated(x_samples, samples_per_anchor=100):      # TODO: Code nochmal durchleuchten
    n_samples, n_features, n_dim = x_samples.shape
    k = samples_per_anchor  # Anzahl Features

    # Für jedes Sample eigene zufällige Indizes
    indices = np.array([
        np.random.choice(n_features, k, replace=False)
        for _ in range(n_samples)
    ])

    # Advanced Indexing
    x_samples_reduced = x_samples[np.arange(n_samples)[:, None], indices]
    return x_samples_reduced

def reduce_sample_size(x_samples, num_anchor_points, samples_per_anchor):
    per_slice = False   # TODO: True war eigentlich gedacht
    x_samples_reduced = random_subsample_along_axis(x=x_samples, k=samples_per_anchor, axis=1, per_slice=per_slice) # TODO: Noch prüfen
    x_samples_reduced = random_subsample_along_axis(x=x_samples_reduced, k=num_anchor_points, axis=0, per_slice=False)

    assert x_samples_reduced.shape == (num_anchor_points,samples_per_anchor, x_samples_reduced.shape[2])
    return x_samples_reduced

def random_subsample_along_axis(x, k, axis=1, per_slice=True, replace=False, rng=None):     # TODO: Code nochmal durchleuchten
    """
    Zufälliges Subsampling entlang einer gegebenen Achse.

    Parameters
    ----------
    x : np.ndarray
    k : int
        Anzahl der auszuwählenden Elemente entlang der Achse
    axis : int
        Achse, entlang der gesampelt wird
    per_slice : bool
        True  -> für jede "Zeile" eigene Zufallsindizes
        False -> gleiche Indizes für alle
    replace : bool
        Sampling mit Zurücklegen
    rng : np.random.Generator (optional)

    Returns
    -------
    np.ndarray
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x)
    n = x.shape[axis]

    if k > n and not replace:
        raise ValueError("k darf nicht größer als Achse sein ohne replace=True")

    # Achse nach vorne holen
    x_swapped = np.swapaxes(x, axis, 0)
    shape = x_swapped.shape  # (n, ...)

    if per_slice:
        # jede "Spalte" bekommt eigene Indizes
        rest = int(np.prod(shape[1:]))
        indices = np.array([
            rng.choice(n, k, replace=replace)
            for _ in range(rest)
        ]).reshape(*shape[1:], k)

        # Fancy indexing vorbereiten
        grid = np.indices(shape[1:])
        result = x_swapped[(indices, *grid)]

        # Achsen zurücktauschen
        result = np.swapaxes(result, 0, axis)

    else:
        # gleiche Indizes für alle
        indices = rng.choice(n, k, replace=replace)
        result = np.take(x, indices, axis=axis)

    return result

def samples(num_anchors, num_runs, d):
    rng = default_rng(123)
    # num_anchors = 6
    # num_runs = 20
    # d = 3
    samples = np.zeros((num_anchors, num_runs, d))
    for i in range(num_anchors):
        samples[i] = rng.normal(i, 1, size=(num_runs, d))
    return samples



if __name__ == "__main__":
    # Ignore warnings
    warnings.filterwarnings("ignore")

    # Hyperparameter
    Wasserstein_exakt = False
    reduce_sample_size_bool = True
    skip_plotting = True

    # Reduction parameter
    shape_of_x_data = (2000, 100, 900)      # TODO: Anpassen/Löschen
    num_anchorpoints_list = [10,50,100,300]
    samples_per_anchor_reduction = 100


    for num_x_ancherpoints in num_anchorpoints_list:
        print('#############################################')
        print(f'Run mit {num_x_ancherpoints} zufälligen anchorpoints')
        main(Wasserstein_exakt, reduce_sample_size_bool, num_x_ancherpoints, samples_per_anchor_reduction, skip_plotting)
