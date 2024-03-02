import numpy as np
from init_val_generator.clustering import k_means_plus_plus


def test_k_means_plus_plus():
    # Generate sample data with a peak at (3, 2)
    data_x = np.repeat(np.arange(8), 6)
    data_y = np.repeat(np.arange(6), 8)
    data = np.zeros((8 * 6,))
    peak_index = (3, 2)
    data[8 * peak_index[1] + peak_index[0]] = 1

    n_clusters = 1

    centroids_x, centroids_y = k_means_plus_plus(data, data_x, data_y, n_clusters)

    # Check if the centroids have correct shape
    assert centroids_x.shape == (n_clusters,)
    assert centroids_y.shape == (n_clusters,)

    # Check the calculated centroid against the expected value
    assert centroids_x[0] == peak_index[0]
    assert centroids_y[0] == peak_index[1]
