import pytest
import numpy as np
from init_val_generator.clustering import k_means, k_means_plus_plus


@pytest.mark.parametrize(
    "peak_pos, peak_val", [([(3, 2)], [1]), ([(3, 2), (6, 5), (1, 4)], [1, 0.8, 0.5])]
)
def test_k_means_plus_plus(peak_pos, peak_val):
    data_x = np.tile(np.arange(8), 6)
    data_y = np.repeat(np.arange(6), 8)
    data = np.zeros((8 * 6,))
    for i, peak_index in enumerate(peak_pos):
        data[8 * peak_index[1] + peak_index[0]] = peak_val[i]
    n_clusters = len(peak_pos)

    centroids_x, centroids_y = k_means_plus_plus(data, data_x, data_y, n_clusters)

    # Check if the centroids have correct shape
    assert centroids_x.shape == (n_clusters,)
    assert centroids_y.shape == (n_clusters,)

    # Check the calculated centroid against the expected value
    for i, peak_index in enumerate(peak_pos):
        assert centroids_x[i] == peak_index[0]
        assert centroids_y[i] == peak_index[1]


def test_k_means():
    data_x = np.array([0, 1, 2, 3])
    data_y = np.array([0, 1, 2, 3])
    data = np.array([1, 1, 2, 2])
    centroid_x = np.array([0, 1])
    centroid_y = np.array([0, 1])

    data_cluster_index = k_means(data, data_x, data_y, centroid_x, centroid_y)

    # Check if the second centroid is moved
    np.testing.assert_array_equal(data_cluster_index, np.array([0, 0, 1, 1]))
