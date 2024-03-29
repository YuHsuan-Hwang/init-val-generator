import numpy as np
import numpy.typing as npt


def k_means_plus_plus(
    data: npt.NDArray[np.float64],
    data_x: npt.NDArray[np.float64],
    data_y: npt.NDArray[np.float64],
    n: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Perform K-means++ initialization to choose initial centroids for K-means clustering.

    Parameters
    ----------
    data
        The input data array.
    data_x
        X coordinates of data points.
    data_y
        Y coordinates of data points.
    n
        Number of centroids to initialize.

    Returns
    -------
    tuple
        X coordinates of the initialized centroids, Y coordinates of the initialized centroids.
    """

    init_centroid_x = np.empty(n)
    init_centroid_y = np.empty(n)

    initIndex = np.argmax(np.abs(data))  # find the max pixel instead of a random pixel
    init_centroid_x[0] = data_x[initIndex]
    init_centroid_y[0] = data_y[initIndex]

    def getWeightedDist(
        centroid_x: np.float64, centroid_y: np.float64
    ) -> npt.NDArray[np.float64]:
        # type hint undefined in np.sqrt
        dist: npt.NDArray[np.float64] = np.sqrt(
            np.square(data_x - centroid_x) + np.square(data_y - centroid_y)
        )
        return np.abs(data) * dist

    for i in range(1, n):
        dist = getWeightedDist(init_centroid_x[0], init_centroid_y[0])
        for j in range(1, i):
            newDist = getWeightedDist(init_centroid_x[j], init_centroid_y[j])
            dist = np.minimum(dist, newDist)

        newIndex = np.argmax(dist)
        init_centroid_x[i] = data_x[newIndex]
        init_centroid_y[i] = data_y[newIndex]

    return init_centroid_x, init_centroid_y


def k_means(
    data: npt.NDArray[np.float64],
    data_x: npt.NDArray[np.float64],
    data_y: npt.NDArray[np.float64],
    centroid_x: npt.NDArray[np.float64],
    centroid_y: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Perform K-means clustering on the input data.

    Parameters
    ----------
    data
        The input data array.
    data_x
        X coordinates of data points.
    data_y
        Y coordinates of data points.
    centroid_x
        X coordinates of initial centroids.
    centroid_y
        Y coordinates of initial centroids.

    tuple
        X coordinates of the initialized centroids, Y coordinates of the initialized centroids, cluster indices for each data point.
    """

    MAX_ITER = 10
    n = len(centroid_x)
    for iter in range(MAX_ITER):

        new_centroid_x = np.zeros(n)
        new_centroid_y = np.zeros(n)
        new_centroid_sum = np.zeros(n)
        data_cluster_index = np.full((data.shape), -1)

        for i in range(len(data)):
            dist = np.abs(data[i]) * np.sqrt(
                (np.square(data_x[i] - centroid_x) + np.square(data_y[i] - centroid_y))
            )
            cluster_index = np.argmin(dist)
            data_cluster_index[i] = cluster_index
            new_centroid_x[cluster_index] += np.abs(data[i]) * data_x[i]
            new_centroid_y[cluster_index] += np.abs(data[i]) * data_y[i]
            new_centroid_sum[cluster_index] += np.abs(data[i])

        new_centroid_x /= new_centroid_sum
        new_centroid_y /= new_centroid_sum

        isCoverged = True
        for i in range(n):
            if (
                new_centroid_x[i] - centroid_x[i] >= 1
                or new_centroid_y[i] - centroid_y[i] >= 1
            ):
                isCoverged = False
                break

        if isCoverged:
            print("k-means clustering converged: {} iterations".format(iter))
            break
        else:
            centroid_x = new_centroid_x
            centroid_y = new_centroid_y

    return data_cluster_index, centroid_x, centroid_y


def mean_distance(
    data_x: npt.NDArray[np.float64],
    data_y: npt.NDArray[np.float64],
    x: float,
    y: float,
) -> float:

    distances = np.sqrt((data_x - x) ** 2 + (data_y - y) ** 2)
    mean_dist: float = np.mean(distances)
    return mean_dist


def get_silhouette_score(
    data: npt.NDArray[np.float64],
    data_x: npt.NDArray[np.float64],
    data_y: npt.NDArray[np.float64],
    centroid_x: npt.NDArray[np.float64],
    centroid_y: npt.NDArray[np.float64],
    data_cluster_index: npt.NDArray[np.float64],
) -> float:

    score = 0.0
    num = len(data)

    for i in range(num):
        x = data_x[i]
        y = data_y[i]
        cluster_index = data_cluster_index[i]

        data_x_cluster = data_x[data_cluster_index == cluster_index]
        data_y_cluster = data_y[data_cluster_index == cluster_index]

        distances = [
            np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            for cx, cy in zip(centroid_x, centroid_y)
        ]
        sorted_indices = np.argsort(distances)
        second_nearest_cluster_index = sorted_indices[1]

        data_x_second_nearest_cluster = data_x[
            data_cluster_index == second_nearest_cluster_index
        ]
        data_y_second_nearest_cluster = data_y[
            data_cluster_index == second_nearest_cluster_index
        ]

        a = mean_distance(data_x_cluster, data_y_cluster, x, y)
        b = mean_distance(
            data_x_second_nearest_cluster, data_y_second_nearest_cluster, x, y
        )

        score += (b - a) / max(a, b)

    score /= num
    return score
