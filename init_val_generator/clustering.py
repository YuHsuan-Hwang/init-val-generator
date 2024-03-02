import numpy as np
import numpy.typing as npt


def k_means_plus_plus(
    data: npt.NDArray[np.float64],
    data_x: npt.NDArray[np.float64],
    data_y: npt.NDArray[np.float64],
    n: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

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
