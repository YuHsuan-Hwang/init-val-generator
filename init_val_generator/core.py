import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .data_selection import SelectionMethod, filter_data
from .clustering import get_silhouette_score, k_means, k_means_plus_plus


def guess(
    data: npt.NDArray[np.float64],
    width: int,
    height: int,
    n: int | None = 1,
    data_selection: SelectionMethod | None = None,
    plot_mode: str = "none",
) -> list[list[float]]:

    x = np.arange(width)
    y = np.arange(height)
    data_x = np.tile(x, height)
    data_y = np.repeat(y, width)

    if data_selection is not None:
        data, data_x, data_y = filter_data(
            data_selection, data, width, height, data_x, data_y, plot_mode
        )

    if n is None:
        MAX_COMPONENT_NUM = 10
        scores = []
        for i in range(MAX_COMPONENT_NUM):
            input_num = i + 1
            print("clustering with component number {}".format(str(input_num)))

            init_centroid_x, init_centroid_y = k_means_plus_plus(
                data, data_x, data_y, input_num
            )
            data_cluster_index, centroid_x, centroid_y = k_means(
                data, data_x, data_y, init_centroid_x, init_centroid_y
            )

            if i != 0:
                score = get_silhouette_score(
                    data, data_x, data_y, centroid_x, centroid_y, data_cluster_index
                )
                scores.append(score)

            if i > 2:
                if scores[i - 1] < scores[i - 2] and scores[i - 2] < scores[i - 3]:
                    break

        if plot_mode == "all":
            print(scores)
            plt.figure()
            plt.plot(list(range(2, len(scores) + 2)), scores)

        n = 1
        if np.max(scores) >= 0.6:
            max_index = np.argmax(scores)
            n = int(max_index) + 2
        print("best component num is {}".format(str(n)))

    if n == 1:
        estimates = [method_of_moments(data, data_x, data_y)]
    elif n < 11:
        init_centroid_x, init_centroid_y = k_means_plus_plus(data, data_x, data_y, n)
        data_cluster_index, centroid_x, centroid_y = k_means(
            data, data_x, data_y, init_centroid_x, init_centroid_y
        )

        estimates = []
        for i in range(n):
            cluster_indexes = np.where(data_cluster_index == i)[0]
            data_cluster = data[cluster_indexes]
            data_x_cluster = data_x[cluster_indexes]
            data_y_cluster = data_y[cluster_indexes]
            estimates.append(
                method_of_moments(data_cluster, data_x_cluster, data_y_cluster)
            )
    else:
        raise Exception("Invalid Gaussian component number.")

    return estimates


def method_of_moments(
    data: npt.NDArray[np.float64],
    data_x: npt.NDArray[np.float64],
    data_y: npt.NDArray[np.float64],
) -> list[float]:
    """
    Estimate parameters of 2D single Gaussian distribution using the method of moments.

    Parameters
    ----------
    data
        The input data array.
    data_x
        X coordinates of data points.
    data_y
        Y coordinates of data points.

    Returns
    -------
    list[float]
        Estimated parameters: amplitude, center x, center y, FWHM x, FWHM y, and position angle.
    """

    m0 = data.sum()
    mx = np.dot(data_x, data) / m0
    my = np.dot(data_y, data) / m0
    mxx = np.dot(np.square(data_x), data) / m0 - mx * mx
    myy = np.dot(np.square(data_y), data) / m0 - my * my
    mxy = np.dot(data_x * data_y, data) / m0 - mx * my

    amp_estimate = m0 * 0.5 * (abs(mxx * myy - mxy * mxy) ** (-0.5)) / np.pi

    SIGMA_TO_FWHM = (8 * math.log(2)) ** 0.5
    tmp = ((mxx - myy) ** 2 + 4 * mxy * mxy) ** 0.5
    sigma_x_estimate = (0.5 * (abs(mxx + myy + tmp))) ** 0.5
    sigma_y_estimate = (0.5 * (abs(mxx + myy - tmp))) ** 0.5
    fwhm_x_estimate = sigma_x_estimate * SIGMA_TO_FWHM
    fwhm_y_estimate = sigma_y_estimate * SIGMA_TO_FWHM

    theta_estimate = np.degrees(-0.5 * np.arctan2(2 * mxy, myy - mxx))

    return [amp_estimate, mx, my, fwhm_x_estimate, fwhm_y_estimate, theta_estimate]
