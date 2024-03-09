import math
import numpy as np
import numpy.typing as npt

from .util import plot_data
from .clustering import k_means, k_means_plus_plus


def guess(
    data: npt.NDArray[np.float64],
    width: int,
    height: int,
    n: int | None = 1,
    data_selection: str = "3-sigma",
    plot_mode: str = "none",
) -> list[list[float]]:

    x = np.arange(width)
    y = np.arange(height)
    data_x = np.tile(x, height)
    data_y = np.repeat(y, width)

    if data_selection == "3-sigma":
        data, data_x, data_y = filter_3_sigma(
            data, width, height, data_x, data_y, plot_mode
        )

    if n is None:
        raise Exception("Unknow Gaussian component number is not supported.")
    elif n == 1:
        estimates = [method_of_moments(data, data_x, data_y)]
    elif n < 11:
        init_centroid_x, init_centroid_y = k_means_plus_plus(data, data_x, data_y, n)
        data_cluster_index = k_means(
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


def filter_3_sigma(
    data: npt.NDArray[np.float64],
    width: int,
    height: int,
    data_x: npt.NDArray[np.float64],
    data_y: npt.NDArray[np.float64],
    plot_mode: str = "none",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:

    std = np.std(data)
    indices = np.where(np.logical_or(data > 3 * std, data < -3 * std))[0]
    data = data[indices]
    data_x = data_x[indices]
    data_y = data_y[indices]

    print("std of the image: {}".format(std))
    print("excluded data within +/- {}".format(3 * std))
    print("selected {} / {}".format(len(data), width * height))

    if plot_mode == "all":
        data_selected_plot = np.full((height, width), np.nan)
        for i in range(len(data)):
            data_selected_plot[data_y[i]][data_x[i]] = data[i]
        plot_data(data_selected_plot, width, height, "Selected Data")

    return data, data_x, data_y


def method_of_moments(
    data: npt.NDArray[np.float64],
    data_x: npt.NDArray[np.float64],
    data_y: npt.NDArray[np.float64],
) -> list[float]:

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

    theta_estimate = np.degrees(np.arctan((myy - mxx + tmp) / 2 / mxy)) + 90

    return [amp_estimate, mx, my, fwhm_x_estimate, fwhm_y_estimate, theta_estimate]
