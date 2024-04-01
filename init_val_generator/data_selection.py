from enum import StrEnum
import numpy as np
import numpy.typing as npt

from .util import plot_data


class SelectionMethod(StrEnum):
    THREE_SIGMA = "3-sigma"
    MAD = "3-mad"


def filter_data(
    method: SelectionMethod,
    data: npt.NDArray[np.float64],
    width: int,
    height: int,
    data_x: npt.NDArray[np.float64],
    data_y: npt.NDArray[np.float64],
    plot_mode: str = "none",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Filter out data points within different method.

    Parameters
    ----------
    method
        The selection method used for filtering out data.
    data
        The input data array.
    width
        Width of the data array.
    height
        Height of the data array.
    data_x
        X coordinates of data points.
    data_y
        Y coordinates of data points.
    plot_mode
        The mode for plotting. Options: "none", "all".

    Returns
    -------
    tuple
        Filtered data array, filtered X coordinates of data points, filtered Y coordinates of data points.
    """

    if method == SelectionMethod.THREE_SIGMA:
        indices = filter_3_sigma(data)
    else:
        indices = filter_3_mad(data)

    data = data[indices]
    data_x = data_x[indices]
    data_y = data_y[indices]

    print("selected {} / {}".format(len(data), width * height))

    if plot_mode == "all":
        data_selected_plot = np.full((height, width), np.nan)
        for i in range(len(data)):
            data_selected_plot[data_y[i]][data_x[i]] = data[i]
        plot_data(data_selected_plot, width, height, "Selected Data")

    return data, data_x, data_y


def filter_3_sigma(
    data: npt.NDArray[np.float64],
) -> npt.NDArray[np.intc]:
    """
    Filter out data points within 3 standard deviations.

    Parameters
    ----------
    data
        The input data array.

    Returns
    -------
    tuple
        Indices of not excluded data.
    """

    std = np.std(data)
    indices = np.where(np.logical_or(data > 3 * std, data < -3 * std))[0]

    print("std of the image: {}".format(std))
    print("excluded data within +/- {}".format(3 * std))

    return indices


def filter_3_mad(
    data: npt.NDArray[np.float64],
) -> npt.NDArray[np.intc]:
    """
    Filter out data points within 3 median absolute deviation (MAD).

    Parameters
    ----------
    data
        The input data array.

    Returns
    -------
    tuple
        Indices of not excluded data.
    """

    mad = 1.4826 * np.median(np.abs(data - np.median(data)))
    indices = np.where(np.logical_or(data > 3 * mad, data < -3 * mad))[0]

    print("mad of the image: {}".format(mad))
    print("excluded data within +/- {}".format(3 * mad))

    return indices
