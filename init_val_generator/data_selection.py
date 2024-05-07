from enum import StrEnum
import numpy as np
import numpy.typing as npt

from .method_of_moments import method_of_moments

from .util import plot_data


class SelectionMethod(StrEnum):
    THREE_SIGMA = "3-sigma"
    MAD = "mad"
    TWO_MAD = "2-mad"
    THREE_MAD = "3-mad"
    FWHM_ESTIMATE = "fwhm-estimate"
    TWO_FWHM_ESTIMATE = "2-fwhm-estimate"
    THREE_FWHM_ESTIMATE = "3-fwhm-estimate"


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
        indices = filter_3_sigma(data, plot_mode)
    elif method == SelectionMethod.MAD:
        indices = filter_mad(data, 1, plot_mode)
    elif method == SelectionMethod.TWO_MAD:
        indices = filter_mad(data, 2, plot_mode)
    elif method == SelectionMethod.THREE_MAD:
        indices = filter_mad(data, 3, plot_mode)
    elif method == SelectionMethod.FWHM_ESTIMATE:
        indices = filter_fwhm(data, data_x, data_y, 1, plot_mode)
    elif method == SelectionMethod.TWO_FWHM_ESTIMATE:
        indices = filter_fwhm(data, data_x, data_y, 2, plot_mode)
    else:
        indices = filter_fwhm(data, data_x, data_y, 3, plot_mode)

    data = data[indices]
    data_x = data_x[indices]
    data_y = data_y[indices]

    if plot_mode == "all":
        print("selected {} / {}".format(len(data), width * height))
        data_selected_plot = np.full((height, width), np.nan)
        for i in range(len(data)):
            data_selected_plot[data_y[i]][data_x[i]] = data[i]
        plot_data(data_selected_plot, width, height, "Selected Data")

    return data, data_x, data_y


def filter_3_sigma(
    data: npt.NDArray[np.float64], plot_mode: str = "none"
) -> npt.NDArray[np.intc]:
    """
    Filter out data points within 3 standard deviations.

    Parameters
    ----------
    data
        The input data array.
    plot_mode
        The mode for plotting. Options: "none", "all".

    Returns
    -------
    tuple
        Indices of not excluded data.
    """

    std = np.std(data)
    indices = np.where(np.logical_or(data > 3 * std, data < -3 * std))[0]
    if plot_mode == "all":
        print("std of the image: {}".format(std))
        print("excluded data within +/- {}".format(3 * std))

    return indices


def filter_mad(
    data: npt.NDArray[np.float64], multiplier: float = 3, plot_mode: str = "none"
) -> npt.NDArray[np.intc]:
    """
    Filter out data points within median absolute deviation (MAD).

    Parameters
    ----------
    data
        The input data array.
    multiplier
        Multiplier used to scale the MAD threshold.
    plot_mode
        The mode for plotting. Options: "none", "all".

    Returns
    -------
    tuple
        Indices of not excluded data.
    """

    mad = 1.4826 * np.median(np.abs(data - np.median(data)))
    indices = np.where(
        np.logical_or(data > multiplier * mad, data < -multiplier * mad)
    )[0]

    if plot_mode == "all":
        print("mad of the image: {}".format(mad))
        print("excluded data within +/- {}".format(3 * mad))

    return indices


def filter_fwhm(
    data: npt.NDArray[np.float64],
    data_x: npt.NDArray[np.float64],
    data_y: npt.NDArray[np.float64],
    multiplier: float = 3,
    plot_mode: str = "none",
) -> npt.NDArray[np.intc]:
    """
    Apply method of moments to the data and filter out data points out of the estimated FWHM.

    Parameters
    ----------
    data
        The input data array.
    data_x
        X coordinates of data points.
    data_y
        Y coordinates of data points.
    multiplier
        Multiplier used to scale the selected area.
    plot_mode
        The mode for plotting. Options: "none", "all".

    Returns
    -------
    tuple
        Indices of not excluded data.
    """

    amp, center_x, center_y, fwhm_x, fwhm_y, pa = method_of_moments(
        data, data_x, data_y
    )
    size = np.max([fwhm_x, fwhm_y])
    indices = np.where(
        np.sqrt((data_x - center_x) ** 2 + (data_y - center_y) ** 2)
        <= size / 2 * multiplier
    )[0]

    if plot_mode == "all":
        print("fwhm of the image: {}, {}".format(fwhm_x, fwhm_y))
        print("excluded data out of radius {}".format(size / 2 * multiplier))

    return indices
