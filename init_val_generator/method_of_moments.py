import math
import numpy as np
import numpy.typing as npt


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
