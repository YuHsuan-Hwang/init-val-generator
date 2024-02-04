import numpy as np
import matplotlib.pyplot as plt


def print_gaussian_param(gaussian_param):
    """
    Print the parameters of Gaussian models.

    Parameters
    ----------
    gaussian_param : list
        List of Gaussian parameters, where each element is a list representing
        (amplitude, center_x, center_y, fwhm_x, fwhm_y, position_angle).

    Returns
    -------
    None
    """
    param_format = "amp: {:.4f}    center x: {:.4f}    center y: {:.4f}    fwhm x: {:.4f}    fwhm y: {:.4f}    pa: {:.4f}"
    for i in range(len(gaussian_param)):
        print(
            "model    ",
            i,
            "    ",
            param_format.format(*gaussian_param[i], sep=", "),
        )


def plot_data(data, width, height, title):
    """
    Plot 2D data.

    Parameters
    ----------
    data : numpy.ndarray
        1D array containing the data to be plotted.
    width : int
        Width of the image.
    height : int
        Height of the image.
    title : str
        Title of the plot.

    Returns
    -------
    None
    """
    plt.imshow(
        np.resize(data, (height, width)),
        origin="lower",
        interpolation="nearest",
    )
    plt.colorbar()
    plt.title(title)
    plt.show()
