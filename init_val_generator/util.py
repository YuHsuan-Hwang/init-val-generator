import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def print_gaussian_param(gaussian_param: list[list[float]]) -> None:
    """
    Print the parameters of Gaussian models.

    Parameters
    ----------
    gaussian_param
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


def plot_data(
    data: npt.NDArray[np.float64], width: int, height: int, title: str
) -> None:
    """
    Plot 2D data.

    Parameters
    ----------
    data
        1D array containing the data to be plotted.
    width
        Width of the image.
    height
        Height of the image.
    title
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

    # type hint bug will be fixed in matplotlib 3.8.1
    plt.show()  # type: ignore


def plot_comparison(
    data: npt.NDArray[np.float64],
    width: int,
    height: int,
    models: list[list[float]],
    estimates: list[list[float]],
) -> None:
    fig, ax = plt.subplots()
    plt.imshow(
        np.resize(data, (height, width)),
        origin="lower",
        interpolation="nearest",
    )
    plt.colorbar()

    for estimate in estimates:
        ellipse = Ellipse(
            (estimate[1], estimate[2]),
            estimate[3],
            estimate[4],
            angle=estimate[5] + 90,
            edgecolor="white",
            facecolor="none",
            linestyle="--",
        )
        plt.gca().add_patch(ellipse)

    for model in models:
        model_ellipse = Ellipse(
            (model[1], model[2]),
            model[3],
            model[4],
            angle=model[5] + 90,
            edgecolor="red",
            facecolor="none",
            linestyle="--",
        )
        plt.gca().add_patch(model_ellipse)

    ax.legend(handles=[ellipse, model_ellipse], labels=["Guess", "Model"])

    # type hint bug will be fixed in matplotlib 3.8.1
    plt.show()  # type: ignore
