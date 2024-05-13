import numpy as np
import numpy.typing as npt

from .init_val_generator import InitValGenerator


def guess(
    data: npt.NDArray[np.float64], width: int, height: int, n: int | None = 1
) -> list[list[float]]:
    """
    Estimates Gaussian components.

    Parameters
    ----------
    data
        The input data array.
    width
        Width of the data array.
    height
        Height of the data array.
    n
        Number of components. If None, the optimal number is estimated.

    Returns
    -------
    list[list[float]]
        List of estimated parameters for the Gaussian components. The estimated parameters are: amplitude, center x, center y, FWHM x, FWHM y, and position angle.
    """
    guesser = InitValGenerator()
    return guesser.estimate(data, width, height, n)
