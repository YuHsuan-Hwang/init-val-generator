import typing
import numpy as np
import numpy.typing as npt
import math

from ..util import plot_data, print_gaussian_param


class GaussianImage:
    """
    Class for generating synthetic images with Gaussian components.

    Parameters
    ----------
    width : int
        Width of the image.
    height : int
        Height of the image.
    n : int, optional
        Number of Gaussian components. If not provided, a random value between 1 and 5 will be used.
    noise : bool, optional
        Flag to add random noise to the generated image. Default is True.
    random_seed : int, optional
        Seed for random number generation.
    plot_mode : {'none', 'all', 'result-only'}, optional
        Plotting mode. 'none' for no plots, 'all' for all plots, 'result-only' for result data plot.

    Attributes
    ----------
    model_components : list
        List of Gaussian model parameters for each component.
    data : numpy.ndarray
        Generated image data.

    Examples
    --------
    >>> gaussian_image = GaussianImage(width=256, height=256, noise=True, random_seed=8, plot_mode='all')
    >>> print(gaussian_image.model_components)
    >>> print(gaussian_image.data)
    """

    def __init__(
        self,
        width: int,
        height: int,
        n: int | None = None,
        noise: bool = True,
        random_seed: int | None = None,
        plot_mode: str = "none",
    ) -> None:
        self.__width = width
        self.__height = height

        self.__x = np.arange(width)
        self.__y = np.arange(height)

        np.random.seed(random_seed)
        self.__n = np.random.randint(5) + 1 if n is None else n
        self.__generate_random_parameters()

        self.__generate_gaussian_components()
        if plot_mode == "all":
            self.__plot_data("Model Data")

        if noise:
            self.__add_noise()
            if plot_mode != "none":
                self.__plot_data("Original Data")

    def __generate_random_parameters(self) -> None:
        """
        Generate random Gaussian model parameters for each component.
        """
        self.model_components = []
        for i in range(self.__n):
            self.model_components.append(self.__get_random_parameters())
        print_gaussian_param(self.model_components)

    def __get_random_parameters(self) -> list[float]:
        """
        Generate random parameters for a single Gaussian component.

        Returns
        -------
        list
            List of random parameters for a Gaussian component.
        """
        amp = np.random.uniform(0.4, 1) * np.random.choice([1, 1, -1])
        center_x = np.random.uniform(self.__width * 0.25, self.__width * 0.75)
        center_y = np.random.uniform(self.__height * 0.25, self.__height * 0.75)
        x_sigma = np.random.uniform(self.__width * 0.01, self.__width * 0.2)
        y_sigma = np.random.uniform(self.__height * 0.01, self.__height * 0.2)
        pa = np.random.uniform(0, 360)
        return [amp, center_x, center_y, x_sigma, y_sigma, pa]

    def __generate_gaussian_components(self) -> None:
        """
        Generate the image data based on the random Gaussian model parameters.
        """
        self.data = np.zeros(self.__width * self.__height)
        for i in range(self.__n):
            self.data += self.__get_gaussian_component(
                self.__x, self.__y, self.model_components[i]
            )

        self.__data_x = np.tile(self.__x, self.__height)
        self.__data_y = np.repeat(self.__y, self.__width)

    def __get_gaussian_component(
        self,
        x: npt.NDArray[np.signedinteger[typing.Any]],
        y: npt.NDArray[np.signedinteger[typing.Any]],
        params: list[float],
    ) -> npt.NDArray[np.float64]:
        """
        Calculate the values of a Gaussian component for given parameters.

        Parameters
        ----------
        x : numpy.ndarray
            x-coordinate values.
        y : numpy.ndarray
            y-coordinate values.
        params : list
            Parameters of the Gaussian component.

        Returns
        -------
        numpy.ndarray
            Calculated values of the Gaussian component.
        """
        SQ_FWHM_TO_SIGMA = 1 / 8 / math.log(2)
        DEG_TO_RAD = math.pi / 180.0

        amp = params[0]
        center_x = params[1]
        center_y = params[2]
        fwhm_x = params[3]
        fwhm_y = params[4]
        pa = params[5]

        dbl_sq_std_x = 2 * fwhm_x * fwhm_x * SQ_FWHM_TO_SIGMA
        dbl_sq_std_y = 2 * fwhm_y * fwhm_y * SQ_FWHM_TO_SIGMA
        theta_radian = (pa - 90.0) * DEG_TO_RAD  # counterclockwise rotation
        a = (
            math.cos(theta_radian) * math.cos(theta_radian) / dbl_sq_std_x
            + math.sin(theta_radian) * math.sin(theta_radian) / dbl_sq_std_y
        )
        dbl_b = 2 * (
            math.sin(2 * theta_radian) / (2 * dbl_sq_std_x)
            - math.sin(2 * theta_radian) / (2 * dbl_sq_std_y)
        )
        c = (
            math.sin(theta_radian) * math.sin(theta_radian) / dbl_sq_std_x
            + math.cos(theta_radian) * math.cos(theta_radian) / dbl_sq_std_y
        )

        data = []

        for j in y:
            for i in x:
                dx = i - center_x
                dy = j - center_y
                data.append(
                    amp * math.exp(-(a * dx * dx + dbl_b * dx * dy + c * dy * dy))
                )

        return np.array(data)

    def __add_noise(self) -> None:
        """
        Add random noise to the generated image.
        """
        noise_std = 0.1
        noise = np.random.normal(0.0, noise_std, self.__width * self.__height)
        self.data += noise

    def __plot_data(self, title: str) -> None:
        """
        Plot the image data.

        Parameters
        ----------
        title : str
            Title of the plot.
        """
        plot_data(self.data, self.__width, self.__height, title)
