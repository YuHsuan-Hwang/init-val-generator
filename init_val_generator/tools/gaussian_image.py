import numpy as np
import math

from ..util import plot_data, print_gaussian_param


class GaussianImage:
    def __init__(
        self, width, height, n=None, noise=True, random_seed=None, plot_mode="none"
    ):
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

    def __generate_random_parameters(self):
        self.model_components = []
        for i in range(self.__n):
            self.model_components.append(self.__get_random_parameters())
        print_gaussian_param(self.model_components)

    def __get_random_parameters(self):
        amp = np.random.uniform(0.4, 1) * np.random.choice([1, 1, -1])
        center_x = np.random.uniform(self.__width * 0.25, self.__width * 0.75)
        center_y = np.random.uniform(self.__height * 0.25, self.__height * 0.75)
        x_sigma = np.random.uniform(self.__width * 0.01, self.__width * 0.2)
        y_sigma = np.random.uniform(self.__height * 0.01, self.__height * 0.2)
        pa = np.random.uniform(0, 360)
        return [amp, center_x, center_y, x_sigma, y_sigma, pa]

    def __generate_gaussian_components(self):
        self.data = np.zeros(self.__width * self.__height)
        for i in range(self.__n):
            self.data += self.__get_gaussian_component(
                self.__x, self.__y, self.model_components[i]
            )

        self.__data_x = np.tile(self.__x, self.__height)
        self.__data_y = np.repeat(self.__y, self.__width)

    def __get_gaussian_component(self, x, y, params):
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

    def __add_noise(self):
        noise_std = 0.1
        noise = np.random.normal(0.0, noise_std, self.__width * self.__height)
        self.data += noise

    def __plot_data(self, title):
        plot_data(self.data, self.__width, self.__height, title)
