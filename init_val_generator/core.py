import numpy as np
import numpy.typing as npt

from .util import plot_data


def guess(
    data: npt.NDArray[np.float64],
    width: int,
    height: int,
    n: int | None = 1,
    data_selection: str = "3-sigma",
    plot_mode: str = "none",
) -> None:

    x = np.arange(width)
    y = np.arange(height)
    data_x = np.tile(x, height)
    data_y = np.repeat(y, width)

    if data_selection == "3-sigma":
        std = np.std(data)
        indices = np.where(np.logical_or(data > 3 * std, data < -3 * std))[0]
        data = data[indices]
        data_x = data_x[indices]
        data_y = data_y[indices]

        print("std of the image: {}".format(std))
        print("selected data within +/- {}".format(3 * std))
        print("selected {} / {}".format(len(data), width * height))

        if plot_mode == "all":
            data_selected_plot = np.full((height, width), np.nan)
            for i in range(len(data)):
                data_selected_plot[data_y[i]][data_x[i]] = data[i]
            plot_data(data_selected_plot, width, height, "Selected Data")

    if n == 1:
        print("TODO: Single Gaussian estimation")
    else:
        raise Exception("Multiple Gaussian or unknow Gaussian number is not supported.")
