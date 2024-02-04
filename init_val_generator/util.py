import numpy as np
import matplotlib.pyplot as plt


def print_gaussian_param(gaussian_param):
    param_format = "amp: {:.4f}    center x: {:.4f}    center y: {:.4f}    fwhm x: {:.4f}    fwhm y: {:.4f}    pa: {:.4f}"
    for i in range(len(gaussian_param)):
        print(
            "model    ",
            i,
            "    ",
            param_format.format(*gaussian_param[i], sep=", "),
        )


def plot_data(data, width, height, title):
    plt.imshow(
        np.resize(data, (height, width)),
        origin="lower",
        interpolation="nearest",
    )
    plt.colorbar()
    plt.title(title)
    plt.show()
