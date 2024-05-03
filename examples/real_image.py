import sys
from astropy.io import fits
import numpy as np
from init_val_generator import guess
from init_val_generator.util import print_gaussian_param, plot_comparison


if len(sys.argv) < 2:
    sys.exit()

image_data = fits.getdata(sys.argv[1])
width = image_data.shape[0]
height = image_data.shape[1]
image_data = np.reshape(image_data, width * height)
estimates = guess(image_data, width, height, None, plot_mode="all")
print_gaussian_param(estimates)
plot_comparison(image_data, width, height, [], estimates)
