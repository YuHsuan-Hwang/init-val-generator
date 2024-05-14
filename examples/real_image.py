import sys
from astropy.io import fits
import numpy as np
from init_val_generator import InitValGenerator
from init_val_generator.util import print_gaussian_param, plot_comparison


if len(sys.argv) < 2:
    sys.exit()

image_data = fits.getdata(sys.argv[1])

index = 0
while image_data.shape[index] == 1:
    index = index + 1
height = image_data.shape[index]
width = image_data.shape[index + 1]
print(image_data.shape)

image_data = np.reshape(image_data, width * height)

guesser = InitValGenerator("2-fwhm-estimate", "3-sigma")
estimates = guesser.estimate(image_data, width, height, 3)

print_gaussian_param(estimates)
plot_comparison(image_data, width, height, [], estimates)
