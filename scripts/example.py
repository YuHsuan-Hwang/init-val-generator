from init_val_generator.tools.gaussian_image import GaussianImage
from init_val_generator.core import guess
from init_val_generator.util import print_gaussian_param, plot_comparison

# generate fake Gaussian image. can be replace by actual image
width = 256
height = 256
# single gaussian
# image = GaussianImage(width, height, random_seed=8, n=1, plot_mode="none")
# multiple gaussian
image = GaussianImage(width, height, random_seed=0, plot_mode="all")

estimates = guess(image.data, width, height, 5, plot_mode="all")

print_gaussian_param(estimates)
plot_comparison(image.data, width, height, image.model_components, estimates)
