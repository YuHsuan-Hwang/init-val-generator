from init_val_generator import InitValGenerator
from init_val_generator.tools.gaussian_image import GaussianImage
from init_val_generator.util import print_gaussian_param, plot_comparison

# generate fake Gaussian image. can be replace by actual image
width = 256
height = 256
# single gaussian
# image = GaussianImage(width, height, n=1, random_seed=8, plot_mode="none")
# image = GaussianImage(
#     width, height, [[1, 128, 128, 40, 20, 35]], plot_mode="none", noise=0.3
# )
# multiple gaussian
image = GaussianImage(width, height, random_seed=0, plot_mode="none")

guesser = InitValGenerator("2-fwhm-estimate", "3-mad")
estimates = guesser.estimate(image.data, width, height, None)

print_gaussian_param(estimates)
plot_comparison(image.data, width, height, image.model_components, estimates)
