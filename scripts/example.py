from init_val_generator.tools.gaussian_image import GaussianImage
from init_val_generator.core import guess

# generate fake Gaussian image. can be replace by actual image
width = 256
height = 256
image = GaussianImage(
    width, height, random_seed=8, n=1, plot_mode="none"
)  # single gaussian
# image = GaussianImage(
#     width, height, random_seed=0, plot_mode="all"
# )  # multiple gaussian

guess(image.data, width, height, 1, plot_mode="all")
