import numpy as np
from init_val_generator.core import guess
from init_val_generator.tools.gaussian_image import GaussianImage


def test_single_gaussian():
    width = 256
    height = 256
    image = GaussianImage(width, height, 1, False, random_seed=8)
    estimates = guess(image.data, width, height, 1, "all")
    estimates[0][5] += 180
    np.testing.assert_allclose(estimates, image.model_components)


def test_single_gaussian_with_noise():
    width = 256
    height = 256
    image = GaussianImage(width, height, 1, random_seed=8)
    estimates = guess(image.data, width, height, 1)
    np.testing.assert_allclose(
        estimates,
        np.array([[1.3421, 94.3963, 112.3251, 27.8443, 14.7146, 41.8023]]),
        atol=1e-4,
    )


def test_multiple_gaussian():
    width = 256
    height = 256
    image = GaussianImage(width, height, random_seed=0)
    estimates = guess(image.data, width, height, 5)
    np.testing.assert_allclose(
        estimates,
        np.array(
            [
                [1.8796, 164.9701, 129.0529, 19.9775, 15.3311, 144.9156],
                [1.2938, 74.8998, 66.8425, 17.4741, 16.8425, 131.4475],
                [1.6088, 102.8455, 124.9227, 20.0898, 14.0249, 112.5173],
                [1.5746, 136.0933, 119.9996, 21.4420, 10.8561, 70.6344],
                [1.7571, 144.9293, 134.1755, 16.0268, 12.3464, 76.1669],
            ]
        ),
        atol=1e-4,
    )