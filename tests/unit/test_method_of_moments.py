import pytest
import numpy as np

from init_val_generator.method_of_moments import method_of_moments
from init_val_generator.tools.gaussian_image import GaussianImage


@pytest.mark.parametrize("pa", np.arange(0, 180, 22.5))
def test_method_of_moments(pa):
    width = 256
    height = 256
    image = GaussianImage(width, height, [[1, 128, 128, 40, 20, pa]], noise=None)
    x = np.arange(width)
    y = np.arange(height)
    data_x = np.tile(x, height)
    data_y = np.repeat(y, width)

    estimates = method_of_moments(image.data, data_x, data_y)
    print(estimates[5])
    if estimates[5] < -1:
        estimates[5] += 180

    np.testing.assert_allclose([estimates], image.model_components, atol=1e-10)
