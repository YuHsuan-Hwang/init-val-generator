import numpy as np

from init_val_generator.core import filter_3_sigma


def test_filter_3_sigma():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    data_x = np.arange(8)
    data_y = np.arange(8)
    data, data_x, data_y = filter_3_sigma(data, 8, 1, data_x, data_y)

    np.testing.assert_array_equal(data, np.array([7.0, 8.0]))
    np.testing.assert_array_equal(data_x, np.array([6.0, 7.0]))
    np.testing.assert_array_equal(data_y, np.array([6.0, 7.0]))
