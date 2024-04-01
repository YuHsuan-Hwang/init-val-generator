import numpy as np

from init_val_generator.data_selection import SelectionMethod, filter_data


def test_filter_data():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    data_x = np.arange(8)
    data_y = np.arange(8)
    data, data_x, data_y = filter_data(
        SelectionMethod.THREE_SIGMA, data, 8, 1, data_x, data_y
    )

    np.testing.assert_array_equal(data, np.array([7.0, 8.0]))
    np.testing.assert_array_equal(data_x, np.array([6.0, 7.0]))
    np.testing.assert_array_equal(data_y, np.array([6.0, 7.0]))
