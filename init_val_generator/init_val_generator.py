import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .data_selection import SelectionMethod, filter_data
from .method_of_moments import method_of_moments
from .clustering import get_silhouette_score, k_means, k_means_plus_plus

MAX_COMPONENT_NUM = 10


class InitValGenerator:
    def __init__(
        self, data_selection: SelectionMethod | None = None, plot_mode: str = "none"
    ):
        self.data_selection = data_selection
        self.plot_mode = plot_mode

    def estimate(
        self, data: npt.NDArray[np.float64], width: int, height: int, n: int | None = 1
    ) -> list[list[float]]:

        x = np.arange(width)
        y = np.arange(height)
        data_x = np.tile(x, height)
        data_y = np.repeat(y, width)

        if self.data_selection is not None:
            data, data_x, data_y = filter_data(
                self.data_selection, data, width, height, data_x, data_y, self.plot_mode
            )

        if n is None:
            scores = []
            for i in range(MAX_COMPONENT_NUM):
                input_num = i + 1
                print("clustering with component number {}".format(str(input_num)))

                init_centroid_x, init_centroid_y = k_means_plus_plus(
                    data, data_x, data_y, input_num
                )
                data_cluster_index, centroid_x, centroid_y = k_means(
                    data, data_x, data_y, init_centroid_x, init_centroid_y
                )

                if i != 0:
                    score = get_silhouette_score(
                        data, data_x, data_y, centroid_x, centroid_y, data_cluster_index
                    )
                    scores.append(score)

                if i > 2:
                    if scores[i - 1] < scores[i - 2] and scores[i - 2] < scores[i - 3]:
                        break

            if self.plot_mode == "all":
                print(scores)
                plt.figure()
                plt.plot(list(range(2, len(scores) + 2)), scores)

            n = 1
            if np.max(scores) >= 0.6:
                max_index = np.argmax(scores)
                n = int(max_index) + 2
            print("best component num is {}".format(str(n)))

        if n == 1:
            estimates = [method_of_moments(data, data_x, data_y)]
        elif n <= MAX_COMPONENT_NUM:
            init_centroid_x, init_centroid_y = k_means_plus_plus(
                data, data_x, data_y, n
            )
            data_cluster_index, centroid_x, centroid_y = k_means(
                data, data_x, data_y, init_centroid_x, init_centroid_y
            )

            estimates = []
            for i in range(n):
                cluster_indexes = np.where(data_cluster_index == i)[0]
                data_cluster = data[cluster_indexes]
                data_x_cluster = data_x[cluster_indexes]
                data_y_cluster = data_y[cluster_indexes]
                estimates.append(
                    method_of_moments(data_cluster, data_x_cluster, data_y_cluster)
                )
        else:
            raise Exception("Invalid Gaussian component number.")

        return estimates
