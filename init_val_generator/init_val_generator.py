import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .data_selection import SelectionMethod, filter_data
from .method_of_moments import method_of_moments
from .clustering import get_silhouette_score, k_means, k_means_plus_plus

MAX_COMPONENT_NUM = 10


class InitValGenerator:
    """
    Generates initial values for Gaussian image fitting.

    Attributes
    ----------
    data_selection
        Method for selecting data for parameter estimation.
    clustering_data_selection
        Method for selecting data for clustering.
    plot_mode
        Plotting mode. 'none' for no plots, 'all' for all plots.
    """

    def __init__(
        self,
        data_selection: SelectionMethod | None = None,
        clustering_data_selection: SelectionMethod | None = None,
        plot_mode: str = "none",
    ):
        """
        Initialize the InitValGenerator.

        Parameters
        ----------
        data_selection
            Method for selecting data for parameter estimation.
        clustering_data_selection
            Method for selecting data for clustering.
        plot_mode
            Plotting mode. 'none' for no plots, 'all' for all plots.
        """
        self.data_selection = data_selection
        self.clustering_data_selection = clustering_data_selection
        self.plot_mode = plot_mode

    def estimate(
        self, data: npt.NDArray[np.float64], width: int, height: int, n: int | None = 1
    ) -> list[list[float]]:
        """
        Estimates Gaussian components.

        Parameters
        ----------
        data
            The input data array.
        width
            Width of the data array.
        height
            Height of the data array.
        n
            Number of components. If None, the optimal number is estimated.

        Returns
        -------
        list[list[float]]
            List of estimated parameters for the Gaussian components. The estimated parameters are: amplitude, center x, center y, FWHM x, FWHM y, and position angle.
        """

        x = np.arange(width)
        y = np.arange(height)
        data_x = np.tile(x, height)
        data_y = np.repeat(y, width)

        if n is None:
            scores = []
            for i in range(MAX_COMPONENT_NUM):
                input_num = i + 1
                print("clustering with component number {}".format(str(input_num)))

                if self.clustering_data_selection is not None:
                    clustering_data, clustering_data_x, clustering_data_y = filter_data(
                        self.clustering_data_selection,
                        data,
                        width,
                        height,
                        data_x,
                        data_y,
                        self.plot_mode,
                    )
                else:
                    clustering_data, clustering_data_x, clustering_data_y = (
                        data,
                        data_x,
                        data_y,
                    )

                init_centroid_x, init_centroid_y = k_means_plus_plus(
                    clustering_data, clustering_data_x, clustering_data_y, input_num
                )
                data_cluster_index, centroid_x, centroid_y = k_means(
                    clustering_data,
                    clustering_data_x,
                    clustering_data_y,
                    init_centroid_x,
                    init_centroid_y,
                )

                if i != 0:
                    score = get_silhouette_score(
                        clustering_data,
                        clustering_data_x,
                        clustering_data_y,
                        centroid_x,
                        centroid_y,
                        data_cluster_index,
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
            if self.data_selection is not None:
                data, data_x, data_y = filter_data(
                    self.data_selection,
                    data,
                    width,
                    height,
                    data_x,
                    data_y,
                    self.plot_mode,
                )

            estimates = [method_of_moments(data, data_x, data_y)]
        elif n <= MAX_COMPONENT_NUM:
            if self.clustering_data_selection is not None:
                clustering_data, clustering_data_x, clustering_data_y = filter_data(
                    self.clustering_data_selection,
                    data,
                    width,
                    height,
                    data_x,
                    data_y,
                    self.plot_mode,
                )

            init_centroid_x, init_centroid_y = k_means_plus_plus(
                clustering_data, clustering_data_x, clustering_data_y, n
            )
            data_cluster_index, centroid_x, centroid_y = k_means(
                clustering_data,
                clustering_data_x,
                clustering_data_y,
                init_centroid_x,
                init_centroid_y,
            )

            if (
                self.data_selection is not None
                and self.data_selection != "fwhm-estimate"
                and self.data_selection != "2-fwhm-estimate"
                and self.data_selection != "3-fwhm-estimate"
            ):
                data, data_x, data_y = filter_data(
                    self.data_selection,
                    data,
                    width,
                    height,
                    data_x,
                    data_y,
                    self.plot_mode,
                )

            data_cluster_index = np.full((data.shape), -1)
            for i in range(len(data)):
                dist = np.abs(data[i]) * np.sqrt(
                    (
                        np.square(data_x[i] - centroid_x)
                        + np.square(data_y[i] - centroid_y)
                    )
                )
                cluster_index = np.argmin(dist)
                data_cluster_index[i] = cluster_index

            estimates = []
            for i in range(n):
                cluster_indexes = np.where(data_cluster_index == i)[0]
                data_cluster = data[cluster_indexes]
                data_x_cluster = data_x[cluster_indexes]
                data_y_cluster = data_y[cluster_indexes]

                if (
                    self.data_selection == "fwhm-estimate"
                    or self.data_selection == "2-fwhm-estimate"
                    or self.data_selection == "3-fwhm-estimate"
                ):
                    data_cluster, data_x_cluster, data_y_cluster = filter_data(
                        self.data_selection,
                        data_cluster,
                        width,
                        height,
                        data_x_cluster,
                        data_y_cluster,
                        self.plot_mode,
                    )

                estimates.append(
                    method_of_moments(data_cluster, data_x_cluster, data_y_cluster)
                )
        else:
            raise Exception("Invalid Gaussian component number.")

        return estimates
