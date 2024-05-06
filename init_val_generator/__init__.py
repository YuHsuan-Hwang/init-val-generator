import numpy as np
import numpy.typing as npt

from .init_val_generator import InitValGenerator


def guess(
    data: npt.NDArray[np.float64], width: int, height: int, n: int | None = 1
) -> list[list[float]]:
    guesser = InitValGenerator()
    return guesser.estimate(data, width, height, n)
