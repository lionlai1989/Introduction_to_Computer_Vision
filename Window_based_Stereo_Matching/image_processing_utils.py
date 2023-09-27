import numpy as np
from skimage import io, color


def coord_is_valid(x_min, x_max, y_min, y_max, width, height):
    if x_min < 0 or y_min < 0:
        # print(f"x_min: {x_min}. y_min: {y_min}")
        return False
    if x_max >= width or y_max >= height:
        # print(f"x_max: {x_max}. y_max: {y_max}")
        return False
    return True


def read_image(input_path):
    # gray-image: MxN
    # RGB-image: MxNx3
    # RGBA-image: MxNx4
    img = io.imread(input_path)
    if img.ndim == 2:
        return rescale_image(img)
    elif img.ndim == 3:
        # rgb2gray automatically converts 0-255 to 0-1.
        return color.rgb2gray(img)


def rescale_image(
    arr: np.ndarray,
    out_range: tuple = (0, 1),
    percentiles: float = None,
    normalized: bool = False,
    bit_depth: int = 8,
) -> np.ndarray:
    """what is the difference of out_range being (0, 1) or (-1, 1)?

    Args:
        arr (np.ndarray): _description_
        out_range (tuple, optional): _description_. Defaults to (0, 1).
        percentiles (float, optional): _description_. Defaults to None.
        normalized (bool, optional): _description_. Defaults to False.
        bit_depth (int, optional): Unsigned integer. Defaults to 8.

    Returns:
        np.ndarray: _description_
    """
    if percentiles is not None:
        mi, ma = np.percentile(arr, (percentiles, 100 - percentiles))
        print("Clip range is from {0} to {1}".format(mi, ma))
        arr = np.minimum(np.maximum(arr, mi), ma)  # clip
    if normalized is True:
        mi = np.min(arr)
        ma = np.max(arr)
        # print(f"min: {mi}, max: {ma}")
    else:
        mi = 0  # Assume 0
        ma = 2**bit_depth - 1

    a = (out_range[1] - out_range[0]) / (ma - mi)
    b = out_range[1] - a * ma
    arr = a * arr + b

    arr[arr <= out_range[0]] = out_range[0]
    arr[arr >= out_range[1]] = out_range[1]

    assert np.all(arr >= out_range[0])
    assert np.all(arr <= out_range[1])
    return arr
