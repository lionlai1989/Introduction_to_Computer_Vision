import numpy as np
from skimage import io, color


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


def rescale_and_clip_raster(arr: np.ndarray, out_range=(0, 254), percentiles=1):
    def clip(img, min, max):
        assert img.ndim == 2
        assert img.shape[0] > 3
        assert img.shape[1] > 3
        img[img < min] = min
        img[img > max] = max
        return img

    assert np.any(arr == NO_DATA_VALUES[str(arr.dtype)]) == True
    # valid_arr is an array whose nodata value are masked out
    valid_arr = np.ma.masked_equal(arr, NO_DATA_VALUES[str(arr.dtype)])
    assert np.any(valid_arr == NO_DATA_VALUES[str(arr.dtype)]) == False

    minimum, maximum = np.percentile(
        valid_arr.flatten(), (percentiles, 100 - percentiles)
    )

    # TODO: minimum must with the range of out_range.
    if minimum <= out_range[0]:
        minimum = out_range[0]

    for i in range(arr.shape[0]):
        arr[i, :, :] = clip(arr[i, :, :], min=minimum, max=maximum)

    a = (out_range[1] - out_range[0]) / (maximum - minimum)
    b = out_range[1] - a * maximum

    arr = a * arr + b
    return arr

    """ Get disparity map from left and right image. navie implementation.
    The disparity search method here is sometimes refer as winner takes all approach. 
    I.e., the disparity value with the lowest SSD gets to be selected.  

    Define the disparity range and maximum disparity value
    # disparity_range = [-32, 32]
    # max_disparity = disparity_range // 2

    Args:
        left_img (_type_): _description_
        right_img (_type_): _description_
        window_size (int, optional): 3, 5, 7 or 9. Defaults to 3.
        disparity_range (tuple, optional): _description_. Defaults to (-32, 32).
    """
