from PIL import Image, ImageDraw
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from skimage import color


def float_to_uint8(img):
    img = img * 255.0
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img


def draw_circles(img: np.ndarray, peaks: List, radius):
    if img.ndim == 2:
        img = color.gray2rgb(img)
        img = float_to_uint8(img)

    assert img.ndim == 3
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    for row_off, col_off, *_ in peaks:
        draw.ellipse(
            (col_off - radius, row_off - radius, col_off + radius, row_off + radius),
            outline=(255, 255, 0),
            width=2,
        )
    return np.asarray(img)


def draw_lines(img: np.ndarray, peaks: List, theta_values, rho_values):
    if img.ndim == 2:
        img = color.gray2rgb(img)
        img = float_to_uint8(img)

    assert img.ndim == 3
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    for rho_idx, theta_idx, *_ in peaks:
        rho = rho_values[rho_idx]
        theta = theta_values[theta_idx]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # print(f"x1, y1: ({x1}, {y1})")
        # print(f"x2, y2: ({x2}, {y2})")

        draw.line([(x1, y1), (x2, y2)], fill=(0, 255, 0), width=2)
    return np.asarray(img)

    """Compute Hough accumulator array for finding lines.

    binary_image: Binary (black (False) and white (True)) image containing edge pixels.
    rho_resolution: Difference between successive rho values, in pixels.
    num_theta: Due to the implementation, theta range is [-90, 90). -90 <= theta < 90
               Users can specify the resolution/steps of theta.

    Note that it has two optional parameters RhoResolution and Theta,
    and returns three values - the hough accumulator array H, theta values
    that correspond to columns of H and rho values that correspond to rows of H.

    Return:
    theta: the angle in degrees between the x-axis and this vector.
    rho: the distance from the origin to the line along a vector
         perpendicular to the line
    """
