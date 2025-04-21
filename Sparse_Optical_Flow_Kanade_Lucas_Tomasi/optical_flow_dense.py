import itertools
from pathlib import Path

import cv2
import imageio

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from natsort import natsorted
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve, convolve2d


def gen_gaussian_pyramid(im, max_level):
    gauss_pyr = [im]
    for i in range(max_level):
        gauss_pyr.append(cv2.pyrDown(gauss_pyr[-1]))
    return gauss_pyr


def expand(img, dst_size, interpolation=None):
    height, width = dst_size[:2]
    return cv2.GaussianBlur(
        cv2.resize(
            img,
            (width, height),
            interpolation=interpolation or cv2.INTER_LINEAR,
        ),
        (5, 5),
        0,
    )


def remap(a, flow):
    height, width = flow.shape[:2]
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    flow_map = np.column_stack(
        (
            x.flatten() + -flow[:, :, 0].flatten(),
            y.flatten() + -flow[:, :, 1].flatten(),
        )
    )
    flow_map = flow_map.reshape((height, width, 2))
    flow_map[:, :, 0] = np.clip(flow_map[:, :, 0], 0, width - 1)
    flow_map[:, :, 1] = np.clip(flow_map[:, :, 1], 0, height - 1)
    flow_map = flow_map.astype(np.float32)
    warped = cv2.remap(a, flow_map, None, cv2.INTER_LINEAR)
    return warped


def hierarchical_lucas_kanade(im1, im2, max_level, window_size):
    gauss_pyr_1 = gen_gaussian_pyramid(im1, max_level)
    gauss_pyr_2 = gen_gaussian_pyramid(im2, max_level)

    g_L = [0 for _ in range(max_level + 1)]
    d_L = [0 for _ in range(max_level + 1)]
    g_L[max_level] = np.zeros(gauss_pyr_1[-1].shape[:2] + (2,)).astype(np.float32)

    for level in range(max_level, -1, -1):
        warped = remap(gauss_pyr_1[level], g_L[level])
        d_L[level] = lucas_kanade(warped, gauss_pyr_2[level], window_size)
        g_L[level - 1] = 2.0 * expand(
            g_L[level] + d_L[level],
            gauss_pyr_2[level - 1].shape[:2] + (2,),
            interpolation=cv2.INTER_LINEAR,
        )
    return g_L[0] + d_L[0]


def lucas_kanade(img1, img2, window_size):
    img1 = np.copy(img1).astype(np.float32)
    img2 = np.copy(img2).astype(np.float32)

    Ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=5)
    It = img1 - img2

    Ix2 = cv2.GaussianBlur(Ix**2, (window_size, window_size), 0)
    Iy2 = cv2.GaussianBlur(Iy**2, (window_size, window_size), 0)
    Ixy = cv2.GaussianBlur(Ix * Iy, (window_size, window_size), 0)
    Ixt = cv2.GaussianBlur(Ix * It, (window_size, window_size), 0)
    Iyt = cv2.GaussianBlur(Iy * It, (window_size, window_size), 0)

    det = Ix2 * Iy2 - Ixy**2
    u = np.where((det > 1e-6), (Iy2 * Ixt - Ixy * Iyt) / (det + 1e-6), 0)
    v = np.where((det > 1e-6), (Ix2 * Iyt - Ixy * Ixt) / (det + 1e-6), 0)

    optical_flow = np.stack((u, v), axis=2)
    return optical_flow.astype(np.float32)


def apply_magnitude_threshold(flow, threshold):
    magnitude = np.linalg.norm(flow, axis=-1)  # Compute magnitude of flow vectors
    magnitude = magnitude.reshape(magnitude.shape + (1,))  # Reshape to match flow shape
    thresholded_flow = np.where(magnitude < threshold, 0, flow)
    return thresholded_flow


def show_flow(img, flow, num_points_per_axis, scale_factor, filename=None):
    x = np.arange(0, img.shape[1], 1)
    y = np.arange(0, img.shape[0], 1)
    x, y = np.meshgrid(x, y)
    plt.figure(figsize=(10, 10))
    fig = plt.imshow(img, cmap="gray", interpolation="bicubic")
    plt.axis("off")
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    step = int(img.shape[0] / num_points_per_axis)

    plt.quiver(
        x[::step, ::step],
        y[::step, ::step],
        flow[::step, ::step, 0],
        -flow[::step, ::step, 1],  # Reverse sign for correct direction
        color="r",
        pivot="tail",
        headwidth=2,
        headlength=3,
        scale=scale_factor,
    )
    if filename is not None:
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)
        plt.close()  # Close the figure to prevent display


if __name__ == "__main__":
    # input_dir = Path("./input/sphere/")
    # output_dir = Path("./output/sphere")
    # image_seq = []
    # for fname in natsorted(input_dir.rglob("*.ppm")):
    #     image_seq.append(cv2.imread(str(fname), cv2.IMREAD_GRAYSCALE))
    # print(image_seq[0].shape)

    input_mp4 = Path("./input/1920_1080_30fps.mp4")
    output_dir = Path("./output/airplane/")

    MAX_LEVEL = [3, 5]
    NUM_AXIS_PTS = [16, 32, 64]
    ARROW_SCALE = [1]
    MIN_WIN_SIZE = [31]
    FLOW_CUTOFF = [1e-2, 2.5e-2]

    combinations = [
        *itertools.product(
            MAX_LEVEL, NUM_AXIS_PTS, ARROW_SCALE, MIN_WIN_SIZE, FLOW_CUTOFF
        )
    ]
    for (
        max_level,
        num_axis_pts,
        arrow_scale,
        min_win_size,
        flow_cutoff,
    ) in combinations:
        cap = cv2.VideoCapture(str(input_mp4))
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        i = 0
        while 1:
            ret, frame = cap.read()
            if not ret:
                print("No frames grabbed!")
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            flow = hierarchical_lucas_kanade(
                old_gray, frame_gray, max_level, min_win_size
            )
            thresholded_flow = apply_magnitude_threshold(flow, threshold=flow_cutoff)
            show_flow(
                frame,
                thresholded_flow,
                num_axis_pts,
                arrow_scale,
                output_dir
                / f"MAXLEVEL={max_level}_MINWINSIZE={min_win_size}_NUMAXSPTS={num_axis_pts}_ARROWSCALE={arrow_scale}_FLOWCUTOFF={flow_cutoff}"
                / f"{i:02}.png",
            )

            old_gray = frame_gray.copy()

            i += 1

            k = cv2.waitKey(30) & 0xFF
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
