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


def lucas_kanade_point(img1, img2, pt, win_size):
    patch1 = cv2.getRectSubPix(img1, (win_size, win_size), tuple(pt))
    patch2 = cv2.getRectSubPix(img2, (win_size, win_size), tuple(pt))
    Ix = cv2.Sobel(patch1, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(patch1, cv2.CV_64F, 0, 1, ksize=5)
    It = patch2.astype(np.float64) - patch1.astype(np.float64)
    A = np.vstack((Ix.ravel(), Iy.ravel())).T
    b = -It.ravel()
    ATA = A.T @ A
    ATb = A.T @ b
    if np.linalg.cond(ATA) < 1e6:
        flow = np.linalg.solve(ATA, ATb)
    else:
        flow = np.zeros(2, dtype=np.float32)
    return flow.astype(np.float32)


def warp_patch(img, pt, flow, win_size):
    # Extract the patch
    patch = cv2.getRectSubPix(img, (win_size, win_size), tuple(pt))
    w = win_size
    gx, gy = np.meshgrid(np.arange(w), np.arange(w))

    # Shift *backwards* by the current flow
    map_x = (gx - flow[0]).astype(np.float32)
    map_y = (gy - flow[1]).astype(np.float32)

    warped = cv2.remap(
        patch,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return warped


def hierarchical_lk_point(pt0, img1, img2, max_level, win_size):
    # build pyramids
    pyr1, pyr2 = [img1], [img2]
    for _ in range(max_level):
        pyr1.append(cv2.pyrDown(pyr1[-1]))
        pyr2.append(cv2.pyrDown(pyr2[-1]))

    flow = np.zeros(2, dtype=np.float32)
    pt = np.array(pt0, dtype=np.float32) / (2**max_level)

    for level in range(max_level, -1, -1):
        I1 = pyr1[level]  # template
        I2 = pyr2[level]  # new frame

        # Warp and extract patches
        patch1_warp = warp_patch(I1, pt, flow / (2**level), win_size)
        patch2 = cv2.getRectSubPix(I2, (win_size, win_size), tuple(pt))

        # Compute residual flow at the patch center
        center = (win_size // 2, win_size // 2)
        d_flow = lucas_kanade_point(patch1_warp, patch2, center, win_size)

        # Accumulate
        flow = flow * 2 + d_flow
        pt += d_flow

    return flow


def track_and_write(video_path, output_path, init_pt, max_level, win_size):
    """
    Track a single point through a video, write out tracking visualization.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read first frame")
    h, w = frame.shape[:2]

    # Use Motion JPEG codec
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(output_path), fourcc, 30.0, (w, h))

    # Setup tracking
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    pt = np.array(init_pt, dtype=np.float32)
    traj = [pt.copy()]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Compute flow and update point
        flow = hierarchical_lk_point(pt, prev_gray, gray, max_level, win_size)
        pt += flow
        traj.append(pt.copy())

        # Draw trajectory
        vis = frame.copy()
        for i in range(1, len(traj)):
            cv2.line(
                vis,
                tuple(traj[i - 1].astype(int)),
                tuple(traj[i].astype(int)),
                (0, 255, 0),
                2,
            )

        # Draw current point
        cv2.circle(vis, tuple(pt.astype(int)), 4, (0, 0, 255), -1)

        # Write frame
        writer.write(vis)
        prev_gray = gray

    cap.release()
    writer.release()
    print(f"Successfully wrote video to {output_path}")


if __name__ == "__main__":
    input_mp4 = Path("./input/1920_1080_30fps.mp4")
    output_dir = Path("./output/airplane")
    output_dir.mkdir(parents=True, exist_ok=True)

    init_pt = (1676, 654)  # initial point to track

    MAX_LEVELS = [3, 4, 5, 6]
    WIN_SIZES = [21, 41, 61, 81, 101]

    for max_lev, win in itertools.product(MAX_LEVELS, WIN_SIZES):
        out_file = output_dir / f"traj_L{max_lev}_W{win}.avi"

        print(f"Tracking with max_level={max_lev}, win_size={win} → {out_file.name}")

        track_and_write(input_mp4, out_file, init_pt, max_lev, win)

    print("All done.")
