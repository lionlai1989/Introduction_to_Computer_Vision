import numpy as np
import matplotlib.pyplot as plt


def flow_to_color(flow, hsv):
    """The code snippet is from OpenCV tutorial.
    https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
    """
    import cv2

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def visualize_polynomial_expansion(image, A, B, C, out_path=None):
    """Visualize polynomial expansion A, B and C associate to `image`.

    Args:
        image: (height, width, 3)
        A: (height, width, 2, 2)
        B: (height, width, 2)
        C: (height, width)
    """

    # Calculate eigenvalues for each 2x2 matrix in A
    eigenvalues = np.linalg.eigvals(A)
    # Sum of eigenvalues magnitude as a visualization metric
    eigen_A = np.sum(np.abs(eigenvalues), axis=-1)

    # B[..., 0] contains x components, B[..., 1] contains y components
    height, width, *_ = B.shape
    # Subsample the array by a factor of 25%
    subsample_ratio = 0.25
    subsampled_height = int(height * subsample_ratio)
    subsampled_width = int(width * subsample_ratio)
    subsampled_B = B[::4, ::4]  # Subsample 25% in both dimensions

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    fig.suptitle("Visualization of polynomial expansion")

    ax[0, 0].set_title("Image")
    ax[0, 0].imshow(image, aspect="equal")
    ax[0, 0].set_axis_off()

    ax[0, 1].set_title("Quadratic: A")
    ax[0, 1].imshow(eigen_A, cmap="hot", aspect="equal")
    ax[0, 1].set_axis_off()

    ax[1, 0].set_title("Linear: B")
    ax[1, 0].quiver(
        np.linspace(0, width, subsampled_width),
        np.linspace(height, 0, subsampled_height),
        subsampled_B[..., 0],
        subsampled_B[..., 1],
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    ax[1, 0].set_xlim([0, width])  # Explicitly set x limits
    ax[1, 0].set_ylim([0, height])  # Explicitly set y limits
    ax[1, 0].set_aspect(
        "equal", adjustable="box"
    )  # Set aspect ratio for the quiver subplot
    ax[1, 0].set_axis_off()

    ax[1, 1].set_title("Constant: C")
    ax[1, 1].imshow(C, cmap="gray", aspect="equal")
    ax[1, 1].set_axis_off()

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path)
    else:
        plt.show()
