import numpy as np

rainbow_colors = [
    (255, 0, 0),  # Bright Red
    (255, 165, 0),  # Bright Orange
    (255, 255, 0),  # Bright Yellow
    (0, 255, 0),  # Bright Green
    (0, 0, 255),  # Bright Blue
    (125, 0, 255),  # Bright Indigo
    (186, 85, 211),  # Bright Violet
]


def compute_epiline_y_coord(normal, x_coord):
    # normal: 3xN
    a = normal[0, :] / normal[2, :]
    b = normal[1, :] / normal[2, :]
    c = normal[2, :] / normal[2, :]

    return (-1 * c - a * x_coord) / b


def compute_epiline_x_coord(normal, y_coord):
    # normal: 3xN
    a = normal[0, :] / normal[2, :]
    b = normal[1, :] / normal[2, :]
    c = normal[2, :] / normal[2, :]
    return (-1 * c - b * y_coord) / a


def get_epipolar_line(fundamenta_matrix, left_point, right_image):
    n_pts = left_point.shape[1]
    height, width, _ = right_image.shape
    epiline = np.dot(fundamenta_matrix, left_point)
    # epiline = epiline / epiline[2, :] does this line matter?

    y1 = compute_epiline_y_coord(normal=epiline, x_coord=0)
    y2 = compute_epiline_y_coord(normal=epiline, x_coord=width)
    return np.zeros(n_pts), y1, width * np.ones(n_pts), y2


def read_keypoints(file: str):
    pts = np.loadtxt(file)  # (N, 2) or (N, 3)
    num_pts = pts.shape[0]
    return np.concatenate((pts.T, np.ones((1, num_pts))), axis=0)  # (3, N) or (4, N)


def normalize_image_points(image_points, avg_distance=np.sqrt(2)):
    # image_points: (3, N)
    # Return homogenous coord (3, N)
    n_pts = image_points.shape[1]

    # Compute the centroid of the image points
    centroid = np.mean(image_points, axis=1, keepdims=True)

    # Translate image points to the origin
    translated_points = image_points - centroid
    assert translated_points[2, 0] == 0 and translated_points[2, 1] == 0

    # Compute the scale factor
    scale_factor = np.mean(np.linalg.norm(translated_points, axis=0))
    scale_factor = scale_factor / avg_distance

    # # Create the similarity transformation matrix
    scale = np.array(
        [
            [1 / scale_factor, 0, 0],
            [0, 1 / scale_factor, 0],
            [0, 0, 1],
        ]
    )
    matrix = np.array(
        [
            [1, 0, -centroid[0, 0]],
            [0, 1, -centroid[1, 0]],
            [0, 0, 1],
        ]
    )
    matrix = np.dot(scale, matrix)
    return np.dot(matrix, image_points), matrix


def normalize_space_points(space_points, avg_distance=np.sqrt(3)):
    # space_points: (4, N)
    # Return homogenous coord (4, N)
    n_pts = space_points.shape[1]

    # Compute the centroid of the image points
    centroid = np.mean(space_points, axis=1, keepdims=True)

    # Translate image points to the origin
    translated_points = space_points - centroid
    assert translated_points[3, 0] == 0 and translated_points[3, 1] == 0

    # Compute the scale factor
    scale_factor = np.mean(np.linalg.norm(translated_points, axis=0))
    scale_factor = scale_factor / avg_distance

    # Create the similarity transformation matrix
    matrix = np.array(
        [
            [1, 0, 0, -centroid[0, 0]] / scale_factor,
            [0, 1, 0, -centroid[1, 0]] / scale_factor,
            [0, 0, 1, -centroid[2, 0]] / scale_factor,
            [0, 0, 0, 1],
        ]
    )
    return np.dot(matrix, space_points), matrix


def average_distance_to_origin(points):
    # Homogeneous points: (3, N) or (4, N)
    assert points.shape[0] == 3 or points.shape[0] == 4
    n_pts = points.shape[1]
    s = np.sum(np.multiply(points[:-1, :], points[:-1, :]), axis=0)
    dist = np.sqrt(s)
    return np.sum(dist) / n_pts


def plot_line(im, x1, y1, x2, y2, colour):
    """
    Plots a line on a rgb image stored as a numpy array

    Args:
        im: 3D numpy array containing the image values. It may be stored as
            uint8 or float32.
        x1, y1, x2, y2: integer coordinates of the line endpoints
        colour: list of length 3 giving the colour used for the plotted line
            (ie [r, g, b])

    Returns:
        a copy of the input numpy array, with the plotted lines on it. It means
        that the intensities of pixels located on the plotted line are changed.
    """
    # colour points of the line pixel by pixel. Loop over x or y, depending on
    # the biggest dimension.
    if np.abs(x2 - x1) >= np.abs(y2 - y1):
        n = np.abs(x2 - x1)
        for i in range(int(n + 1)):
            x = int(x1 + i * (x2 - x1) / n)
            y = int(np.round(y1 + i * (y2 - y1) / n))
            try:
                im[y, x] = colour
            except IndexError:
                pass
    else:
        n = np.abs(y2 - y1)
        for i in range(int(n + 1)):
            y = int(y1 + i * (y2 - y1) / n)
            x = int(np.round(x1 + i * (x2 - x1) / n))
            try:
                im[y, x] = colour
            except IndexError:
                pass


def draw_epipolar_line_in_image(left_points, right_image, fundamental_matrix):
    ret = right_image.copy()
    # Draw epipolar line of the `point` in the `image` by using fundamental_matrix
    assert fundamental_matrix.shape == (3, 3)
    assert left_points.shape[0] == 3
    n_pts = left_points.shape[1]

    x1, y1, x2, y2 = get_epipolar_line(
        fundamenta_matrix=fundamental_matrix,
        left_point=left_points,
        right_image=ret,
    )

    for i in range(n_pts):
        plot_line(
            im=ret,
            x1=x1[i],
            y1=y1[i],
            x2=x2[i],
            y2=y2[i],
            colour=rainbow_colors[i % len(rainbow_colors)],
        )
    return ret
