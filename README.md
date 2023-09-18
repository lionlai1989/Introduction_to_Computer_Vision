# Introduction to Computer Vision

In the era of deep learning, neural networks have revolutionized computer vision,
demonstrating their effectiveness in solving complex visual problems. Nevertheless, I
believe that conventional methods can still contribute to algorithmic improvement. For
example, [YOLO](https://arxiv.org/abs/1506.02640), a neural network-based object
detection method, can recognize objects in images but treats each image independently
without considering temporal connections. However, I propose that introducing a
continuous-time aspect, such as vehicle dynamics, into the detection process could
enhance precision and efficiency. This repository reflects my efforts to reacquaint
myself with traditional computer vision methodology.

## Description

This repository is organized by various computer vision topics. While some topics may
have connections (e.g., SIFT and RANSAC), they are discussed separately in different
sections for pedagogical clarity. I utilize open-source Python packages like Numpy,
Pillow, Scipy, scikit-image, etc. as needed, but it's important to note that
**`OpenCV`** is intentionally prohibited. The aim here is to foster a deep understanding
of the mathematical foundations behind each topic by implementing them from scratch.

### Table of Contents

-   [Images as Functions](https://htmlpreview.github.io/?https://github.com/lionlai1989/Introduction_to_Computer_Vision/blob/master/00-Images_as_Functions/images_as_functions.html):
    Images are not just collections of pixel values; they can be represented as
    mathematical functions $f(x, y)$. This representation forms the foundation for
    various mathematical operations, including filtering, Laplace/Fourier transforms,
    and convolution. The initial notebook within this repository introduces the notation
    used and demonstrates fundamental image operations such as reading, writing,
    addition, subtraction, and standardization.

<div style="text-align:center">
  <img src="./00-Images_as_Functions/images/green_channel_original.png" alt="Your Image Description" width="196" height="196">
  <img src="./00-Images_as_Functions/images/green_channel_standardized.png" alt="Your Image Description" width="196" height="196">
  <p style="font-size: 14px; color: #777;">Left: original image. Right: image after standardization. Can you explain that why it looks so gray?</p>
</div>

-   [Hough Transform]():
-   [Window-based Stereo Matching]():
-   [Stereo Geometry]():
-   [Lucas-Kanade Optical Flow]():

## Getting Started

All the results in Jupyter Notebook can be reproduced by following the instructions
below.

### Dependencies

Before you start, you need to make sure that **Python3** is installed. **Python-3.10**
is used throughout all the developments to the problems.

### Downloading

-   To download this repository, run the following command:

```shell
git clone https://github.com/lionlai1989/Introduction_to_Computer_Vision.git
```

### Install Python Dependencies

-   Create and activate a Python virtual environment

```
python3.10 -m venv venv_computer_vision && source venv_computer_vision/bin/activate
```

-   Update `pip` and `setuptools`:

```
python3 -m pip install --upgrade pip setuptools
```

-   Install required Python packages in `requirements.txt`.

```
python3 -m pip install -r requirements.txt
```

### Running Jupyter Notebook

Now you are ready to go to each Jupyter Notebook and run the code. Please remember to
select the kernel you just created in your virtual environment `venv_computer_vision`.

## Contributing

Any feedback, comments, and questions about this repository are welcome.

## Authors

[@lionlai](https://github.com/lionlai1989)

## Version History

-   0.0.1
    -   Initial Release

## Acknowledgments

The materials in this repository are primarily sourced from two courses: 'Introduction
to Computer Vision' offered by Udacity and Georgia Tech, and 'Computer Vision' from ETH
Zürich. It's essential to emphasize that I do not possess any ownership rights over the
materials included in this repository. For a more in-depth exploration of the original
course contents, please refer to the following links:

-   Georgia Tech:

    -   https://www.udacity.com/course/introduction-to-computer-vision--ud810
    -   https://docs.google.com/spreadsheets/d/1ecUGIyhYOfQPi3HPXb-7NndrLgpX_zgkwsqzfqHPaus/pubhtml

-   ETH Zürich:
    -   https://cvg.ethz.ch/teaching/compvis/
