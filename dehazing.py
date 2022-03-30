"""Grayscale Implementation of Single Image Haze Removal Using Dark Channel Prior

Based one:
Bria, A. et al. (2017). Spatial Enhancement by Dehazing for Detection of
Microcalcifications with Convolutional Nets. Image Analysis and
Processing - ICIAP 2017  . ICIAP 2017. Lecture Notes in Computer Science(),
vol 10485. Springer, Cham. https://doi.org/10.1007/978-3-319-68548-9_27
"""

import numpy as np
from cv2.ximgproc import guidedFilter
from scipy.ndimage import minimum_filter


def dehaze(image: np.ndarray, omega: float, window_size: int, radius=40, eps=1e-3):
    """Dehazes given grayscale image

    Args:
        image (np.ndarray): 2D image with normalized intensities to the scale [0,1]
        omega (float): Dehaizing factor controlling the amount of contrast introduced
            in the final dehazed image. Scale:[0,1]
        window_size (int): Window size for the minimum filter (darck prior neighborhood)
        radius (int): Guided filter parameter. Defaults to 40
        eps (float): Guided filter parameter. Defaults to 1e-3

    Returns:
        np.ndarray: Dehazed iamge
    """

    darck_ch = minimum_filter(image, window_size, mode='reflect')

    recovered_image = 1 - (1 - image)/(1 - omega*darck_ch)

    return guidedFilter(recovered_image, recovered_image, radius=radius, eps=eps)
