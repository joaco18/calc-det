import numpy as np
import cv2


def dehaze(image: np.ndarray, omega: float, window_size: int, radius=40, eps=1e-3):
    """Dehazes given grayscale image
    Args:
        image (np.ndarray): 2D image with normalized intensities to the scale [0,1].
            Should be of type: np.float32 | np.uint8 | np.uint16
        omega (float): Dehaizing factor controlling the amount of contrast introduced
            in the final dehazed image. Scale:[0,1]
        window_size (int): Window size for the minimum filter (darck prior neighborhood)
        radius (int): Guided filter parameter. Defaults to 40
        eps (float): Guided filter parameter. Defaults to 1e-3
    Returns:
        np.ndarray: Dehazed iamge
    """
    # minimum filtering OpenCV
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    darck_ch = cv2.erode(image, kernel)

    recovered_image = 1 - (1 - image)/(1 - omega*darck_ch)
    filtered_image = cv2.ximgproc.guidedFilter(recovered_image, recovered_image,
                                               radius=radius, eps=eps)
    return filtered_image
