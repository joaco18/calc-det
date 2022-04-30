import cv2
import numpy as np


def load_point(point_string: str, dtype: str = 'float'):
    return tuple(
        [int(num) for num in point_string.strip('()').split(', ')]
    )


def load_coords(point_string: str, dtype: str = 'float'):
    return tuple(
        [load_point(num, 'int') for num in point_string.strip('[]').split('), (')]
    )


def z_score_norm(img: np.ndarray, mean: float = None, std: float = None):
    if (mean is None) and (std is None):
        mean, std = cv2.meanStdDev(img)
    img = (img - mean) / std
    # TODO: Decide what to do with the floats
    return img


def min_max_norm(img: np.ndarray, max_val: int = None):
    """
    Scales images to be in range [0, 2**bits]

    Args:
        img (np.ndarray): Image to be scaled.
        max_val (int, optional): Value to scale images
            to after normalization. Defaults to None.

    Returns:
        np.ndarray: Scaled image with values from [0, max_val]
    """
    if max_val is None:
        max_val = np.iinfo(img.dtype).max
    img = (img - img.min()) / (img.max() - img.min()) * max_val
    return img

def sobel_gradient(img):
    """ Calculates Sobel gradient magnitude of the image.
    
    Uses np.sqrt(grad_x**2 + grad_y**2) for magniture evaluation
    """
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    return grad

def crop_center_coords(cx, cy, image, patch_size=100):
    """Returns coordinates of the patch, cropping the image at cx,cy location
    with a given patch size.
    
    Crops to image boundaries if patch excceeds image dimensions.

    Args:
        cx (int): x coordinate
        cy (int): y coordinate
        image (np.ndarray): image to crop patches from
        patch_size (int, optional): patch size. Defaults to 100.

    Returns:
        tuple[int]: (x1, x2, y1, y2) coordinates of the patch to crop
    """
    x1 = max(0, cx-patch_size)
    x2 = min(image.shape[1], cx+patch_size)
    y1 = max(0, cy-patch_size)
    y2 = min(image.shape[0], cy+patch_size)
    return x1, x2, y1, y2