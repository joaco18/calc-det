import cv2
import numpy as np
import logging
from numba import njit

logging.basicConfig(level=logging.INFO)


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


def get_center_bbox(pt1, pt2):
    """Gets the center of the bbox given by the tl and br inputed points tuples
    Returns:
        (tuple): (xc, yc)
    """
    xc = pt1[0] + (pt2[0] - pt1[0]) // 2
    yc = pt1[1] + (pt2[1] - pt1[1]) // 2
    return (xc, yc)


def get_center_bboxes(bboxes: np.ndarray):
    """Applies the previous function across all the bboxes present the array
    Args:
        bboxes (np.ndarray): array with bboxes as rows [(x0,y0), (x1,y1)]
    Returns:
        (np.ndarray): array with centers as rows [(xc,yc)]
    """
    return np.asarray([get_center_bbox(bbox[0], bbox[1]) for bbox in bboxes])


@njit(cache=True)
def our_hist_numba(vector, bins):
    freqs = np.zeros((bins-1,))
    for i in range(len(vector)):
        freqs[vector[i]] += 1
    return np.arange(0, bins-1), freqs


@njit(cache=True)
def get_trianglular_threshold(histogram: np.ndarray):
    # Get the left and rightmost nonzero values of the histogram
    idxs = np.nonzero(histogram)[0]
    minv = idxs.min()
    maxv = idxs.max()

    # Reduce min by one increase max by one to cactch the ends
    minv = minv - 1 if minv > 0 else minv
    minv2 = maxv + 1 if maxv < histogram.size else maxv

    # Find whether the histogram is left or right skewed
    maxv = np.argmax(histogram)
    inverted = False
    if (maxv - minv) < (minv2 - maxv):
        inverted = True
        histogram = np.flip(histogram)
        minv = histogram.size - 1 - minv2
        maxv = histogram.size - 1 - maxv

    if minv == maxv:
        return minv

    nx = histogram[maxv]
    ny = minv - maxv
    d = np.sqrt(nx*nx + ny*ny)
    nx /= d
    ny /= d
    d = nx * minv + ny * histogram[minv]

    # find the split point
    split = minv
    split_distance = 0
    for i in range(minv + 1, maxv + 1):
        new_distance = nx * i + ny * histogram[i] - d
        if new_distance > split_distance:
            split = i
            split_distance = new_distance
    split -= 1

    # reverse back the histogram
    if inverted:
        histogram = np.flip(histogram)
        return histogram.size - 1 - split
    else:
        return split
