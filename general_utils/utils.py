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


def crop_center_coords(cx, cy, image_shape, patch_size=100):
    """Returns coordinates of the patch, cropping the image at cx,cy location
    with a given patch size.

    Crops to image boundaries if patch excceeds image dimensions.

    Args:
        cx (int): x coordinate
        cy (int): y coordinate
        image_shape (tuple): image shape
        patch_size (int, optional): patch size. Defaults to 100.

    Returns:
        tuple[int]: (x1, x2, y1, y2) coordinates of the patch to crop
    """
    x1 = max(0, cx-patch_size)
    x2 = min(image_shape[1], cx+patch_size)
    y1 = max(0, cy-patch_size)
    y2 = min(image_shape[0], cy+patch_size)
    return x1, x2, y1, y2


def patch_coordinates_from_center(
    center: tuple, image_shape: tuple, patch_size: int, use_padding: bool = True,
    image: np.ndarray = None, mask: np.ndarray = None
):
    """Returns coordinates of the patch, cropping the image at center location
    with a given patch size. If the center is in the left or upper border shift 
    the center and crop fixed size patch. If the center is in the left bottom border
    do the same if use_padding=False, or pad with zeros and crop

    Crops to image boundaries if patch excceeds image dimensions.

    Args:
        center (tuple): (x coordinate, y coordinate)
        image_shape (tuple): shape of image to crop patches from
        patch_size (int, optional): patch size
        use_padding (bool): If the center falls in right bottob border, optionally
            padd the image and give the centered patch, if False, shift the center
            to fit inside the image
    Returns:
        x1, x2, y1, y2: coordinates of the patch to crop
        if use_padding:
            image (np.ndarray, optional): padded image
            mask (np.ndarray, optional): padded mask
    """
    if use_padding:
        assert (image is not None) and (mask is not None), \
            'If padding method is used, image and mask should be provided,' \
                ' utils.patch_coordinates_from_center'
    patch_half_size = patch_size // 2

    x1 = center[0] - patch_half_size
    x2 = center[0] + patch_size - patch_half_size
    if x1 < 0:
        x1 = 0
        x2 = patch_size

    y1 = center[1] - patch_half_size
    y2 = center[1] + patch_size - patch_half_size
    if y1 < 0:
        y1 = 0
        y2 = patch_size

    if not use_padding:
        if x2 > image_shape[1]:
            x2 = image_shape[1]
            x1 = image_shape[1] - patch_size
        if y2 > image_shape[0]:
            y2 = image_shape[0]
            y1 = image_shape[0] - patch_size
        return x1, x2, y1, y2
    else:
        if x2 > image_shape[1]:
            image = np.pad(image, ((0, 0), (0, patch_size)), mode='constant', constant_values=0)
            mask = np.pad(mask, ((0, 0), (0, patch_size)), mode='constant', constant_values=0)
        if y2 > image_shape[0]:
            image = np.pad(image, ((0, patch_size), (0, 0)), mode='constant', constant_values=0)
            mask = np.pad(mask, ((0, patch_size), (0, 0)), mode='constant', constant_values=0)
        return x1, x2, y1, y2, image, mask


@njit(cache=True)
def integral_img(img_arr):
    shape = img_arr.shape
    row_sum = np.zeros(shape)
    int_img = np.zeros((shape[0] + 1, shape[1] + 1))
    for x in range(shape[1]):
        for y in range(shape[0]):
            row_sum[y, x] = row_sum[y-1, x] + img_arr[y, x]
            int_img[y+1, x+1] = int_img[y+1, x-1+1] + row_sum[y, x]
    return int_img.astype(np.int8)


@njit(cache=True)
def diagonal_integral_img(img_arr):
    shape = img_arr.shape
    diag_int_img = np.zeros((shape[0] + 3, shape[1] + 3))
    img_arr_ = np.zeros((shape[0] + 1, shape[1] + 2))
    img_arr_[1:, 1:-1] = img_arr
    for y in range(shape[0]):
        for x in range(img_arr_.shape[1]-1):
            diag_int_img[y+2, x+2] = \
                diag_int_img[y+1, x+1] + diag_int_img[y+1, x+3] - \
                diag_int_img[y, x+2] + img_arr_[y+1, x+1] + \
                img_arr_[y, x+1]
            diag_int_img[y+2, 1] = diag_int_img[y+1, 2]
            diag_int_img[y+2, -1] = diag_int_img[y+1, -2]
    return diag_int_img[1:-1, 1:-1].astype(np.int8)


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
