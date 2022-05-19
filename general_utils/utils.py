import cv2
import numpy as np
import logging
from numba import njit
from itertools import zip_longest

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


@njit(cache=True)
def patch_coordinates_from_center(
    center: tuple, image_shape: tuple, patch_size: int, use_padding: bool = True,
    image: np.ndarray = None, mask: np.ndarray = None
):
    """Returns coordinates of the patch, cropping the image at center location
    with a given patch size. If the center is in the left or upper border shift
    the center and crop fixed size patch.
    Args:
        center (tuple): (x coordinate, y coordinate)
        image_shape (tuple): shape of image to crop patches from
        patch_size (int, optional): patch size
    Returns:
        x1, x2, y1, y2: coordinates of the patch to crop
    """
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

    if x2 > image_shape[1]:
        x2 = image_shape[1]
        x1 = image_shape[1] - patch_size
    if y2 > image_shape[0]:
        y2 = image_shape[0]
        y1 = image_shape[0] - patch_size
    return x1, x2, y1, y2


def patch_coordinates_from_center_w_padding(
    center: tuple, image_shape: tuple, patch_size: int,
    image: np.ndarray = None, mask: np.ndarray = None
):
    """Returns coordinates of the patch, cropping the image at center location
    with a given patch size. If the center is in the left or upper border shift
    the center and crop fixed size patch. If the center is in the right or bottom
    border do the same pad with zeros and crop
    Args:
        center (tuple): (x coordinate, y coordinate)
        image_shape (tuple): shape of image to crop patches from
        patch_size (int, optional): patch size
    Returns:
        x1, x2, y1, y2: coordinates of the patch to crop
        image (np.ndarray, optional): padded image
        mask (np.ndarray, optional): padded mask
    """
    assert (image is not None) and (mask is not None), \
        'If padding method is used, image and mask should be provided,' \
        ' utils.patch_coordinates_from_center_w_padding'
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
    return int_img


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
    return diag_int_img[1:-1, 1:-1]


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


def crop_patch_around_center(patch_x1, patch_x2, patch_y1, patch_y2, center_crop_size):
    """Calculates coordinates of the crop around center of the given patch of given size.

    Args:
        patch_x1 (int): patch coordinate
        center_crop_size (int): size of the croppes patch

    Returns:
        tuple: center_px1, center_px2, center_py1, center_py2
    """
    p_center_y = patch_y1 + (patch_y2 - patch_y1)//2
    p_center_x = patch_x1 + (patch_x2 - patch_x1)//2
    center_py1 = p_center_y - center_crop_size//2
    center_py2 = p_center_y + center_crop_size//2 + center_crop_size % 2
    center_px1 = p_center_x - center_crop_size//2
    center_px2 = p_center_x + center_crop_size//2 + center_crop_size % 2
    return center_px1, center_px2, center_py1, center_py2


@njit(cache=True)
def img_to_patches_array(image: np.ndarray, candidates: np.ndarray, patch_size: int):
    """Crop the fix size patches arround the detections and stack them in an array
    Args:
        image (np.ndarray): image to process
        candidates (np.ndarray): [x,y,radius]
        patch_size (int): patch size to use
    Returns:
        np.ndarray: [n_patches, patch_size, patch_size]
    """
    images = np.empty((len(candidates), patch_size, patch_size))
    for j, coords in enumerate(candidates):
        # Get the patch arround center
        x1, x2, y1, y2 = patch_coordinates_from_center(
            center=(coords[0], coords[1]), image_shape=image.shape,
            patch_size=patch_size)
        images[j, :, :] = image[y1:y2, x1:x2]
    return images


@njit(cache=True)
def get_an_example_array(image: np.ndarray, patch_size: int, candidates: np.ndarray):
    """Given an image, generate an array of fixed size patches centered on the candidates.
    Args:
        image (np.ndarray): image to process
        patch_size (int): indx
        candidates (np.ndarray): [x, y, radius]
    Returns:
        np.ndarray: array of patches [n_patches, size, size]
    """
    # candidate selection
    images = img_to_patches_array(image, candidates, patch_size)
    return images


def blockwise_retrieval(t, size=2, fillvalue=None):
    it = iter(t)
    return zip_longest(*[it]*size, fillvalue=fillvalue)


def get_bbox_of_lesions_in_patch(mask):
    rois_idxs = np.unique(mask)
    lesion_bboxes = []
    for idx in rois_idxs:
        if idx == 0:
            continue
        [y, x] = np.where(mask == idx)
        tl = (x.min(), y.min())
        br = (x.max(), y.max())
        lesion_bboxes.append((tl, br))
    return lesion_bboxes