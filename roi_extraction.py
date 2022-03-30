import numpy as np
import pandas as pd
from cv2 import BORDER_CONSTANT, copyMakeBorder
from skimage.util import montage
from skimage.util.shape import view_as_windows

LEISION_TYPES = ['asymmetry', 'calcification',
                 'cluster', 'distortion', 'mass', 'normal']


def extract_ROIs(image: np.ndarray, mask: np.ndarray, roi_df: pd.DataFrame, window_size: int, stride: int, case_id: str, img_id: int):
    """Extracts ROIs and returns their description for given image and mask.

    Args:
        image (np.ndarray): Image to slice
        mask (np.ndarray): Mask corresponding to the image
        roi_df (pd.DataFrame): DataFrame describing each roi of the database
        window_size (int): Window size
        stride (int): Stride

    Returns:
        np.ndarray: image_slices array of shape (n_patches, window_size, window_size)
        np.ndarray: mask_slices array of shape (n_patches, window_size, window_size)
        pd.DataFrame: slices_descr describing each ROI.
    """

    if image.shape[0] % window_size != 0 or image.shape[1] % window_size != 0:
        print("[WARNING] Loosing information.")
        print("Image shapes should be divible by window_size. Consider padding")

    # slice equally image and the mask
    image_slices = slice_image(image, window_size=window_size, stride=stride)
    mask_slices = slice_image(mask, window_size=window_size, stride=stride)

    # count number of classes per slice
    index_freqs = [dict(zip(*np.unique(m, return_counts=True))) for m in mask_slices]
    slices_descr = pd.DataFrame(index_freqs)

    image_rois = roi_df[(roi_df['case_id'] == case_id) & (roi_df['img_id'] == img_id)]

    breast_pixels = np.array([(p != 0).sum() for p in image_slices])

    # mapping indexes in image to type name
    rois_types = pd.Series(image_rois.lesion_type.values,
                           index=image_rois.index_in_image).to_dict()
    rois_types[0] = "normal"
    slices_descr = slices_descr.rename(columns=rois_types)

    # grouping rois with the same type
    slices_descr = slices_descr.groupby(lambda x: x, axis=1).sum() / \
        (image_slices.shape[1]*image_slices.shape[2])

    # select only rois from LEISION_TYPES
    slices_descr = slices_descr[slices_descr.columns.intersection(LEISION_TYPES)]

    # standartize df to habe always same number and types of columns
    for lt in LEISION_TYPES:
        if lt not in slices_descr.columns:
            slices_descr[lt] = 0

    slices_descr['breast'] = breast_pixels / \
        (image_slices.shape[1]*image_slices.shape[2])

    # calculating patches coordinates
    row_num, col_num, _, __ = view_as_windows(image, window_size, stride).shape
    bbox_coordinates = []
    for col in range(row_num):
        row_idx = [((row*stride, col*stride),
                    (window_size+row*stride,
                    window_size+col*stride)) for row in range(col_num)]
        bbox_coordinates.extend(row_idx)
    slices_descr['bbox'] = bbox_coordinates

    return image_slices, mask_slices, slices_descr


def padd_image(image: np.ndarray, window_size: int, padd_type=BORDER_CONSTANT):
    """Padds given image with '0' on the bottom and right sides to ensure safe slicing
    of the image with a given window size and any stride multiple of it

    Args:
        image (np.ndarray): Image array to be padded. Expects 2D images.
        window_size (int): Window size of slicing.
        padd_type (int, optional): OpenCV padding. Defaults to cv2.BORDER_CONSTANT.


    Returns:
        np.ndarray: Padded array
    """

    padded_x_len = np.ceil(image.shape[0] / window_size) * window_size - image.shape[0]
    padded_y_len = np.ceil(image.shape[1] / window_size) * window_size - image.shape[1]

    return copyMakeBorder(image, 0, int(padded_x_len), 0, int(padded_y_len), padd_type)


def slice_image(image: np.ndarray, window_size: int, stride: int):
    """Slices image with a sliding window of given size and stride

    Args:
            image (np.ndarray): Image to be sliced.
            window_size (int): Window size. Should be divisible by stride.
            stride (int): Stride. Must be multiple of the window_size.

    Returns:
            np.ndarray: Numpy view of the sliced image patches
                        of shape (n_patches, w_size, w_size)
    """
    if window_size % stride != 0:
        raise(ValueError("Window size must be a multiple of stride"))
    img_pathces = view_as_windows(image, window_size, stride)
    return img_pathces.reshape((img_pathces.shape[0] * img_pathces.shape[1],
                                img_pathces.shape[2], img_pathces.shape[3]))


def reconstruct_patches(patches: np.ndarray, original_image: np.ndarray, window_size: int, stride: int):
    """Reconstructs images from an array of image patches obtained
    by slicing the image with a sliding window of given size and stride

    Args:
            patches (np.ndarray): Image patches of shape (n_patches, w_size, w_size).
            original_image (np.ndarray): Original image used for slicing
            window_size (int): Size of the window used for slicing
            stride (int): Stride of the window used for slicing

    Returns:
            np.ndarray: Reconstruced image with shape equal to original_image
    """
    if stride == window_size:
        patched_image_shape = view_as_windows(
            original_image, window_size, stride).shape
    else:
        # get the original slicing grid for window=stridetride
        patched_image_shape = view_as_windows(
            original_image, window_size, window_size).shape
        # reconstruct patches for the split window=stridetride
        patches = destride_array(patches, original_image, window_size, stride)

    reconstructed_image = montage(patches, grid_shape=(
        patched_image_shape[0], patched_image_shape[1]))
    return reconstructed_image


def destride_array(patches: np.ndarray, original_image: np.ndarray, window_size: int, stride: int):
    """Removes overlapping patches after slicing  an image with a stride < window_size.

    Args:
            patches (np.ndarray): Array of patches of shape (n_patches, w_size, w_size)
            original_image (np.ndarray): Image that was sliced into patches
            window_size (int): Window size used during patching
            stride (int): Stride of the patching window

    Returns:
            np.ndarray: Array of non-overlapping image patches
    """
    striding_factor = int(window_size/stride)
    if striding_factor > 1:
        patched_image_shape = view_as_windows(
            original_image, window_size, stride).shape
        mask = np.zeros((patched_image_shape[0], patched_image_shape[1]))
        mask[::striding_factor, ::striding_factor] = True
        return patches[mask.flatten().astype(bool)]
    else:
        return patches
