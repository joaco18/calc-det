import cv2
import numpy as np
import logging
import torch
import torchvision

import pandas as pd
import scipy.ndimage as ndi
import SimpleITK as sitk

from numba import njit
from functools import partial
from itertools import zip_longest
from pathlib import Path
from skimage.measure import label


logging.basicConfig(level=logging.INFO)


def load_point(point_string: str):
    return tuple(
        [int(num) for num in point_string.strip('()').split(', ')]
    )


def load_coords(point_string: str):
    return tuple(
        [load_point(num) for num in point_string.strip('[]').split('), (')]
    )


def load_patch_point(point_string: str):
    return tuple(
        [int(num) for num in point_string.strip('[]').split(' ') if num != '']
    )


def load_patch_coords(point_string: str):
    return tuple(
        [load_patch_point(num) for num in point_string.strip('[]').split(']\n [')]
    )


def z_score_norm(
    img: np.ndarray, mean: float = None, std: float = None, non_zero_region: bool = False
):
    if (mean is None):
        if non_zero_region:
            mean = img[img != 0].mean()
        else:
            mean = img.mean()
    if (std is None):
        if non_zero_region:
            std = img[img != 0].std()
        else:
            std = img.std()
    img = (img - mean) / std
    return img.astype('float32')


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
def patch_coordinates_from_center(center: tuple, image_shape: tuple, patch_size: int):
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


def get_bbox_of_lesions_in_patch(mask: np.ndarray, ignored_lesions: bool = False):
    if ignored_lesions:
        mask = np.where(mask == -1, 255, 0)
        mask = label(mask, background=0, connectivity=1)
    rois_idxs = np.unique(mask)
    lesion_bboxes = []
    for idx in rois_idxs:
        if (idx == 0) or (idx == -1):
            continue
        [y, x] = np.where(mask == idx)
        tl = (x.min(), y.min())
        br = (x.max(), y.max())
        lesion_bboxes.append((tl, br))
    return lesion_bboxes


def peak_local_max(
    image: np.ndarray, footprint: np.ndarray, threshold_abs: float, threshold_rel: float = None,
    num_peaks: int = np.inf, additional_mask: np.ndarray = None
):
    """Finds peaks in an image as coordinate list.
    If both `threshold_abs` and `threshold_rel` are provided, the maximum
    of the two is chosen as the minimum intensity threshold of peaks.
    Based on skimage's function, but modified to be more efficient
    Args:
        image (np.ndarray): 3d image where local peaks are searched
        threshold_abs (float, optional): Minimum intensity of peaks.
        threshold_rel (float, optional): Minimum intensity of peaks,
            calculated as `max(image) * threshold_rel`. Defaults to None.
        num_peaks (int, optional): Maximum number of peaks. When the
            number of peaks exceeds `num_peaks`, return `num_peaks` peaks
            based on highest peak intensity. Defaults to np.inf.
        additional_mask (np.ndarray): Mask to filter the peaks,
            where the mask is zero the peaks are ignored.
    Returns:
        (np.ndarray): (row, column, ...) coordinates of peaks.
    """
    threshold = threshold_abs
    if threshold_rel is not None:
        threshold = max(threshold, threshold_rel * image.max())

    # Non maximum filter
    mask = get_peak_mask(image, footprint, threshold)
    if additional_mask is not None:
        mask = mask * np.where(additional_mask != 0, 1, 0)
    coordinates = get_high_intensity_peaks(
        image, mask, num_peaks)

    return coordinates


def get_peak_mask(image, footprint, threshold):
    """
    Return the mask containing all peak candidates above thresholds.
    """
    image_max = ndi.maximum_filter(
        image, footprint=footprint, mode='constant')
    out = image == image_max
    # no peak for a trivial image
    image_is_trivial = np.all(out)
    if image_is_trivial:
        out[:] = False
    out &= image > threshold
    return out


def get_high_intensity_peaks(image: np.ndarray, mask, num_peaks):
    """
    Return the highest intensity peak coordinates
    """
    # Get coordinates of peaks
    coord = np.nonzero(mask)
    intensities = image[coord]
    # Sort peaks descending order
    idx_maxsort = np.argsort(-intensities)
    coord = np.transpose(coord)[idx_maxsort]
    if len(coord) > num_peaks:
        coord = coord[:num_peaks]
    return coord


def adjust_gamma_float(image: np.ndarray, gamma: float = 1.0):
    """
    Apply gamma adjustment to image in rage [0, 1] float32
    Args:
        image (np.ndarray): _description_
        gamma (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    max_val = image.max()
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    if max_val == 1:
        return image ** invGamma
    # elif max_val == 255:
    #     table = [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    #     table = np.array(table).astype("uint8")
    #     # apply gamma correction using the lookup table
    #     return cv2.LUT(image, table)
    # else:
    #     logging.warning('Datatype not supported, either float or uint8')


def adjust_bbox_to_fit(img_shape: tuple, bbox: tuple, k: int):
    """Adjust the coordinates of a bbox with extra size k, to fit in the image
    Args:
        img_shape (tuple): shape of the image.
        bbox (tuple): ((x1, y1), (x2, y2))
        k (int): extra size.
    Returns:
        tl (tuple): top left coords (x1, y1)
        br (tuple): bottom right coords (x2, y2)
    """
    ((x1, y1), (x2, y2)) = bbox
    patch_size = (x2 - x1 + k, y2 - y1 + k)
    x1 = x1 - k
    x2 = x2 + k
    if x1 < 0:
        x1 = 0
        x2 = patch_size[0]
    y1 = y1 - k
    y2 = y2 + k
    if y1 < 0:
        y1 = 0
        y2 = patch_size[1]
    if x2 > img_shape[1]:
        x2 = img_shape[1]
        x1 = img_shape[1] - patch_size[0]
    if y2 > img_shape[0]:
        y2 = img_shape[0]
        y1 = img_shape[0] - patch_size[1]
    tl = (int(x1), int(y1))
    br = (int(x2), int(y2))
    return tl, br


def non_max_supression(
    detections: np.ndarray, iou_threshold: float = 0.5, return_indexes: bool = False
):
    """Filters the detections bboxes using NMS.
    Args:
        detections (np.ndarray): [x1, x2, y1, y2, score] or [xc, yc, x1, x2, y1, y2, score]
        iou_threshold (float, optional): Iou threshold value. Defaults to 0.5.
        return_indexes (bool, optional): Whether to return the indexes or the filtered detections.
    Returns:
        detections (np.ndarray): [x1, x2, y1, y2, score] or [xc, yc, x1, x2, y1, y2, score]
    """
    if detections.shape[1] == 5:
        x1, x2, y1, y2, score = 0, 1, 2, 3, 4
    elif detections.shape[1] == 7:
        x1, x2, y1, y2, score = 2, 3, 4, 5, 6
    bboxes = np.asarray(
        [detections[:, x1], detections[:, y1], detections[:, x2], detections[:, y2]]).T

    bboxes = torch.from_numpy(bboxes).to(torch.float)
    scores = torch.from_numpy(detections[:, score]).to(torch.float)
    indxs = torchvision.ops.nms(bboxes, scores, iou_threshold=iou_threshold)
    if return_indexes:
        return indxs
    else:
        return detections[indxs, :]


def detections_mask(
    image: np.ndarray, candidates: pd.DataFrame, conf_thr: float = 0.1, k: int = 10
):
    """Labels the candidates and plots them accordingly over the provided image
    Args:
        image (np.ndarray): Image to plot the results
        candidates (pd.DataFrame): [x, y, radius, score]
        conf_thr (float, optional): final threshold to select candidates.
            Only those with confidence higher will be considered for labelling and
            display. Defaults to 0.1.
        k (int, optional): increase in the size of the plotted bboxes.
            Plotted bboxe will have side + k by side + k size. Defaults to 10.
    Returns:
        np.ndarray: binary image with detection bboxes
    """
    candidates = candidates.loc[~(candidates.score < conf_thr), :]

    # format candidate coordinates to plotting format
    get_bbox_from_center = partial(
        patch_coordinates_from_center, image_shape=image.shape, patch_size=14+k)
    centers_it = zip(candidates['x'].values, candidates['y'].values)
    bbox_coordinates = [get_bbox_from_center(center) for center in centers_it]
    candidates[['x1', 'x2', 'y1', 'y2']] = bbox_coordinates
    candidates.drop(columns=['x', 'y'], inplace=True)
    mask = np.zeros(image.shape, dtype='uint8')

    # overlay bboxes
    for _, [_, score, x1, x2, y1, y2] in candidates.iterrows():
        tl = (int(x1), int(y1))
        br = (int(x2), int(y2))
        mask = cv2.rectangle(mask, tl, br, 255, 2)
        bbox_tag = f'{score:.3f}'
        y = tl[1]-15 if (tl[1]-15) > 15 else tl[1]+15
        mask = cv2.putText(
            mask, bbox_tag, (int(x1), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    return mask


def store_as_dcm(
    image: np.ndarray, detections_df: pd.DataFrame, original_dcm_filepath: Path,
    output_filepath: Path, breast_bbox: tuple
):
    """Stores the bboxes mask as dcm image in order to visualize them in dicom viewer
    Args:
        image (np.ndarray): actual breastiamge processed
        detections_df (pd.DataFrame): colums ---> ['x', 'y', 'radius', scores]
        original_dcm_filepath (Path): Path to the original dicom, that can be
            generated from the info kept in the img_df of dataset
        output_filepath (Path): Path to the destiny dcm file
        breast_bbox (tuple): Bbox of the breast region in the original image
    """
    assert original_dcm_filepath.exists(), \
        f'Dcm image missing check the path {original_dcm_filepath}'
    if isinstance(breast_bbox, str):
        breast_bbox = load_coords(breast_bbox)
    assert not isinstance(breast_bbox, (str, list, np.ndarray)), 'Breast bbox wrong, fix it'

    # get the mask of detection bboxes
    detections_df[['x', 'y', 'radius']] = detections_df[['x', 'y', 'radius']].astype('int')
    mask = detections_mask(image, detections_df, conf_thr=0.28785, k=15)

    # turn back the image to the right side if necessary
    dcm_img_name = original_dcm_filepath.name
    if dcm_img_name.split('_')[3] == 'R':
        mask = np.fliplr(mask)

    # read the original dicom to get the positional metadata
    ref_itkimage = sitk.ReadImage(str(original_dcm_filepath))
    ref_itkimage_array = sitk.GetArrayFromImage(ref_itkimage)
    complete_mask = np.zeros(ref_itkimage_array.shape, dtype='uint8')

    # replace the breast bbox region an empty image with the mask
    tl = breast_bbox[0]
    br = breast_bbox[1]
    complete_mask[0, tl[1]:br[1], tl[0]:br[0]] = mask

    # copy the positional metadata
    origin = ref_itkimage.GetOrigin()
    spacing = ref_itkimage.GetSpacing()
    complete_mask = sitk.GetImageFromArray(complete_mask, isVector=False)
    complete_mask.SetSpacing(spacing)
    complete_mask.SetOrigin(origin)

    # write the image
    sitk.WriteImage(complete_mask, str(output_filepath))
