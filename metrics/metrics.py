import cv2
import numpy as np
import math


def circle_comparison(predicted_roi_circles, mask, return_counts=True):
    """Finds TP, FP and FN among predicted circles based on the true image mask.

    More precise circle comparison. Checks the intersection of each predicted circle
    with the true leision bbox and counts a TP if >=1 roi pixel is in the circle.

    Args:
        true_bboxes (np.ndarray): Array of shape (n_rois, 2) containing
            top_left and bottom_right bbox coordinates in tuples
        predicted_roi_circles (np.ndarray): Array of shape (n_predicted_circ, 3)
            with circle_x, circle_y and circle_radius values
        mask (np.ndarray): Image mask containing indexes of rois
        return_counts (bool, optional): Whether to return counts of TP, FP, FN arrays
            or corresponding indexes. Defaults to True.

    Returns:
        If return_counts=True:
            (TP, FP, FN) (tuple[int]): number of TP, FP, FN
        else:
            TP (set): contains indexes of ROIs correctly selected by predicted circles
            FP (set): contains indexes of circles that weren't mapped to any roi
            FN (set): indexes of ROIs not mapped to any of the predicted circles
    """
    TP = set()
    FP = []

    # set containing background pixel value in the mask
    back_set = set([0])

    for circle_idx, circle in enumerate(predicted_roi_circles.astype(int)):
        circle_roi_mask = cv2.circle(np.zeros(mask.shape),
                                     (circle[0], circle[1]),
                                     circle[2], 1, -1).astype(np.bool8)

        # retrieving roi idxs from the mask from the circle
        mapped_rois_idxs = set(
            np.unique(mask[circle_roi_mask])).difference(back_set)
        if len(mapped_rois_idxs) > 0:
            TP = TP.union(mapped_rois_idxs)
        else:
            FP.append(circle_idx)
    FN = set(mask.ravel()).difference(back_set).difference(TP)

    if return_counts:
        return (len(TP), len(FP), len(FN))
    else:
        return TP, FP, FN


def quick_circle_comparison(predicted_roi_circles, mask, return_counts=True):
    """Finds TP, FP and FN among predicted circles based on the true image mask.

    Quick version that checks if a rectangular bbox around each circle in the image
    mask contains any roi indexes and counts a TP if >= 1 roi pixel is in that bbox.

    Args:
        true_bboxes (np.ndarray): Array of shape (n_rois, 2) containing
            top_left and bottom_right bbox coordinates in tuples
        predicted_roi_circles (np.ndarray): Array of shape (n_predicted_circ, 3)
            with circle_x, circle_y and circle_radius values
        mask (np.ndarray): Image mask containing indexes of rois
        return_counts (bool, optional): Whether to return counts of TP, FP, FN arrays
            or corresponding indexes. Defaults to True.

    Returns:
        If return_counts=True:
            (TP, FP, FN) (tuple[int]): number of TP, FP, FN
        else:
            TP (set): contains indexes of ROIs correctly selected by predicted circles
            FP (set): contains indexes of circles that weren't mapped to any roi
            FN (set): indexes of ROIs not mapped to any of the predicted circles
    """
    TP = set()
    FP = []

    # set containing background pixel value in the mask
    back_set = set([0])

    for circle_idx, circle in enumerate(predicted_roi_circles.astype(int)):
        # finds a bbox around each circles and checks if in the mask it contains any rois
        cricle_tl = (max(0, circle[0] - circle[2]),
                     max(0, circle[1] - circle[2]))
        circle_br = (min(mask.shape[1], circle[0] + circle[2]),
                     min(mask.shape[0], circle[1] + circle[2]))

        # looks at pixels in the mask at the circle's bbox
        intersected_mask_idxs = set(
            mask[cricle_tl[1]:circle_br[1], cricle_tl[0]:circle_br[0]].ravel()).difference(back_set)
        if len(intersected_mask_idxs) > 0:
            TP = TP.union(intersected_mask_idxs)
        else:
            FP.append(circle_idx)
    TP = set(TP).difference(back_set)
    FP = set(FP)
    FN = set(mask.ravel()).difference(set([0])).difference(TP)

    if return_counts:
        return (len(TP), len(FP), len(FN))
    else:
        return TP, FP, FN


def create_binary_mask_from_blobs(
    original_img_shape: tuple, blobs_x_y_sigma: np.ndarray, n_jobs: int=4
):
    blobs_x_y_sigma = blobs_x_y_sigma.astype('int')
    sigmas = np.unique(blobs_x_y_sigma[:, 2])
    # all_detections_mask = np.zeros(original_img_shape)
    all_detections_mask = []
    for sigma in sigmas:
        mask = np.zeros(original_img_shape)
        coords = blobs_x_y_sigma[np.where(blobs_x_y_sigma[:, 2] == sigma)]
        mask[coords[:, 0], coords[:, 1]] = 1

        radius = int(math.sqrt(2)*sigma)
        kernel_size = 2*radius
        if kernel_size % 2 != 1:
            kernel_size += 1
        template_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))
        mask = cv2.dilate(mask, template_circle)
        all_detections_mask = mask & all_detections_mask
    return all_detections_mask


def cal_det_se(no_tp, no_fn):
    return no_tp/(no_tp+no_fn)


def fp_per_unit_area(image_shape, no_fp):
    return no_fp/(image_shape[0] * image_shape[1] * (0.070**2)/100)
