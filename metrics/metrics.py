import cv2
import math

import numpy as np
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm
from functools import partial

from metrics.metrics_utils import non_maximum_suppression_w_labels


def circle_comparison(predicted_roi_circles, mask, return_counts=True):
    """Finds TP, FP and FN among predicted circles based on the true image mask.
    More precise circle comparison. Checks the intersection of each predicted circle
    with the true leision bbox and counts a TP if >=1 roi pixel is in the circle.
    Args:
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
            mask[cricle_tl[1]:circle_br[1], cricle_tl[0]:circle_br[0]].ravel()
        ).difference(back_set)
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


def create_binary_mask_from_blobs(shape: tuple, blobs_x_y_sigma: list):
    img_binary_blobs = np.zeros(shape)
    for blob in blobs_x_y_sigma:
        img_binary_blobs = cv2.circle(
            img_binary_blobs, (blob[1], blob[0]),
            int(math.sqrt(2) * blob[2]), 255, -1
        )
    return img_binary_blobs


def cal_det_se(no_tp, no_fn):
    return no_tp/(no_tp+no_fn)


def fp_per_unit_area(image_shape, no_fp):
    return no_fp/(image_shape[0] * image_shape[1] * (0.070**2)/100)


def froc_curve(
    froc_df: pd.DataFrame, thresholds: np.ndarray = None, cut_on_50fpi: bool = False,
    non_max_supression: bool = True
):
    """Using the complete dataset for froc computation containing all the images, obtains the
    Sensitivity and the Average False positives per image at each threshold.
    If thresholds is given then those thresholds are checked if not all posible ones.
    If cut_on_50fpi is True, then stop computing when 50 fpi are reached.
    Args:
        froc_df (pd.DataFrame): Dataframe with prediction results,
            check function: 'get_froc_df_of_img'
        thresholds (np.ndarray, optional): Array of thresholds to check. Defaults to None.
        cut_on_50fpi (bool, optional): Whether to stop computation at 50 ftpi or not.
            Defaults to False.
        non_max_supression (bool): Whether to perfom NMS over the labels (piking just the
            candidate with larger prediciton for each labeled mc). Defaults to True
    Returns:
        sensitivities (np.ndarray): sensitivities at each threshold
        avgs_fp_per_image (np.ndarray): average number of fp per image at each threshold
        thresholds (np.ndarray): thresholds
    """
    if non_max_supression:
        froc_df = non_maximum_suppression_w_labels(froc_df)

    total_n_images = len(froc_df.img_id.unique())
    sensitivities = []
    avgs_fp_per_image = []
    if thresholds is None:
        thresholds = np.sort(froc_df.pred_scores.unique())[::-1]
    for th in thresholds:
        froc_df.loc[froc_df.pred_scores >= th, 'pred_binary'] = True
        froc_df.loc[froc_df.pred_scores < th, 'pred_binary'] = False

        classif_as_pos = froc_df.pred_binary
        froc_df.loc[classif_as_pos & (froc_df.detection_labels == 'TP'), 'class_labels'] = 'TP'
        froc_df.loc[classif_as_pos & (froc_df.detection_labels == 'FP'), 'class_labels'] = 'FP'
        # This may only occur on th=0
        froc_df.loc[classif_as_pos & (froc_df.detection_labels == 'FN'), 'class_labels'] = 'FN'

        classif_as_neg = ~froc_df.pred_binary
        froc_df.loc[classif_as_neg & (froc_df.detection_labels == 'FN'), 'class_labels'] = 'FN'
        froc_df.loc[classif_as_neg & (froc_df.detection_labels == 'TP'), 'class_labels'] = 'FN'
        froc_df.loc[classif_as_neg & (froc_df.detection_labels == 'FP'), 'class_labels'] = 'TN'

        n_TP = len(froc_df.loc[froc_df.class_labels == 'TP'])
        n_FP = len(froc_df.loc[froc_df.class_labels == 'FP'])
        n_FN = len(froc_df.loc[froc_df.class_labels == 'FN'])

        sens = n_TP / (n_TP + n_FN)
        avg_nfp_per_image = n_FP / total_n_images

        sensitivities.append(sens)
        avgs_fp_per_image.append(avg_nfp_per_image)

        if cut_on_50fpi and avg_nfp_per_image > 50:
            break
    return sensitivities, avgs_fp_per_image, thresholds


def froc_curve_bootstrap(
    original_froc_df: pd.DataFrame, n_sets: int = 1000, n_jobs: int = -1,
    non_max_supression: bool = True
):
    """Using the complete dataset for froc computation containing all the images,
    generates 'n_sets' bootstrap samples. For each of them compute the froc curve.
    Args:
        original_froc_df (pd.DataFrame): Dataframe with prediction results,
            check function: 'get_froc_df_of_img'
        n_sets (int, optional): Number of bootstrap samples to take. Defaults to 1000.
        n_jobs (int, optional): Number of processes to use. If -1 all available
            cores will be used. Defaults to -1.
        non_max_supression (bool): Whether to perfom NMS over the labels (piking just the
            candidate with larger prediciton for each labeled mc). Defaults to True
    Returns:
        avg_sensitivities (np.ndarray): Average (over btrsp sets) sensitivity at each threshold
        std_sensitivities (np.ndarray): Standat deviation (over btrsp sets) of sensitivity
            at each threshold
        avg_avgs_fp_per_image (np.ndarray): Average (over btrsp sets) false positives per image
            at each threshold.
        std_avgs_fp_per_image (np.ndarray): Standard deviation (over btrsp sets) of false positives
            per image at each threshold.
        thresholds (np.ndarray): Thresholds
    """
    if non_max_supression:
        original_froc_df = non_maximum_suppression_w_labels(original_froc_df)

    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    thresholds = np.sort(original_froc_df.pred_scores.unique())[::-1]

    # Get btstrp samples
    n_samples = len(original_froc_df)
    btstrp_froc_df = []
    for i in range(n_sets):
        btstrp_froc_df.append(original_froc_df.sample(n=n_samples, replace=True, random_state=i))

    # compute individual FROCS
    partial_func_froc_curve = partial(froc_curve, thresholds=thresholds, non_max_supression=False)
    res = []
    with mp.Pool(n_jobs) as pool:
        for result in tqdm(pool.imap(partial_func_froc_curve, btstrp_froc_df), total=n_sets):
            res.append(result)

    # Reorganize results
    sensitivities = np.zeros((n_sets, len(thresholds)))
    avgs_fp_per_image = np.zeros((n_sets, len(thresholds)))
    for i in range(n_sets):
        sensitivities[i, :], avgs_fp_per_image[i, :], _ = res[i]

    # Compute summaries
    avg_sensitivities = np.mean(sensitivities, axis=0)
    std_sensitivities = np.std(sensitivities, axis=0)
    avg_avgs_fp_per_image = np.mean(avgs_fp_per_image, axis=0)
    std_avgs_fp_per_image = np.std(avgs_fp_per_image, axis=0)
    return avg_sensitivities, std_sensitivities, \
        avg_avgs_fp_per_image, std_avgs_fp_per_image, thresholds
