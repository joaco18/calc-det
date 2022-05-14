import math

import cv2
import numpy as np
import pandas as pd
import multiprocessing as mp

from scipy import spatial
from tqdm import tqdm
from functools import partial

from metrics.metrics_utils import evaluate_pairs_iou_appox, evaluate_pairs_iou_exact
import general_utils.utils as utils


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


def get_tp_fp_fn(
    lesion_bboxes: np.ndarray, radiuses: np.ndarray, detections: np.ndarray,
    max_dist: int, min_iou: float = None, exact: bool = False,
    lesion_mask: np.ndarray = None
):
    """Gets the true positives, false positives, false negatives, values from
        ground truth actually correctly predicted and the false positives matching
        the distance condition but not the area one.

    Args:
        lesion_bboxes (np.ndarray): ground truth lesion bboxes array as
            coming out from INBreastDataset item
        radiuses (np.ndarray): ground truth lesion radiuses array as
            coming out from INBreastDataset item
        detections (np.ndarray): detections given by (x,y,sigma)
        max_dist (int): max distance criterium between the centers of
            candidate lesion and ground truth to be mapped
        min_iou (float): minimun intersection over union criterion
            between the centers of candidate lesion and ground truth
        exact (bool): If True, the true shape of the lesion is used
            else the circular approximation is used
        lesion_mask (np.ndarray): if the exact method is used the lesion mask needs
            to be used
    Returns:
        tp (np.ndarray): [(x,y,sigma)]
        fp (np.ndarray): [(x,y,sigma)]
        fn (np.ndarray): [(x,y,radius)]
        gt_predicted (np.ndarray): ground truth lesion actually detected
            [(x,y,radius)]
        close_fp (np.ndarray): false positives that match distance criteria
            but not iou (only if iou value is indicated)
    """
    if exact:
        assert (lesion_mask is not None), 'Exact method requires the mask of lesions'

    # Get ground truth approximate circles
    radiuses = np.expand_dims(radiuses.astype(int), 1)
    gt_centers = utils.get_center_bboxes(lesion_bboxes)
    gt_circles = np.concatenate([gt_centers, radiuses], axis=1)

    # Get the distance tree
    datapoints = np.concatenate([gt_circles, detections])
    tree = spatial.cKDTree(datapoints)

    # Get the indexes among all points of ground truth points
    gt_idxs = np.arange(len(gt_circles))

    # Get the pairs closer than the required distance
    pairs = tree.query_pairs(max_dist)
    if len(pairs) == 0:
        fn = gt_circles
        fp = detections
        return [], fp, fn, [], []

    # Get the pairs matching the intersection over union condition
    min_iou = 1 if min_iou is None else min_iou
    if exact:
        tp_idx, fp_idx, detected_gts = evaluate_pairs_iou_exact(
            pairs, gt_idxs, datapoints, min_iou, lesion_mask, lesion_bboxes
        )
    else:
        tp_idx, fp_idx, detected_gts = evaluate_pairs_iou_appox(
            pairs, gt_idxs, datapoints, min_iou
        )

    # Get TP, FP, FN and missed sets of points
    detected_gts = list(set(detected_gts))
    missed_idx = [idx for idx in gt_idxs if idx not in detected_gts]
    tp = datapoints[tp_idx, :]
    close_fp = datapoints[fp_idx, :]
    fn = datapoints[missed_idx, :]
    gt_predicted = datapoints[list(detected_gts), :]
    fp_idx = np.full(len(datapoints), True)
    fp_idx[tp_idx] = False
    fp_idx[gt_idxs] = False
    fp = datapoints[fp_idx, :]
    return tp, fp, fn, gt_predicted, close_fp


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


def get_froc_df_of_many_imgs_features(
    candidates_df: pd.DataFrame, fns_df: pd.DataFrame, predictions: np.ndarray 
):
    """Get the standard froc dataframe out of the candidates dataframe of many images
    and the dataframe of false negatives for those images.
    Args:
        candidates_df (pd.DataFrame): 
            Rows: candidates (tp + fp)
            Columns: ['candidate_coordinates', 'labels', 'img_id']
        fns_df (pd.DataFrame): 
            Rows: fn lesion
            Columns: ['x', 'y', 'radius', 'labels', 'img_id']
        predictions (np.ndarray): prediction scores for the candidates
    Returns:
        pd.DataFrame:
            Rows: tp, fp, fn objects
            Columns: [
                x, y, radius, detection_labels, pred_scores,
                image_ids, pred_binary, class_labels
            ]
    """
    df = candidates_df.copy()
    values = candidates_df.loc[:, 'candidate_coordinates'].values.copy()
    df.loc[:, ['x', 'y', 'radius']] = np.vstack(values)
    df.loc[df.labels, 'detection_labels'] = 'TP'
    df.loc[~df.labels, 'detection_labels'] = 'FP'
    df.loc[:, 'pred_scores'] = predictions

    fns_df_temp = fns_df.copy()
    fns_df_temp.loc[:, 'detection_labels'] = 'FN'
    fns_df_temp.loc[:, 'pred_scores'] = 0.
    columns = ['x', 'y', 'radius', 'detection_labels', 'pred_scores', 'img_id']

    df = pd.concat([df[columns], fns_df_temp[columns]], ignore_index=True)
    df.columns = ['x', 'y', 'radius', 'detection_labels', 'pred_scores', 'image_ids']
    df.loc[:, 'pred_binary'] = False
    df.loc[:, 'class_labels'] = False
    return df


def get_froc_df_of_img(
    tp: list, fp: list, fn: list, predictions: np.ndarray, image_id: int
):
    """Get the basic dataframe used for froc calculation from the detections
    of an image.
    Args:
        tp (list): True positive candidates [[x, y, radius]]
        fp (list): False positive candidates [[x, y, radius]]
        fn (list): False Negative lesions [[x, y, radius]]
        predictions (np.ndarray): scores outputed by the model for the candidate
            (tp+fp) detections
        image_id (int): id of the image considered

    Returns:
        pd.DataFrame:
            Rows: tp, fp, fn objects
            Columns: [
                x, y, radius, detection_labels, pred_scores,
                image_ids, pred_binary, class_labels
            ]
    """
    objects = tp + fp + fn
    objects = np.asarray(objects)
    df = pd.DataFrame(objects, columns=['x', 'y', 'radius'])
    df['detection_labels'] = ['TP']*len(tp) + ['FP']*len(fp) + ['FN']*len(fn)
    df['pred_scores'] = 0.
    df.loc[(df.detection_labels == 'TP') | (df.detection_labels == 'FP'), 'pred_scores'] = \
        predictions
    df['image_ids'] = image_id
    df['pred_binary'] = False
    df['class_labels'] = False
    return df


def froc_curve(froc_df: pd.DataFrame, thresholds: np.ndarray = None, cut_on_50fpi: bool = False):
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
    Returns:
        sensitivities (np.ndarray): sensitivities at each threshold
        avgs_fp_per_image (np.ndarray): average number of fp per image at each threshold
        thresholds (np.ndarray): thresholds
    """
    total_n_images = len(froc_df.image_ids.unique())
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
    original_froc_df: pd.DataFrame, n_sets: int = 1000, n_jobs: int = -1
):
    """Using the complete dataset for froc computation containing all the images,
    generates 'n_sets' bootstrap samples. For each of them compute the froc curve.
    Args:
        original_froc_df (pd.DataFrame): Dataframe with prediction results,
            check function: 'get_froc_df_of_img'
        n_sets (int, optional): Number of bootstrap samples to take. Defaults to 1000.
        n_jobs (int, optional): Number of processes to use. If -1 all available
            cores will be used. Defaults to -1.
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

    if n_jobs is None:
        n_jobs = mp.cpu_count()
    thresholds = np.sort(original_froc_df.pred_scores.unique())[::-1]

    # Get btstrp samples
    n_samples = len(original_froc_df)
    btstrp_froc_df = []
    for i in range(n_sets):
        btstrp_froc_df.append(original_froc_df.sample(n=n_samples, replace=True, random_state=i))

    # compute individual FROCS
    partial_func_froc_curve = partial(froc_curve, thresholds=thresholds)
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
