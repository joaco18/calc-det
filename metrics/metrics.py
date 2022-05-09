import cv2
import numpy as np
import math
from scipy import spatial
from general_utils.utils import get_center_bboxes
from metrics.metrics_utils import evaluate_pairs_iou_appox, evaluate_pairs_iou_exact


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
    min_dist: int, min_iou: float = None, exact: bool = False,
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
        min_dist (int): minimun distance criterium between the centers of
            candidate lesion and ground truth
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
    gt_centers = get_center_bboxes(lesion_bboxes)
    gt_circles = np.concatenate([gt_centers, radiuses], axis=1)

    # Get the distance tree
    datapoints = np.concatenate([gt_circles, detections])
    tree = spatial.cKDTree(datapoints)

    # Get the indexes among all points of ground truth points
    gt_idxs = np.arange(len(gt_circles))

    # Get the pairs closer than the required distance
    pairs = tree.query_pairs(min_dist)
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
