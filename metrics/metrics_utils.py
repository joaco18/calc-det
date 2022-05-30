import cv2
import math
import logging
import warnings

import numpy as np
import pandas as pd

import general_utils.utils as utils

from numba import njit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from typing import List

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
logging.basicConfig(level=logging.INFO)


@njit
def clip(val):
    if val > 1.0:
        return 1.0
    elif val < -1.0:
        return -1.0
    else:
        return val


@njit(cache=True)
def compute_disk_overlap(d: float, r1: float, r2: float, iou: bool = False):
    """Compute fraction of surface overlap between two disks of radius
    r1 and r2, with centers separated by a distance d.
    Args:
        d (float): Distance between centers.
        r1 (float): Radius of the first disk.
        r2 (float): Radius of the second disk.
        iou (bool): Whether to return the intersection over union.
            Defaults to False
    Returns:
        (float): Fraction of area of the overlap between the two disks.
            Or intersection over union if indicated
    """
    if d == 0:
        if iou:
            a1 = (math.pi * (r1) ** 2)
            a2 = (math.pi * (r2) ** 2)
            a_min = np.minimum(a1, a2)
            return a_min / a1 + a2 - a_min
        else:
            return 1.
    ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
    ratio1 = clip(ratio1)
    acos1 = math.acos(ratio1)

    ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
    ratio2 = clip(ratio2)
    acos2 = math.acos(ratio2)

    a = -d + r2 + r1
    b = d - r2 + r1
    c = d + r2 - r1
    d = d + r2 + r1
    area = (r1 ** 2 * acos1 + r2 ** 2 * acos2 -
            0.5 * math.sqrt(abs(a * b * c * d)))
    if iou:
        if ((math.pi * (r1) ** 2) + (math.pi * (r2) ** 2) - area) == 0:
            print((math.pi * (r1) ** 2), (math.pi * (r2) ** 2), area)
        return area / ((math.pi * (r1) ** 2) + (math.pi * (r2) ** 2) - area)

    return area / (math.pi * (min(r1, r2) ** 2))


@njit(cache=True)
def blob_overlap(blob1, blob2):
    """Finds the overlapping area fraction between two blobs.

    Returns a float representing fraction of overlapped area. Note that 0.0
    is *always* returned for dimension greater than 3.

    Parameters
    ----------
    blob1 : sequence of arrays
        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
        where ``row, col`` (or ``(pln, row, col)``) are coordinates
        of blob and ``sigma`` is the standard deviation of the Gaussian kernel
        which detected the blob.
    blob2 : sequence of arrays
        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
        where ``row, col`` (or ``(pln, row, col)``) are coordinates
        of blob and ``sigma`` is the standard deviation of the Gaussian kernel
        which detected the blob.

    Returns
    -------
    f : float
        Fraction of overlapped area (or volume in 3D).
    """
    root_ndim = math.sqrt(2)

    # Here we divide the coordinates and the sigmas by the largest sigmas, to
    # normalize de radious to 1 for the largest blob and sth < 1 for the other blob
    # TODO: I don't think this is necessary
    if blob1[-1] == blob2[-1] == 0:
        return 0.0
    elif blob1[-1] > blob2[-1]:
        max_sigma = blob1[-1]
        r1 = 1
        r2 = blob2[-1] / blob1[-1]
    else:
        max_sigma = blob2[-1]
        r2 = 1
        r1 = blob1[-1] / blob2[-1]
    pos1 = blob1[:2] / (max_sigma * root_ndim)
    pos2 = blob2[:2] / (max_sigma * root_ndim)

    d = np.sqrt(np.sum((pos2 - pos1)**2))
    # No overlap case
    if d > r1 + r2:
        return 0.0

    # One blob is inside the other
    if d <= abs(r1 - r2):
        return 1.0

    return compute_disk_overlap(d, r1, r2)


@njit(cache=True)
def compare_and_filter_pairs(
    pairs: np.ndarray, blobs_array: np.ndarray, overlap: float, min_distance: float
):
    """Check if close detections have an overlapping greater than the threshold
    if so, keep the largest
    Args:
        pairs (np.ndarray): indexes of a pair of close points
        blobs_array (np.ndarray): array with blobs as rows
            (y, x, sigma) = (row, col, sigma)
        overlap (float): minimum overlap allowed
        min_distance (float): minimum distance allowed between centers
    Returns:
        (np.ndarray): filtered array with blobs as rows
            (y, x, sigma) = (row, col, sigma)
    """
    for (i, j) in pairs:
        # Check overlapping
        blob1, blob2 = blobs_array[i], blobs_array[j]
        if (blob1[-1] == 0) or (blob2[-1] == 0):
            continue
        pair_overlap = blob_overlap(blob1, blob2)
        if pair_overlap > overlap:
            if blob1[-1] > blob2[-1]:
                blob2[-1] = 0
            else:
                blob1[-1] = 0
        # Check distance btween centers
        elif min_distance != 0:
            centers_distance = math.sqrt((np.power(blob1[:-1]-blob2[:-1], 2)).sum())
            if centers_distance < min_distance:
                if blob1[-1] > blob2[-1]:
                    blob2[-1] = 0
                else:
                    blob1[-1] = 0
    return blobs_array[np.where(blobs_array[:, -1] > 0)]


@njit(cache=True)
def evaluate_pairs_iou_appox(
    pairs: set, gt_idxs: list, datapoints: np.ndarray, min_iou: float
):
    """
    Evaluate pairs of gt objects and detected ones over the overlap
    """
    fp_idx, tp_idx, detected_gts = [], [], []
    for i, j in pairs:
        if ((i not in gt_idxs) and (j not in gt_idxs)) or (i in gt_idxs and j in gt_idxs):
            continue
        gt_idx = i if ((i in gt_idxs) and (j not in gt_idxs)) else j
        det_idx = j if ((i in gt_idxs) and (j not in gt_idxs)) else i

        gt = datapoints[gt_idx]
        det = datapoints[det_idx]

        d = math.sqrt(np.sum((gt[:-1] - det[:-1])**2))
        # No intersection
        r1, r2 = gt[-1], det[-1]

        # Don't check overlaping
        if min_iou > 1:
            fp_idx.append(det_idx)
        elif min_iou == 0:
            tp_idx.append(det_idx)
            detected_gts.append(gt_idx)
        # Check overlaping
        else:
            # No overlapping
            if d > r1 + r2:
                fp_idx.append(det_idx)
            # Overlapping
            else:
                # Dot mc/det inside det/mc
                if (r1 == 0) or (r2 == 0):
                    tp_idx.append(det_idx)
                    detected_gts.append(gt_idx)
                # Overlapping greater than threshold
                elif compute_disk_overlap(d, r1, r2, True) > min_iou:
                    detected_gts.append(gt_idx)
                    tp_idx.append(det_idx)
                    # Overlapping less than threshold
                else:
                    fp_idx.append(det_idx)
    return tp_idx, fp_idx, detected_gts


def evaluate_pairs_iou_exact(
    pairs: set, gt_idxs: list, datapoints: np.ndarray, min_iou: float,
    lesion_mask: np.ndarray, lesion_bboxes: np.ndarray
):
    """
    Evaluate pairs of gt objects and detected ones over the overlap
    """
    img_shape = lesion_mask.shape
    fp_idx, tp_idx, detected_gts = [], [], []
    for i, j in pairs:
        if ((i not in gt_idxs) and (j not in gt_idxs)):
            continue
        gt_idx = i if ((i in gt_idxs) and (j not in gt_idxs)) else j
        det_idx = j if (i in gt_idxs) and (j not in gt_idxs) else i

        gt = datapoints[gt_idx]
        det = datapoints[det_idx]

        d = math.sqrt(np.sum((gt[:-1] - det[:-1])**2))
        # No intersection
        r1, r2 = gt[-1], det[-1]

        # Don't check overlaping
        if min_iou >= 1:
            tp_idx.append(det_idx)
            detected_gts.append(gt_idx)
        # Check overlaping
        else:
            if (d == 0) and (r1 == 0) and (r2 == 0):
                tp_idx.append(det_idx)
                detected_gts.append(gt_idx)
            # No overlapping
            elif d > r1 + r2:
                fp_idx.append(det_idx)
            # Overlapping
            else:
                # Dot mc inside det
                if r1 == 0:
                    detected_gts.append(gt_idx)
                    tp_idx.append(det_idx)
                else:
                    iou = get_exact_iou(
                        gt, gt_idx, det, img_shape, lesion_mask, lesion_bboxes
                    )
                    if iou > min_iou:
                        detected_gts.append(gt_idx)
                        tp_idx.append(det_idx)
                        # Overlapping less than threshold
                    else:
                        fp_idx.append(det_idx)
    return tp_idx, fp_idx, detected_gts


def get_exact_iou(
    gt: np.ndarray, gt_idx: int, det: np.ndarray, img_shape: tuple,
    lesion_mask: np.ndarray, lesion_bboxes: np.ndarray
):
    """"Given lesion and detection coordinates compute
    the exact iou using the lesion mask"""
    gt = gt.astype(int)
    det = det.astype(int)
    gt_val = lesion_mask[gt[:-1]]
    gt_bbox = lesion_bboxes[gt_idx]
    det_bbox = center_sigma_to_bbox(det, img_shape)
    bbox = get_max_bbox(gt_bbox, det_bbox)
    det_coords_adjusted = det.copy()
    det_coords_adjusted[0] = det[0] - bbox[0][0]
    det_coords_adjusted[1] = det[1] - bbox[0][1]

    gt_patch = lesion_mask[
        bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]
    ]
    detection_patch = np.zeros_like(gt_patch)
    gt_patch = np.where(gt_patch == gt_val, 1, 0)
    detection_patch = cv2.circle(
        detection_patch,
        (det_coords_adjusted[0], det_coords_adjusted[1]),
        int(det_coords_adjusted[0]), 1, -1
    )
    return (gt_patch & detection_patch).sum() / \
        (gt_patch | detection_patch).sum()


def center_sigma_to_bbox(coords: np.ndarray, img_shape: tuple):
    """
    Convert (x, y, s) coordinates to a bbox
    """
    x, y, s = coords
    x0 = np.minimum(x - s, 0)
    x1 = np.minimum(x + s, img_shape[0])
    y0 = np.minimum(y - s, 0)
    y1 = np.minimum(y + s, img_shape[1])
    return [(x0, y0), (x1, y1)]


def get_max_bbox(gt_bbox: np.ndarray, det_bbox: np.ndarray):
    """Get the maimum bbox containing the two of them"""
    x_min = np.minimum(gt_bbox[0][0], det_bbox[0][0])
    x_max = np.maximum(gt_bbox[1][0], det_bbox[1][0])
    y_min = np.minimum(gt_bbox[0][1], det_bbox[0][1])
    y_max = np.maximum(gt_bbox[1][1], det_bbox[1][1])
    return [(x_min, y_min), (x_max, y_max)]


@njit(cache=True)
def patches_intersections_with_labels(
    candidates: np.ndarray, roi_mask: np.ndarray, center_region_size: int, patch_size: int,
    binary: bool = False
):
    """Get the number of pixels of the intersection between the candidates and the ground
    truth mask, looking at the center (of size center_region_size) of a patch (of size
    patch_size) centered at each candidate.
    If binary just return if there's intersection of not.
    Args:
        candidates (np.ndarray): [x, y, radius]
        roi_mask (np.ndarray): mask of lesion labels (each one identified independently)
        center_region_size (int): region in the center of the patch to consider for labeling.
            If None, all the patch is cosidered.
        patch_size (int): size of the patch to evaluate
        binary (bool, optional): Whether to return True or False if there was an intersection
            instead of the counts. Defaults to False.
    Returns:
        intersections (np.ndarray): number of pixels of intersection with the gt mask
            for each candidate. If Binary then this is binarized.
    """
    intersections = np.empty(len(candidates))
    for coords_idx, coords in enumerate(candidates):
        # getting patch coordinates
        patch_x1, patch_x2, patch_y1, patch_y2 = utils.patch_coordinates_from_center(
            (coords[0], coords[1]), roi_mask.shape, patch_size)

        # getting coordinates of the patch center. Necessary if the patch is
        # in border and shifted
        if center_region_size is not None:
            p_center_y = patch_y1 + (patch_y2 - patch_y1)//2
            p_center_x = patch_x1 + (patch_x2 - patch_x1)//2
            patch_x1, patch_x2, patch_y1, patch_y2 = utils.patch_coordinates_from_center(
                (p_center_x, p_center_y), roi_mask.shape, center_region_size)

        intersection = np.sum(roi_mask[patch_y1:patch_y2, patch_x1:patch_x2] > 0)
        intersections[coords_idx] = intersection
    if binary:
        intersection = np.where(intersection > 0, 1, 0)
    return intersections


def get_tp_fp_fn_center_patch_criteria(
    candidates: np.ndarray, roi_mask: np.ndarray, center_region_size: int, patch_size: int,
    use_euclidean_dist: bool = False
):
    """
    Given an array of candidates and the mask of labels, it computes the itersection of a
    patch of patch_size centered on each candidate and -if the center crop of center_regio_size
    inside that patch matches a lesion in the gt mask- its counted as a tp if not as a fp.
    If center_region_size is None, then the intersection on the original patch is computed.
    At the end it gets the labels that weren't matched (FN).
    The returned dataframe for TP containes duplicated detections, for the cases in whic one
    detection matches more than one label. To include them in the sentivity metrics and to
    make an easier computation of froc NMS this duplicates are kept. If they are not desired
    performing a drop_duplicates(subset=['repeted_idxs']) will do the job.
    Args:
        candidates (np.ndarray): [x, y, radius]
        roi_mask (np.ndarray): mask of lesion labels (each one identified independently)
        center_region_size (int): region in the center of the patch to consider for labeling.
            If set to None the hole patch is considered for the intersection.
        patch_size (int): size of the patch to evaluate
        use_euclidean_dist (bool): whether to use euclidean distance (True - aka circular patch)
            or p=infinity minkowsky distance (False - aka rectangular patch), between the center
            of the detect candidate and the closest point of sorrounding ground truth labels.
            Defaults to False
    Returns:
        tp (pd.DataFrame): Columns: ['x', 'y', 'radius', 'label', 'matching_gt','repeted_idxs']
        fp (pd.DataFrame): Columns: ['x', 'y', 'radius', 'label', 'matching_gt','repeted_idxs']
        fn (pd.DataFrame): Columns: ['x', 'y', 'radius', 'label', 'matching_gt','repeted_idxs']
        ignored_candidates (pd.DataFrame):
            Columns: ['x', 'y', 'radius', 'label', 'matching_gt','repeted_idxs']
    """
    ignored_candidates, tp, fp, fn = [], [], [], []
    matching_gt, repeted_tp, repeted_fp = [], [], []
    if use_euclidean_dist:
        if patch_size % 2 == 0:
            patch_size = patch_size + 1
        center = patch_size//2
        circular_mask = np.zeros((patch_size, patch_size), dtype='uint8')
        circular_mask = cv2.circle(
            circular_mask, center=(center, center),radius=center, color=1, thickness=-1)
    detected_labels = set()
    tp_count, fp_count = 0, 0
    for coords in candidates:
        # getting patch coordinates
        patch_x1, patch_x2, patch_y1, patch_y2 = utils.patch_coordinates_from_center(
            (coords[0], coords[1]), roi_mask.shape, patch_size)

        # getting coordinates of the patch center. Necessary if the patch is
        # in border and shifted
        if center_region_size is not None:
            p_center_y = patch_y1 + (patch_y2 - patch_y1)//2
            p_center_x = patch_x1 + (patch_x2 - patch_x1)//2
            patch_x1, patch_x2, patch_y1, patch_y2 = utils.patch_coordinates_from_center(
                (p_center_x, p_center_y), roi_mask.shape, center_region_size)

        # Get the overlap with the rois mask and determine the unique labels
        overlap_on_labels = roi_mask[patch_y1:patch_y2, patch_x1:patch_x2]
        if use_euclidean_dist:
            overlap_on_labels = overlap_on_labels * circular_mask
        unique_labels = [label for label in np.unique(overlap_on_labels) if label != 0]
        detected_labels.update(set(unique_labels))
        intersection = (overlap_on_labels > 0).any()
        just_label_to_ignore_matched = (overlap_on_labels < 0).any()

        # If there's intersection repeat the candidate for each gt that it matched
        # This will help a faster NMS and correct computation of metrics
        if intersection:
            count = 0
            for label in unique_labels:
                tp.append(coords)
                matching_gt.append(label)
                # Keep which ones are repeted to avoid feature extraction later
                repeted_tp.append(tp_count)
                count += 1
            tp_count += count
        elif just_label_to_ignore_matched:
            ignored_candidates.append(coords)
        else:
            fp.append(coords)
            repeted_fp.append(fp_count)
            fp_count += 1

    # Obtain the false negatives
    detected_labels = list(detected_labels)
    gt_labels = np.unique(roi_mask)
    for label in gt_labels:
        if (label == 0) or (label in detected_labels):
            continue
        y, x = np.where(roi_mask == label)
        y1, y2 = y.min(), y.max()
        x1, x2 = x.min(), x.max()
        w, h = x2 - x1, y2 - y1
        center_x = x1 + w//2
        center_y = y1 + h//2
        radius_aprox = np.maximum(w, h) / 2
        fn.append((center_x, center_y, radius_aprox))

    # Generate the resulting dataframe
    tp = pd.DataFrame(tp, columns=['x', 'y', 'radius'])
    tp['label'] = 'TP'
    tp['repeted_idxs'] = repeted_tp
    tp['matching_gt'] = matching_gt

    fp = pd.DataFrame(fp, columns=['x', 'y', 'radius'])
    fp['label'] = 'FP'
    fp['repeted_idxs'] = np.asarray(repeted_fp) + len(tp)
    fp['matching_gt'] = None

    fn = pd.DataFrame(fn, columns=['x', 'y', 'radius'])
    fn['label'] = 'FN'
    fn['matching_gt'] = None
    fn['repeted_idxs'] = False

    ignored_candidates = pd.DataFrame(ignored_candidates, columns=['x', 'y', 'radius'])
    ignored_candidates['label'] = 'ignored'
    ignored_candidates['matching_gt'] = None
    ignored_candidates['repeted_idxs'] = False

    return tp, fp, fn, ignored_candidates


def get_froc_df_of_many_imgs_features(
    candidates_df: pd.DataFrame, fns_df: pd.DataFrame, predictions: np.ndarray,
    normal_img_ids: List[str]
):
    """Get the standard froc dataframe out of the candidates dataframe of many images
    and the dataframe of false negatives for those images.
    Args:
        candidates (pd.DataFrame): Dataframe of candidates (tp + fp).
            Check 'get_tp_fp_fn_center_patch_criteria' for more deails
        fn (pd.DataFrame): Dataframe of fn. Check 'get_tp_fp_fn_center_patch_criteria'
            for more deails
        predictions (np.ndarray): prediction scores for the candidates
        normal_img_ids (List[str]): array of bools indicating thered
    Returns:
        pd.DataFrame:
            Rows: tp, fp, fn objects
            Columns: [
                x, y, radius, detection_labels, pred_scores, img_id,
                matching_gt, repeted_idxs, pred_binary, class_labels,
                is_normal
            ]
    """
    # Fill metadata for TP and FP
    df = candidates_df.copy()
    values = candidates_df.loc[:, 'candidate_coordinates'].values.copy()
    df.loc[:, ['x', 'y', 'radius']] = np.vstack(values)
    df.loc[df.label, 'detection_labels'] = 'TP'
    df.loc[~df.label, 'detection_labels'] = 'FP'
    df.loc[:, 'pred_scores'] = predictions
    df.loc[:, 'matching_gt'] = candidates_df.loc[:, 'matching_gt'].values.copy()
    df.loc[:, 'repeted_idxs'] = candidates_df.loc[:, 'repeted_idxs'].values.copy()

    # Fill metadata for FN
    if len(fns_df) != 0:
        fns_df.loc[:, 'detection_labels'] = 'FN'
        fns_df.loc[:, 'pred_scores'] = 0.
        fns_df.loc[:, 'matching_gt'] = None
        fns_df.loc[:, 'repeted_idxs'] = False
        columns = [
            'x', 'y', 'radius', 'detection_labels', 'pred_scores',
            'img_id', 'matching_gt', 'repeted_idxs'
        ]
        df = pd.concat([df[columns], fns_df[columns]], ignore_index=True)

    # Add useful columns for froc computation
    df.loc[:, 'pred_binary'] = False
    df.loc[:, 'class_labels'] = False

    df.loc[:, 'is_normal'] = False
    df.loc[df.img_id.isin(normal_img_ids), 'is_normal'] = True

    return df


def get_froc_df_of_img(
    candidates: pd.DataFrame, fn: pd.DataFrame, predictions: np.ndarray, image_id: int,
    is_normal: bool
):
    """Get the basic dataframe used for froc calculation from the detections
    of an image.
    Args:
        candidates (pd.DataFrame): Dataframe of candidates (tp + fp).
            Check 'get_tp_fp_fn_center_patch_criteria' for more deails
        fn (pd.DataFrame): Dataframe of fn. Check 'get_tp_fp_fn_center_patch_criteria'
            for more deails
        predictions (np.ndarray): Array of predictions (of the non-duplicated dataset)
        image_id (int): id of the image considered
        is_normal (bool): whether the image is normal or not

    Returns:
        pd.DataFrame:
            Rows: tp, fp, fn objects
            Columns: [
                x, y, radius, detection_labels, pred_scores, img_id,
                matching_gt, repeted_idxs, pred_binary, class_labels,
                is_normal
            ]
    """
    # Restore repeted detections
    df = candidates.drop_duplicates(subset='repeted_idxs')
    df.loc[:, 'pred_scores'] = predictions
    df = df.loc[candidates.repeted_idxs.tolist(), :]

    # Fill metadata
    df['img_id'] = image_id
    df['repeted_idxs'] = candidates.repeted_idxs.tolist()
    df['matching_gt'] = candidates.matching_gt.tolist()
    df['detection_labels'] = candidates.label.tolist()

    # Fill metadata for FN
    if len(fn) != 0:
        fn['detection_labels'] = 'FN'
        fn['img_id'] = image_id
        fn['pred_scores'] = 0.
        fn['matching_gt'] = None
        fn['repeted_idxs'] = False
        df = pd.concat([df, fn], axis=0, ignore_index=True)

    # Add useful columns for froc computation
    df['pred_binary'] = False
    df['class_labels'] = False
    df['is_normal'] = is_normal

    return df


def non_maximum_suppression_w_labels(froc_df: pd.DataFrame):
    """ For each grund truth label matched by a detection, keep the detection with
    the highest prediction score
    Args:
        froc_df (pd.DataFrame): Dataframe with prediction results,
            check function 'get_froc_df_of_img' for more details.
    Returns:
        (pd.DataFrame): filtered version of the Dataframe with prediction results.
    """
    filtered_froc_df = []
    for img_id in froc_df.img_id.unique():
        # get the data for one image
        img_df = froc_df.loc[froc_df.img_id == img_id].copy()
        img_df.reset_index(inplace=True, drop=True)
        # Get the duplicated gts and keep the one with largest pred score
        img_df.loc[:, 'to_drop'] = False
        for gt_idx in img_df.matching_gt.dropna().unique():
            condition = (img_df.matching_gt == gt_idx)
            if (gt_idx is None) or len(img_df.loc[condition]) == 1:
                continue
            else:
                max_pred = img_df.loc[condition, 'pred_scores'].max()
                img_df.loc[(condition) & (img_df.pred_scores != max_pred), 'to_drop'] = True
        # Drop unnecessary stuf
        img_df = img_df.loc[~img_df.to_drop, img_df.columns != 'to_drop']

        # Add the image data to the filtered dataframe
        filtered_froc_df.append(img_df)
    filtered_froc_df = pd.concat(filtered_froc_df, axis=0, ignore_index=True)

    return filtered_froc_df
