import cv2
import math
import logging
import warnings

import numpy as np

from numba import njit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

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
    pairs: np.ndarray, blobs_array: np.ndarray, overlap: float
):
    """Check if closes detections have an overlapping greater than the threshold
    Args:
        pairs (np.ndarray): indexes of a pair of close points
        blobs_array (np.ndarray): array with blobs as rows
            (y, x, sigma) = (row, col, sigma)
        overlap (float): minimum overlap allowed
    Returns:
        (np.ndarray): filtered array with blobs as rows
            (y, x, sigma) = (row, col, sigma)
    """
    for (i, j) in pairs:
        blob1, blob2 = blobs_array[i], blobs_array[j]
        if (blob1[-1] == 0) or (blob2[-1] == 0):
            continue
        if blob_overlap(blob1, blob2) > overlap:
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
            # No overlapping
            if d > r1 + r2:
                fp_idx.append(det_idx)
            # Overlapping
            else:
                # Dot mc inside det
                if r1 == 0:
                    detected_gts.append(gt_idx)
                    tp_idx.append(det_idx)
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
            # No overlapping
            if d > r1 + r2:
                fp_idx.append(det_idx)
            # Overlapping
            else:
                # Dot mc inside det
                if r1 == 0:
                    detected_gts.append(gt_idx)
                    tp_idx.append(det_idx)
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
