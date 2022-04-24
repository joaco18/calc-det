import cv2
import numpy as np

def circle_comparison(true_bboxes, predicted_roi_circles, mask, n_jobs=6):
    """Finds TP, FP and number of FN for a prediction of circles given image mask
    More precise circle comparison. Checks the intersection of each predicted circle
    with the true leision bbox

    Args:
        true_bboxes (np.ndarray): Array of shape (n_rois, 2) containing
            tl and br bbox coordinates in tuples
        predicted_roi_circles (np.ndarray): Array of shape (n_predicted_circ, 3)
            with circle_x, circle_y and circle_radius values
        mask (np.ndarray): Image mask containing indexes of rois

    Returns:
        TP (set): contains TP roi indexes
        FP (set): contains FP circle indexes (thate weren't mapped to any rois)
        FN (int): number of rois not mapped to any of the predicted circles
    """
    TP = set()
    FP = []
    
    for circle_idx, circle in enumerate(predicted_roi_circles.astype(int)):
        circle_roi_mask=cv2.circle(np.zeros(mask.shape),
                                     (circle[0], circle[1]),
                                     circle[2], 1, -1).astype(np.bool8)

        mapped_rois_idxs=set(np.unique(mask[circle_roi_mask])).difference(set([0]))
        if len(mapped_rois_idxs) > 0:
            TP = TP.union(mapped_rois_idxs)
        else:
            FP.append(circle_idx)
    FN = len(true_bboxes) - len(TP)
    return TP, FP, FN



def quick_circle_comparison(true_bboxes, predicted_roi_circles, mask, n_jobs=6):
    """Finds TP, FP and number of FN for a prediction of circles given image mask
    Quick vesrsion that looks at intersection between given true lesion mask
    and predicted bbox mask created by overlapping all predicted bboxes in one mask
    Args:
        true_bboxes (np.ndarray): Array of shape (n_rois, 2) containing
            tl and br bbox coordinates in tuples
        predicted_roi_circles (np.ndarray): Array of shape (n_predicted_circ, 3)
            with circle_x, circle_y and circle_radius values
        mask (np.ndarray): Image mask containing indexes of rois

    Returns:
        TP (set): contains TP roi indexes
        FP (int): contains FP circle indexes (thate weren't mapped to any rois)
        FN (int): number of rois not mapped to any of the predicted circles
    """
    TP = []
    FP = 0
    FN = 0
    
    for circle in predicted_roi_circles.astype(int):
        cricle_tl = (max(0, circle[0] - circle[2]), max(0, circle[1] - circle[2]))
        circle_br = (min(mask.shape[1], circle[0] + circle[2]), min(mask.shape[0], circle[1] + circle[2]))
        
        intersected_mask_idxs = np.unique(mask[cricle_tl[1]:circle_br[1], cricle_tl[0]:circle_br[0]])
        if intersected_mask_idxs.sum()>0:
            TP.extend(intersected_mask_idxs)
        else:
            FP+=1
    TP = set(np.unique(TP)).difference(set([0]))
    FN = len(true_bboxes) - len(TP)
    return TP, FP, FN
