import cv2
import numpy as np


def circle_comparison(true_bboxes, predicted_roi_circles, mask):
    """Finds TP, FP and number of FN for a prediction of circles given image mask

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

    true_mask = mask

    for circle_idx, circle in enumerate(predicted_roi_circles.astype(int)):
        circle_roi_mask=cv2.circle(np.zeros(mask.shape),
                                     (circle[0], circle[1]),
                                     circle[2], 1, -1).astype(np.bool8)

        mapped_rois_idxs=set(
            np.unique(true_mask[circle_roi_mask])).difference(set([0]))
        if len(mapped_rois_idxs) > 0:
            TP = TP.union(mapped_rois_idxs)
        else:
            FP.append(circle_idx)
    FN = len(true_bboxes) - len(TP)
    return TP, FP, FN

def bbox_comparison_draft(
    mask,
    rois_df,
    true_bboxes: np.ndarray,
    predicted_bboxes: np.ndarray,
    prediction_confidences: np.ndarray,
    max_centre_dist: float,
    min_IoU: float,
):
    """ [NOT FINISHED YET]Calculates data for fROC curve plot based on bounding boxes.

    Args:
        true_bboxes (np.ndarray): Array of predicted bboxes containing for each bbox
            [(top_left_coordinates), (bottom_right_coordinates), (centre_coordinates)]
        predicted_bboxes (np.ndarray): Array of predicted bboxes of same shape as true_bboxes
        max_centre_dist (float): Maximum distance between bbox centres for a pair of bboxes to be
            considered a match
        min_IoU (float): Minimum IoU area of true and predicted bboxes for them to be considered mapped

    Returns:
        np.ndarray: 2D fROC data array
    """

    mapping_results = []
    # go through predicted bboxes and try to find ROI that correspond to them
    for bbox_idx, bbox in predicted_bboxes:

        mapping_result = {
            "prediction_confidence": prediction_confidences[bbox_idx]}
        mapping_result["bbox_idx"] = bbox_idx
        mapping_result["mapped_rois"] = []

        # slice mask in the place of the predicted bbox and find which ROIs intersect with it
        intersected_rois = np.unique(
            mask[bbox[0][0]: bbox[1][0], bbox[0][1]: bbox[1][1]]
        )

        # flag to check if able to find a true ROI for a given predicted bbox
        found_match = False

        # check each ROI for the correspondance to parameters
        for roi_idx, roi in roi_df[
            roi_df.index_in_image.isin(intersected_rois)
        ].iterrows():

            # calculate IoU
            pred_bbox_mask = cv2.rectangle(
                np.zeros(mask.shape), bbox[0], bbox[1], 1, -1
            ).astype(np.bool8)
            roi_mask = mask == roi_idx
            iou = (pred_bbox_mask & roi_mask).sum() / \
                (pred_bbox_mask | roi_mask).sum()

            roi_center = np.array(roi["center_crop"])
            pred_center_ = np.array(bbox[2])
            if (
                np.linalg.norm(roi_center - pred_center_) <= max_centre_dist
                and iou >= min_IoU
            ):
                mapping_result["mapped_rois"].append(roi["index_in_image"])

    mapping_results.append(mapping_result)
    mapping_results_df = pd.DataFrame(mapping_results, index="bbox_idx")

    metrics = []
    for conf_thr in np.linspace(0.01, 0.99, 99):
        thresholded_df = mapping_results_df[
            mapping_results_df.prediction_confidence > conf_thr
        ]
        TP_num = len(thresholded_df[thresholded_df.mapped_rois.apply(len) > 0])
        FP_num = len(
            thresholded_df[thresholded_df.mapped_rois.apply(len) == 0])

        # getting roi_idx of all successfully mapped ROIs
        succs_mapped_rois = np.unique(
            np.concatenate(thresholded_df["mapped_rois"].values)
        )

        FN_num = len(true_bboxes) - len(succs_mapped_rois)

        sensitivity = TP_num / (TP_num + FN_num)
        metrics.append(
            {
                "conf_thr": conf_thr,
                "TP_num": TP_num,
                "FP_num": FP_num,
                "FN_num": FN_num,
                "sensitivity": sensitivity,
            }
        )
    return metrics