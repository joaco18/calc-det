def bbox_comparison(mask, rois_df, true_bboxes:np.ndarray, predicted_bboxes:np.ndarray, prediction_confidences:np.ndarray, max_centre_dist:float, min_IoU:float):
    """Calculates data for fROC curve plot based on bounding boxes.

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

        mapping_result = {'prediction_confidence':prediction_confidences[bbox_idx]}
        mapping_result['bbox_idx'] = bbox_idx
        mapping_result['mapped_rois'] = []
        
        # slice mask in the place of the predicted bbox and find which ROIs intersect with it
        intersected_rois = np.unique(mask[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1]])
        
        # flag to check if able to find a true ROI for a given predicted bbox
        found_match = False 

        # check each ROI for the correspondance to parameters
        for roi_idx, roi in roi_df[roi_df.index_in_image.isin(intersected_rois)].iterrows():
            
            # calculate IoU
            pred_bbox_mask = cv2.rectangle(np.zeros(mask.shape), bbox[0], bbox[1], 1, -1).astype(np.bool8)
            roi_mask = mask == roi_idx
            iou = (pred_bbox_mask & roi_mask).sum() / (pred_bbox_mask | roi_mask).sum()

            roi_center = np.array(roi['center_crop'])
            pred_center_= np.array(bbox[2])
            if np.linalg.norm(roi_center - pred_center_) <= max_centre_dist and \
                iou >= min_IoU:
                mapping_result['mapped_rois'].append(roi['index_in_image'])

    mapping_results.append(mapping_result)
    mapping_results_df = pd.DataFrame(mapping_results, index='bbox_idx')
    
    metrics = []
    for conf_thr in np.linspace(0.01, 0.99, 99):
        thresholded_df = mapping_results_df[mapping_results_df.prediction_confidence > conf_thr]
        TP_num = len(thresholded_df[thresholded_df.mapped_rois.apply(len)>0])
        FP_num = len(thresholded_df[thresholded_df.mapped_rois.apply(len)==0])

        # getting roi_idx of all successfully mapped ROIs
        succs_mapped_rois = np.unique(np.concatenate(thresholded_df['mapped_rois'].values))
        
        FN_num = len(true_bboxes) - len(succs_mapped_rois)

        sensitivity = TP_num/(TP_num + FN_num)
        metrics.append({'conf_thr':conf_thr, 'TP_num':TP_num, 'FP_num':FP_num,
                       'FN_num':FN_num, 'sensitivity':sensitivity})
    return metrics

def plot_bbox(image, true_bboxes:np.ndarray, predicted_bboxes:np.ndarray, prediction_confidences:np.ndarray):
    for tr_bbb in true_bboxes:
        image = cv2.rectangle(image, tr_bbb[0], tr_bbb[1], cv2.green, thickness=2)
        
    for pr_bb_idx, pr_bb in enumerate(predicted_bboxes):
        image = cv2.rectangle(image, pr_bb[0], pr_bb[1], cv2.red, thickness=2)

        image = cv2.putText(image, f'{prediction_confidences[pr_bb_idx]}', pr_bb[1])
    simple_im_show(image)
    return image