
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import numpy as np


# ploting functions

def simple_im_show(img, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

def plot_bboxes_over_image(image, bboxes, colors, types, thickness=2, alpha=0.8):
    """Overimposes bboxes on the image. Can work with multiple groups of bboxes
    Args:
        image (np.ndarray): Grayscale image for overimposition
        bboxes (List[np.ndarray]): List of bboxes arrays.
            Should be one of the specific shape depending of bbox type:
                (n_bboxes, top_left_coord, top_right_coordinate) - rect type
                    top_left_coord, top_right_coordinate - tuples with 2 integer coords
                (n_bboxes, centre_x, centre_y, radius) - circ type
                    center coordinates and radiuses should be integers
        colors (list): List of colors to be assigned to the bboxes. BGR convention
        types (list): List of types: 'rect' OR 'circ'
    """
    img_uint8 = (255*(image/image.max())).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

    for bbox_group_idx, bbox_group in enumerate(bboxes):
        bbox_mask = np.zeros(
            (image.shape[0], image.shape[1], 3)).astype(np.uint8)
        for bbox in bbox_group:
            if types[bbox_group_idx] == 'rect':
                bbox_mask = cv2.rectangle(bbox_mask, bbox[0], bbox[1],
                                          color=colors[bbox_group_idx],
                                          thickness=thickness)
            elif types[bbox_group_idx] == 'circ':
                bbox_mask = cv2.circle(bbox_mask, (bbox[0], bbox[1]),
                                       bbox[2], color=colors[bbox_group_idx],
                                       thickness=thickness)
        beta = 1 - alpha
        img_bgr = cv2.addWeighted(img_bgr, alpha, bbox_mask, beta, 0.0)
    return img_bgr

def plot_img_rois(index: int, colors = ['yellow','orange'], linewidth = 1,radius = 6):
    """
    Plots the original image with rois as markers. Rectangles for rois with bounding boxes
    and circles with given radius for point lesions. Both markers with given linewidth and colors.

    Args:
        index (int): index of item in the database instance INBreast_Dataset
        colors (list, optional): color of the marker [rectangles, circles]. Defaults to ['yellow','orange'].
        linewidth (int, optional): width of line. Defaults to 1.
        radius (int, optional): radius of circles. Defaults to 6.
    """
    f,ax = plt.subplots(1,2,figsize=(10,8))
    ax[0].imshow(db[index]['img'],cmap='gray') # display image
    
    #bbox[1][0]-bbox[0][0],bbox[1][1]-bbox[0][1]] \
    lesion_bbxs = [[bbox[0],bbox[1][0]-bbox[0][0],bbox[1][1]-bbox[0][1]] \
         for bbox in db[index]['lesion_bboxes'] if bbox[0] != bbox[1]] # get lesion boxes

    lesion_pts = [bbox[0] for bbox in db[index]['lesion_bboxes'] if bbox[0] == bbox[1]] # get lesion points
    
    for coords,width,height in lesion_bbxs:
        rec = plt.Rectangle(coords, width=width, height=height, color=colors[0], linewidth=linewidth, fill=False)
        ax[0].add_patch(rec)
    for coords in lesion_pts:
        c = plt.Circle(coords, radius=radius, color=colors[1], linewidth=linewidth, fill=False)
        ax[0].add_patch(c)
    ax[0].set_title('Image with ROIs')
    ax[0].axis('off')
    ax[1].imshow(db[index]['lesion_mask'])
    ax[1].set_title('Image mask')
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()
    
def plot_img_hist(img,img_array):
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax[0].imshow(img,cmap='gray')
    sns.histplot(img_array.flatten(), ax=ax[1], bins=1000, element='poly', alpha=0.2,kde=True)
    ax[1].set_yscale("log")

def plot_blobs(image,image_blobs):
    f,ax = plt.subplots(1,1,figsize=(10,10))
    ax.imshow(image,cmap='gray')
    for blob in tqdm(image_blobs):
        y,x,r = blob
        c = plt.Circle((x, y), r, color='yellow', linewidth=1, fill=False)
        ax.add_patch(c)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
# other functions

def norm_image(image):
    return (image - image.min())/(image.max()-image.min()) # image normalization

def cal_det_se(no_tp,no_fn):
    return no_tp/(no_tp+no_fn)

def fp_per_unit_area(image_shape,no_fp):
    return no_fp/(image_shape[0] * image_shape[1] * (0.070**2)/100)

def quick_circle_comparison(true_bboxes, predicted_roi_circles, mask, n_jobs=6):
    """Finds TP, FP and number of FN for a prediction of circles given image mask

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
        FP (set): contains FP circle indexes (that weren't mapped to any rois)
        FN (int): number of rois not mapped to any of the predicted circles
    """
    TP = set()
    FP = []

    true_mask = mask

    for circle_idx, circle in enumerate(tqdm(predicted_roi_circles.astype(int))):
        circle_roi_mask = cv2.circle(np.zeros(mask.shape),
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