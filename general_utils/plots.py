import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from general_utils.utils import min_max_norm
from sklearn.metrics import auc

cmap = plt.get_cmap("tab10")


def simple_im_show(img, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def simple_im_show2(img, mask, figsize=(10, 10)):
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    img_cmap = 'gray' if len(img.shape) == 2 else None
    mask_cmap = 'gray' if len(img.shape) == 2 else None
    ax[0].imshow(img, cmap=img_cmap)
    ax[1].imshow(mask, cmap=mask_cmap)
    ax[0].axis('off')
    ax[1].axis('off')
    plt.show()


def plot_blobs(image: np.ndarray, image_blobs: np.ndarray, ax=None):
    """Overlay blob circles over the image.
    Args:
        image (np.ndarray): image to which overlay the blobs
        image_blobs (np.ndarray): blobs to overlay over the image
            each row is a candidate (x, y, radius)
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(20, 20))
    # image = min_max_norm(image, 255)
    image = image.astype('uint8')
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for blob in image_blobs:
        x, y, r = blob
        x, y = int(x), int(y)
        image = cv2.circle(
            image, (x, y), int(math.sqrt(2) * r)+10, (255, 0, 0), 2
        )
    ax.imshow(image)
    plt.axis('off')
    plt.tight_layout()
    if ax is None:
        plt.show()


def plot_blobs2(image, blobs_a, blobs_b, ax=None):
    """Overlay two sets of blob circles over the image.
    Args:
        image (np.ndarray): image to which overlay the blobs
        tp_blobs (np.ndarray): blobs to overlay over the image in red
            each row is a candidate (x, y, radius)
        gt_blobs (np.ndarray): blobs to overlay over the image in green.
            each row is a candidate (x, y, radius)
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(20, 20))
    # image = min_max_norm(image, 255)
    image = image.astype('uint8')
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for blob in blobs_a:
        x, y, r = blob
        x, y = int(x), int(y)
        image = cv2.circle(
            image, (x, y), int(math.sqrt(2) * r)+10, (255, 0, 0), 2
        )
    for blob in blobs_b:
        x, y, r = blob
        x, y = int(x), int(y)
        image = cv2.circle(
            image, (x, y), int(math.sqrt(2) * r)+25, (0, 255, 0), 2
        )
    ax.imshow(image)
    plt.axis('off')
    plt.tight_layout()
    if ax is None:
        plt.show()


def plot_bboxes_over_image(image, bboxes, colors, types, thickness=2, alpha=0.2):
    """Overimposes bboxes on the image. Can work with multiple groups and types of bboxes.
    Args:
        image (np.ndarray): Single channel grayscale image for overimposition.
        bboxes (list[np.ndarray]): List of bboxes. Each element of the list
            should have one of the specific shapes depending of bbox type:
                (n_bboxes, top_left_coord, top_right_coordinate) - 'rect' type, where
                    top_left_coord, top_right_coordinate - are tuples with 2 integer coords
                (n_bboxes, centre_x, centre_y, radius) - 'circ' type
                    center coordinates - and radiuses should be integers
        colors (list[tuple]): List of colors to be assigned to the bboxes. In BGR convention.
        types (list[str]): List of bbox types: 'rect' OR 'circ'.
        thickness (int, optional): Bbox plot thickness. Defaults to 2.
        alpha (float, optional): Transparency of the bboxes. Defaults to 0.2.

    Returns:
        np.ndaray: 3 channel colored image with overimposed bboxes
    """
    img_bgr = (255*(image/image.max())).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)

    for bbox_group_idx, bbox_group in enumerate(bboxes):
        bbox_mask = np.zeros_like(img_bgr, dtype=np.uint8)
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
        img_bgr = cv2.addWeighted(img_bgr, beta, bbox_mask, alpha, 0.0)
    return img_bgr


def plot_gabor_filters(filters, plots_columns=3):
    """Plots Gabor filter
    Args:
        filters (list[np.ndarray]): List of Gabot filters to plot
        plots_columns (int, optional): Number of columns in the subplots image.
            Rows are scaled automatically . Defaults to 3.
    """
    plots_rows = int(np.ceil(len(filters)/plots_columns))
    fig, axs = plt.subplots(plots_rows, plots_columns, tight_layout=True, figsize=(10,10),)
    # fig.set_constrained_layout_pads(w_pad=0, h_pad=10, hspace=0, wspace=0)
    for ax_idx, (r, c) in enumerate(np.indices((plots_columns, plots_rows)).reshape((2, plots_rows*plots_columns)).T):
        if ax_idx < len(filters):
            axs[c,r].imshow(filters[ax_idx], cmap='gray')
        axs[c,r].set_axis_off()
    
    plt.show()


def plot_froc(
    fpis: np.ndarray, tprs: np.ndarray, total_mC: int = None, label: str = '', new_figure: bool = True
):
    """Plot FROC curve
    Args:
        fpis (np.ndarray): Average false positives per image at different thresholds
        tprs (np.ndarray): Sensitivity at different thresholds
        total_mC (int, optional): Total number of ground truth mC. Defaults to None.
        label (str, optional): Label of the line. Defaults to ''.
        new_figure (bool, optional): Whether to generate a new figure or not, allows overlapping.
            Defaults to True.
    """
    # limit point calculation till fpis<=50
    area_lim_mask = np.array(fpis)<=50
    fpis = np.array(fpis)[area_lim_mask]
    tprs = np.array(tprs)[area_lim_mask]
    
    if new_figure:
        plt.figure(figsize=(8, 8))
    plt.xlabel('FPpI')
    if total_mC is not None:
        plt.ylabel(f'TPR ({total_mC}) mC')
    else:
        plt.ylabel('TPR')
    plt.title('FROC curve')
    plt.plot(fpis, tprs, c=cmap(0))
    plt.xlim(0, 50)
    plt.ylim((0, 1))
    plt.legend([f"{label} AUC: {auc(fpis, tprs)}"])
    sns.despine()
    if new_figure:
        plt.show()


def plot_bootstrap_froc(
    fpis: np.ndarray, tprs: np.ndarray, std_tprs: np.ndarray,
    total_mC: int = None, label: str = '', new_figure: bool = True
):
    """Plot FROC curve
    Args:
        fpis (np.ndarray): Average false positives per image at different thresholds
        tprs (np.ndarray): Sensitivity at different thresholds
        std_tprs (np.ndarray): Standard deviation of sensitivity at different thresholds
            over the bootstrap samples.
        total_mC (int, optional): Total number of ground truth mC. Defaults to None.
        label (str, optional): Label of the line. Defaults to ''.
        new_figure (bool, optional): Whether to generate a new figure or not, allows overlapping.
            Defaults to True.
    """
    # limit point calculation till fpis<=50
    area_lim_mask = np.array(fpis)<=50
    fpis = np.array(fpis)[area_lim_mask]
    tprs = np.array(tprs)[area_lim_mask]
    
    max_tprs = tprs + std_tprs
    min_tprs = tprs - std_tprs

    if new_figure:
        plt.figure(figsize=(8, 8))
    plt.xlabel('FPpI')
    if total_mC is not None:
        plt.ylabel(f'TPR ({total_mC}) mC')
    else:
        plt.ylabel('TPR')
    plt.title('FROC curve')
    plt.plot(fpis, tprs, c=cmap(0))
    plt.xlim(0, 50)
    plt.fill_between(fpis, min_tprs, max_tprs, alpha=0.3, color=cmap(0))
    plt.ylim((0, 1))
    plt.legend([f"{label} AUC: {auc(fpis, tprs)}"])
    sns.despine()
    if new_figure:
        plt.show()
