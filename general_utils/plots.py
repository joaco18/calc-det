import math
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from feature_extraction.haar_features.haar_modules import Feature
from functools import partial
from matplotlib.lines import Line2D
from metrics.metrics_utils import get_tp_fp_fn_center_patch_criteria
from scipy.ndimage.morphology import binary_fill_holes
from sklearn.metrics import auc

import general_utils.utils as utils

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


def plot_blobs_2_sets(
    image: np.ndarray, blobs_a: np.ndarray, blobs_b: np.ndarray, ax=None,
    label_a: str = 'set 1', label_b: str = 'set 2'
):
    """Overlay two sets of blob circles over the image.
    Args:
        image (np.ndarray): image to which overlay the blobs
        blobs_a (np.ndarray): blobs to overlay over the image in red
            each row is a candidate (x, y, radius)
        blobs_b (np.ndarray): blobs to overlay over the image in green.
            each row is a candidate (x, y, radius)
        ax (ax element, optional): To use in subplots. Defaults to None.
        label_a (str, optional): Name for the first set. Defaults to 'set 1'.
        label_b (str, optional): Name for the second set. Defaults to 'set 2'.
    """
    legend_elements = [
        Line2D(
            [0], [0], marker='o', ls='None', c='w', label=label_a, mfc='k', mec='r', ms=10, mew=2),
        Line2D(
            [0], [0], marker='o', ls='None', c='w', label=label_b, mfc='k', mec='g', ms=10, mew=2)
    ]
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
    ax.legend(handles=legend_elements, loc='upper right',
              frameon=False, labelcolor='w')
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
    fig, axs = plt.subplots(plots_rows, plots_columns,
                            tight_layout=True, figsize=(10, 10))
    indices = np.indices((plots_columns, plots_rows))
    for ax_idx, (r, c) in enumerate(indices.reshape((2, plots_rows*plots_columns)).T):
        if ax_idx < len(filters):
            axs[c, r].imshow(filters[ax_idx], cmap='gray')
        axs[c, r].set_axis_off()
    plt.show()


def plot_froc(
    fpis: np.ndarray, tprs: np.ndarray, total_mC: int = None,
    label: str = '', ax: int = None, title: str = None,
    cut_on_50fpi: bool = True, color=cmap(0)
):
    """Plot FROC curve
    Args:
        fpis (np.ndarray): Average false positives per image at different thresholds
        tprs (np.ndarray): Sensitivity at different thresholds
        total_mC (int, optional): Total number of ground truth mC. Defaults to None.
        label (str, optional): Label of the line. Defaults to ''.
        ax (bool, optional): Whether to plot the figure in the ax of another plot.
            Defaults to None.
        title (str, optional): Optional title to include in the figure
        cut_on_50fpi (bool): whether to cut the froc plot at 50fpi. Defaults to True
    """
    ax_ = ax
    if ax_ is None:
        f, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlabel('FPpI')
    if total_mC is not None:
        ax.set_ylabel(f'TPR ({total_mC}) mC')
    else:
        ax.set_ylabel('TPR')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('FROC curve')

    fpis = np.asarray(fpis)
    ax.plot(fpis, tprs, c=color)
    ax.set_ylim((0, 1))
    if cut_on_50fpi:
        ax.set_xlim(-0.01, 50)
    ax.legend([f"{label} AUC: {auc(fpis/fpis.max(), tprs):.4f}"])
    sns.despine()
    if ax_ is None:
        plt.show()


def plot_bootstrap_froc(
    fpis: np.ndarray, tprs: np.ndarray, std_tprs: np.ndarray,
    total_mC: int = None, label: str = '',  ax: int = None,
    title: str = None, cut_on_50fpi: bool = True
):
    """Plot FROC curve
    Args:
        fpis (np.ndarray): Average false positives per image at different thresholds
        tprs (np.ndarray): Sensitivity at different thresholds
        std_tprs (np.ndarray): Standard deviation of sensitivity at different thresholds
            over the bootstrap samples.
        total_mC (int, optional): Total number of ground truth mC. Defaults to None.
        label (str, optional): Label of the line. Defaults to ''.
        ax (bool, optional): Whether to plot the figure in the ax of another plot.
            Defaults to None.
        title (str, optional): Defaults to None.
        cut_on_50fpi (bool): whether to cut the froc l¿plt at 50fpi. Defaults to True
    """
    max_tprs = tprs + std_tprs
    min_tprs = tprs - std_tprs

    ax_ = ax
    if ax_ is None:
        f, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlabel('FPpI')
    if total_mC is not None:
        ax.set_ylabel(f'TPR ({total_mC}) mC')
    else:
        ax.set_ylabel('TPR')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('FROC curve')
    ax.plot(fpis, tprs, c=cmap(0))
    if cut_on_50fpi:
        ax.set_xlim(-0.01, 50)
    ax.fill_between(fpis, min_tprs, max_tprs, alpha=0.3, color=cmap(0))
    ax.set_ylim((0, 1))
    ax.legend([f"{label} mean-AUC: {auc(fpis/fpis.max(), tprs):.4f}"])
    sns.despine()
    if ax_ is None:
        plt.show()


def plot_several_frocs(
    data: dict, ax: int = None, title: str = None, cut_on_50fpi: bool = True
):
    """Plot FROC curve
    Args:
        data (Dict[dict]):
            {
              'exp1':{
                'tprs': np.ndarray,
                'fpis': np.ndarray,
                'ths': np.ndarray
              }
            }
        ax (bool, optional): Whether to plot the figure in the ax of another plot.
            Defaults to None.
        title (str, optional): Optional title to include in the figure
        cut_on_50fpi (bool): whether to cut the froc plot at 50fpi. Defaults to True
    """
    ax_ = ax
    if ax_ is None:
        f, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlabel('FPpI')
    ax.set_ylabel('TPR')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('FROC curve')
    for k, (label, data_) in enumerate(data.items()):
        fpis = np.asarray(data_['fpis'])
        tprs = data_['tprs']
        ax.plot(fpis, tprs, c=cmap(k), label=f"{label} AUC: {auc(fpis/fpis.max(), tprs):.4f}")
        ax.set_ylim((0, 1))
        if cut_on_50fpi:
            ax.set_xlim(-0.01, 50)
    ax.legend()
    sns.despine()
    if ax_ is None:
        plt.show()


def draw_our_haar_like_features(
    image: np.ndarray, haar_feature: Feature, alpha: float = 0.5,
    rot: bool = False
):
    """Draws the Rotated Haar Feature kernel over the image with the fading given by
    alpha.
    Args:
        image (np.ndarray): image over which to draw the kernel
        haar_feature (Feature): Rotated Haar Feature
        alpha (float, optional): Fading factor with which to plot the kernel.
            Defaults to 0.5.
        rot (bool, optional): Whether the feature is a rotated one or not.
            Defaults to False
    Returns:
        (np.ndarray): image + overlap
    """
    image = utils.min_max_norm(image, 255).astype('uint8')
    result = np.zeros(image.shape).astype(int)
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    xby4 = utils.blockwise_retrieval(haar_feature.coords_x, size=4)
    yby4 = utils.blockwise_retrieval(haar_feature.coords_y, size=4)
    cby4 = utils.blockwise_retrieval(haar_feature.coeffs, size=4)

    for rect_pts_x, rect_pts_y, rect_coeff in zip(xby4, yby4, cby4):
        if rot:
            rect_pts_x = np.asarray(rect_pts_x) - np.asarray([1, 2, 0, 1])
            rect_pts_x = np.append(rect_pts_x, [rect_pts_x[2], rect_pts_x[1]])

            rect_pts_y = np.asarray(rect_pts_y) - np.asarray([0, 0, 0, 1])
            rect_pts_y = np.append(
                rect_pts_y, [rect_pts_y[2] - 1, rect_pts_y[1] - 1])

        rect_points = list(zip(rect_pts_x, rect_pts_y))
        rect_points = np.asarray(rect_points, dtype='int32')

        a = int(abs(rect_coeff[0]))

        if rect_coeff[0] < 0:
            temp = np.zeros(result.shape).astype('uint8')
            ch = cv2.convexHull(rect_points)
            bin_ = cv2.drawContours(temp, [ch], -1, 1, -1)
            result -= (binary_fill_holes(bin_)*a).astype(int)
        else:
            temp = np.zeros(result.shape).astype('uint8')
            ch = cv2.convexHull(rect_points)
            bin_ = cv2.drawContours(temp, [ch], -1, 1, -1)
            result += (binary_fill_holes(bin_)*a).astype(int)
    mask = np.zeros(image.shape)
    mask[:, :, 0] = np.zeros_like(result)
    mask[:, :, 1] = np.where(result < 0, 255, 0)
    mask[:, :, 0] = np.where(result > 0, 255, 0)
    mask = mask.astype('uint8')
    image = cv2.addWeighted(image, (1-alpha), mask, alpha, 0.0)
    return image


def plot_detections(
    detections: np.ndarray, image: np.ndarray, k=15,
    gt_bboxes: List[tuple] = None, ax: int = None,
    color=(255, 0, 0), return_image=False
):
    """Draws red a rectangle (increased in k pixels on each side) on each
    detection, also writes the corresponding score.
    Args:
        detections (np.ndarray): array of detections bboxes [x1, x2, y1, y2, score].
        image (np.ndarray): image to use as a basis.
        k (int, optional): Extra size added in both direcitons to the bbox.
            Defaults to 10.
        gt_bboxes (List[tuple], optional): array of bboxes coordinates [((x1, y1), (x2, y2))].
            If provided plotted in green. Defaults to None.
        ax (bool, optional): Whether to plot the figure in the ax of another plot.
            Defaults to None.
        color (tuple, optional): Color use to plot bboxes and text.
            Defaults to red  (255, 0, 0).
        return_image (bool, optional): Whether to plot the figure or to return the image.
            Defaults to False (plot figure).

    Returns:
        image (np.ndarray): image with plotted bboxes if return_image=True, otherwise None
    """
    if len(image.shape) < 3:
        image = utils.min_max_norm(image, 255).astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for [x1, x2, y1, y2, score] in detections:
        tl, br = utils.adjust_bbox_to_fit(image.shape, ((x1, y1), (x2, y2)), k)
        image = cv2.rectangle(image, tl, br, color, 2)
        label = f'{score:.3f}'
        y = tl[1]-15 if (tl[1]-15) > 15 else tl[1]+15
        image = cv2.putText(
            image, label, (int(x1), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    if gt_bboxes is not None:
        for ((x1, y1), (x2, y2)) in gt_bboxes:
            tl, br = utils.adjust_bbox_to_fit(
                image.shape, ((x1, y1), (x2, y2)), k)
            image = cv2.rectangle(image, tl, br, (0, 255, 0), 3)
            label = 'GT'
            y = tl[1]-15 if (tl[1]-15) > 15 else tl[1]+15
            image = cv2.putText(
                image, label, (int(x1), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    if return_image:
        return image
    else:
        ax_ = ax
        if ax_ is None:
            f, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(image)
        ax.axis('off')
        if ax_ is None:
            plt.show()


def add_detections_overlay(
    image: np.ndarray, candidates: pd.DataFrame, mask: np.ndarray = None,
    conf_thr: float = 0.1, k: int = 10, need_labeling: bool = True,
):
    """Labels the candidates and plots them accordingly over the provided image
    Args:
        image (np.ndarray): Image to plot the results
        candidates (pd.DataFrame):
            if labeling is needed:
                [x, y, radius, score]
            if labeling is not needed:
                [x, y, radius, score, label, 'matching_gt','repeted_idxs']
        mask (np.ndarray, optional):
            image lesion mask used for candidate labelling, only needed if
            'need_labeling' == True
        conf_thr (float, optional): final threshold to select candidates.
            Only those with confidence higher will be considered for labelling and
            display. Defaults to 0.1.
        k (int, optional): increase in the size of the plotted bboxes.
            Plotted bboxe will have side + k by side + k size. Defaults to 10.
    Returns:
        np.ndarray: image with plotted labelled candidates (BGR)
    """
    colors_code = {
        'TP': (0, 255, 0),
        'FP': (0, 0, 255),
        'FN': (255, 0, 0)
    }

    # label candidates
    if need_labeling:
        tp, fp, fn, ignored_candidates = get_tp_fp_fn_center_patch_criteria(
            candidates, mask, None, patch_size=14, use_euclidean_dist=True, scores_passed=True)
        candidates = pd.DataFrame()
        for frame in [tp, fp, fn]:
            candidates = pd.concat([candidates, frame])
    candidates = candidates[['x', 'y', 'score', 'label']]

    # adjust labels based on confidence threshold
    new_fn_position = (candidates.label == 'TP') & (candidates.score < conf_thr)
    candidates.loc[new_fn_position, 'label'] = 'FN'
    detections_to_drop = ((candidates.label != 'FN') & (candidates.score < conf_thr))
    candidates = candidates.loc[~detections_to_drop, :]

    # format candidate coordinates to plotting format
    get_bbox_from_center = partial(
        utils.patch_coordinates_from_center, image_shape=image.shape, patch_size=14+k)
    centers_it = zip(candidates['x'].values, candidates['y'].values)
    bbox_coordinates = [get_bbox_from_center(center) for center in centers_it]
    candidates[['x1', 'x2', 'y1', 'y2']] = bbox_coordinates
    candidates.drop(columns=['x', 'y'], inplace=True)

    # convert the image to rgb if necessary
    if len(image.shape) < 3:
        image = utils.min_max_norm(image, 255).astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # overlay bboxes
    for idx, [score, label, x1, x2, y1, y2] in candidates.iterrows():
        tl = (int(x1), int(y1))
        br = (int(x2), int(y2))
        image = cv2.rectangle(image, tl, br, colors_code[label], 2)
        if label == 'FN':
            bbox_tag = '-1' if score is None else f'{score:.3f}'
        else:
            bbox_tag = f'{score:.3f}'
        y = tl[1]-15 if (tl[1]-15) > 15 else tl[1]+15
        image = cv2.putText(
            image, bbox_tag, (int(x1), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX, 1, colors_code[label], 2)

    image = cv2.putText(image, 'TP', (image.shape[1]-150, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    image = cv2.putText(image, 'FP', (image.shape[1]-150, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    image = cv2.putText(image, 'FN (-1 not detected)', (image.shape[1]-400, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    image = cv2.putText(image, 'FN (* detected)', (image.shape[1]-300, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return image
