import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

from general_utils.utils import min_max_norm


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
