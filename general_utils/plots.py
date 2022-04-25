import cv2
import numpy as np
import matplotlib.pyplot as plt


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
