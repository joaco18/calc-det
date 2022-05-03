import cv2
import numpy as np
from skimage.measure import label
from pathlib import Path


class MorphologyDetection:
    def __init__(
        self, rbd_img_path: str, threshold: float, alternative: bool = False,
        store_intermediate: bool = True
    ):
        self.use_alternative = alternative
        self.rbd_img_path = Path(rbd_img_path)
        self.threshold = threshold
        self.store_intermediate = store_intermediate
        if self.use_alternative:
            self.dilation_k_size = 14
        else:
            self.dilation_k_size = 20

    def detect(self, image: np.ndarray, image_id: int, ):

        self.image = image

        # load or create reconstructed by dialation image
        rbd_image = None
        if (self.rbd_img_path/f'{image_id}.tiff').exists():
            rbd_image = cv2.imread(
                str(self.rbd_img_path/f'{image_id}.tiff'),  cv2.IMREAD_ANYDEPTH)

        if rbd_image is None:
            rbd_image = self.reconstruction_by_dialation(
                image, circle_size=self.dilation_k_size)
            if self.store_intermediate:
                cv2.imwrite(str(self.rbd_img_path/f'{image_id}.tiff'), rbd_image)

        # erode breast boundary to avoid FP there
        rbd_image_no_bbound = self.breast_boundary_erosion(rbd_image)

        # intensity thresholding
        trheshold = np.quantile(
            rbd_image_no_bbound[rbd_image_no_bbound != 0].ravel(), q=self.threshold)
        thr1_rbd = rbd_image_no_bbound.copy()
        thr1_rbd[thr1_rbd <= trheshold] = 0

        if self.use_alternative:
            quant = 0.8
            trheshold = np.quantile(rbd_image_no_bbound[thr1_rbd > 0].ravel(), q=quant)
            thr_rbd = rbd_image_no_bbound.copy()
            thr_rbd[thr_rbd <= trheshold] = 0
        else:
            thr_rbd = thr1_rbd

        # connected components extraction and filtering
        markers = self.connected_components_extraction(thr1_rbd)
        cc_mask = self.connected_components_filtering(markers)
        return cc_mask

    def reconstruction_by_dialation(
        self, mask: np.ndarray, rect_size: int = 3, circle_size: int = 20
    ):
        """Reconstructs image using grayscale dialation
        Args:
            mask (np.ndarray): Image arre of type float or np.uint8
            rect_size (int, optional):
                Size of the SE used for geodesic reconstruction. Defaults to 3.
            circle_size (int, optional):
                Size of the SE used for creating a marker image. Defaults to 20.
        """
        rect_SE = cv2.getStructuringElement(
            cv2.MORPH_RECT, (rect_size, rect_size))
        circle_SE = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (circle_size, circle_size))

        marker_cur = cv2.morphologyEx(mask, cv2.MORPH_OPEN, circle_SE)
        marker_prev = np.zeros_like(marker_cur)
        while (not (marker_prev == marker_cur).all()):
            marker_prev = marker_cur.copy()
            marker_cur = cv2.min(cv2.dilate(marker_prev, rect_SE), mask)
        return mask - marker_cur

    def breast_boundary_erosion(self, rbd_image: np.ndarray):
        """Use breast mask and remove its contour from the detection image"""
        erosion_size = 5
        erosion_iter = 10

        breast_mask = (self.image != 0).astype(np.uint8)
        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                        (erosion_size, erosion_size))
        breast_boundary_mask = cv2.erode(breast_mask, structuring_element,
                                         iterations=erosion_iter)
        rbd_image_no_bbound = rbd_image.copy()
        rbd_image_no_bbound[breast_boundary_mask == 0] = 0
        return rbd_image_no_bbound

    def connected_components_extraction(self, thr1_rbd: np.ndarray):
        """Finds connected components"""
        # binarize and perform connected components labeling
        thr1_rbd_bin = self.to_uint8(255*(thr1_rbd > 0))
        markers, _ = label(thr1_rbd_bin, background=0,
                           return_num=True, connectivity=1)
        return markers

    def connected_components_filtering(self, markers: np.ndarray):
        """Filter connected components"""
        # connected components filtering
        selected_cc = []
        candidate_blobs = []
        out = np.zeros_like(markers, dtype='uint16')
        if self.use_alternative:
            contours, _ = cv2.findContours(np.where(markers > 0, 255, 0).astype('uint8'),
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for i, obj in enumerate(contours):
                A = cv2.contourArea(obj)
                A = 1 if A == 0 else A
                if A > 14 * 14:
                    continue
                p = cv2.arcLength(obj, True)
                p = 1 if p == 0 else p
                c = (4 * np.pi * A) / (p * p)
                if c > 0.6:
                    selected_cc.append(obj)
                    center, r = cv2.minEnclosingCircle(obj)
                    candidate_blobs.append((center[0], center[1], r))
                    # out = cv2.drawContours(out, contours, i, i+1, -1)
            out = cv2.drawContours(out, selected_cc, -1, 255, -1)
            # markers[~candidates_mask] = 0
            return out, candidate_blobs
        else:
            contours, _ = cv2.findContours(
                np.where(markers > 0, 255, 0).astype('uint8'),
                cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            for i, obj in enumerate(contours):
                A = cv2.contourArea(obj)
                if 14*14 > A >= 0:
                    selected_cc.append(obj)
                    center, r = cv2.minEnclosingCircle(obj)
                    candidate_blobs.append((center[0], center[1], r))
                    # out = cv2.drawContours(out, contours, i, i+1, -1)
            out = cv2.drawContours(out, selected_cc, -1, 255, -1)
            # markers[out == 0] = 0
            return out, candidate_blobs

    @staticmethod
    def min_max_norm(img):
        return (img - img.min())/(img.max() - img.min())

    def to_uint8(self, img):
        return (255*self.min_max_norm(img)).astype(np.uint8)
