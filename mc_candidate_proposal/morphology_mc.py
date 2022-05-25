import cv2
import numpy as np
from skimage.measure import label
from pathlib import Path
from scipy import spatial
from numba import njit
from mc_candidate_proposal.candidate_utils import filter_dets_from_muscle_region


@njit
def filter_by_distance(centers, pairs):
    for i, j in pairs:
        if (centers[i, -1] == 0) or (centers[j, -1] == 0):
            continue
        if centers[i, -1] > centers[j, -1]:
            centers[j, -1] = 0
        else:
            centers[i, -1] = 0
    indxs = np.where(centers[:, -1] != 0)[0]
    return indxs


class MorphologyCalcificationDetection:
    def __init__(
        self, rbd_img_path: str, threshold: float, min_distance: int, area: int,
        store_intermediate: bool = True, filter_muscle_region: bool = False
    ):
        """Constructor for MorphologyCalcificationDetection class
        Args:
            rbd_img_path (str): path for the reconstructed by dilation image
            threshold (float): quatile to use to threshold intensities anfter rbd
            min_distance (int): minimum distance between detections
            area (int): maximum area of a mc
            store_intermediate (bool, optional): whether to store rbd to accelerate tests.
                Defaults to True.
            filter_muscle_region (bool, optional): whether to filter candidates inside the
                pectoral muscle region. Defaults to False.
        """
        self.rbd_img_path = Path(rbd_img_path)
        self.threshold = threshold
        self.store_intermediate = store_intermediate
        self.min_distance = min_distance
        self.area = area
        self.dilation_k_size = 14
        self.filter_muscle_region = filter_muscle_region

    def detect(
        self, image: np.ndarray, image_id: int, muscle_mask: np.ndarray = None, max_radius: int = 10
    ):
        """Detects mC for a given image
        Args:
            image (np.ndarray): image to process
            image_id (int): id
            muscle_mask (np.ndarray, optional): pectoral muscle mask. Only necessary
                if the filtering is indicated in constructor. Defaults to None
            max_radius (int): maximum radius allowed in candidates.
                                Defaults to 10 (in accordance to filter convention)
        Returns:
            candidate_blobs (np.ndarray): [x, y, radius]
        """
        if self.filter_muscle_region:
            assert muscle_mask is not None, \
                'If filtering of muscle region is required the muscle region mask should'\
                ' be provided'
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
        self.output = rbd_image_no_bbound

        # intensity thresholding
        trheshold = np.quantile(
            rbd_image_no_bbound[rbd_image_no_bbound != 0].ravel(), q=self.threshold)
        thr1_rbd = rbd_image_no_bbound.copy()
        thr1_rbd[thr1_rbd <= trheshold] = 0

        trheshold = np.quantile(rbd_image_no_bbound[thr1_rbd > 0].ravel(), q=0.8)
        thr_rbd = rbd_image_no_bbound.copy()
        thr_rbd[thr_rbd <= trheshold] = 0

        # connected components extraction and filtering
        markers = self.connected_components_extraction(thr_rbd)
        candidate_blobs = self.connected_components_filtering(markers)

        if self.filter_muscle_region:
            candidate_blobs = filter_dets_from_muscle_region(
                candidate_blobs.astype(int), muscle_mask)

        # filter by max_radius
        candidate_blobs = candidate_blobs[candidate_blobs[:, 2] <= max_radius]

        return candidate_blobs

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
        thr1_rbd_bin = np.where(thr1_rbd > 0, 255, 0).astype('uint8')
        markers, _ = label(thr1_rbd_bin, background=0, return_num=True, connectivity=1)
        return markers

    def connected_components_filtering(self, markers: np.ndarray):
        """Filter connected components"""
        # connected components filtering
        candidate_blobs = []
        centers = []
        # out = np.zeros_like(markers, dtype='uint16')

        contours, _ = cv2.findContours(
            np.where(markers > 0, 255, 0).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # filter by area
        contours = list(contours)
        for i, obj in enumerate(contours):
            center, r = cv2.minEnclosingCircle(obj)
            A = cv2.contourArea(obj)
            A = 1 if A == 0 else A
            if A > self.area:
                centers.append([0, 0, 0])
            else:
                candidate_blobs.append((center[0], center[1], r))
                centers.append((center[0], center[1], r))
        centers = np.asarray(centers)
        candidate_blobs = np.asarray(candidate_blobs)
        indxs = np.where(centers[:, -1] != 0)[0]
        contours = [contours[i] for i in range(len(contours)) if i not in indxs]

        # filter by distance
        if self.min_distance != 0:
            centers = centers[indxs, :]
            tree = spatial.cKDTree(centers[:, :-1])
            pairs = np.array(list(tree.query_pairs(self.min_distance)))
            indxs = filter_by_distance(centers, pairs)
            if len(indxs) == 0:
                return None
            contours = [contours[i] for i in range(len(contours)) if i not in indxs]
            candidate_blobs = candidate_blobs[indxs, :]
        # out = cv2.drawContours(out, contours, -1, 255, -1)
        return candidate_blobs
