
import sys
sys.path.insert(0, '..')

import cv2
import numpy as np

from skimage.measure import label


class MorpholohyDetection:
    def __init__(self, rbd_img_path, threshold) -> None:
        self.rbd_img_path = rbd_img_path
        self.threshold = threshold
    
    def predict(self, image:np.ndarray, image_id:int):
        
        self.image = image
        
        # load or create reconstructed by dialation image
        rbd_image = cv2.imread(str(self.rbd_img_path/f'{image_id}.tiff'),  cv2.IMREAD_ANYDEPTH)
        if rbd_image is None:
            rbd_image = self.reconstruction_by_dialation(image)
        
        # erode breast boundary to avoid FP there
        rbd_image_no_bbound = self.breast_boundary_erosion(rbd_image)
        
        # intensity thresholding 
        trheshold = np.quantile(rbd_image_no_bbound[rbd_image_no_bbound!=0].ravel(), q=self.threshold)
        thr1_rbd = rbd_image_no_bbound.copy()
        thr1_rbd[thr1_rbd <= trheshold] = 0
        
        # connected components extraction and filtering
        cc_mask = self.connected_components_extraction(thr1_rbd)
        
        return cc_mask
        
        
    def reconstruction_by_dialation(self, image, rect_size=3, circle_size=20):
        """Reconstructs image using grayscale dialation

        Args:
            image (np.ndarray): Image arre of type float or np.uint8
            rect_size (int, optional): Size of the SE used for geodesic reconstruction. Defaults to 3.
            circle_size (int, optional): Size of the SE used for creating a marker image. Defaults to 20.
        """
        rect_SE = cv2.getStructuringElement(cv2.MORPH_RECT, (rect_size, rect_size))
        circle_SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (circle_size, circle_size))

        marker = cv2.morphologyEx(image, cv2.MORPH_OPEN, circle_SE)
        mask = image.copy()

        marker_cur = marker.copy()

        while(True):
            marker_prev = marker_cur.copy()
            marker_cur = cv2.min(cv2.dilate(marker_prev, rect_SE), mask) 
            if np.all(marker_prev==marker_cur):
                break
        return mask - marker_cur
        
        
    def breast_boundary_erosion(self, rbd_image):
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
    
    def connected_components_extraction(self, thr1_rbd):
        # binarize and perform connected components labeling
        thr1_rbd_bin = self.to_uint8(255*(thr1_rbd>0))
        markers, ret = label(thr1_rbd_bin, background=0, return_num=True,)
    
        # connected components filtering
        selected_cc = []
        # selecting only no wholes candidates
        marker_sizes = dict(zip(*np.unique(markers, return_counts=True)))
        for marker, msize in marker_sizes.items():
            # max/min mC area filtering
            if  350 > msize >= 2 and marker != 0:
                if msize >=8:
                    cc_mask = (markers == marker).astype(np.uint8)
                    contours,_ = cv2.findContours(cc_mask, method=cv2.CHAIN_APPROX_NONE, mode=cv2.RETR_EXTERNAL)
                    
                    contour_mask = cv2.drawContours(np.zeros(markers.shape), contours, -1, 1, -1)
                    if not np.all(cc_mask == contour_mask):
                        continue
                selected_cc.append(marker)
                
        selected_cc_mask = np.isin(markers, selected_cc)
        markers[~selected_cc_mask] = 0
        return markers
        
    @staticmethod
    def min_max_norm(img):
        return (img - img.min())/(img.max() - img.min())
    def to_uint8(self, img):
            return (255*self.min_max_norm(img)).astype(np.uint8)