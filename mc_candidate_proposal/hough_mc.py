import cv2
import multiprocessing as mp
import numpy as np

from general_utils.utils import patch_coordinates_from_center, min_max_norm, sobel_gradient
from general_utils.dehazing import dehaze
from pathlib import Path
from skimage import restoration

DEHAZING_PARAMS = {'omega': 0.9, 'window_size': 11, 'radius': 40, 'eps': 1e-5}

HOUGH1_PARAMS = {'method': cv2.HOUGH_GRADIENT, 'dp': 1, 'minDist': 20,
                 'param1': 300, 'param2': 8,  'minRadius': 2, 'maxRadius': 20}

HOUGH2_PARAMS = {'method': cv2.HOUGH_GRADIENT, 'dp': 1, 'minDist': 20,
                 'param1': 300, 'param2': 10,  'minRadius': 2, 'maxRadius': 20}

BACK_EXT_RADIOUS = 50

EROSION_ITER = 20
EROSION_SIZE = 5
this_file_path = Path(__file__).resolve()


class HoughCalcificationDetection:
    """Microcalcification detection using Hough Transforms
    """

    def __init__(self, dehazing_params: dict = DEHAZING_PARAMS,
                 back_ext_radius: int = BACK_EXT_RADIOUS,
                 processed_imgs_path: str = this_file_path.parent.parent.parent / 'data/hough_img',
                 hough1_params: dict = HOUGH1_PARAMS,
                 hough2_params: dict = HOUGH2_PARAMS,
                 erosion_iter: int = EROSION_ITER,
                 erosion_size: int = EROSION_SIZE,
                 n_jobs: int = 6
    ):
        """Constructor for detHoughCalcificationDetection class

        Args:
            dehazing_params (dict): parameters used for dehazing
                ex: {'omega': 0.9, 'window_size': 11, 'radius': 40, 'eps': 1e-5}
            back_ext_radius (int): rolling ball radius used for background extraction
            processed_imgs_path (str): path where to load/save preprocessed images
            hough1_params (dict): parameters for first globabl hough transform search
                ex: {'method':cv2.HOUGH_GRADIENT, 'dp':1, 'minDist':20, 'param1':300,
                     'param2':10,  'minRadius':2, 'maxRadius':20}
            hough2_params (dict): parameters for second local hough transform search
                ex: see hough1_params
            erosion_iter (int, optional): Erosions iterations for breast
                boundary removal. Defaults to 30.
            erosion_size (int, optional): Erosion sizes for breast boundary removals.
                Defaults to 6.
        """

        self.dehazing_params = dehazing_params
        self.back_ext_radius = back_ext_radius
        self.processed_imgs_path = Path(processed_imgs_path)
        self.hough1_params = hough1_params
        self.hough2_params = hough2_params
        self.erosion_iter = erosion_iter
        self.erosion_size = erosion_size
        self.n_jobs = n_jobs

    def detect(self, image: np.ndarray, image_id: int, load_processed_images=True, hough2=False):
        """Detects mC for a given image

        Args:
            image (np.ndarray): Grayscale image for detection.
            image_id (int): Image id used to save/load images
            load_processed_images (bool, optional): Whether to load image from
                processed_imgs_path or to process them again. Defaults to True.
            hough2 (bool): Whether to calculate and output results of second
                (local) hough circles search

        Returns:
            h1_circles (np.ndarray): of shape (#_detected_circles, 3) corresponding
                to the first global hough transform where second axis contains
                circle_x, circle_y, circle_radius parameters
            h2_circles (np.ndarray): same as h1_circles but for the second search
        """
        # 1.-4. Image Enhancement
        processed_image = self.load_preprocessed_image(image, image_id,
                                                       load_processed_images)
        # 5. Global Hough
        h1_circles = self.hough1(processed_image)

        if hough2:
            # 6. Local Hough
            h2_circles = self.hough2(processed_image, h1_circles)
        else:
            h2_circles = None

        return h1_circles, h2_circles

    def load_preprocessed_image(self, image, image_id, load_processed_images):
        """Loads images and performs image enhancing needed for Hough transform.
        Either loads already preprocessed images or loads raw images, enhances them
        and saves them into a given folder.
        """
        img_path = self.processed_imgs_path/f'{image_id}.tiff'
        processed_image = None
        if load_processed_images:
            if not self.processed_imgs_path.exists():
                print(f"{self.processed_imgs_path} not found - creating one")
                self.processed_imgs_path.mkdir(parents=True, exist_ok=True)

            if img_path.exists():
                processed_image = cv2.imread(str(img_path),
                                             cv2.IMREAD_ANYDEPTH)
            else:
                processed_image = self.enhance_image(image)
                cv2.imwrite(str(img_path), processed_image)
        else:
            processed_image = self.enhance_image(image)
            cv2.imwrite(str(img_path), processed_image)
        return processed_image

    def enhance_image(self, image):
        """Performs image enhancment needed for Hough Detection
        """
        # 1. CONTRAST ENHANCEMENT - EQUALIZATION
        normalized_image = min_max_norm(image, max_val=1).astype('float32')
        dehazed_image = dehaze(normalized_image, **self.dehazing_params)
        #  2. BACKGROUND EXTRACTION
        background = restoration.rolling_ball(
            dehazed_image, radius=self.back_ext_radius)
        background_substracted = dehazed_image - background

        # 3. SOBEL-GAUSSIAN-SOBEL
        sobel_f1 = sobel_gradient(background_substracted)
        blured_iamge = cv2.GaussianBlur(sobel_f1, (0, 0), 2)
        sobel_f2 = sobel_gradient(blured_iamge)

        # 4. BREAST BOUNDARY EROSION
        breast_mask = (image != 0).astype(np.uint8)
        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                        (self.erosion_size, self.erosion_size))
        breast_boundary_mask = cv2.erode(breast_mask, structuring_element,
                                         iterations=self.erosion_iter)
        sobel_f2[breast_boundary_mask == 0] = 0
        return sobel_f2

    def hough1(self, processed_image, alpha1=0.97):
        """First global hough circles detection on saturated image
        """

        # 5. FIRST [GLOBAL] THRESHOLDING and HOUGH
        # 5.1 Saturation to alpha1
        alpha1_intensity = np.quantile(processed_image, q=alpha1)
        processed_image[processed_image <= alpha1_intensity] = 0

        # 5.2 Converting image to supported type by HoughCircles and hough search
        gradient_normalized = min_max_norm(processed_image, max_val=255).astype(np.uint8)
        processed_image_bin = 255*(gradient_normalized > 0).astype(np.uint8)
        h1_circles = cv2.HoughCircles(
            processed_image_bin, **self.hough1_params)
        return h1_circles[0].astype(int)

    def hough2(self, processed_image, hough1_circles, alpha2=0.95, patch_size=100):
        """Performs local hough circles search on 200x200 patches around
        circle centres detected by global hough transform.
        """
        # 6. SECOND [LOCAL] HOUGH
        hough2_circles = []
        self.processed_image = processed_image

        with mp.Pool(self.n_jobs) as pool:
            for result in pool.map(self.process_patch_hough2, hough1_circles):
                hough2_circles.extend(result)

        return np.asarray(hough2_circles).astype(int)

    def process_patch_hough2(self, circle, alpha2=0.95, patch_size=100):

        cx, cy, cr = circle
        # get coordinates of 200*200 cropped patch aroung circle
        x1, x2, y1, y2 = patch_coordinates_from_center((cx, cy), self.processed_image.shape, patch_size=patch_size*2, use_padding=False)

        h2_normalized_patch = self.processed_image[y1:y2, x1:x2].copy()

        image_circle_mask = cv2.circle(np.zeros(self.processed_image.shape),
                                       (cx, cy), cr, 1, -1).astype(bool)
        patch_circle_mask = image_circle_mask[y1:y2, x1:x2].copy()

        # saturation of circle intensities to the mean in whole window
        mean_window_int = np.mean(h2_normalized_patch)
        circle_values = h2_normalized_patch[patch_circle_mask]
        circle_values[circle_values <= mean_window_int] = mean_window_int
        h2_normalized_patch[patch_circle_mask] = circle_values

        # saturation to alpha2 of all window
        alpha2_intens = np.quantile(h2_normalized_patch[h2_normalized_patch > 0],
                                    q=alpha2)
        h2_normalized_patch[h2_normalized_patch <=
                            alpha2_intens] = alpha2_intens

        h2_normalized_patch = min_max_norm(h2_normalized_patch, max_val=1)
        h2_normalized_patch = 255 * \
            (min_max_norm(h2_normalized_patch, max_val=255).astype(np.uint8) > 0)
        h2_normalized_patch = h2_normalized_patch.astype(np.uint8)
        h2_circ = cv2.HoughCircles(
            h2_normalized_patch, **self.hough2_params)
        h2_circles_scaled = []
        if h2_circ is not None:
            # scaling back circle coordinates to whole image scale from patch scale
            h2_circles_scaled = [[c[0] + x1, c[1] + y1, c[2]]
                                 for c in h2_circ[0]]
        return h2_circles_scaled