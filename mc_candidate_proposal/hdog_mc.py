import cv2
import h5py
import logging
import subprocess
import math

# import sys; sys.path.insert(0, '../')  # TODO: Remove this ugly thing

import numpy as np
import skimage.feature as skift
import scipy.ndimage as ndi

from itertools import combinations_with_replacement
from skimage.util.dtype import img_as_float
from pathlib import Path
from scipy import spatial
from metrics.metrics_utils import compare_and_filter_pairs
from general_utils.utils import min_max_norm
from mc_candidate_proposal.candidate_utils import filter_dets_from_muscle_region


logging.basicConfig(level=logging.INFO)

# Default parameters
dog_parameters = {
    'min_sigma': 1.18,
    'max_sigma': 3.1,
    'sigma_ratio': 1.05,
    'n_scales': None,
    'dog_blob_th': 0.006,
    'dog_overlap': 1,
    'dog_min_dist': 0,
}

hessian_parameters = {
    'method': 'alex',
    'hessian_threshold': 1.4,
    'hessian_th_divider': 200.
}


class HDoGCalcificationDetection:
    """
    Microcalcification detection using Hessian of Differential of
        Gaussians (Marasinou2021 and variations)
    """
    def __init__(
        self,
        dog_parameters: dict = dog_parameters,
        hessian_parameters: dict = hessian_parameters,
        processed_imgs_path: str = None,
        detections_path: str = None,
        filter_muscle_region: bool = False
    ):
        """Constructor

        Args:
            dog_parameters (dict, optional): Parameters for the DoG blob extraction.
                Defaults to:
                dog_parameters = {
                    'min_sigma': 1.18,
                    'max_sigma': 3.1,
                    'sigma_ratio': 1.05,
                    'n_scales': None,
                    'dog_blob_th': 0.1,
                    'dog_overlap': 1,
                    'dog_min_dist': 0
                } If dog_min_dist == 0 no filtering is done
            hessian_parameters (dict, optional): Parameters for the hessian filtering
                of blobs. Defaults to:
                hessian_parameters = {
                    'method': 'marasinou',
                    'hessian_threshold': 1.4,
                    'hessian_th_divider': 500.
                }
            processed_imgs_path (str, optional): Directory where the intermediate
                DoG and Hessians are going to be stored if required. If not
                specified it a directory data/hdog_preprocessed_images will
                be generated in the parent of calc-det. Defaults to None.
            detections_path (str, optional): Directory where the intermediate and final
                detections are going to be stored if required. If not specified it a
                directory data/hdog_detections will be generated in the parent of
                calc-det. Defaults to None.
            filter_muscle_region
        """
        if processed_imgs_path is None:
            self.processed_imgs_path = \
                Path(__file__).parent.parent.absolute()/'data'/'hdog_preprocessed_images'
        else:
            self.processed_imgs_path = Path(processed_imgs_path)
        if detections_path is None:
            self.detections_path = \
                Path(__file__).parent.parent.absolute()/'data'/'hdog_detections'
        else:
            self.detections_path = Path(detections_path)

        self.dog_parameters = dog_parameters
        self.hessian_parameters = hessian_parameters
        # DOG parameters
        self.min_sigma = dog_parameters['min_sigma']
        self.max_sigma = dog_parameters['max_sigma']
        self.sigma_ratio = dog_parameters['sigma_ratio']
        self.n_scales = dog_parameters['n_scales']
        self.dog_blob_th = dog_parameters['dog_blob_th']
        self.dog_overlap = dog_parameters['dog_overlap']
        self.dog_min_dist = dog_parameters['dog_min_dist']

        # Hessian parameters
        self.method = hessian_parameters['method']
        self.h_th_div = hessian_parameters['hessian_th_divider']
        self.h_thr = hessian_parameters['hessian_threshold']
        self.filter_muscle_region = filter_muscle_region

        # Generate paths for the stored precomputed steps based on parameters
        self.hdog_image_path = self.processed_imgs_path / \
            f'dog_ms-{self.min_sigma}_sr-{self.sigma_ratio}_' \
            f'Ms-{self.max_sigma}_m-{self.method}.hdf5'
        self.raw_detections_path = self.detections_path / \
            f'det_ms-{self.min_sigma}_sr-{self.sigma_ratio}_Ms-{self.max_sigma}_' \
            f'm-{self.method}_dth-{self.dog_blob_th}.hdf5'
        self.final_detections_path = self.detections_path / \
            f'det_ms-{self.min_sigma}_sr-{self.sigma_ratio}_Ms-{self.max_sigma}_' \
            f'm-{self.method}_dth-{self.dog_blob_th}_hdiv-{self.h_th_div}_' \
            f'hth-{self.h_thr}.hdf5'

    def get_sigmas(self):
        """Based on the inputed parameter (n_scales or sigma ratio) it generates
            the array of sigmas to consider
        Returns:
            (np.ndarray): array of sigmas
        """
        # Get number of scales or get the sigma ratio depending the arguments inputed.
        if self.n_scales is None:
            self.n_scales = int(
                np.log(self.max_sigma / self.min_sigma) / np.log(self.sigma_ratio) + 1)
        else:
            self.sigma_ratio = (self.max_sigma/self.min_sigma) ** (1 / self.n_scales)
        # Progression of standard deviations for gaussian kernels
        sigma_array = self.min_sigma * (self.sigma_ratio ** np.arange(self.n_scales + 1))
        return sigma_array

    def detect(
        self, image: np.ndarray, img_id: int, use_preprocessed: bool = True,
        save_results: bool = True, muscle_mask: np.ndarray = None
    ):
        """Method to obtain the microcalcification detections
        Args:
            image (np.ndarray): Image to process.
            img_id (int): Id of the image
            load_processed (bool, optional): Whether to use preprocessed data
                (dog, hessians, raw_detections) if available to accelerate
                experiments. Defaults to True.
            save_results (bool, optional): Whether to store the final detections for
                further use. Defaults to True.
            muscle_mask (np.ndarray, optional): pectoral muscle mask. Only needed if
                the filtering is indicated in the constructor
        Returns:
            detections (np.ndarray): Array with filtered detections as rows
                (x, y, sigma) = (col, row, sigma)
            candidate_detections (np.ndarray): Array with raw detections as rows
                (x, y, sigma) = (col, row, sigma)
        """
        if self.filter_muscle_region:
            assert muscle_mask is not None, \
                'If filtering of muscle region is required the muscle region mask should'\
                ' be provided'
        
        image = min_max_norm(image, 1)

        self.use_preprocessed = use_preprocessed
        self.image_id = img_id

        # Get the sigmas controlling the scale dimension of DoG
        self.sigma_array = self.get_sigmas()

        det_available = False
        if self.use_preprocessed:
            det_available = self.detections_available()
        if det_available:
            if self.method == 'marasinou':
                candidate_detections, hessian_cube = self.load_blob_candidates()
            else:
                candidate_detections, hessian_eigval = self.load_blob_candidates()
        else:
            # Get the DoG and hessian (or hessian eigenvalues depending method)
            if self.method == 'marasinou':
                dog_cube, hessian_cube = self.get_dog_and_hessian_info(image, img_id)
            else:
                dog_cube, hessian_eigval = self.get_dog_and_hessian_info(image, img_id)

            # Get candidate detections
            candidate_detections = self.get_blob_candidates(dog_cube)

            if self.use_preprocessed and (not det_available):
                self.store_detections(candidate_detections)

        # Filter candidate detections
        if self.method == 'marasinou':
            detections = self.filter_blob_candidates(candidate_detections, hessian_cube)
        else:
            detections = self.filter_blob_candidates(candidate_detections, hessian_eigval)

        detections = self.convert_yxs2xys(detections)
        # candidate_detections = self.convert_yxs2xys(candidate_detections)

        # convert to radius
        detections[:, 2] = detections[:, 2]*math.sqrt(2)
        detections = detections.astype(int)

        if save_results:
            self.store_final_detections(candidate_detections, detections)

        if self.filter_muscle_region:
            detections = filter_dets_from_muscle_region(detections, muscle_mask)

        return detections

    def detections_available(self):
        """Check if the detection file exists and it contains the desired image"""
        if not self.raw_detections_path.exists():
            return False
        with h5py.File(self.raw_detections_path, 'r') as f:
            data_in_file = f'{self.image_id}/raw_detections' in f
        return data_in_file

    def load_blob_candidates(self):
        """Load blob candidates to accelerate experiments"""
        with h5py.File(self.raw_detections_path, 'r') as f:
            raw_detections = f[f'{self.image_id}/raw_detections'][:]
        with h5py.File(self.hdog_image_path, 'r') as f:
            hessian_info = f[f'{self.image_id}/hessian_info'][:]
        return raw_detections, hessian_info

    def get_dog_and_hessian_info(self, img: np.ndarray, img_id: int):
        """Computes the DoG and hessian of the image based on the scales preindicated
            if the eigenvalues method is indicated (not Marasinou), then the eigenvalues
            of the hessian are returned insted.
            The computation of DoG, and Hessian eigenvalues, is done in place modifying
            the gaussians and the hessians, to avoid unnecessary memory usage.
            If the DoG and Hessian/Hessian eigenvalues are available in disk and it was
            idicated to use them, then the computation is avoided.
        Args:
            img (np.ndarray): image to be processed
            img_id (int): image id
        Returns:
            dog (np.ndarray): DoG cube with many DoG as sigmas-1 are evaluated.
            essian_info (np.ndarray): Hessian of DoG or eigenvalues of the Hessian
                depending on the filtering method used
        """
        available = False
        if self.use_preprocessed:
            available = self.preprocessed_available()
        if available:
            return self.load_preprocessed()
        else:
            dog = []
            hessian_info = []
            for m, s in enumerate(self.sigma_array):
                # Get kernel size 4 times std plus an offset from skimage implementation
                ks = int(4 * s + 0.5)
                ks = ks if ks % 2 == 1 else ks + 1

                # Get the gaussian smoothing
                dog.append(cv2.GaussianBlur(img, (ks, ks), s))
                if m != 0:
                    idx = m - 1
                    # Get the DoG and store in place to avoid using innecesary mem
                    norm_factor = self.sigma_array[idx]
                    norm_factor /= (self.sigma_array[idx + 1] - self.sigma_array[idx])
                    dog[idx] = (dog[idx] - dog[idx + 1]) * norm_factor

                    # Compute the Hessian of DoG
                    hessian_info.append(self.hessian_matrix(dog[idx]))

                    # Compute the eigenvalues if necessary and store inplace
                    if self.method != 'marasinou':
                        hessian_info[idx] = \
                            skift.hessian_matrix_eigvals(hessian_info[idx])

            # Get arrays from the lists
            hessian_info = np.asarray(hessian_info)
            dog = np.stack(dog[:-1], axis=-1)

            # If needed for future, store
            if self.use_preprocessed and (not available):
                self.store_preprocessed(dog, hessian_info)
        return dog, hessian_info

    def preprocessed_available(self):
        """Check if DoG and Hessian are available in disk
        """
        # Check if the file exists
        if not self.hdog_image_path.exists():
            return False
        # Check if the data for the particular image exists
        with h5py.File(self.hdog_image_path, 'r') as f:
            data_in_file = f'{self.image_id}/dog' in f
            data_in_file = data_in_file and f'{self.image_id}/hessian_info' in f
        return data_in_file

    def load_preprocessed(self):
        """Loads the DoG and Hessian from disk
        """
        with h5py.File(self.hdog_image_path, 'r') as f:
            dog = f[f'{self.image_id}/dog'][:]
            hessian_info = f[f'{self.image_id}/hessian_info'][:]
        return dog, hessian_info

    @staticmethod
    def hessian_matrix(image: np.ndarray):
        """Compute the hessian of the image
        Args:
            image (np.ndarray): image to process
        Returns:
            (list): List of 3 np.ndarrays corresponding to:
                H[0,0], H[0,1], H[1,1] of the hessian elements of the image
        """
        if image.dtype != float:
            image = img_as_float(image)

        gradients = np.gradient(image)
        axes = reversed(range(image.ndim))
        return [
            np.gradient(gradients[ax0], axis=ax1)
            for ax0, ax1 in combinations_with_replacement(axes, 2)
        ]

    def store_preprocessed(self, dog: np.ndarray, hessian_info: np.ndarray):
        """Store DoG and Hessian (eigenvals) for future use
        """
        self.hdog_image_path.parent.mkdir(parents=True, exist_ok=True)
        mode = 'a' if self.hdog_image_path.exists() else 'w'
        with h5py.File(self.hdog_image_path, mode) as f:
            _ = f.create_dataset(f'{self.image_id}/dog', data=dog)
            _ = f.create_dataset(f'{self.image_id}/hessian_info', data=hessian_info)

    def get_blob_candidates(self, dog_cube: np.ndarray):
        """Computes the local maxima with a 3x3x3 window over the DoG cube
            to find detection candidates
        Args:
            dog_cube (np.ndarray): multiscal DoG of the orginal image
        Returns:
            (np.ndarray): Array with candidate detections as rows
                (y, x, sigma) = (row, col, sigma)
        """
        # Get the local maximum
        local_maxima = \
            self.peak_local_max(dog_cube, threshold_abs=self.dog_blob_th)

        # Catch no peaks
        if local_maxima.size == 0:
            return np.empty((0, 3)), np.empty((0, 3))

        # Transform indexes in sigmas
        local_maxima = local_maxima.astype(np.float64)
        local_maxima[:, -1] = self.sigma_array[local_maxima[:, -1].astype('int')]
        return local_maxima

    def peak_local_max(
        self, image: np.ndarray, threshold_abs: float, min_distance: int = 1,
        threshold_rel: float = None, num_peaks: int = np.inf, p_norm: int = np.inf
    ):
        """Finds peaks in an image as coordinate list.
        If both `threshold_abs` and `threshold_rel` are provided, the maximum
        of the two is chosen as the minimum intensity threshold of peaks.
        Based on skimage's function, but modified to be more efficient
        Args:
            image (np.ndarray): 3d image where local peaks are searched
            min_distance (int, optional): minimum distance between peaks.
                Defaults to 1.
            threshold_abs (float, optional): Minimum intensity of peaks.
            threshold_rel (float, optional): Minimum intensity of peaks,
                calculated as `max(image) * threshold_rel`. Defaults to None.
            num_peaks (int, optional): Maximum number of peaks. When the
                number of peaks exceeds `num_peaks`, return `num_peaks` peaks
                based on highest peak intensity. Defaults to np.inf.
            p_norm (int, optional): Which Minkowski p-norm to use to get the
                distance between peaks. Should be in the range [1, inf].
                Defaults to np.inf.
        Returns:
            (np.ndarray): (row, column, ...) coordinates of peaks.
        """
        threshold = threshold_abs
        if threshold_rel is not None:
            threshold = max(threshold, threshold_rel * image.max())
        footprint = np.ones((3, 3, 3))

        # Non maximum filter
        mask = self.get_peak_mask(image, footprint, threshold)

        # TODO: Check if we can delete it
        # Select highest intensities (num_peaks)
        coordinates = self.get_high_intensity_peaks(
            image, mask, num_peaks, min_distance, p_norm)

        return coordinates

    @staticmethod
    def get_peak_mask(image, footprint, threshold):
        """
        Return the mask containing all peak candidates above thresholds.
        """
        image_max = ndi.maximum_filter(
            image, footprint=footprint, mode='constant')
        out = image == image_max
        # no peak for a trivial image
        image_is_trivial = np.all(out)
        if image_is_trivial:
            out[:] = False
        out &= image > threshold
        return out

    @staticmethod
    def get_high_intensity_peaks(
        image: np.ndarray, mask, num_peaks, min_distance, p_norm
    ):
        """
        Return the highest intensity peak coordinates and check the
        minimum distance between peaks
        """
        # Get coordinates of peaks
        coord = np.nonzero(mask)
        intensities = image[coord]
        # Sort peaks descending order
        idx_maxsort = np.argsort(-intensities)
        coord = np.transpose(coord)[idx_maxsort]
        if len(coord) > num_peaks:
            coord = coord[:num_peaks]
        return coord

    def store_detections(self, raw_detections):
        """Stores raw detections for futher use
        """
        self.raw_detections_path.parent.mkdir(parents=True, exist_ok=True)
        mode = 'a' if self.raw_detections_path.exists() else 'w'
        with h5py.File(self.raw_detections_path, mode) as f:
            _ = f.create_dataset(f'{self.image_id}/raw_detections', data=raw_detections)

    def filter_blob_candidates(self, blobs: np.ndarray, hessian_info: np.ndarray):
        """Filters blob candidates based on the overlapping of the blobs and
            on the hessian conditions
        Args:
            blobs (np.ndarray): array with blob candidates as rows
                (y, x, sigma) = (row, col, sigma)
            hessian_info (np.ndarray): containing the HDoG or the eigenvalues
                of the HDoG depending the method
        Returns:
            (np.ndarray): array with blob candidates as rows
                (y, x, sigma) = (row, col, sigma)
        """
        # Filter blobs according to overlapping
        blobs = self.prune_blobs(blobs, self.dog_overlap)

        # Filter blobs by hessian condition
        blobs = self.filter_by_hessian_condition(blobs, hessian_info)
        return blobs

    def prune_blobs(self, blobs_array: np.ndarray, overlap: float):
        """Eliminates blobs by overlap area fraction.
        Args:
            blobs_array (np.ndarray): Array with blob candidates as rows
                (y, x, sigma) = (row, col, sigma)
            overlap (float): A value in [0, 1[. If the fraction of area overlapping
                for 2 blobs is greater than `overlap` the smaller blob is eliminated.
                If overlap == 1, then no filtering is done
        Returns:
            (np.ndarray): Array with blob candidates as rows
                (y, x, sigma) = (row, col, sigma)
        """
        if overlap == 1:
            return blobs_array
        sigma = blobs_array[:, -1].max()
        distance = 2 * sigma * np.sqrt(2)
        tree = spatial.cKDTree(blobs_array[:, :-1])
        pairs = np.array(list(tree.query_pairs(distance)))
        if len(pairs) == 0:
            return blobs_array
        blobs_array = compare_and_filter_pairs(
            pairs, blobs_array, self.dog_overlap, self.dog_min_dist)
        return blobs_array

    def filter_by_hessian_condition(
        self, blob_candidates: np.ndarray, hessian_info: np.ndarray
    ):
        """
        Args:
            blob_candidates (np.ndarray): array with blob candidates as rows
                (y, x, sigma) = (row, col, sigma)
            hessian_info (np.ndarray): HDoG or eigenvalues of HDoG depending the method
        Returns:
            (np.ndarray): array with blob candidates as rows
                (y, x, sigma) = (row, col, sigma)
        """
        blobs_filtered = []
        for sigma in np.unique(blob_candidates[:, 2]):
            idx = np.where(self.sigma_array == sigma)[0][0]

            # Get binary mask from plaussible coordinates by hessian
            if self.method == 'marasinou':
                hessian_mask = self.apply_marasinou_conditions(hessian_info[idx])
            else:
                hessian_mask = self.apply_hess_eigenval_conditions(hessian_info[idx])

            # Get binary mask from blob coordinates
            selected_blobs = np.where(blob_candidates[:, 2] == sigma)[0]
            blob_coords = blob_candidates[selected_blobs, :2].astype(np.int16)
            blob_mask = np.zeros(hessian_mask.shape, dtype=np.int16)
            blob_mask[blob_coords[:, 0], blob_coords[:, 1]] = 1

            # Filter by intersecting the two masks
            filtered_blob_x, filtered_blob_y = np.where(blob_mask * hessian_mask)
            filtered_blobs = np.zeros((len(filtered_blob_x), 3))
            filtered_blobs[:, 0] = filtered_blob_x.astype(np.int16)
            filtered_blobs[:, 1] = filtered_blob_y.astype(np.int16)
            filtered_blobs[:, 2] = sigma
            blobs_filtered.append(filtered_blobs)
        return np.concatenate(blobs_filtered)

    def apply_marasinou_conditions(self, hessian):
        """Filter by determinant and trace conditions over the HDoG
            based on Marasinou et al. 2021 https://arxiv.org/pdf/2102.00754.pdf
        Args:
            hessian (np.ndarray): HDoG
        Returns:
            (np.ndarray): boolean mask of the filtered detections
        """
        # Get trace and determinant
        trace = hessian[0] + hessian[2]
        det = (hessian[0] * hessian[2]) - hessian[1] * hessian[1]
        # Check conditions
        hessian_mask = (np.abs(det) / (trace * trace + 2.23e-16)) <= self.h_thr
        hessian_mask = hessian_mask | (det < 0)
        hessian_mask = hessian_mask & (trace < 0)
        return hessian_mask

    def apply_hess_eigenval_conditions(self, hessian_eigval: np.ndarray):
        """Filter by eigenvalues of HDoG. Based on Muthavel
        Args:
            hessian_eigenval (np.ndarray): eigenvalues of HDoG
        Returns:
            (np.ndarray): boolean mask of the filtered detections
        """
        # Get the maximum persistent eigenvalue
        eig1_ms = hessian_eigval[0] * hessian_eigval[0]
        eig2_ms = hessian_eigval[1] * hessian_eigval[1]

        # Get the proportional threshold
        thrs_1 = np.max(eig1_ms) / self.h_th_div
        thrs_2 = np.max(eig2_ms) / self.h_th_div

        # Check conditions
        hessian_mask = (eig1_ms > thrs_1) & (eig2_ms > thrs_2)
        return hessian_mask

    @staticmethod
    def convert_yxs2xys(detections: np.ndarray):
        """turns (y, x, sigma) = (row, col, sigma) into
                (x, y, sigma) = (col, row, sigma)
        """
        temp = detections.copy()
        detections[:, 0] = temp[:, 1]
        detections[:, 1] = temp[:, 0]
        return detections

    def store_final_detections(
        self, raw_detections: np.ndarray, hess_detections: np.ndarray
    ):
        """ Save the final predictions for easy further data analysis
        Args:
            raw_detections (np.ndarray): Array with detections as rows (x, y, sigma).
                Detections before filtering
            hess_detections (np.ndarray): Array with detections as rows (x, y, sigma).
                Detections after filtering
        """
        self.final_detections_path.parent.mkdir(parents=True, exist_ok=True)
        mode = 'a' if self.final_detections_path.exists() else 'w'
        with h5py.File(self.final_detections_path, mode) as f:
            if f'{self.image_id}/raw_detections' not in f:
                _ = f.create_dataset(
                    f'{self.image_id}/raw_detections', data=raw_detections
                )
            if f'{self.image_id}/hessian_detections' not in f:
                _ = f.create_dataset(
                    f'{self.image_id}/hessian_detections', data=hess_detections
                )

    def delete_hdog_file(self):
        subprocess.call(['rm', str(self.hdog_image_path)])
        subprocess.call(['rm', str(self.raw_detections_path)])
