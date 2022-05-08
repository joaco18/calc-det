import multiprocessing as mp

import cv2
import numpy as np
import SimpleITK as sitk
from general_utils.utils import crop_center_coords, min_max_norm
from radiomics import featureextractor
from scipy.stats import kurtosis, skew
from tqdm import tqdm

# machine epsillon used to avoid zero errors
epsillon = np.finfo(float).eps


class CandidatesFeatureExtraction:
    def __init__(self, patch_size: int, gabor_params=None):
        """Defines which features to extract

        # TODO: add FE parameters to specify what FE to extract

        Args:
            patch_size (int): Size of the patch extracted around each candidate and
                used for FE
        """
        self.patch_size = patch_size
        self.gabor_params = gabor_params

    def extract_features(self, candidates: np.ndarray, image: np.ndarray, roi_mask: np.ndarray, fp2tp_sample=None):
        """Extracts features from image patches cropped around given candidates.

        Args:
            candidates (np.ndarray): of dtype=int containing candidats for FE of shape (n_candidates, 3).
                Second axis should contain (x_coord, y_coord, radius).
            image (np.ndarray): 2D image used for cropping and FE
            roi_mask (np.ndarray): true rois mask
            fp2tp_sample (int, optional): number of FP candidates to sample for each TP candidate.
                If not None, total number of sampled candidates will be len(TP_candidates)*(fp2tp_sample + 1).
                Defaults to None which means no sampling is performed and features for all candidates are
                extracted.

        Returns:
            list[dict]: containing dictionaries with candidate features
        """
        # sample candidates to a given size if needed
        if fp2tp_sample is not None:
            cand_idxs = self.split_sample_candidates(
                candidates, roi_mask, fp2tp_sample)
            candidates = candidates[cand_idxs]

        image = min_max_norm(image, max_val=1.)
        candidates_features = []
        # iterating over candidates and cropping patches

        if self.gabor_params:
            gabor_kernels = gabor_kernels = self.gabor_feature_bank(
                **self.gabor_params)
            gabored_images = [cv2.filter2D(
                image, ddepth=cv2.CV_32F, kernel=k) for k in gabor_kernels]

        # TODO: paralelize with Pool
        for coords in candidates:

            # calculating canidate cropping patch coordinates
            patch_x1, patch_x2, patch_y1, patch_y2 = crop_center_coords(
                coords[0], coords[1], image.shape, self.patch_size//2)
            image_patch = image[patch_y1:patch_y2, patch_x1:patch_x2]

            # extracting features
            features = {}

            # First order statistics
            features = features | self.first_order_statistics(image_patch)

            # Gabor features
            if self.gabor_params:
                features = features | self.gabor_features(
                    gabored_images, patch_x1, patch_x2, patch_y1, patch_y2)

            # TODO: Other features extraction
            # features = features | other_features
            features['candidate_coordinates'] = coords
            features['patch_coordinates'] = (
                (patch_y1, patch_y2), (patch_x1, patch_x2))
            features['patch_mask_intersection'] = (roi_mask[patch_y1:patch_y2,
                                                            patch_x1:patch_x2] > 0).sum()
            candidates_features.append(features)

        return candidates_features

    def split_sample_candidates(self, candidates, roi_mask, sample):
        """Samples given candidates list to obtain a given proportion of TPxFP in it.

        First selects all TP canidates and then randomly sampled a required number of FP.
        """
        TP_idxs = []
        FP_idxs = []
        for coords_idx, coords in enumerate(candidates):
            patch_x1, patch_x2, patch_y1, patch_y2 = crop_center_coords(
                coords[0], coords[1], roi_mask.shape, self.patch_size//2)
            if np.any(roi_mask[patch_y1:patch_y2, patch_x1:patch_x2] > 0):
                TP_idxs.append(coords_idx)
            else:
                FP_idxs.append(coords_idx)
        TP_idxs.extend(np.random.choice(
            FP_idxs, size=len(TP_idxs)*sample, replace=False))
        return TP_idxs

    @staticmethod
    def entropy_uniformity(image_patch):
        """Calculates image entropy and uniformity"""
        _, counts = np.unique(image_patch, return_counts=True)
        norm_counts = counts / counts.sum()
        uniformity = (norm_counts**2).sum()
        return -(norm_counts * np.log2(norm_counts + epsillon)).sum(), uniformity

    @staticmethod
    def first_order_statistics(image_patch, flag=''):
        """Calculates first-order statistics as defined in 
        https://pyradiomics.readthedocs.io/en/latest/features.html#radiomics.firstorder.RadiomicsFirstOrder

        Args:
            image_patch (np.ndarray): Image array

        Returns:
            dict: with feature values and names
        """
        patch_size = image_patch.shape[0]*image_patch.shape[1]

        img_energy = (image_patch**2).sum()
        img_entropy, img_uniformity = CandidatesFeatureExtraction.entropy_uniformity(
            image_patch)
        img_min = image_patch.min()
        img_10th_perc = np.quantile(image_patch, q=0.1)
        img_90th_perc = np.quantile(image_patch, q=0.9)
        img_max = np.quantile(image_patch, q=0.1)
        img_mean = np.mean(image_patch)
        img_meadian = np.median(image_patch)
        img_inter_quartile_range = np.quantile(
            image_patch, q=0.75) - np.quantile(image_patch, q=0.25)
        img_range = img_max - img_min
        img_mean_abs_deviation = np.abs(
            image_patch - img_mean).sum()/patch_size

        robust_img = image_patch[(image_patch <= img_90th_perc) & (
            image_patch >= img_10th_perc)]
        img_robust_mean_abs_deviation = np.abs(
            robust_img - robust_img.mean()).sum()/len(robust_img)

        img_rms = np.sqrt((image_patch**2).sum()/patch_size)
        img_std = np.std(image_patch)
        img_skew = skew(image_patch.ravel())
        img_kurt = kurtosis(image_patch.ravel())

        return {f'img_energy{flag}': img_energy,
                f'img_entropy{flag}': img_entropy,
                f'img_uniformity{flag}': img_uniformity,
                f'img_min{flag}': img_min,
                f'img_10th_perc{flag}': img_10th_perc,
                f'img_90th_perc{flag}': img_90th_perc,
                f'img_max{flag}': img_max,
                f'img_mean{flag}': img_mean,
                f'img_meadian{flag}': img_meadian,
                f'img_inter_quartile_range{flag}': img_inter_quartile_range,
                f'img_range{flag}': img_range,
                f'img_mean_abs_deviation{flag}': img_mean_abs_deviation,
                f'img_robust_mean_abs_deviation{flag}': img_robust_mean_abs_deviation,
                f'img_rms{flag}': img_rms,
                f'img_std{flag}': img_std,
                f'img_skew{flag}': img_skew,
                f'img_kurt{flag}': img_kurt}

    def gabor_feature_bank(self, scale, orientation, max_freq=0.2, ksize=(50, 50), sigma=1, gamma=0.5, psi=0):
        orientations = [(i*np.pi)/orientation for i in range(orientation)]
        frequencies = [(max_freq)/(np.sqrt(2)**i) for i in range(scale)]
        gabor_kernels = []
        for orient in orientations:
            for freq in frequencies:
                gabor_kernels.append(cv2.getGaborKernel(
                    ksize=ksize, sigma=sigma, theta=orient, lambd=1/freq, gamma=gamma, psi=psi))

        return gabor_kernels

    def gabor_features(self, gabored_images, patch_x1, patch_x2, patch_y1, patch_y2):
        features = {}
        for img_idx, filtered_image in enumerate(gabored_images):
            img_patch = filtered_image[patch_y1:patch_y2, patch_x1:patch_x2]

            features[f'gabor_energy_{img_idx}'] = (img_patch**2).sum()
            features[f'gabor_mean_{img_idx}'] = np.mean(img_patch)
            features[f'gabor_std_{img_idx}'] = np.std(img_patch)
            features[f'gabor_skew_{img_idx}'] = skew(img_patch.ravel())
            features[f'gabor_kurt_{img_idx}'] = kurtosis(img_patch.ravel())
        return features
