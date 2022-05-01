import multiprocessing as mp

import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from scipy.stats import kurtosis, skew
from tqdm import tqdm

from general_utils.utils import crop_center_coords, min_max_norm

epsillon = np.finfo(float).eps


class CandidatesFeatureExtraction:
    def __init__(self, patch_size: int):
        """Defines which features to extract and from which images.
        
        Args:
            patch_size (int): Size of the patch extracted around each candidate and
                used for FE
        """
        self.patch_size = patch_size

    def extract_features(self, candidates: np.ndarray, image: np.ndarray, roi_mask: np.ndarray, sample=None):
        """Extracts features from image patches cropped around given candidates.

        Args:
            candidates (np.ndarray): containing candidate coordinates for FE of shape (n_candidates, 3)
            image (np.ndarray): 2D image used for cropping and FE
            roi_mask (np.ndarray): mask containing whether given pixel corresponds to a positive class (mC)
            sample (int, optional): number of candidates FP to sample per each TP candidate

        Returns:
            list[dict]: containing dictionaries with candidates features
        """
        # sample candidates to a given size if needed
        if sample is not None:
            cand_idxs = self.split_sample_candidates(candidates, roi_mask, sample)
            candidates = candidates[cand_idxs]
        
        image = min_max_norm(image, max_val=1.)
        candidates_features = []
        # iterating over candidates and cropping patches
    
        for coords in candidates:
            
            # calculating patch coords
            patch_x1, patch_x2, patch_y1, patch_y2 = crop_center_coords(coords[0], coords[1], image.shape, self.patch_size//2)
            image_patch = image[patch_y1:patch_y2, patch_x1:patch_x2]
            
            # extracting features
            features = {}
            # First order statistics
            features = features | self.fof(image_patch)
            
            # TODO: Other features extraction
            # features = features | other_features
            
            features['coordinates'] = (
                (patch_y1, patch_y2), (patch_x1, patch_x2))
            features['patch_mask_intersection'] = (roi_mask[patch_y1:patch_y2,
                                                            patch_x1:patch_x2] > 0).sum()
            candidates_features.append(features)

        return candidates_features

    def split_sample_candidates(self, candidates, roi_mask, sample):
        TP_idxs = []
        FP_idxs = []
        for coords_idx, coords in enumerate(candidates):
            patch_x1, patch_x2, patch_y1, patch_y2 = crop_center_coords(coords[0], coords[1], roi_mask.shape, self.patch_size//2)
            if np.any(roi_mask[patch_y1:patch_y2, patch_x1:patch_x2] > 0):
                TP_idxs.append(coords_idx)
            else:
                FP_idxs.append(coords_idx)
        TP_idxs.extend(np.random.choice(FP_idxs, size=len(TP_idxs)*sample, replace=False))
        return TP_idxs
    
    @staticmethod
    def entropy_uniformity(image_patch):
        _, counts = np.unique(image_patch, return_counts=True)
        norm_counts = counts / counts.sum()
        uniformity = (norm_counts**2).sum()
        return -(norm_counts * np.log2(norm_counts + epsillon)).sum(), uniformity

    @staticmethod
    def fof(image_patch):
        """Calculates first-order statistics as defined in 
        https://pyradiomics.readthedocs.io/en/latest/features.html#radiomics.firstorder.RadiomicsFirstOrder

        Args:
            image_patch (np.ndarray): Image array

        Returns:
            dict: with feature values and names
        """
        patch_size = image_patch.shape[0]*image_patch.shape[1]

        img_energy = (image_patch**2).sum()
        img_entropy, img_uniformity = CandidatesFeatureExtraction.entropy_uniformity(image_patch)
        img_min = image_patch.min()
        img_10th_perc = np.quantile(image_patch, q=0.1)
        img_90th_perc = np.quantile(image_patch, q=0.9)
        img_max = np.quantile(image_patch, q=0.1)
        img_mean = np.mean(image_patch)
        img_meadian = np.median(image_patch)
        img_inter_quartile_range = np.quantile(
            image_patch, q=0.75) - np.quantile(image_patch, q=0.25)
        img_range = img_max - img_min
        img_mean_abs_deviation = np.abs(image_patch - img_mean).sum()/patch_size

        robust_img = image_patch[(image_patch <= img_90th_perc) & (
            image_patch >= img_10th_perc)]
        img_robust_mean_abs_deviation = np.abs(
            robust_img - robust_img.mean()).sum()/len(robust_img)

        img_rms = np.sqrt((image_patch**2).sum()/patch_size)
        img_std = np.std(image_patch)
        img_skew = skew(image_patch.ravel())
        img_kurt = kurtosis(image_patch.ravel())

        return {'img_energy': img_energy,
                'img_entropy': img_entropy,
                'img_uniformity': img_uniformity,
                'img_min': img_min,
                'img_10th_perc': img_10th_perc,
                'img_90th_perc': img_90th_perc,
                'img_max': img_max,
                'img_mean': img_mean,
                'img_meadian': img_meadian,
                'img_inter_quartile_range': img_inter_quartile_range,
                'img_range': img_range,
                'img_mean_abs_deviation': img_mean_abs_deviation,
                'img_robust_mean_abs_deviation': img_robust_mean_abs_deviation,
                'img_rms': img_rms,
                'img_std': img_std,
                'img_skew': img_skew,
                'img_kurt': img_kurt}