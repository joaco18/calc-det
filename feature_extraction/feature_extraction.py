import multiprocessing as mp

import cv2
import numpy as np
import SimpleITK as sitk
from general_utils.utils import crop_center_coords, min_max_norm
from pywt import dwt2
from radiomics import featureextractor
from scipy.stats import kurtosis, skew
from skimage.feature import greycomatrix, greycoprops
from tqdm import tqdm

# machine epsillon used to avoid zero errors
epsillon = np.finfo(float).eps


class CandidatesFeatureExtraction:
    def __init__(self, patch_size: int, gabor_params=None, wavelt_features=None):
        """Defines which features to extract

        # TODO: add FE parameters to specify what FE to extract

        Args:
            patch_size (int): Size of the patch extracted around each candidate and
                used for FE
        """
        self.patch_size = patch_size
        self.gabor_params = gabor_params
        self.wavelt_features = wavelt_features

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

        # hhar_features = J(candidates, image)
        # TODO: paralelize with Pool
        for coords in candidates:

            # calculating canidate cropping patch coordinates
            patch_x1, patch_x2, patch_y1, patch_y2 = crop_center_coords(
                coords[0], coords[1], image.shape, self.patch_size//2)
            image_patch = image[patch_y1:patch_y2, patch_x1:patch_x2]

            # extracting features
            features = {}

            # First order statistics features
            features = features | self.first_order_statistics(image_patch)

            # Gabor features
            if self.gabor_params:
                features = features | self.gabor_features(
                    gabored_images, patch_x1, patch_x2, patch_y1, patch_y2)

            # Wavelet and GLCM features
            if self.wavelt_features:
                features = features | self.get_wavelet_features(image_patch)

            # TODO: Other features extraction
            # features = features | other_features

            features['candidate_coordinates'] = coords
            features['patch_coordinates'] = (
                (patch_y1, patch_y2), (patch_x1, patch_x2))
            features['patch_mask_intersection'] = (roi_mask[patch_y1:patch_y2,
                                                            patch_x1:patch_x2] > 0).sum()
            candidates_features.append(features)
        # merge J and candidates_features
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
        img_median = np.median(image_patch)
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
                f'img_meadian{flag}': img_median,
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

    @staticmethod
    def get_wavelet_features(patch: np.ndarray):
        """ Extracts features from an patch's haar 2-level wavelet decomposition (8 decompositions)
        First order statistics: mean, skewness, standard deviation, kurtosis, entropy, uniformity, relative smoothness
        GLCM features (D=5,theta = 0) [note: only in LH1, HL1 and HH1]: energy, correlation, homogeneity, contrast, 
        dissimilarity, (NOTRELEVANT: ASM, entropy, uniformity, sum of squares, autocorrelation)    

        Args:
            patch (np.ndarray): image patch (normalized)

        Returns:
            features (dict): dictionary containing all extracted features
        """
        eight_decomp = CandidatesFeatureExtraction.get_wavelet_decomp(patch)
        fo_features = {}
        for idx, single_decomp in enumerate(eight_decomp):
            fo_features = fo_features | CandidatesFeatureExtraction.wav_first_order(
                single_decomp, idx)
        glcm_features = {}
        for idx, single_decomp in enumerate(eight_decomp[1:4]):
            glcm_features = glcm_features | CandidatesFeatureExtraction.wav_glcm_features(
                single_decomp, idx)

        return fo_features | glcm_features

    @staticmethod
    def get_wavelet_decomp(patch: np.ndarray, wavelet_type='haar'):
        """ Gets wavelet decomposition of two levels with approximation and detail coefficients.
        Uses decimated 2D forward Discrete Wavelet Transform

        Args:
            patch (np.ndarray): image patch (normalized)
            wavelet_type (str, optional): type of wavelet for decomposition. Defaults to 'haar'.

        Returns:
            decompositions (list[np.ndarray]): list of decompositions [LL1,LH1,HL1,HH1,LL2,LH2,HL2,HH2]
        """
        LL1, (LH1, HL1, HH1) = dwt2(patch, wavelet_type)
        LL2, (LH2, HL2, HH2) = dwt2(LL1, wavelet_type)

        return [LL1, LH1, HL1, HH1, LL2, LH2, HL2, HH2]

    @staticmethod
    def wav_first_order(single_decomp: np.ndarray, idx: int):
        """ Extracts first order statistics from a single wavelet decomposition
        Features: mean, skewness, standard deviation, kurtosis, entropy, uniformity, relative smoothness

        Args:
            single_decomp (np.ndarray): single decomposition from the list [LL1,LH1,HL1,HH1,LL2,LH2,HL2,HH2]
            Note: expected to be used on the eight decompositions for correct feature naming
            idx (int): index of the decomposition from list above

        Returns:
            first_order_features (dict): dictionary containing first order features
        """
        decomp_names = ['LL1', 'LH1', 'HL1', 'HH1', 'LL2', 'LH2', 'HL2', 'HH2']
        patch_mean = np.mean(single_decomp)
        patch_std = np.std(single_decomp)
        patch_skew = skew(single_decomp.ravel())
        patch_kurt = kurtosis(single_decomp.ravel())
        patch_entropy, patch_unif = CandidatesFeatureExtraction.entropy_uniformity(
            single_decomp)
        patch_relsmooth = 1 - 1/(1+patch_std)

        return {f'patch_mean_{decomp_names[idx]}': patch_mean,
                f'patch_skew_{decomp_names[idx]}': patch_skew,
                f'patch_std_{decomp_names[idx]}': patch_std,
                f'patch_kur_{decomp_names[idx]}': patch_kurt,
                f'patch_entropy_{decomp_names[idx]}': patch_entropy,
                f'patch_uniformity_{decomp_names[idx]}': patch_unif,
                f'patch_relsmooth_{decomp_names[idx]}': patch_relsmooth}

    @staticmethod
    def wav_glcm_features(single_decomp: np.ndarray, idx: int):
        """Extracts features from a Gray Level Co-occurence Matrix (D=5,theta = 0) from a single wavelet decomposition
        Features: energy, correlation, homogeneity, contrast, dissimilarity,(DROPPED: ASM, entropy, uniformity, sum of squares, autocorrelation)

        Args:
            single_decomp (np.ndarray): single decomposition from list [LH1, HL1,HH1]
            Note: expected to be used on three decompositions for correct feature naming
            idx (int): index of the decomposition from list above

        Returns:
            glcm_features (dict): dictionary containing glcm features
        """
        decomp_names = ['LH1', 'HL1', 'HH1']
        skimage_glcm_features = ['energy', 'correlation',
                                 'homogeneity', 'contrast', 'dissimilarity']  # 'ASM'

        single_decomp_glcm = greycomatrix(min_max_norm(
            single_decomp, max_val=256).astype(np.uint8), [2], [0], normed=True)

        glcm_features_1 = {}
        for feature_name in skimage_glcm_features:
            feature_results = greycomatrix(
                single_decomp_glcm, prop=feature_name)
            for fv in feature_results.ravel():
                glcm_features_1[f'patch_glcm_{feature_name}_{decomp_names[idx]}'] = fv

        # glcm_features_2 = {}
        # for glcm in single_decomp_glcm[:,:,0,:]:
        #     entropy, uniformity = CandidatesFeatureExtraction.entropy_uniformity(glcm)
        #     sum_squares = (glcm*(1- glcm.mean())**2).sum()
        #     idx_grid = np.indices((glcm.shape[0], glcm.shape[1]))
        #     autocorrelation = (idx_grid[0, :, :]*idx_grid[1, :, :]*glcm).sum()

        #     glcm_features_2[f'patch_glcm_entropy_{decomp_names[idx]}'] = entropy
        #     glcm_features_2[f'patch_glcm_uniformity_{decomp_names[idx]}'] = uniformity
        #     glcm_features_2[f'patch_glcm_sum_squares_{decomp_names[idx]}'] = sum_squares
        #     glcm_features_2[f'patch_glcm_autocorrelation_{decomp_names[idx]}'] = autocorrelation

        return glcm_features_1
