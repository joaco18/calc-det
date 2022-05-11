import multiprocessing as mp

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from dask import delayed
from general_utils.utils import (crop_patch_around_center, min_max_norm,
                                 patch_coordinates_from_center)
from joblib import delayed
from pywt import dwt2
from radiomics import featureextractor
from scipy.stats import kurtosis, skew
from skimage.feature import greycomatrix, greycoprops
from tqdm import tqdm

from feature_extraction.haar_features.haar_extractor import (
    HaarFeatureExtractor, extract_haar_feature_image_skimage)

# machine epsillon used to avoid zero errors
epsillon = np.finfo(float).eps

wavelet_decomp_names = ['LL1', 'LH1', 'HL1', 'HH1', 'LL2', 'LH2', 'HL2', 'HH2']
glcm_decompositions = ['LH1', 'HL1', 'HH1']
skimage_glcm_features = ['energy', 'correlation',
                         'homogeneity', 'contrast', 'dissimilarity']


class CandidatesFeatureExtraction:
    def __init__(self, patch_size: int, fos=True, gabor_params=None, wavelt_features=None, haar_params=None, center_crop_size=7):
        """Defines which features to extract

        Args:
            patch_size (int): size of the patch extracted around each candidate and
                used for FE
            fos (bool): whether to extract 17 first order statistics features
            gabor_params (dict): parameters for gabor feature bank
            wavelt_features (dict): parameters for wavelt glcm f.e.
                needs key 'angle' with a list of angles to calculate GLCMs for.
            haar_params (dict): parameters for haar features extractor
            center_crop_size (int): size of the patch center crop to consider
                when looking for intersection in mask while defining a TP
        """
        self.patch_size = patch_size
        self.fos = fos
        self.gabor_params = gabor_params
        self.wavelt_features = wavelt_features
        self.haar_params = haar_params
        self.center_crop_size = center_crop_size

        # used to store calculated features names
        self.get_feature_names()

    def get_feature_names(self):
        self.feature_names = []

        if self.fos:
            self.feature_names.extend(['img_energy', 'img_entropy', 'img_uniformity',
                                       'img_min', 'img_10th_perc', 'img_90th_perc',
                                       'img_max', 'img_mean', 'img_median',
                                       'img_inter_quartile_range', 'img_range',
                                       'img_mean_abs_deviation',
                                       'img_robust_mean_abs_deviation', 'img_rms',
                                       'img_std', 'img_skew', 'img_kurt'])
        if self.gabor_params:
            for img_idx in range(self.gabor_params['scale']*self.gabor_params['orientation']):
                self.feature_names.extend([f'gabor_energy_{img_idx}',
                                           f'gabor_mean_{img_idx}',
                                           f'gabor_std_{img_idx}',
                                           f'gabor_skew_{img_idx}',
                                           f'gabor_kurt_{img_idx}'])

        if self.wavelt_features:
            for decomp_name in wavelet_decomp_names:
                self.feature_names.extend([f'patch_mean_{decomp_name}',
                                           f'patch_skew_{decomp_name}',
                                           f'patch_std_{decomp_name}',
                                           f'patch_kur_{decomp_name}',
                                           f'patch_entropy_{decomp_name}',
                                           f'patch_uniformity_{decomp_name}',
                                           f'patch_relsmooth_{decomp_name}'])
            for fn in skimage_glcm_features:
                for dn in glcm_decompositions:
                    for angle_idx in range(len(self.wavelt_features['angles'])):
                        self.feature_names.append(
                            f'patch_glcm_{fn}_{dn}_{angle_idx}')

        # some debug features for now
        self.feature_names.extend(['candidate_coordinates',
                                   'patch_coordinates',
                                   'center_patch_mask_intersection'])

    def extract_features(
            self, candidates: np.ndarray, image: np.ndarray, roi_mask: np.ndarray, fp2tp_sample=None):
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

        if self.haar_params:
            features_haar = self.haar_features_extraction(image, candidates)
            candidate_coordinates = []
            patch_coordinates = []
            patch_mask_intersection = []

        # TODO: paralelize with Pool
        for coords in candidates:
            # calculating canidate cropping patch coordinates
            patch_x1, patch_x2, patch_y1, patch_y2 = patch_coordinates_from_center(
                (coords[0], coords[1]), image.shape, self.patch_size, use_padding=False)

            # getting coordinates of the patch center crop
            center_px1, center_px2, center_py1, center_py2 = crop_patch_around_center(
                patch_x1, patch_x2, patch_y1, patch_y2, self.center_crop_size)
            image_patch = image[patch_y1:patch_y2, patch_x1:patch_x2]

            if not self.haar_params:
                # extracting features
                features = []

                # First order statistics features
                if self.fos:
                    features.extend(self.first_order_statistics(image_patch))

                # Gabor features
                if self.gabor_params:
                    features.extend(self.gabor_features(
                        gabored_images, patch_x1, patch_x2, patch_y1, patch_y2))

                # Wavelet and GLCM features
                if self.wavelt_features:
                    features.extend(self.get_wavelet_features(image_patch))

                features.append(coords)
                features.append(((patch_y1, patch_y2), (patch_x1, patch_x2)))
                # cropping center of the patch now
                features.append(
                    (roi_mask[center_py1:center_py2, center_px1:center_px2] > 0).sum())
                candidates_features.append(features)
            else:
                candidate_coordinates.append(coords)
                patch_coordinates.append(
                    ((patch_y1, patch_y2), (patch_x1, patch_x2)))

                # saving the intersection between the center of the patch and the mask
                patch_mask_intersection.append(
                    (roi_mask[center_py1:center_py2, center_px1:center_px2] > 0).sum())

        if self.haar_params:
            candidates_features = pd.DataFrame(
                features_haar, columns=[f'f{i}' for i in range(features_haar.shape[1])])
            candidates_features['candidate_coordinates'] = candidate_coordinates
            candidates_features['patch_coordinates'] = patch_coordinates
            candidates_features['patch_mask_intersection'] = patch_mask_intersection
        return candidates_features

    def split_sample_candidates(self, candidates, roi_mask, sample):
        """Samples given candidates list to obtain a given proportion of TPxFP in it.

        First selects all TP canidates and then randomly sampled a required number of FP.
        """
        TP_idxs = []
        FP_idxs = []
        for coords_idx, coords in enumerate(candidates):
            # getting patch coordinates
            patch_x1, patch_x2, patch_y1, patch_y2 = patch_coordinates_from_center(
                (coords[0], coords[1]), roi_mask.shape, self.patch_size, use_padding=False)
            # getting patch centre crop coordinates 
            center_px1, center_px2, center_py1, center_py2 = crop_patch_around_center(
                patch_x1, patch_x2, patch_y1, patch_y2, self.center_crop_size)

            if np.any(roi_mask[center_py1:center_py2, center_px1:center_px2] > 0):
                TP_idxs.append(coords_idx)
            else:
                FP_idxs.append(coords_idx)
        TP_idxs.extend(np.random.choice(
            FP_idxs, size=len(TP_idxs)*sample, replace=False))
        return TP_idxs

    def haar_features_extraction(self, image: np.ndarray, detections: np.ndarray):
        """Get horizontal haar features from skimage and rotated ones from our code"""
        images = np.empty((len(detections), 14, 14))
        # generate a patches array to distribute computation
        for j, location in enumerate(detections):
            # Get the patch arround center
            x1, x2, y1, y2 = patch_coordinates_from_center(
                center=(location[0], location[1]), image_shape=image.shape,
                patch_size=14, use_padding=False)
            images[j, :, :] = image[y1:y2, x1:x2]
        # Generate computational graph
        X = delayed(extract_haar_feature_image_skimage(img) for img in images)
        # Compute the result
        X = np.array(X.compute(scheduler='processes'))

        # Rotated haar_features
        haarfe = HaarFeatureExtractor(14, False, True)
        X_r = []
        for img in images:
            X_r.append(haarfe.extract_features_from_crop(img))
        X_r = np.asarray(X_r)

        X = np.concatenate([X, X_r], axis=1)
        return X

    @staticmethod
    def entropy_uniformity(image_patch):
        """Calculates image entropy and uniformity"""
        _, counts = np.unique(image_patch, return_counts=True)
        norm_counts = counts / counts.sum()
        uniformity = (norm_counts**2).sum()
        return -(norm_counts * np.log2(norm_counts + epsillon)).sum(), uniformity

    def first_order_statistics(self, image_patch):
        """Calculates first-order statistics as defined in 
        https://pyradiomics.readthedocs.io/en/latest/features.html#radiomics.firstorder.RadiomicsFirstOrder

        Args:
            image_patch (np.ndarray): Image array

        Returns:
            np.ndarray: with feature values and names
        """
        patch_size = image_patch.shape[0]*image_patch.shape[1]

        img_energy = (image_patch**2).sum()
        img_entropy, img_uniformity = CandidatesFeatureExtraction.entropy_uniformity(
            image_patch)
        img_min = image_patch.min()
        img_10th_perc = np.quantile(image_patch, q=0.1)
        img_90th_perc = np.quantile(image_patch, q=0.9)
        img_max = np.max(image_patch)
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

        fos_features = np.asarray([img_energy, img_entropy, img_uniformity, img_min, img_10th_perc,
                                   img_90th_perc, img_max, img_mean, img_median,
                                   img_inter_quartile_range, img_range, img_mean_abs_deviation,
                                   img_robust_mean_abs_deviation, img_rms, img_std, img_skew,
                                   img_kurt])
        return fos_features

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
        """Extracts energy, mean, std, skeweness and kurtosis from patches
            of images filtered with gabor kernel

        Args:
            gabored_images (list): list of images filtered with Gabor kernels
        Returns:
            np.ndarray: of 4_features*n_gabored_images 
        """
        features = []
        for filtered_image in gabored_images:
            img_patch = filtered_image[patch_y1:patch_y2, patch_x1:patch_x2]

            features.extend([(img_patch**2).sum(),
                             np.mean(img_patch),
                             np.std(img_patch),
                             skew(img_patch.ravel()),
                             kurtosis(img_patch.ravel())])
        return np.asarray(features)

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
        wavelt_features = []

        for idx, single_decomp in enumerate(eight_decomp):
            wavelt_features.extend(
                CandidatesFeatureExtraction.wav_first_order(single_decomp, idx))

        for idx, single_decomp in enumerate(eight_decomp[1:4]):
            wavelt_features.extend(
                CandidatesFeatureExtraction.wav_glcm_features(single_decomp, idx))

        return np.asarray(wavelt_features)

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
        patch_mean = np.mean(single_decomp)
        patch_std = np.std(single_decomp)
        patch_skew = skew(single_decomp.ravel())
        patch_kurt = kurtosis(single_decomp.ravel())
        patch_entropy, patch_unif = CandidatesFeatureExtraction.entropy_uniformity(
            single_decomp)
        patch_relsmooth = 1 - 1/(1+patch_std)

        return np.asarray([patch_mean,
                           patch_skew,
                           patch_std,
                           patch_kurt,
                           patch_entropy,
                           patch_unif,
                           patch_relsmooth])

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

        single_decomp_glcm = greycomatrix(min_max_norm(
            single_decomp, max_val=256).astype(np.uint8), [2], [0], normed=True)

        glcm_features_1 = []
        for feature_name in skimage_glcm_features:
            feature_results = greycoprops(
                single_decomp_glcm, prop=feature_name)
            for fv in feature_results.ravel():
                glcm_features_1.append(fv)

        return np.asarray(glcm_features_1)
