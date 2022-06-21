import sys
from pathlib import Path
sys.path.insert(0, Path.cwd().parent)
import cv2
import pickle 
import logging
import numpy as np
import multiprocessing as mp

from dask import delayed
from pywt import dwt2
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops, haar_like_feature_coord

import general_utils.utils as utils
from feature_extraction.haar_features.haar_extractor import \
    extract_haar_feature_image_skimage, HaarFeatureExtractor

import feature_extraction.haar_features.haar_modules as hm

# machine epsillon used to avoid zero errors
epsillon = np.finfo(float).eps

wavelet_decomp_names = ['LL1', 'LH1', 'HL1', 'HH1', 'LL2', 'LH2', 'HL2', 'HH2']
glcm_decompositions = ['LH1', 'HL1', 'HH1']
skimage_glcm_features = ['correlation',
                         'homogeneity', 'contrast', 'dissimilarity']


# FE hyperparameters
CENTER_CROP_PATCH = 14
PATCH_SIZE = 14

GABOR_PARAMS = {'scale': 2, 'orientation': 3,
                'max_freq': 0.2, 'ksize': (20, 20), 'sigma': 3}
WAVELET_PARAMS = {'angles': [0, np.pi/4, np.pi/2]}

this_path = Path(__file__).parent.resolve()
haar_feat_path = this_path.parent / 'data/final_haar_500_feat_selection_gsm.p' #'/home/vzalevskyi/projects/calc-det/data/final_haar_500_feat_selection_gsm.p'
with open(haar_feat_path, 'rb') as f:
    selection = pickle.load(f)
    
HAAR_PARAMS = {
    'skimage': {
        'feature_type': selection['skimage_haar_feature_types_sel'],
        'feature_coord': selection['skimage_haar_feature_coords_sel']
    },
    'ours': {
        'horizontal_feature_selection': selection['hor_feats_selection'].tolist(),
        'rotated_feature_selection': selection['rot_feats_selection'].tolist(),
        'rotated_feature_types': None,
        'horizontal_feature_types': None
    },
    'patch_size': 14
}


logging.basicConfig(level=logging.INFO)


class CandidatesFeatureExtraction:
    def __init__(self, patch_size: int, fos=True, gabor_params=None, wavelet_params=None,
                 haar_params=None):
        """Defines which features to extract

        Args:
            patch_size (int): size of the patch extracted around each candidate and
                used for FE
            fos (bool): whether to extract 17 first order statistics features
            gabor_params (dict): parameters for gabor feature bank
            wavelet_params (dict): parameters for wavelt glcm f.e.
                needs key 'angle' with a list of angles to calculate GLCMs for.
            haar_params (dict): parameters for haar features extractor

        """
        self.patch_size = patch_size
        self.fos = fos
        self.gabor_params = gabor_params
        self.wavelet_params = wavelet_params
        self.haar_params = haar_params

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
                                           f'gabor_max_{img_idx}',
                                           f'gabor_min_{img_idx}',
                                           f'gabor_mean_{img_idx}',
                                           f'gabor_std_{img_idx}',
                                           f'gabor_skew_{img_idx}',
                                           f'gabor_kurt_{img_idx}'])

        if self.wavelet_params:
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
                    for angle_idx in range(len(self.wavelet_params['angles'])):
                        self.feature_names.append(
                            f'patch_glcm_{fn}_{dn}_{angle_idx}')
        # some debug features for now
        self.feature_names.extend(['candidate_coordinates',
                                   'patch_coordinates'])

    def extract_features(self, candidates: np.ndarray, image: np.ndarray):
        """Extracts features from image patches cropped around given candidates.
        Args:
            candidates (np.ndarray): of dtype=int containing candidats for FE of shape
                (n_candidates, 3). Second axis should contain (x_coord, y_coord, radius).
            image (np.ndarray): 2D image used for cropping and FE
        Returns:
            np.array: rows candidates, columns (features + 3 columns of metadata)
        """
        image = utils.min_max_norm(image, max_val=1.)
        candidates_features = []

        if self.gabor_params:
            gabor_kernels = gabor_kernels = self.gabor_feature_bank(
                **self.gabor_params)
            gabored_images = [cv2.filter2D(
                image, ddepth=cv2.CV_32F, kernel=k) for k in gabor_kernels]

        if self.haar_params:
            features_haar = self.haar_features_extraction(image, candidates)

        for coords in candidates:
            # calculating canidate cropping patch coordinates
            patch_x1, patch_x2, patch_y1, patch_y2 = utils.patch_coordinates_from_center(
                (coords[0], coords[1]), image.shape, self.patch_size)
            image_patch = image[patch_y1:patch_y2, patch_x1:patch_x2]

            # Extracting non-haar features
            features = []

            # First order statistics features
            if self.fos:
                features.extend(self.first_order_statistics(image_patch))

            # Gabor features
            if self.gabor_params:
                features.extend(self.gabor_features(
                    gabored_images, patch_x1, patch_x2, patch_y1, patch_y2))

            # Wavelet and GLCM features
            if self.wavelet_params:
                features.extend(self.get_wavelet_features(image_patch))

            # Append metadata columns
            features.append(coords)
            features.append(((patch_y1, patch_y2), (patch_x1, patch_x2)))
            candidates_features.append(features)
        
        candidates_features = np.asarray(candidates_features, dtype=object)
        
        if self.haar_params:
            candidates_features = np.concatenate(
                [features_haar, candidates_features], axis=1)
        return candidates_features

    def haar_features_extraction(self, image: np.ndarray, candidates: np.ndarray):
        """Get horizontal haar features from skimage and rotated ones from our code.
        features names are modified to include this features.
        Args:
            image (np.ndarray): Image array
            candidates (np.ndarray): candidate detections [[x,y,r]]
        Returns:
            np.ndarray: with feature values
        """
        images = utils.img_to_patches_array(
            image, candidates, self.haar_params['patch_size'])

        # Generate a list of features for easy access for feature selection
        if ((self.haar_params['skimage']['feature_coord'] is None) and
                (self.haar_params['skimage']['feature_type'] is None)):
            self.skimage_haar_feature_coords, self.skimage_haar_feature_types = \
                haar_like_feature_coord(
                    width=self.haar_params['patch_size'], height=self.haar_params['patch_size'],
                    feature_type=['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4'])
        else:
            self.skimage_haar_feature_coords = self.haar_params['skimage']['feature_coord']
            self.skimage_haar_feature_types = self.haar_params['skimage']['feature_type']

        # Generate computational graph
        have_skimage_haars = len(self.skimage_haar_feature_coords) != 0
        if have_skimage_haars:
            X = delayed(extract_haar_feature_image_skimage(
                img, self.haar_params['skimage']['feature_type'],
                self.haar_params['skimage']['feature_coord']
            ) for img in images)
            # Compute the result
            X = np.array(X.compute(scheduler='processes'))

        # Our haar_features
        haarfe = HaarFeatureExtractor(
            self.haar_params['patch_size'], True, True,
            self.haar_params['ours']['horizontal_feature_types'],
            self.haar_params['ours']['rotated_feature_types'],
            self.haar_params['ours']['horizontal_feature_selection'],
            self.haar_params['ours']['rotated_feature_selection']
        )
        self.our_haar_feature_types_h = haarfe.features_h
        self.our_haar_feature_types_r = haarfe.features_r

        have_our_haars = (len(haarfe.features_h) != 0) or (
            len(haarfe.features_r) != 0)
        if have_our_haars:
            # Preallocate results holder
            X_extension = np.empty((len(candidates), len(
                haarfe.features_h)+len(haarfe.features_r)))

            # Obtain the features
            for k, img in enumerate(images):
                X_extension[k, :] = haarfe.extract_features_from_crop(img)

        # Adjust columns names if necessary
        if 'haar' not in self.feature_names[0]:
            haar_feature_names = [f'haar_{i}' for i in range(
                len(self.skimage_haar_feature_coords))]
            our_h_haar_feature_names = [
                f'hor_haar_{i}' for i in range(len(haarfe.features_h))]
            our_r_haar_feature_names = [
                f'rot_haar_{i}' for i in range(len(haarfe.features_r))]
            self.feature_names = \
                haar_feature_names + our_h_haar_feature_names + \
                our_r_haar_feature_names + self.feature_names

        # Merge our and skimage features
        if have_our_haars and have_skimage_haars:
            return np.concatenate([X, X_extension], axis=1)
        elif have_our_haars and (not have_skimage_haars):
            return X_extension
        elif (not have_our_haars) and have_skimage_haars:
            return X
        else:
            logging.warning('No haars where extracted')

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

    def gabor_feature_bank(
        self, scale, orientation, max_freq=0.2, ksize=(50, 50), sigma=1, gamma=0.5, psi=0
    ):
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
            np.ndarray: of 6_features*n_gabored_images 
        """
        features = []
        for filtered_image in gabored_images:
            img_patch = filtered_image[patch_y1:patch_y2, patch_x1:patch_x2]

            features.extend([(img_patch**2).sum(),
                             np.max(img_patch),
                             np.min(img_patch),
                             np.mean(img_patch),
                             np.std(img_patch),
                             skew(img_patch.ravel()),
                             kurtosis(img_patch.ravel())])
        return np.asarray(features)

    def get_wavelet_features(self, patch: np.ndarray):
        """ Extracts features from an patch's haar 2-level wavelet
        decomposition (8 decompositions)
        First order statistics: mean, skewness, standard deviation, kurtosis,
        entropy, uniformity, relative smoothness
        GLCM features (D=5,theta = 0) [note: only in LH1, HL1 and HH1]: energy,
         correlation, homogeneity, contrast,
        dissimilarity, (NOTRELEVANT: ASM, entropy, uniformity, sum of squares, autocorrelation)

        Args:
            patch (np.ndarray): image patch (normalized)

        Returns:
            features (dict): dictionary containing all extracted features
        """
        eight_decomp = CandidatesFeatureExtraction.get_wavelet_decomp(patch)
        wavelet_params = []

        for idx, single_decomp in enumerate(eight_decomp):
            wavelet_params.extend(
                CandidatesFeatureExtraction.wav_first_order(single_decomp, idx))

        for idx, single_decomp in enumerate(eight_decomp[1:4]):
            wavelet_params.extend(self.wav_glcm_features(single_decomp, idx))

        return np.asarray(wavelet_params)

    def wav_glcm_features(self, single_decomp: np.ndarray, idx: int):
        """Extracts features from a Gray Level Co-occurence Matrix (D=5,theta = 0)
            from a single wavelet decomposition
        Features: energy, correlation, homogeneity, contrast, dissimilarity,
            (DROPPED: ASM, entropy, uniformity, sum of squares, autocorrelation)

        Args:
            single_decomp (np.ndarray): single decomposition from list [LH1, HL1,HH1]
            Note: expected to be used on three decompositions for correct feature naming
            idx (int): index of the decomposition from list above

        Returns:
            glcm_features (dict): dictionary containing glcm features
        """

        all_glcm_decomp = graycomatrix(utils.min_max_norm(
            single_decomp, max_val=255).astype(np.uint8), [2], self.wavelet_params['angles'], normed=True)

        glcm_features = []

        
        for feature_name in skimage_glcm_features:
            feature_results = graycoprops(
                all_glcm_decomp, prop=feature_name)
            for fv in feature_results.ravel():
                glcm_features.append(fv)

        return np.asarray(glcm_features)

    @staticmethod
    def get_wavelet_decomp(patch: np.ndarray, wavelet_type='haar'):
        """ Gets wavelet decomposition of two levels with approximation and detail coefficients.
        Uses decimated 2D forward Discrete Wavelet Transform

        Args:
            patch (np.ndarray): image patch (normalized)
            wavelet_type (str, optional): type of wavelet for decomposition. Defaults to 'haar'.

        Returns:
            decompositions (list[np.ndarray]): list of decompositions
            [LL1,LH1,HL1,HH1,LL2,LH2,HL2,HH2]
        """
        LL1, (LH1, HL1, HH1) = dwt2(patch, wavelet_type)
        LL2, (LH2, HL2, HH2) = dwt2(LL1, wavelet_type)

        return [LL1, LH1, HL1, HH1, LL2, LH2, HL2, HH2]

    @staticmethod
    def wav_first_order(single_decomp: np.ndarray, idx: int):
        """ Extracts first order statistics from a single wavelet decomposition
        Features: mean, skewness, standard deviation, kurtosis, entropy, uniformity, relative
        smoothness

        Args:
            single_decomp (np.ndarray): single decomposition from the list
            [LL1,LH1,HL1,HH1,LL2,LH2,HL2,HH2]
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

class CandidatesFeatureExtraction_MP(CandidatesFeatureExtraction):
    """ Multiprocessing version of the CandidatesFeatureExtraction.
    
    Works around 4x faster than the one above.
    """
    def __init__(self, patch_size: int = PATCH_SIZE, fos=True, gabor_params=GABOR_PARAMS, wavelet_params=WAVELET_PARAMS, haar_params=HAAR_PARAMS, n_jobs=8):
        super().__init__(patch_size, fos, gabor_params, wavelet_params, haar_params)
        self.n_jobs = n_jobs
    
    def slice_image_in_patches(self, image, gabored_images, candidates):
        image_patches = []
        for coords in candidates:
            patch_x1, patch_x2, patch_y1, patch_y2 = utils.patch_coordinates_from_center(
                (coords[0], coords[1]), image.shape, self.patch_size)
            image_patches.append((image[patch_y1:patch_y2, patch_x1:patch_x2],
                                  [gb[patch_y1:patch_y2, patch_x1:patch_x2] for gb in gabored_images],
                                  [coords[0], coords[1], coords[2]],
                                  ((patch_y1, patch_y2), (patch_x1, patch_x2))))
        return image_patches
    
    def extract_features(self, candidates: np.ndarray, image: np.ndarray):
        """Extracts features from image patches cropped around given candidates.
        Args:
            candidates (np.ndarray): of dtype=int containing candidats for FE of shape
                (n_candidates, 3). Second axis should contain (x_coord, y_coord, radius).
            image (np.ndarray): 2D image used for cropping and FE
        Returns:
            np.array: rows candidates, columns (features + 3 columns of metadata)
        """
        image = utils.min_max_norm(image, max_val=1.)

        if self.gabor_params:
            gabor_kernels = gabor_kernels = self.gabor_feature_bank(
                **self.gabor_params)
            gabored_images = [cv2.filter2D(
                image, ddepth=cv2.CV_32F, kernel=k) for k in gabor_kernels]
        else:
            gabored_images = []

        if self.haar_params:
            features_haar = self.haar_features_extraction(image, candidates)

        image_patches = self.slice_image_in_patches(image, gabored_images, candidates)
        
        candidates_features = []
        with mp.Pool(self.n_jobs) as pool:
            for result in pool.map(self.extract_patch_nonHaar_features, image_patches):
                candidates_features.append(result)


        candidates_features = np.asarray(candidates_features, dtype=object)
        
        if self.haar_params:
            candidates_features = np.concatenate(
                [features_haar, candidates_features], axis=1)
        return candidates_features
    
    def extract_patch_nonHaar_features(self, patches):

        
        image_patch , gabored_image_patches, coords, pc = patches
        (patch_y1, patch_y2), (patch_x1, patch_x2) = pc
        # Extracting non-haar features
        features = []

        # First order statistics features
        if self.fos:
            features.extend(self.first_order_statistics(image_patch))

        # Gabor features
        if self.gabor_params:
            features.extend(self.gabor_features(gabored_image_patches))

        # Wavelet and GLCM features
        if self.wavelet_params:
            features.extend(self.get_wavelet_features(image_patch))

        # Append metadata columns
        features.append(coords)
        features.append(((patch_y1, patch_y2), (patch_x1, patch_x2)))
        return features
        
    def gabor_features(self, gabored_image_patches):
        """Extracts energy, mean, std, skeweness and kurtosis from patches
            of images filtered with gabor kernel

        Args:
            gabored_images (list): list of images filtered with Gabor kernels
        Returns:
            np.ndarray: of 6_features*n_gabored_images 
        """
        features = []
        for img_patch in gabored_image_patches:

            features.extend([(img_patch**2).sum(),
                             np.max(img_patch),
                             np.min(img_patch),
                             np.mean(img_patch),
                             np.std(img_patch),
                             skew(img_patch.ravel()),
                             kurtosis(img_patch.ravel())])
        return np.asarray(features)