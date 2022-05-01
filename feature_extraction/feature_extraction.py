import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from sqlalchemy import false
from tqdm import tqdm
import multiprocessing as mp

class CandidatesFeatureExtraction:
    def __init__(self, feature_types: dict, image_types: dict, patch_size: int):
        """Defines which features to extract and from which images.
        
        Refer to https://pyradiomics.readthedocs.io/en/latest/customization.html
        for information about supported  feature_types and image_types.
        
        Args:
            feature_types (dict): pyradiomics feature types. Should contain
                {'feature_group_name':['FeatureName']}. Empty list or None indicates
                all feature from the group.
            image_types (dict): pyradiomics image types. {'Original':{}} will indicate
                calculate features on original image without any preprocessing.
            patch_size (int): Size of the patch extracted around each candidate and
                used for FE
        """
        self.patch_size = patch_size
        self.feature_types = feature_types
        self.image_types = image_types

        # initializing FE class
        self.fe = featureextractor.RadiomicsFeatureExtractor()
        self.fe.disableAllFeatures()
        self.fe.disableAllImageTypes()
        self.fe.enableFeaturesByName(**self.feature_types)
        self.fe.enableImageTypes(**self.image_types)

    def extract_features(self, candidates: np.ndarray, image: np.ndarray, roi_mask: np.ndarray, sample=None):
        """Extracts features from image patches cropped around given candidates.

        Args:
            candidates (np.ndarray): containing candidate coordinates for FE of shape (n_candidates, 3)
            image (np.ndarray): 2D image used for cropping and FE
            roi_mask (np.ndarray): mask containing whether given pixel corresponds to a positive class (mC)
            sample (int, optional): number of candidates to sample. 
                If None: extracts features for all candidates. Defaults to None.

        Returns:
            list[dict]: containing dictionaries with candidates features
        """
        # sample candidates to a given size if needed
        if sample is not None:
            cand_idxs = np.random.choice(
                len(candidates), size=sample, replace=False)
            candidates = candidates[cand_idxs]

        # pyradiomics image and roi mask should be sitk images
        sitk_patch_mask = sitk.GetImageFromArray(np.zeros(image.shape))
        sitk_image = sitk.GetImageFromArray(image)


        candidates_features = []
        # iterating over candidates and cropping patches
        for coords in tqdm(candidates):

            patch_half_size = int(self.patch_size / 2)

            # get the coordinates of the patch centered on the lesion
            patch_x1 = coords[1] - patch_half_size
            patch_y1 = coords[0] - patch_half_size

            if patch_x1 < 0:
                patch_x1 = 0
                patch_x2 = self.patch_size
            else:
                patch_x2 = coords[1] + self.patch_size - patch_half_size
            if patch_y1 < 0:
                patch_y1 = 0
                patch_y2 = self.patch_size
            else:
                patch_y2 = coords[0] + self.patch_size - patch_half_size

            if patch_x2 > image.shape[1]:
                image = np.pad(
                    image, ((0, 0), (0, self.patch_size)), mode='constant', constant_values=0
                )

            if patch_y2 > image.shape[0]:
                image = np.pad(
                    image, ((0, self.patch_size), (0, 0)), mode='constant', constant_values=0
                )

            # croping the patch
            sitk_patch_mask[patch_y1:patch_y2, patch_x1:patch_x2] = 1
            features = self.fe.execute(sitk_image, sitk_patch_mask)
            sitk_patch_mask[patch_y1:patch_y2, patch_x1:patch_x2] = 0

            features['coordinates'] = (
                (patch_y1, patch_y2), (patch_x1, patch_x2))
            features['patch_mask_intersection'] = (roi_mask[patch_y1:patch_y2,
                                                           patch_x1:patch_x2]>0).sum()
            # save the coordinates
            candidates_features.append(features)

        return candidates_features

class CandidatesFeatureExtractionMP:
    def __init__(self, feature_types: dict, image_types: dict, patch_size: int, n_jobs=6):
        """Defines which features to extract and from which images.
        
        Refer to https://pyradiomics.readthedocs.io/en/latest/customization.html
        for information about supported  feature_types and image_types.
        
        Args:
            feature_types (dict): pyradiomics feature types. Should contain
                {'feature_group_name':['FeatureName']}. Empty list or None indicates
                all feature from the group.
            image_types (dict): pyradiomics image types. {'Original':{}} will indicate
                calculate features on original image without any preprocessing.
            patch_size (int): Size of the patch extracted around each candidate and
                used for FE
            n_jobs (int, optional): number of threads to use. Defaults to 6.
        """
        self.patch_size = patch_size
        self.feature_types = feature_types
        self.image_types = image_types
        self.n_jobs = n_jobs

        # initializing FE class
        self.fe = featureextractor.RadiomicsFeatureExtractor()
        self.fe.disableAllFeatures()
        self.fe.disableAllImageTypes()
        self.fe.enableFeaturesByName(**self.feature_types)
        self.fe.enableImageTypes(**self.image_types)

    def extract_features(self, candidates: np.ndarray, image: np.ndarray, roi_mask: np.ndarray, sample=None):
        """Extracts features from image patches cropped around given candidates.

        Args:
            candidates (np.ndarray): containing candidate coordinates for FE of shape (n_candidates, 3)
            image (np.ndarray): 2D image used for cropping and FE
            roi_mask (np.ndarray): mask containing whether given pixel corresponds to a positive class (mC)
            sample (int, optional): number of candidates to sample. 
                If None: extracts features for all candidates. Defaults to None.

        Returns:
            list[dict]: containing dictionaries with candidates features
        """
        
        # sample candidates to a given size if needed
        if sample is not None:
            cand_idxs = np.random.choice(
                len(candidates), size=sample, replace=False)
            candidates = candidates[cand_idxs]

        # pyradiomics image and roi mask should be sitk images
        self.sitk_image = sitk.GetImageFromArray(image)
        self.image_shape = image.shape
        self.roi_mask = roi_mask
        
        candidates_features = []
        with mp.Pool(self.n_jobs) as pool:
            for result in pool.map(self.process_candidate, candidates):
                candidates_features.append(result)        


        return candidates_features
    
    def process_candidate(self, coords):
        
        # since different threads might simultaneously change the
        # sitk_patch_mask values it is better to create new one for each thread
        # it doesn't affect the speed that much anyway
        sitk_patch_mask = sitk.GetImageFromArray(np.zeros(self.image_shape))
        # iterating over candidates and cropping patches

        patch_half_size = int(self.patch_size / 2)

        # get the coordinates of the patch centered on the lesion
        patch_x1 = coords[1] - patch_half_size
        patch_y1 = coords[0] - patch_half_size

        if patch_x1 < 0:
            patch_x1 = 0
            patch_x2 = self.patch_size
        else:
            patch_x2 = coords[1] + self.patch_size - patch_half_size
        if patch_y1 < 0:
            patch_y1 = 0
            patch_y2 = self.patch_size
        else:
            patch_y2 = coords[0] + self.patch_size - patch_half_size

        if patch_x2 > self.image_shape[1]:
            patch_x2 = self.image_shape[1]

        if patch_y2 > self.image_shape[0]:
            patch_y2 = self.image_shape[0]

        # croping the patch
        sitk_patch_mask[patch_y1:patch_y2, patch_x1:patch_x2] = 1
        features = self.fe.execute(self.sitk_image, sitk_patch_mask)
        sitk_patch_mask[patch_y1:patch_y2, patch_x1:patch_x2] = 0

        features['coordinates'] = (
            (patch_y1, patch_y2), (patch_x1, patch_x2))
        features['patch_mask_intersection'] = (self.roi_mask[patch_y1:patch_y2,
                                                        patch_x1:patch_x2]>0).sum()
        # save the coordinates
        return features