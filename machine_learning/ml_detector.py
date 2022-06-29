import numpy as np
import pandas as pd

from feature_extraction.feature_extraction import CandidatesFeatureExtraction_MP
from machine_learning.cascade_classifier import CascadeClassifier
from candidate_proposal.morphology_mc import MorphologyCalcificationDetection, GSM_DEFAULT_PARAMS
from general_utils.utils import non_max_supression


class DetectorML:
    """Calcification Detection Model based on Grayscale Morphology"""
    def __init__(self, cascade_model_path: str):
        """Constructor for the DetectorML class
        Args:
            cascade_model_path (str): Path to trained cascade models
                trained on all features extracted from GSM detector.
        """
        # instantiate the candidate proposer with the default configuration
        # which contains the best parameters found duering experiments
        self.detector = MorphologyCalcificationDetection(**GSM_DEFAULT_PARAMS)

        # instantiate the feature extractor with the default configuration
        # which contains the best parameters found duering experiments
        self.feature_extractor = CandidatesFeatureExtraction_MP()

        # instantiate the cascade classifier with weights trained during experiments
        self.classifier = CascadeClassifier(cascade_model_path)

    def detect(self, image, IoU=0.5):
        """Detects calcification on a given mammography image
        Args:
            image (np.ndarray): image to use for detection
            IoU (floag): threshold used for NMS. Defaluts to 0.5.
        Returns:
            candidates: pd.DataFrame with detected calcifications in a form
                a dataftame with columns:
                    'candidate_coordinates' == (x, y, radius of the candidate)
                    'patch_coordinates' == (y1, y2, x1, x2 coordinates of 14x14 patch)
                    'confidence' == cascade model probability of the candidate
                        being a micro-calcification
        """
        candidates = self.detector.detect(image, 0)
        cand_features = self.feature_extractor.extract_features(candidates, image)
        cand_features = pd.DataFrame(
            data=cand_features, columns=self.feature_extractor.feature_names)

        prb, cand_mask = self.classifier.predict(cand_features, 'all_features', True)
        cand_features = cand_features.loc[cand_mask, :].reset_index()
        cand_features['confidence'] = prb

        # perform NMS over detected candidates
        nms_bboxes = [[x[1][0], x[1][1], x[0][0], x[0][1], cand_features.confidence.values[xidx]]
                      for xidx, x in enumerate(cand_features.patch_coordinates.values)]
        cand_features = cand_features.loc[non_max_supression(np.stack(nms_bboxes), IoU, True), :]

        return cand_features[['candidate_coordinates', 'patch_coordinates', 'confidence']]
