import numpy as np
import pandas as pd
from path import Path

from feature_extraction.feature_extraction import CandidatesFeatureExtraction_MP
from machine_learning.cascade_classifier import CascadeClassifier
from candidate_proposal.morphology_mc import MorphologyCalcificationDetection, GSM_DEFAULT_PARAMS

class DetectorML:
    """Calcification Detection Model based on Grayscale Morphology"""
    def __init__(self, cascade_model_path=''):
        """Constructor for the DetectorML class

        Args:
            cascade_model_path (str, optional): Path to trained cascade models
                trained on all features extracted from GSM detector. Defaults to ''.
        """

        self.detector = MorphologyCalcificationDetection(**GSM_DEFAULT_PARAMS)
        self.feature_extractor = CandidatesFeatureExtraction_MP()
        self.classifier = CascadeClassifier(cascade_model_path)
    
    def detect(self, image):
        """Detects calcification on a given mammography image

        Args:
            image (np.ndarray): image to use for detection

        Returns:
            candidates: pd.DataFrame with detected calcifications in a form
                a dataftame with columns 
                'candidate_coordinates' == (x, y, radius of the candidate) 
                'patch_coordinates' == (y1, y2, x1, x2 coordinates of 14x14 patch)
                'confidence' == cascade model probability of the candidate being a positive one
        """
        candidates = self.detector.detect(image, 0)
        cand_features = self.feature_extractor.extract_features(candidates, image)
        cand_features = pd.DataFrame(data=cand_features, columns=self.feature_extractor.feature_names)
        cand_features['confidence'] = self.classifier.predict(cand_features)
        return cand_features[['candidate_coordinates', 'patch_coordinates',
                              'confidence']]
        