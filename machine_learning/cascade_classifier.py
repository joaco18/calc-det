import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve


class CascadeClassifier:
    def __init__(self, models_path:str):
        """Constructor for CascadeClassifier

        Args:
            models_path (str): path to the pickle file with models and metadata.
                If file does not exist, will use the path to save trained models.
        """
        self.models_path = Path(models_path)
        
        if self.models_path.exists():
            print('Loading model...')
            self.load_models()
        else:
            print('Model not found. Need to call fit before predict.')
            self.first_model  = None
            self.second_model = None

    def predict(self, candidate_features: pd.DataFrame, features_set:str = 'all_features'):
        """Predicts the probablility of given candidates to contain a mC

        Args:
            candidate_features (pd.DataFrame): containing candidate features and metadata.
                Output of CandidatesFeatureExtraction().extract_features()
            feature_set (list[str]): names of features to be used by models

        Returns:
            np.ndarray: probablilities of candidates being positive (containing mC)
        """
        # feature groups
        fos_cols = [x for x in candidate_features if 'img' in x and x != 'img_id']
        gabor_cols = [x for x in candidate_features if 'gabor_' in x]
        wavelet_cols = [x for x in candidate_features if x[:6]
                        == 'patch_' and x != 'patch_coordinates']
        haar_cols = [x for x in candidate_features if 'haar' in x]

        features_sets = {'fos': fos_cols, 'gabor': gabor_cols, 'wavelet': wavelet_cols,
                        'haar': haar_cols,
                        'all_features': fos_cols+gabor_cols+wavelet_cols+haar_cols}

        if self.first_model  is None:
            raise Exception('Model nor trained nor loaded. Provide a path to the model or perform training') 
        return self.second_model.predict_proba(candidate_features[features_sets[features_set]])[:, 1]
        
    def fit(self, clf, train_features:pd.DataFrame, features:list[str], sens_threshold:float, kfolds=5, FP2TP_rate=10):
        """Trains cascaded models for candidate prediction

        Args:
            clf (_type_): sklearn classifier model instance used for fit/predict 
            train_features (pd.DataFrame): containing candidate features and metadata.
                Output of CandidatesFeatureExtraction().extract_features()
            features (list[str]): names of features to be used by models
            sens_threshold (float): desired threshold of kept sensitivity
            kfolds (int, optional): number of folds used for CV. Defaults to 5.
            FP2TP_rate (int, optional): balancing ration between FP:TP. Defaults to 10.
        """
        np.random.seed(42)

        # 1. FIRST STAGE OF THE CASCADE 
        
        # splitting data into kfolds of train/validation case-wise
        all_train_case_ids = train_features.case_id.unique()
        kfolds_case_splits = np.array_split(all_train_case_ids, kfolds)

        predicted_test_df = []

        # kfold CV
        for valid_case_ids in kfolds_case_splits:

            # split data into test
            test_split_mask = train_features.case_id.isin(valid_case_ids)

            # split into train and take into account cleaned data with no mC in the borders of the patch
            cleaned_features_data = train_features[~test_split_mask]

            # and sampling to a predefined 1:FP2TP_rate TP:FP samples
            positive_mask = (cleaned_features_data.label > 0)
            positive_train_part = cleaned_features_data[positive_mask]
            negative_train_part = cleaned_features_data[~positive_mask].sample(
                FP2TP_rate*positive_mask.sum())

            train_df = pd.concat(
                [positive_train_part, negative_train_part]).sample(frac=1.)

            test_df = train_features[test_split_mask]

            train_y = train_df.label

            scaler = MinMaxScaler()
            train_X_scaled = scaler.fit_transform(train_df[features].values)
            test_X_scaled = scaler.transform(test_df[features].values)

            clf.fit(train_X_scaled, train_y)

            test_y_predicted = clf.predict_proba(test_X_scaled)[:, 1]

            test_df['cv_fold_predict_proba'] = test_y_predicted
            predicted_test_df.append(test_df)
        
        predicted_test_df = pd.concat(predicted_test_df)
        
        
        # 2. SELECTING CONFIDENCE THRESHOLD TO FILTER OUT FP
        fpr, tpr, thrs = roc_curve(predicted_test_df.label,
                           predicted_test_df.cv_fold_predict_proba)
        
        total_fp = np.sum(~predicted_test_df.label)

        self.c1_tpr = tpr.copy()
        self.c1_thrs = thrs.copy()
        self.c1_labels = predicted_test_df.label.values
        self.c1_probas = predicted_test_df.cv_fold_predict_proba.values

        self.max_conf_thr_required = self.c1_thrs[np.argmax(self.c1_tpr >= sens_threshold)]

        filtered_fp = np.sum((~self.c1_labels) & (
            self.c1_probas <= self.max_conf_thr_required))/total_fp


        msg = (
            f"Selected keep_sens_thr={sens_threshold}\n"
            f'max_conf_thr_required to keep given sensitivity is {self.max_conf_thr_required:.5f}\n'
            f'Filtering out all candidates with confidence <={self.max_conf_thr_required:.5f} is estimated to reduce FP by {100*filtered_fp:.2f} %'
        )
        print(msg)
        
        # 3. TRAIN FINAL FIRST STAGE MODEL
        positive_mask = (train_features.label > 0)
        positive_train_part = train_features[positive_mask]
        negative_train_part = train_features[~positive_mask].sample(
            FP2TP_rate*positive_mask.sum())


        train_df = pd.concat(
            [positive_train_part, negative_train_part]).sample(frac=1.)

        self.first_model = Pipeline([('scaler', MinMaxScaler()), ('svc', clf)])
        self.first_model.fit(train_df[features], train_df.label)
        

        # 4. TRAIN FINAL SECOND STAGE MODEL
        train_df_stage2 = predicted_test_df[predicted_test_df.cv_fold_predict_proba >
                                    self.max_conf_thr_required]
        
        positive_mask = (train_df_stage2.label > 0)
        positive_train_part = train_df_stage2[positive_mask]
        negative_train_part = train_df_stage2[~positive_mask].sample(
            FP2TP_rate*positive_mask.sum())


        train_df = pd.concat(
            [positive_train_part, negative_train_part]).sample(frac=1.)

        self.second_model = Pipeline([('scaler', MinMaxScaler()), ('svc', clf)])
        self.second_model.fit(train_df[features], train_df.label)


    def load_models(self):
        """Loads cascade models from pickle at models_path"""
        with open(self.models_path, 'rb') as f:
            c1m, c2m, tpr, thrs, labels, probas, keep_sens_thr = pickle.load(f)
        
        self.sens_threshold = keep_sens_thr
        self.first_model = c1m
        self.second_model = c2m
        
        self.c1_tpr = tpr
        self.c1_thrs = thrs
        self.c1_labels = labels
        self.c1_probas = probas
            
                
        self.max_conf_thr_required = self.c1_thrs[np.argmax(self.c1_tpr >= self.sens_threshold)]
        total_fp = np.sum(~self.c1_labels)

        filtered_fp = np.sum((~self.c1_labels) & (
            self.c1_probas <= self.max_conf_thr_required))/total_fp

        
        msg = (
            f'Selected sensitivity threshold to keep {self.sens_threshold}\n'
            f'Confidence threshold to keep given sensitivity is {self.max_conf_thr_required:.5f}\n'
            f'Filtering out all candidates with confidence <={self.max_conf_thr_required:.5f} is estimated to reduce FP by {100*filtered_fp:.2f} %'
        )
        print(msg)