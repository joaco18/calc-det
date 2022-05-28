from operator import length_hint
from pathlib import Path

THISPATH = Path(__file__).resolve()
import sys; sys.path.insert(0, str(THISPATH.parent.parent))

import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import logging
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from database.dataset import INBreast_Dataset
from mc_candidate_proposal.hough_mc import HoughCalcificationDetection
from mc_candidate_proposal.morphology_mc import MorphologyCalcificationDetection
from feature_extraction.feature_extraction import CandidatesFeatureExtraction
import feature_extraction.haar_features.haar_modules as hm
import general_utils.utils as utils
from metrics.metrics_utils import get_tp_fp_fn_center_patch_criteria

logging.basicConfig(level=logging.INFO)


def sort_relevances(feat_importances):
    """ Get the mean importance for each feature across the runs and then sort
    them in decreasing order.
    """
    feature_importances_array = np.asarray(feat_importances)
    mean_feature_importances_array = np.mean(feature_importances_array, axis=0)
    sorted_features = np.argsort(mean_feature_importances_array)[::-1]
    sorted_mean_feature_importance = mean_feature_importances_array[sorted_features]
    return sorted_features, sorted_mean_feature_importance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector")
    args = parser.parse_args()
    detector = args.detector

    data_path = THISPATH.parent.parent.parent/'data'
    print(THISPATH.parent.parent.parent/'data')
    rbd_path = data_path/'gsm_imgs'

    logging.info('Intantiating db...')
    db = INBreast_Dataset(
        return_lesions_mask=True,
        level='image',
        max_lesion_diam_mm=None,
        extract_patches=False,
        partitions=['train', 'validation'],
        lesion_types=['calcification', 'cluster'],
        cropped_imgs=True,
        keep_just_images_of_lesion_type=False,
        use_muscle_mask=True,
        ignore_diameter_px=15
    )

    # Candidate Proposals detector
    logging.info('Intantiating candidate proposer...')
    if detector == 'hough':
        hd = HoughCalcificationDetection(filter_muscle_region=True)
    elif detector == 'gsm':
        hd = MorphologyCalcificationDetection(
            rbd_img_path=rbd_path, threshold=0.95, min_distance=6, area=14*14,
            store_intermediate=True, filter_muscle_region=True
        )

    # Feature extractor
    logging.info('Intantiating features extractor...')
    haar_params = {
        'skimage': {
            'feature_type': None, 'feature_coord': None
        },
        'ours': {
            'horizontal_feature_types': [
                (hm.Feature4h, 1, 4), (hm.Feature4v, 4, 1), (hm.Feature3h3v, 3, 3)],
            'rotated_feature_types': None,
            'horizontal_feature_selection': None,
            'rotated_feature_selection': None,
        },
        'patch_size': 14
    }

    cfe = CandidatesFeatureExtraction(patch_size=14, fos=False, haar_params=haar_params)

    # Extract features for all images
    logging.info('Extracting features for images:')
    for idx in tqdm(range(100)):
        # Get image to process
        db_sample = db[idx]
        image = db_sample['img']
        image_id = db.df.iloc[idx].img_id
        image_mask = db_sample['lesion_mask']
        muscle_mask = db_sample['muscle_mask']

        # Avoid reprocessing
        path = (data_path / f'haar_features_{detector}'/f'{image_id}.fth')
        path.parent.mkdir(exist_ok=True, parents=True)
        if path.exists():
            continue
        else:
            # candidate detection
            candidates = hd.detect(image, image_id, muscle_mask=muscle_mask)

            # labeling of candidates:
            tp, fp, fn, ignored_candidates = get_tp_fp_fn_center_patch_criteria(
                candidates, image_mask, None, 14)
            n_TPs = len(tp)
            if n_TPs == 0:
                n_FP = len(fp)
                sample_size = int(n_FP * 0.1)
            else:
                sample_size = n_TPs * 10
            if len(fp) >= sample_size:
                fp = fp.sample(sample_size, random_state=0, replace=False)

            candidates = pd.concat([tp, fp], axis=0, ignore_index=True)

            # Extracting features
            labels = np.where(
                candidates.drop_duplicates(subset='repeted_idxs').label.values == 'TP', True, False)

            # Avoid extracting features for repeted detections
            X = candidates.drop_duplicates(subset='repeted_idxs')
            X = cfe.extract_features(X.loc[:, ['x', 'y', 'radius']].values.astype(int), image)

            # Get features dfs
            res = pd.DataFrame(data=X, columns=cfe.feature_names)
            res['img_id'] = image_id
            res['labels'] = labels

            res.to_feather(path)

    logging.info('Feature extraction completed')

    # Fraction features in two dfs

    logging.info('Organizing database...')
    logging.info('Joining first half...')
    if not (data_path / f'haar_features_{detector}/all_feats_10to1_pt0.fth').exists():
        for idx in tqdm(range(50)):
            # Get case
            image_id = db.df.iloc[idx].img_id

            # Load data
            path = data_path / f'haar_features_{detector}/{image_id}.fth'
            if path.exists():
                data = pd.read_feather(path)

                # Sample fp to reduce size
                TPs = data.loc[data.labels]
                n_TPs = len(TPs)
                if n_TPs == 0:
                    n_FP = len(data.loc[~data.labels])
                    sample_size = int(n_FP * 0.1)
                else:
                    sample_size = n_TPs * 10
                if len(data.loc[~data.labels]) >= sample_size:
                    FPs = data.loc[~data.labels].sample(sample_size, random_state=0, replace=False)
                else:
                    FPs = data.loc[~data.labels]

                # Rejoin
                data = pd.concat([TPs, FPs], axis=0).sample(frac=1)
                if idx == 0:
                    fdf = data
                else:
                    fdf = pd.concat([fdf, data])
                del data, TPs, FPs
            else:
                logging.warning(f'File {image_id}.fth does not exist')
        # Store
        fdf.reset_index(drop=True).to_feather(
            data_path / f'haar_features_{detector}/all_feats_10to1_pt0.fth')
        del fdf

    logging.info('Joining second half...')
    if not (data_path / f'haar_features_{detector}/all_feats_10to1_pt1.fth').exists():
        for idx in tqdm(range(50, 100)):
            # Get case
            image_id = db.df.iloc[idx].img_id

            # Load data
            path = (Path.cwd().parent.parent / f'data/haar_features_{detector}/{image_id}.fth')
            if path.exists():
                data = pd.read_feather(path)

                # Sample fp to reduce size
                TPs = data.loc[data.labels]
                n_TPs = len(TPs)
                sample_size = n_TPs * 10
                if len(data.loc[~data.labels]) >= sample_size:
                    FPs = data.loc[~data.labels].sample(sample_size, random_state=0, replace=False)
                else:
                    FPs = data.loc[~data.labels]

                # Rejoin
                data = pd.concat([TPs, FPs], axis=0).sample(frac=1)
                if idx == 50:
                    fdf = data
                else:
                    fdf = pd.concat([fdf, data])
                del data, TPs, FPs
            else:
                logging.warning(f'File {image_id}.fth does not exist')

        # Store
        fdf.reset_index(drop=True).to_feather(
            data_path / f'haar_features_{detector}/all_feats_10to1_pt1.fth')
        del fdf

    # Separate skimage features from ours

    logging.info('Splitting first half in skimage and ours...')
    if ((not (data_path / f'haar_features_{detector}/all_feats_10to1_pt0_skimage.fth').exists()) and
            (not (data_path / f'haar_features_{detector}/all_feats_10to1_pt0_our.fth').exists())):
        data = pd.read_feather(
            data_path / f'haar_features_{detector}/all_feats_10to1_pt0.fth')

        metadata_cols = [col for col in data.columns if ('haar' not in col)]
        skimage_cols = [col for col in data.columns if (
            ('haar' in col) and ('rot' not in col) and ('hor' not in col))]
        our_cols = [col for col in data.columns if ('rot' in col) or ('hor' in col)]

        data.loc[:, skimage_cols+metadata_cols].reset_index(drop=True).to_feather(
            data_path / f'haar_features_{detector}/all_feats_10to1_pt0_skimage.fth')
        data.loc[:, our_cols+metadata_cols].reset_index(drop=True).to_feather(
            data_path / f'haar_features_{detector}/all_feats_10to1_pt0_our.fth')
        del data

    logging.info('Splitting second half in skimage and ours...')
    if ((not (data_path / f'haar_features_{detector}/all_feats_10to1_pt1_skimage.fth').exists()) and
            (not (data_path / f'haar_features_{detector}/all_feats_10to1_pt1_our.fth').exists())):
        data = pd.read_feather(
            data_path / f'haar_features_{detector}/all_feats_10to1_pt1.fth')

        metadata_cols = [col for col in data.columns if ('haar' not in col)]
        skimage_cols = [col for col in data.columns if (
            ('haar' in col) and ('rot' not in col) and ('hor' not in col))]
        our_cols = [col for col in data.columns if ('rot' in col) or ('hor' in col)]

        data.loc[:, skimage_cols+metadata_cols].reset_index(drop=True).to_feather(
            data_path / f'haar_features_{detector}/all_feats_10to1_pt1_skimage.fth')
        data.loc[:, our_cols+metadata_cols].reset_index(drop=True).to_feather(
            data_path / f'haar_features_{detector}/all_feats_10to1_pt1_our.fth')
        del data

    # OUR FEATURES

    logging.info('Starting analysis of our features...')
    if not (data_path / f'haar_features_{detector}/all_cases_our_features_10to1_1000_selection.fth').exists():
        # Load all aour features
        logging.info('Loading data...')
        data = pd.concat([
            pd.read_feather(
                data_path / f'haar_features_{detector}/all_feats_10to1_pt0_our.fth'),
            pd.read_feather(
                data_path / f'haar_features_{detector}/all_feats_10to1_pt1_our.fth')
        ], ignore_index=True)

        # CHOOSE THE NUMBER OF CASES TO CONSIDER:
        N = 100
        data = data.loc[data.img_id.isin(data.img_id.unique()[:N]), :]

        # SELECT THE NUMBER OF FOLDS FOR TO USE IN THE EXPERIMENTS:
        folds = 5
        n = int(len(data.img_id.unique()) / folds)

        # Train the Forests and store the features importances
        aucs_test = []
        aucs_train = []
        feat_importances = []
        clfs = []

        models_path = data_path / f'haar_models_{detector}'
        models_path.mkdir(exist_ok=True, parents=True)

        logging.info('Starting training iterations...')
        for k, test_img_id in tqdm(enumerate(utils.blockwise_retrieval(data.img_id.unique(), n))):
            if k == folds + 1:
                continue
            
            # Divide train and test based on cases (cross validation image wise)
            train_df = data[~data.img_id.isin(test_img_id)]
            train_df = train_df.sample(frac=1, random_state=0)
            test_df = data[data.img_id.isin(test_img_id)]

            # Generate features and labels datasets
            train_X = train_df.drop(
                columns=['patch_coordinates', 'candidate_coordinates', 'labels', 'img_id'])
            train_y = (train_df.labels).astype(int)
            del train_df
            test_X = test_df.drop(
                columns=['patch_coordinates', 'candidate_coordinates', 'labels', 'img_id'])
            test_y = (test_df.labels).astype(int)
            del test_df

            # Train a random forest classifier
            filename = f'RF_our_{k}.sav'
            if not (models_path / filename).exists():
                clf = RandomForestClassifier(
                    n_estimators=1000, max_depth=data.shape[1]-4, n_jobs=-1, random_state=0)
                clf.fit(train_X, train_y)

                # Store the trained models
                with open(models_path / filename, 'wb') as f:
                    pickle.dump(clf, f)
            else:
                with open(models_path / filename, 'rb') as f:
                    clf = pickle.load(f)

            # Asses performance
            test_y_predicted = clf.predict_proba(test_X)[:, 1]
            train_y_predicted = clf.predict_proba(train_X)[:, 1]

            auc_full_features_test = roc_auc_score(test_y, test_y_predicted)
            auc_full_features_train = roc_auc_score(train_y, train_y_predicted)
            aucs_test.append(auc_full_features_test)
            aucs_train.append(auc_full_features_train)

            # Store on memory the classifier and the feature importances
            clfs.append(clf)
            feat_importances.append(clf.feature_importances_)

        del train_X, train_y, test_X, test_y, test_y_predicted, train_y_predicted
        del data
        # Store the auroc data to disk
        auc_data = pd.DataFrame(aucs_test + aucs_train, columns=['auc'])
        auc_data['set'] = ['val']*len(aucs_test) + ['train']*len(aucs_train)
        auc_data.to_csv(models_path/f'aucs_our_haar_{folds}_runs.csv')
        # Store the feature importances to disk
        filename = f'feature_importances_our_haar_{folds}_runs.p'
        with open(models_path / filename, 'wb') as f:
            pickle.dump(feat_importances, f)

        logging.info('Sorting features by importance...')
        sorted_features, mean_importance = sort_relevances(feat_importances)
        OUR_CUT = 1000

        # Load all aour features
        logging.info('Storing selection...')
        data = pd.concat([
            pd.read_feather(data_path / f'haar_features_{detector}/all_feats_10to1_pt0_our.fth'),
            pd.read_feather(data_path / f'haar_features_{detector}/all_feats_10to1_pt1_our.fth')
        ], ignore_index=True)

        metadata_cols = [col for col in data.columns if ('haar' not in col)]
        selected_feats_cols = data.columns.values[sorted_features[:OUR_CUT]].tolist()
        data.loc[:, selected_feats_cols+metadata_cols].reset_index(drop=True).to_feather(
            data_path / f'haar_features_{detector}/all_cases_our_features_10to1_1000_selection.fth')
        del data

    # SKIMAGE FEATURES

    logging.info('Starting analysis of skimage features...')
    if not (data_path / f'haar_features_{detector}/all_cases_skimage_features_10to1_2000_selection.fth').exists():
        # Load all skimage features
        logging.info('Loading data...')

        data = pd.concat([
            pd.read_feather(data_path / f'haar_features_{detector}/all_feats_10to1_pt0_skimage.fth'),
            pd.read_feather(data_path / f'haar_features_{detector}/all_feats_10to1_pt1_skimage.fth')
        ], ignore_index=True)

        # CHOOSE THE NUMBER OF CASES TO CONSIDER: IN MY CASE I COULD HANDDLE 100
        N = 100
        # Reduce the number of examples to be manageable
        data = data.loc[data.img_id.isin(data.img_id.unique()[:N]), :]

        # SELECT THE NUMBER OF FOLDS FOR TO USE IN THE EXPERIMENTS:
        folds = 5
        n = int(len(data.img_id.unique()) / folds)

        # Train the Forests and store the features importances
        aucs_test = []
        aucs_train = []
        feat_importances = []
        clfs = []

        models_path = data_path / f'haar_models_{detector}'
        models_path.mkdir(exist_ok=True, parents=True)
        logging.info('Starting training iterations...')
        for k, test_img_id in tqdm(enumerate(utils.blockwise_retrieval(data.img_id.unique(), n))):
            if k == folds + 1:
                continue
            # Divide train and test based on cases (cross validation image wise)
            train_df = data[~data.img_id.isin(test_img_id)]
            train_df = train_df.sample(frac=1, random_state=0)
            test_df = data[data.img_id.isin(test_img_id)]

            # Generate features and labels datasets
            train_X = train_df.drop(
                columns=['patch_coordinates', 'candidate_coordinates', 'labels', 'img_id'])
            train_y = (train_df.labels).astype(int)
            del train_df
            test_X = test_df.drop(
                columns=['patch_coordinates', 'candidate_coordinates', 'labels', 'img_id'])
            test_y = (test_df.labels).astype(int)
            del test_df

            # Train a random forest classifier
            filename = f'RF_skimage_{k}.sav'
            if not (models_path / filename).exists():
                clf = RandomForestClassifier(
                    n_estimators=1000, max_depth=data.shape[1]-4, n_jobs=-1, random_state=0)
                clf.fit(train_X, train_y)

                # Store the trained models
                with open(models_path / filename, 'wb') as f:
                    pickle.dump(clf, f)
            else:
                with open(models_path / filename, 'rb') as f:
                    clf = pickle.load(f)

            # Asses performance
            test_y_predicted = clf.predict_proba(test_X)[:, 1]
            train_y_predicted = clf.predict_proba(train_X)[:, 1]

            auc_full_features_test = roc_auc_score(test_y, test_y_predicted)
            auc_full_features_train = roc_auc_score(train_y, train_y_predicted)
            aucs_test.append(auc_full_features_test)
            aucs_train.append(auc_full_features_train)

            # Store on memory the classifier and the feature importances
            clfs.append(clf)
            feat_importances.append(clf.feature_importances_)
        del train_X, train_y, test_X, test_y, test_y_predicted, train_y_predicted

        # Store the auroc data to disk
        auc_data = pd.DataFrame(aucs_test + aucs_train, columns=['auc'])
        auc_data['set'] = ['val']*len(aucs_test) + ['train']*len(aucs_train)
        auc_data.to_csv(models_path/f'aucs_skimage_haar_{folds}_runs.csv')
        # Store the feature importances to disk
        filename = f'feature_importances_skimage_haar_{folds}_runs.p'
        with open(models_path/filename, 'wb') as f:
            pickle.dump(feat_importances, f)

        logging.info('Sorting features by importance...')
        sorted_features, mean_importance = sort_relevances(feat_importances)
        OUR_CUT = 2000

        # Load all aour features
        logging.info('Storing selection...')
        del data
        data = pd.concat([
            pd.read_feather(data_path / f'haar_features_{detector}/all_feats_10to1_pt0_skimage.fth'),
            pd.read_feather(data_path / f'haar_features_{detector}/all_feats_10to1_pt1_skimage.fth')
        ], ignore_index=True)

        metadata_cols = [col for col in data.columns if ('haar' not in col)]
        selected_feats_cols = data.columns.values[sorted_features[:OUR_CUT]].tolist()
        data.loc[:, selected_feats_cols+metadata_cols].reset_index(drop=True).to_feather(
            data_path / f'haar_features_{detector}/all_cases_skimage_features_10to1_2000_selection.fth')
        del data

    # COMBINED ANALYSIS
    logging.info('Starting combined analysis...')
    logging.info('Loading data...')
    if not(data_path / f'haar_features_{detector}/all_cases_all_features_10to1_3000_selection.fth').exists():
        our_features = pd.read_feather(
            data_path / f'haar_features_{detector}/all_cases_our_features_10to1_1000_selection.fth')
        feature_cols = [col for col in our_features.columns if ('haar' in col)]
        skimage_features = pd.read_feather(
            data_path / f'haar_features_{detector}/all_cases_skimage_features_10to1_2000_selection.fth')
        data = pd.concat(
            [our_features.loc[:, feature_cols], skimage_features], axis=1, ignore_index=True)
        data.columns = feature_cols + skimage_features.columns.tolist()
        data.reset_index(drop=True).to_feather(
            data_path / f'haar_features_{detector}/all_cases_all_features_10to1_3000_selection.fth')
        del our_features, skimage_features
    else:
        data = pd.read_feather(
            data_path / f'haar_features_{detector}/all_cases_all_features_10to1_3000_selection.fth')

    # CHOOSE THE NUMBER OF CASES TO CONSIDER:
    N = 100
    # Reduce the number of examples to be manageable
    data = data.loc[data.img_id.isin(data.img_id.unique()[:N]), :]

    # SELECT THE NUMBER OF FOLDS FOR TO USE IN THE EXPERIMENTS:
    folds = 5
    n = int(len(data.img_id.unique()) / folds)

    # Train the Forests and store the features importances
    aucs_test = []
    aucs_train = []
    feat_importances = []
    clfs = []

    models_path = (data_path / f'haar_models_{detector}')
    models_path.mkdir(exist_ok=True, parents=True)
    logging.info('Starting training iterations...')
    for k, test_img_id in tqdm(enumerate(utils.blockwise_retrieval(data.img_id.unique(), n))):
        if k == folds + 1:
            continue
        # Divide train and test based on cases (cross validation image wise)
        train_df = data[~data.img_id.isin(test_img_id)]
        train_df = train_df.sample(frac=1, random_state=0)
        test_df = data[data.img_id.isin(test_img_id)]

        # Generate features and labels datasets
        train_X = train_df.drop(
            columns=['patch_coordinates', 'candidate_coordinates', 'labels', 'img_id'])
        train_y = (train_df.labels).astype(int)
        del train_df
        test_X = test_df.drop(
            columns=['patch_coordinates', 'candidate_coordinates', 'labels', 'img_id'])
        test_y = (test_df.labels).astype(int)
        del test_df

        filename = f'RF_all_{k}.sav'
        if not (models_path / filename).exists():
            # Train a random forest classifier
            clf = RandomForestClassifier(
                n_estimators=1000, max_depth=data.shape[1]-4, n_jobs=-1, random_state=0)
            clf.fit(train_X, train_y)

            # Store the trained models
            with open(models_path / filename, 'wb') as f:
                pickle.dump(clf, f)
        else:
            with open(models_path / filename, 'rb') as f:
                clf = pickle.load(f)

        # Asses performance
        test_y_predicted = clf.predict_proba(test_X)[:, 1]
        train_y_predicted = clf.predict_proba(train_X)[:, 1]

        auc_full_features_test = roc_auc_score(test_y, test_y_predicted)
        auc_full_features_train = roc_auc_score(train_y, train_y_predicted)
        aucs_test.append(auc_full_features_test)
        aucs_train.append(auc_full_features_train)

        # Store on memory the classifier and the feature importances
        clfs.append(clf)
        feat_importances.append(clf.feature_importances_)
    del train_X, train_y, test_X, test_y, test_y_predicted, train_y_predicted
    del data
    # Store the auroc data to disk
    auc_data = pd.DataFrame(aucs_test + aucs_train, columns=['auc'])
    auc_data['set'] = ['val']*len(aucs_test) + ['train']*len(aucs_train)
    auc_data.to_csv(models_path/f'aucs_all_haar_{folds}_runs.csv')
    # Store the feature importances to disk
    filename = f'feature_importances_all_haar_{folds}_runs.p'
    with open(models_path/filename, 'wb') as f:
        pickle.dump(feat_importances, f)

    logging.info('Sorting features by importance...')
    sorted_features, mean_importance = sort_relevances(feat_importances)

    # SELECTION OF MOST IMPORTANT FEATURES
    logging.info('Starting analysis importance vs performace...')
    logging.info('Loading_data...')
    data = pd.read_feather(
        data_path / f'haar_features_{detector}/all_cases_all_features_10to1_3000_selection.fth')

    # CHOOSE THE NUMBER OF CASES TO CONSIDER:
    N = 100
    # Reduce the number of examples to be manageable
    data = data.loc[data.img_id.isin(data.img_id.unique()[:N]), :]

    folds = 5
    n = int(len(data.img_id.unique()) / folds)

    aucs_test_feat_sel_all = {}
    aucs_train_feat_sel_all = {}
    aupr_test_feat_sel_all = {}
    aupr_train_feat_sel_all = {}
    clfs_feat_sel_all = {}

    models_path = data_path / f'haar_models_{detector}'
    models_path.mkdir(exist_ok=True, parents=True)
    logging.info('Starting training iteration...')
    condition = (
        (models_path/'aupr_test_all_sorted.csv').exists() and
        (models_path/'aucs_test_all_sorted.csv').exists() and
        (models_path/'aupr_train_all_sorted.csv').exists() and
        (models_path/'auc_train_all_sorted.csv').exists())
    if not condition:
        for i in tqdm([len(sorted_features), 1600, 800, 400, 200, 100, 50, 25, 10, 5, 2], total=11):
            clfs_feat_sel_all[i] = []
            aucs_test_feat_sel_all[i] = []
            aucs_train_feat_sel_all[i] = []
            aupr_test_feat_sel_all[i] = []
            aupr_train_feat_sel_all[i] = []
            feature_selection = sorted_features[:i]

            data_ = data.iloc[:, feature_selection.astype(int).tolist() + [3000, 3001, 3002, 3003]]
            for k, test_img_id in tqdm(enumerate(utils.blockwise_retrieval(data_.img_id.unique(), n))):
                if k == folds + 1:
                    continue

                # Divide train and test based on cases (cross validation image wise)
                train_df = data_[~data_.img_id.isin(test_img_id)]
                train_df = train_df.sample(frac=1, random_state=0)
                test_df = data_[data_.img_id.isin(test_img_id)]

                # Generate features and labels datasets
                train_X = train_df.drop(
                    columns=['patch_coordinates', 'candidate_coordinates', 'labels', 'img_id'])
                train_y = (train_df.labels).astype(int)
                del train_df
                test_X = test_df.drop(
                    columns=['patch_coordinates', 'candidate_coordinates', 'labels', 'img_id'])
                test_y = (test_df.labels).astype(int)
                del test_df

                # Train a random forest classifier
                filename = f'RF_all_performace_{i}_{k}.sav'
                if not (models_path / filename).exists():
                    clf = RandomForestClassifier(
                        n_estimators=1000, max_depth=data.shape[1]-4, n_jobs=-1, random_state=0)
                    clf.fit(train_X, train_y)

                    # Save the model to disk
                    with open(models_path / filename, 'wb') as f:
                        pickle.dump(clf, f)
                else:
                    with open(models_path / filename, 'rb') as f:
                        clf = pickle.load(f)

                # Performance
                test_y_predicted = clf.predict_proba(test_X)[:, 1]
                train_y_predicted = clf.predict_proba(train_X)[:, 1]

                # AUROC
                aucs_test_feat_sel_all[i].append(roc_auc_score(test_y, test_y_predicted))
                aucs_train_feat_sel_all[i].append(roc_auc_score(train_y, train_y_predicted))

                # PR
                pr, rc, th = precision_recall_curve(test_y, test_y_predicted)
                aupr_test_feat_sel_all[i].append(auc(rc, pr))
                pr, rc, th = precision_recall_curve(train_y, train_y_predicted)
                aupr_train_feat_sel_all[i].append(auc(rc, pr))

                # Store classifiers on memory
                clfs_feat_sel_all[i].append(clf)

        aupr_test = pd.DataFrame.from_dict(aupr_test_feat_sel_all)
        aupr_test.to_csv(models_path/'aupr_test_all_sorted.csv')

        aucs_test = pd.DataFrame.from_dict(aucs_test_feat_sel_all)
        aucs_test.to_csv(models_path/'aucs_test_all_sorted.csv')

        aupr_train = pd.DataFrame.from_dict(aupr_train_feat_sel_all)
        aupr_train.to_csv(models_path/'aupr_train_all_sorted.csv')

        aucs_train = pd.DataFrame.from_dict(aucs_train_feat_sel_all)
        aucs_train.to_csv(models_path/'auc_train_all_sorted.csv')


if __name__ == '__main__':
    main()
