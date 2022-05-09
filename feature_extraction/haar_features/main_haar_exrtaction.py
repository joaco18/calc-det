import cv2
import sys; sys.path.insert(0, '../..')

from database.dataset import INBreast_Dataset
from mc_candidate_proposal.hough_mc import HoughCalcificationDetection
from models.bria2014.haar_extractor import HaarFeatureExtractor
from metrics.metrics import get_tp_fp_fn
import general_utils.utils as utils
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from time import time

from dask import delayed
from skimage.transform import integral_image
from skimage.feature import haar_like_feature


@delayed
def extract_feature_image(img, feature_type=None, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)


def main():
    db = INBreast_Dataset(
        return_lesions_mask=True,
        level='image',
        max_lesion_diam_mm=1.,
        extract_patches=False,
        extract_patches_method='all',  # 'centered'
        patch_size=256,
        stride=256,
        min_breast_fraction_roi=0.5,
        normalize=None,
        n_jobs=-1,
        partitions=['train', 'validation']
    )

    dehazing_params = {'omega': 0.9, 'window_size': 11, 'radius': 40, 'eps': 1e-5}

    hough1_params = {'method': cv2.HOUGH_GRADIENT, 'dp': 1, 'minDist': 20,
                     'param1': 300, 'param2': 8,  'minRadius': 2, 'maxRadius': 20}

    hough2_params = {'method': cv2.HOUGH_GRADIENT, 'dp': 1, 'minDist': 20,
                     'param1': 300, 'param2': 10,  'minRadius': 2, 'maxRadius': 20}
    back_ext_radius = 50
    erosion_iter = 20
    erosion_size = 5

    path = Path('/home/jseia/Desktop/ml-dl/data/hough')
    path.mkdir(exist_ok=True, parents=True)

    hd = HoughCalcificationDetection(
        dehazing_params, back_ext_radius,
        path,
        hough1_params, hough2_params,
        erosion_iter=erosion_iter,
        erosion_size=erosion_size
    )

    BASE_PATH = Path('/home/jseia/Desktop/ml-dl/data_rois/haar_features')
    for idx in tqdm(range(len(db)), total=len(db)):
        case = db[idx]
        image = case['img']
        image_id = db.df.iloc[idx].img_id
        radiouses = case['radiuses']
        true_bboxes = db[idx]['lesion_bboxes']

        _, h2_circles = hd.detect(
            image, image_id, load_processed_images=True, hough2=True)

        tp, fp, fn, gt_d, close_fp = get_tp_fp_fn(true_bboxes, radiouses, h2_circles, 7, 0.0)

        h2_circles = np.concatenate((tp, fp), axis=0)

        images = np.empty((len(h2_circles), 14, 14))
        # generate a patches array to distribute computation
        for j, location in enumerate(h2_circles):
            # Get the patch arround center
            x1, x2, y1, y2 = utils.patch_coordinates_from_center(
                center=(location[0], location[1]), image_shape=image.shape,
                patch_size=14, use_padding=False)
            images[j, :, :] = image[y1:y2, x1:x2]

        labels = np.array([1] * len(tp) + [0] * len(fp))

        # feature_types = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4']

        X = delayed(extract_feature_image(img) for img in images)

        # Compute the result
        t_start = time()
        X = np.array(X.compute(scheduler='processes'))
        time_full_feature_comp = time() - t_start

        haarfe = HaarFeatureExtractor(14, False, True)
        t_start = time()
        X_r = []
        for img in images:
            X_r.append(haarfe.extract_features_from_crop(img))
        X_r = np.asarray(X_r)
        time_full_feature_comp_rot = time() - t_start

        X = np.concatenate([X, X_r], axis=1)
        features = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
        features['label'] = labels

        print(f'Image {idx+1} -- Total computation horizontal haar {time_full_feature_comp},'
              f' rotated haar:{time_full_feature_comp_rot}')

        features.to_feather(BASE_PATH/f'{image_id}.fth')


if __name__ == '__main__':
    main()
