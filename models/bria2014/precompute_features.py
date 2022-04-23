import cv2
import logging

import numpy as np
import concurrent.futures as ft
import feature_extraction as fe
import multiprocessing as mp
import pandas as pd


from pathlib import Path
from numba import njit
from tqdm import tqdm

BASE_PATH = Path('../../../data_rois/')
BASE_PATH_BRIA = Path('../../data/Bria')
logging.basicConfig(level=logging.INFO)
N_BATCHES = 1000


def open_img(filename: str):
    filename = str(BASE_PATH / filename)
    return cv2.imread(filename, cv2.IMREAD_ANYDEPTH)


@njit
def normalize(img: np.ndarray, mean: float, std: float):
    return (img - mean) / std


def open_img_normalize(filename: str, mean: float = 1438.1613, std: float = 261.5595):
    return normalize(open_img(filename), mean, std)


def load_imgs(img_files: list, normalize: bool = False):
    with ft.ThreadPoolExecutor() as executor:
        if normalize:
            xs = list(executor.map(open_img_normalize, img_files))
        else:
            xs = list(executor.map(open_img, img_files))
        return np.asarray(xs)


@njit()
def sequential_integral(arr: np.ndarray):
    mapped = np.zeros((arr.shape[0], arr.shape[1] + 1, arr.shape[2] + 1))
    for i in range(arr.shape[0]):
        mapped[i, :, :] = fe.to_integral(arr[i, :, :])
    return mapped


@njit()
def sequential_integral_diag(arr: np.ndarray):
    mapped = np.zeros((arr.shape[0], arr.shape[1] + 1, arr.shape[2] + 1))
    for i in range(arr.shape[0]):
        mapped[i, :, :] = fe.to_diag_integral(arr[i, :, :])
    return mapped


def main():
    # n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
    n_jobs = 8
    data = pd.read_feather(BASE_PATH_BRIA/'small_dataset.fth')
    # batch_size = int(data.shape[0]/N_BATCHES)
    for k in range(1):
        logging.info(f'Stage {k}')

        # Load the images
        xs = load_imgs(data.filename.tolist(), normalize=True)
        # xs = load_imgs(data.filename[(batch_size*k):(batch_size*(k+1))].tolist(), normalize=True)

        # Get the integral images
        xsi = sequential_integral(xs)

        # Generate the horizontal features
        features_h = fe.feature_instantiator(14, 'hor')
        features_db = []
        for j, feature in tqdm(enumerate(features_h), total=len(features_h)):
            with mp.Pool(n_jobs) as pool:
                res = pool.map(feature, xsi)
            features_db.append(res)
        del xsi

        # Get the diagonal integral images
        xs = sequential_integral_diag(xs)

        # Generate the rotated features
        features_r = fe.feature_instantiator(14, 'rot')
        for j, feature in tqdm(enumerate(features_r, j+1), total=len(features_r)):
            with mp.Pool(n_jobs) as pool:
                res = pool.map(feature, xs)
            features_db.append(res)

        # Store the dataframe
        db_filename = BASE_PATH_BRIA/'features'/f'features_{k}.fth'
        db_filename.parent.mkdir(exist_ok=True, parents=True)
        features_db = pd.DataFrame(
            data=np.asarray(features_db).T, columns=[f'f{i}' for i in range(len(features_db))]
        )
        features_db.reset_index(drop=True, inplace=True)

        features_db.to_feather(db_filename)


if __name__ == '__main__':
    main()
