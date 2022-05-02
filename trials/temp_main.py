import cv2
import logging
import time

import numpy as np
import skimage.feature as skift

import sys; sys.path.insert(0, '../')
from general_utils.utils import min_max_norm
from itertools import combinations_with_replacement
from numba import njit
from skimage.util.dtype import img_as_float
from database.dataset import INBreast_Dataset

logging.basicConfig(level=logging.INFO)


def hessian_matrix(image, sigma=1, mode='constant', cval=0):
    if image.dtype != float:
        image = img_as_float(image)

    # gaussian_filtered = ndi.gaussian_filter(
    #     image, sigma=sigma, mode=mode, cval=cval
    # )
    gradients = np.gradient(image)
    axes = reversed(range(image.ndim))
    return [
        np.gradient(gradients[ax0], axis=ax1)
        for ax0, ax1 in combinations_with_replacement(axes, 2)
    ]


@njit(cache=True)
def get_sigmas(min_sigma: float, sigma_ratio: float, k: int):
    return min_sigma * (sigma_ratio ** np.arange(k+1))


def get_dog_and_hessian_eigenvalues(image: np.ndarray, sigma_array: np.ndarray, method: str):
    # ----------------------------------------------------
    gaussian_images = []
    img_eigval = []
    for m, s in enumerate(sigma_array):
        # Get kernel size
        ks = int(4 * s + 0.5)
        ks = ks if ks % 2 == 1 else ks + 1

        # Get the gaussian smoothing
        gaussian_images.append(cv2.GaussianBlur(image, (ks, ks), s))
        if m != 0:
            idx = m - 1

            # Get the DoG and store in place to avoid using innecesary mem
            gaussian_images[idx] = (gaussian_images[idx] - gaussian_images[idx + 1])
            gaussian_images[idx] *= sigma_array[idx] / (sigma_array[idx + 1] - sigma_array[idx])

            # Compute the hessian
            img_eigval.append(hessian_matrix(gaussian_images[idx]))
            # hessian_matrix(
            # gaussian_images[idx], sigma=sigma_array[idx],
            #  mode='constant')

            # Compute the eigenvalues
            if method != 'marasinou':
                img_eigval[idx] = skift.hessian_matrix_eigvals(img_eigval[idx])

    # Get arrays from the lists
    img_eigval = np.asarray(img_eigval)
    dog_cube = np.stack(gaussian_images[:-1], axis=-1)
    return dog_cube, img_eigval


def filter_by_hessian_condition(img_blobs, img_eigval, method, sigma_array, h_thr, divider):
    blobs_filtered = []
    for sigma in np.unique(img_blobs[:, 2]):
        idx = np.where(sigma_array == sigma)[0][0]
        if method == 'marasinou':
            # Get trace and determinant
            trace = img_eigval[idx][0] + img_eigval[idx][2]
            det = (img_eigval[idx][0] * img_eigval[idx][2])
            det -= img_eigval[idx][1] * img_eigval[idx][1]
            # Check conditions
            hessian_mask = (np.abs(det) / (trace * trace + 2.23e-16)) <= h_thr
            hessian_mask = hessian_mask | (det < 0)
            hessian_mask = hessian_mask & (trace < 0)
            # logging.info(hessian_mask.sum())
        else:
            # Get the maximum persistent eigenvalue
            eig1_ms = img_eigval[idx][0] * img_eigval[idx+1][0]
            eig2_ms = img_eigval[idx][1] * img_eigval[idx+1][1]

            # Get the proportional threshold
            thrs_1 = np.max(eig1_ms) / divider
            thrs_2 = np.max(eig2_ms) / divider

            # Check conditions
            hessian_mask = (eig1_ms > thrs_1) & (eig2_ms > thrs_2)
            # logging.info(hessian_mask.sum())

        selected_blobs = np.where(img_blobs[:, 2] == sigma)[0]
        blob_coords = img_blobs[selected_blobs, :2].astype(np.int16)
        blob_mask = np.zeros(hessian_mask.shape, dtype=np.int16)
        blob_mask[blob_coords[:, 0], blob_coords[:, 1]] = 1

        filtered_blob_x, filtered_blob_y = np.where(blob_mask * hessian_mask)
        filtered_blobs = np.zeros((len(filtered_blob_x), 3))
        filtered_blobs[:, 0] = filtered_blob_x.astype(np.int16)
        filtered_blobs[:, 1] = filtered_blob_y.astype(np.int16)
        filtered_blobs[:, 2] = sigma
        blobs_filtered.append(filtered_blobs)
    blobs_filtered = np.concatenate(blobs_filtered)
    return blobs_filtered


def get_blob_candidates(
    dog_cube, dog_blob_th, sigma_array, dog_overlap, img_eigval, method, h_thr, divider
):

    # Get the local maximum
    local_maxima = skift.peak.peak_local_max(
        dog_cube,
        threshold_abs=dog_blob_th,
        footprint=np.ones((3, 3, 3)),
        threshold_rel=0.0,
        exclude_border=(0, 0, 0),
    )
    # print(len(local_maxima))
    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    # Transform indexes in sigmas
    lm = local_maxima.astype(np.float64)
    sigmas_of_peaks = sigma_array[local_maxima[:, -1]]
    lm[:, -1] = sigmas_of_peaks

    # Filter blobs according to overlapping
    img_blobs = skift.blob._prune_blobs(lm, dog_overlap)

    logging.info('Filter blobs')
    logging.info(img_eigval.shape)

    # Filter blobs by hessian condition
    blobs_filtered = filter_by_hessian_condition(
        img_blobs, img_eigval, method, sigma_array, h_thr, divider
    )

    logging.info(f'No. of blobs: {img_blobs.shape}')
    logging.info(f'No. of blobs after hessian condition: {blobs_filtered.shape}')

    return blobs_filtered, img_blobs


def hdog_blob_extraction(
    image: np.ndarray,
    img_type: str = 'uint8',  # 'float'
    min_sigma: float = 1.18,
    max_sigma: float = 3.1,
    sigma_ratio: float = 1.05,
    n_scales: int = None,
    dog_blob_th: float = 0.06,
    dog_overlap: float = 1,
    divider: float = 500.,
    method: str = 'marasinou',
    h_thr: float = 1.4
):
    logging.info('Start')
    # convert the image to the desired intensity range
    if 'float' in img_type:
        image = img_as_float(image)
    elif 'uint8' in img_type:
        image = min_max_norm(image, 1)
        # image = img_as_float(image)

    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    if n_scales is None:
        k = int(np.log(max_sigma / min_sigma) / np.log(sigma_ratio) + 1)
    else:
        k = n_scales
        sigma_ratio = (max_sigma/min_sigma) ** (1 / n_scales)

    # Progression of standard deviations for gaussian kernels
    sigma_array = get_sigmas(min_sigma, sigma_ratio, k)

    # Comute in the same iterative process the smoothed image,
    #   the diferential of gaussian and the eigenvalues

    logging.info('Get gaussian')

    dog_cube, img_eigval = \
        get_dog_and_hessian_eigenvalues(image, sigma_array, method)

    # logging.info(f'{dog_cube.min()}, {dog_cube.max()}, {np.mean(dog_cube)}')

    logging.info('Get Blob candidates')

    blobs_filtered, img_blobs = get_blob_candidates(
        dog_cube, dog_blob_th, sigma_array, dog_overlap, img_eigval, method, h_thr, divider
    )
    return blobs_filtered, img_blobs


def create_binary_mask_from_blobs(image: np.ndarray, blobs_x_y_sigma: list):
    img_binary_blobs = np.zeros(image.shape)
    for blob in blobs_x_y_sigma:
        img_binary_blobs = cv2.circle(
            img_binary_blobs, (blob[1], blob[0]), (blob[2] / 2), 255, -1
        )


def main():
    db = INBreast_Dataset(
        return_lesions_mask=True,
        level='image',
        extract_patches=False,
        normalize=None,
        n_jobs=-1,
    )

    start = time.time()
    _, _ = hdog_blob_extraction(db[0]['img'])  # , method='alex')
    print(f'TIME: {time.time()-start}')


if __name__ == '__main__':
    main()
