import cv2
import logging

import numpy as np
import skimage.feature as skift

from database.utils import min_max_norm
from itertools import combinations_with_replacement
from numba import njit
from skimage.util.dtype import img_as_float


def hessian_matrix(image, sigma=1, mode='constant', cval=0):
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


@njit
def get_sigmas(min_sigma: float, sigma_ratio: float, k: int):
    return min_sigma * (sigma_ratio ** np.arange(k+1))


def hdog_blob_extraction(
    image: np.ndarray,
    img_type: str = 'uint8',  # 'float'
    min_sigma: float = 1.18,
    max_sigma: float = 3.1,
    sigma_ratio: float = 1.05,
    n_scales: int = None,
    dog_blob_th: float = 0.006,
    dog_overlap: float = 1,
    exclude_border: int = False,
    divider: float = 500.,
    method: str = 'marasinou',
    h_thr: float = 1.4
):
    # convert the image to the desired intensity range
    if 'float' in img_type:
        image = img_as_float(image)
    elif 'uint8' in img_type:
        image = min_max_norm(image, 2**8)

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

    # TODO: convert this in a fuction get_dog_and_hessian_eigenvalues()
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
            gaussian_images[idx] = (
                gaussian_images[idx] - gaussian_images[idx + 1]
            )
            gaussian_images[idx] *= sigma_array[idx] / 0.109

            # Compute the hessian
            img_eigval.append(hessian_matrix(gaussian_images[idx]))
            # hessian_matrix(
            # gaussian_images[idx], sigma=sigma_array[idx],
            #  mode='constant')

            # Compute the eigenvalues
            img_eigval[idx] = skift.hessian_matrix_eigvals(img_eigval[idx])

    # Get arrays from the lists
    img_eigval = np.asarray(img_eigval)
    dog_cube = np.stack(gaussian_images[:-1], axis=-1)
    # ----------------------------------------------------

    # TODO: convert this in a fuction get_blob_candidates()
    # ----------------------------------------------------
    # Get the local maximum
    local_maxima = skift.peak.peak_local_max(
        dog_cube,
        threshold_abs=dog_blob_th,
        footprint=np.ones((3,) * (image.ndim + 1)),
        threshold_rel=0.0,
        exclude_border=(0,) * (image.ndim + 1),
    )

    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    lm = local_maxima.astype(np.float64)
    # transform indexes in sigmas
    sigmas_of_peaks = sigma_array[local_maxima[:, -1]]
    # Remove sigma index and replace with sigmas
    lm[:, :-1] = sigmas_of_peaks

    # Filter blobs according to overlapping
    img_blobs = skift.blob._prune_blobs(lm, dog_overlap)
    # ----------------------------------------------------

    # Get condition and create hessian mask
    # TODO: convert this in a fuction filter_blobs()
    # TODO: TODO: TODO: use njit
    # ----------------------------------------------------
    img_hdog_filter = []
    print(img_eigval)
    for idx in range(img_eigval.shape[0]-1):
        if method == 'marasinou':
            # Get trace and determinant
            trace = img_eigval[idx][0] * img_eigval[idx+1][2]
            det = (img_eigval[idx][0] * img_eigval[idx][3])
            det -= img_eigval[idx][1] * img_eigval[idx][0]
            # Check conditions
            condition_a = (trace < 0)
            condition_ba = (det < 0)
            condition_bb = (np.abs(det) / (trace * trace)) <= h_thr
            hessian_mask = condition_a & (condition_ba | condition_bb)
            # TODO: check homogeneity of bool masks and unit8 masks
        else:
            # Get the maximum persistent eigenvalue
            eig1_ms = img_eigval[idx][0] * img_eigval[idx+1][0]
            eig2_ms = img_eigval[idx][1] * img_eigval[idx+1][1]

            # Get the proportional threshold
            print(np.max(eig1_ms))
            thrs_1 = np.max(eig1_ms) / divider
            thrs_2 = np.max(eig2_ms) / divider

            # Check conditions
            hessian_mask = (eig1_ms > thrs_1) & (eig2_ms > thrs_2)
            # TODO: check homogeneity of bool masks and unit8 masks

            img_hdog_filter.append(hessian_mask)

    img_hdog_filter = np.asarray(img_hdog_filter)

    blobs_filtered = []
    blob_coords = img_blobs[:, :2].astype(int)

    for sigma in np.unique(img_blobs[:, 2]):
        idx = np.where(sigma_array == sigma)
        hess_mask = img_hdog_filter[idx][0]  # TODO: Check this zero
        hess_mask = np.where(hess_mask, 1, 0)

        blob_coords = img_blobs[np.where(img_blobs[:, 2] == sigma), :2]
        blob_mask = np.zeros_like(hess_mask)
        blob_mask[blob_coords] = 1

        # TODO: Fix this
        filtered_blob_idx = np.where(blob_mask * hess_mask)
        filtered_blob_idx = np.expand_dims(filtered_blob_idx, -1)
        filtered_blob_idx[-1] = sigma
        blobs_filtered.expand(filtered_blob_idx.tolist())
    # ----------------------------------------------------

    logging.info(f'No. of blobs: {img_blobs.shape[0]}')
    logging.info(f'No. of blobs after hessian condition: {len(blobs_filtered)[0]}')

    return blobs_filtered, img_blobs


def create_binary_mask_from_blobs(image: np.ndarray, blobs_x_y_sigma: list):
    img_binary_blobs = np.zeros(image.shape)
    for blob in blobs_x_y_sigma:
        img_binary_blobs = cv2.circle(
            img_binary_blobs, (blob[1], blob[0]), (blob[2] / 2), 255, -1
        )


def main():
    hdog_blob_extraction()


if __name__ == '__main__':
    main()
