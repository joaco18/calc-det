import numpy as np
import math
from numba import njit
import logging

logging.basicConfig(level=logging.INFO)


@njit
def clip(val):
    if val > 1.0:
        return 1.0
    elif val < -1.0:
        return -1.0
    else:
        return val


@njit(cache=True)
def compute_disk_overlap(d, r1, r2):
    """
    Compute fraction of surface overlap between two disks of radii
    ``r1`` and ``r2``, with centers separated by a distance ``d``.

    Parameters
    ----------
    d : float
        Distance between centers.
    r1 : float
        Radius of the first disk.
    r2 : float
        Radius of the second disk.

    Returns
    -------
    fraction: float
        Fraction of area of the overlap between the two disks.
    """

    ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
    ratio1 = clip(ratio1)
    acos1 = math.acos(ratio1)

    ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
    ratio2 = clip(ratio2)
    acos2 = math.acos(ratio2)

    a = -d + r2 + r1
    b = d - r2 + r1
    c = d + r2 - r1
    d = d + r2 + r1
    area = (r1 ** 2 * acos1 + r2 ** 2 * acos2 -
            0.5 * math.sqrt(abs(a * b * c * d)))
    return area / (math.pi * (min(r1, r2) ** 2))


@njit(cache=True)
def blob_overlap(blob1, blob2):
    """Finds the overlapping area fraction between two blobs.

    Returns a float representing fraction of overlapped area. Note that 0.0
    is *always* returned for dimension greater than 3.

    Parameters
    ----------
    blob1 : sequence of arrays
        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
        where ``row, col`` (or ``(pln, row, col)``) are coordinates
        of blob and ``sigma`` is the standard deviation of the Gaussian kernel
        which detected the blob.
    blob2 : sequence of arrays
        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
        where ``row, col`` (or ``(pln, row, col)``) are coordinates
        of blob and ``sigma`` is the standard deviation of the Gaussian kernel
        which detected the blob.

    Returns
    -------
    f : float
        Fraction of overlapped area (or volume in 3D).
    """
    root_ndim = math.sqrt(2)

    # Here we divide the coordinates and the sigmas by the largest sigmas, to
    # normalize de radious to 1 for the largest blob and sth < 1 for the other blob
    # TODO: I don't think this is necessary
    if blob1[-1] == blob2[-1] == 0:
        return 0.0
    elif blob1[-1] > blob2[-1]:
        max_sigma = blob1[-1]
        r1 = 1
        r2 = blob2[-1] / blob1[-1]
    else:
        max_sigma = blob2[-1]
        r2 = 1
        r1 = blob1[-1] / blob2[-1]
    pos1 = blob1[:2] / (max_sigma * root_ndim)
    pos2 = blob2[:2] / (max_sigma * root_ndim)

    d = np.sqrt(np.sum((pos2 - pos1)**2))
    # No overlap case
    if d > r1 + r2:
        return 0.0

    # One blob is inside the other
    if d <= abs(r1 - r2):
        return 1.0

    return compute_disk_overlap(d, r1, r2)


@njit(cache=True)
def compare_and_filter_pairs(pairs: np.ndarray, blobs_array: np.ndarray, overlap: float):
    """Check if closes detections have an overlapping greater than the threshold
    Args:
        pairs (np.ndarray): indexes of a pair of close points
        blobs_array (np.ndarray): array with blobs as rows (x, y, sigma)
        overlap (float): overlapping threshold
    Returns:
        (np.ndarray): filtered array with blobs as rows (x, y, sigma)
    """
    for (i, j) in pairs:
        blob1, blob2 = blobs_array[i], blobs_array[j]
        if (blob1[-1] == 0) or (blob2[-1] == 0):
            continue
        if blob_overlap(blob1, blob2) > overlap:
            if blob1[-1] > blob2[-1]:
                blob2[-1] = 0
            else:
                logging.info(blob1[-1])
                blob1[-1] = 0
                # TODO: Check if this is ocurring in place
                logging.info(blob1[-1])
                logging.info(blobs_array[i])
    return blobs_array[np.where(blobs_array[:, -1] > 0)]


def min_max_norm(img: np.ndarray, max_val: int = None):
    """
    Scales images to be in range [0, 2**bits]

    Args:
        img (np.ndarray): _description_
        max_val (int, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if max_val is None:
        max_val = np.iinfo(img.dtype).max
    img = (img - img.min()) / (img.max() - img.min()) * max_val
    return img
