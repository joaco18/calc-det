import numpy as np


def balance_candidates(
    tp: np.ndarray, fp: np.ndarray, fp2tp_sample: float, minimum_fp: float = 0.2
):
    """ Samples false positives to the desired proportion with respect to true positives.
    If the requiered number of false positives is larger than the ones available, then
    return all of them, if there are no TP among the candidates return 'minimum_fp'
    fraction of the false positives.
    Args:
        tp (list): [(x, y, r)]
        fp (list): [(x, y, r)]
        fp2tp_sample (float): Ratio of fp/tp to be sampled.
        minimum_fp (float): If no TP were deetected, still sample this
            fraction of the FP of that case.
    """
    # Sample candidates to the desired proportion between TP and FP
    np.random.seed(20)
    sample_size = len(tp) * fp2tp_sample
    sample_size = int(minimum_fp * len(fp)) if sample_size == 0 else sample_size
    # check if required fraction of candidates is possible if not return the closest
    fp_indxs = np.arange(len(fp))
    if sample_size <= len(fp):
        fp_indxs = np.random.choice(fp_indxs, size=sample_size, replace=False)
        fp = [fp[i] for i in range(len(fp)) if i in fp_indxs]
    return tp, fp


def filter_dets_from_muscle_region(candidates: np.ndarray, muscle_mask: np.ndarray):
    if not muscle_mask.any():
        return candidates
    muscle_mask = np.where(muscle_mask > 0, False, True)
    return candidates[muscle_mask[candidates[:, 1], candidates[:, 0]]]
