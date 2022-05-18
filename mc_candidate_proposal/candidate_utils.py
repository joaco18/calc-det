import numpy as np
import pandas as pd


def balance_candidates(
    candidates: pd.DataFrame, fp2tp_sample: float, minimum_fp: float = 0.2
):
    """ Samples false positives to the desired proportion with respect to true positives.
    If the requiered number of false positives is larger than the ones available, then
    return all of them, if there are no TP among the candidates return 'minimum_fp'
    fraction of the false positives.
    Args:
        candidates (pd.DataFrame): Rows: TP + FP candidates
            Columns: ['x', 'y', 'radius', 'label', 'matching_gt','repeted_idxs']
        fp2tp_sample (float): Ratio of fp/tp to be sampled.
        minimum_fp (float): If no TP were deetected, still sample this
            fraction of the FP of that case.
    """
    # Sample candidates to the desired proportion between TP and FP
    tp = candidates.loc[candidates.label == 'TP']
    fp = candidates.loc[candidates.label == 'FP']
    n_tp = len(tp)
    n_fp = len(fp)

    sample_size = n_tp * fp2tp_sample
    sample_size = int(minimum_fp * n_fp) if sample_size == 0 else sample_size

    # check if required fraction of candidates is possible if not return the closest
    if sample_size <= n_fp:
        fp = fp.sample(sample_size, replace=False, random_state=20)
    return pd.concat([tp, fp], axis=0)


def filter_dets_from_muscle_region(candidates: np.ndarray, muscle_mask: np.ndarray):
    if not muscle_mask.any():
        return candidates
    muscle_mask = np.where(muscle_mask > 0, False, True)
    return candidates[muscle_mask[candidates[:, 1], candidates[:, 0]]]
