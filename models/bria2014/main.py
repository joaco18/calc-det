
import numpy as np
import feature_modules as fm
import feature_extraction as fe
import typing as tp

from numba import njit
from sklearn.metrics import confusion_matrix


WeakClassifier = tp.NamedTuple(
    'WeakClassifier', [
        ('threshold', float), ('alpha', float), ('feature_indx', int),
        ('classifier', tp.Callable[[np.ndarray], float])
    ])

StrongClassifier = tp.NamedTuple(
    'StrongClassifier', [
        ('threshold', float), ('weak_classifiers', tp.List[WeakClassifier])
    ])

PredictionStats = tp.NamedTuple(
    'PredictionStats', [('tn', int), ('fp', int), ('fn', int), ('tp', int)]
)


@njit
def initialize_sample_weights(labels: np.ndarray):
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    return np.where(labels, 1 / n_pos, 1 / n_neg)


@njit
def s(labels):
    return np.where(labels, 1, -1)


@njit
def compute_potentials(weights: np.ndarray, labels: np.ndarray):
    pos_loc = np.where(labels == 1, True, False)
    sum_weights_pos = weights[pos_loc].sum()
    sum_weights_neg = weights[~pos_loc].sum()
    third_term = np.where(labels == 1, sum_weights_neg, sum_weights_pos)
    return s(labels) * weights * third_term


@njit
def best_weak_ranking(features: np.ndarray, potentials: np.ndarray):
    theta = None
    feature_idx = None
    max_val = 0
    for k, feature in enumerate(features):
        indx = np.argsort(feature)
        pot_sums = np.flip(np.cumsum(np.flip(potentials[indx])))
        theta_idx = np.argmax(np.abs(pot_sums))
        val = np.abs(pot_sums[theta_idx])
        if val > max_val:
            max_val = val
            # needed: f(x) > theta not >=
            theta = 0 if (theta_idx - 1 < 0) else feature[theta_idx - 1]
            feature_idx = k
            r = pot_sums[theta_idx]
    return feature_idx, theta, r


@njit
def get_alpha(r: float):
    return .5 * np.ln((1 + r)/(1 - r))


@njit
def update_weights(
    predictions: np.ndarray, labels: np.ndarray, weights: np.ndarray, alpha: np.ndarray
):
    numerator = weights * np.exp(-s(labels) * alpha * predictions)
    pos_loc = np.where(labels, True, False)
    sum_pos_weights = np.sum(numerator[pos_loc])
    sum_neg_weights = np.sum(numerator[~pos_loc])
    return np.where(pos_loc, numerator/sum_pos_weights, numerator/sum_neg_weights)


@njit
def weak_classifier(x: np.ndarray, f: fm.Feature, theta: float):
    return np.where(f(x) > theta, 1, 0)


@njit
def run_weak_classifier(x: np.ndarray, c: WeakClassifier):
    return weak_classifier(x=x, f=c.classifier, theta=c.threshold)


@njit
def run_strong_classifier(
    x: np.ndarray, weak_classifiers: tp.List[WeakClassifier],
    compute_features: bool = False, theta=None
):
    predictions = np.empty(x.shape[0])
    for c in weak_classifiers:
        if compute_features:
            predictions += c.alpha * c.classifier(x)
        else:
            # TODO: try majority voting
            predictions += c.alpha * x[:, c.feature_indx] # > c.theta
    if theta is not None:
        return np.where(predictions > theta, 1, 0)
    return predictions


@njit
def get_optimal_theta_val(predictions: np.ndarray, val_labels: np.ndarray):
    get_d_f()
    return theta_val, D_curr, F_curr


@njit
def remove_easy_negatives(
    features: np.ndarray, labels: np.ndarray, strong_classifier: StrongClassifier
):
    prediction = run_strong_classifier(
        features, strong_classifier.weak_classifiers,
        compute_features=False, theta=strong_classifier.threshold
    )
    idxs_to_delete = np.where((labels == 0) & (prediction == labels))
    features = np.delete(features, idxs_to_delete, axis=0)
    labels = np.delete(labels, idxs_to_delete, axis=0)
    return features, labels


@njit
def prediction_stats(y_true: np.ndarray, y_pred: np.ndarray):
    c = confusion_matrix(y_true, y_pred)
    return c, PredictionStats(*(c.ravel()))


@njit
def false_positive_rate(pred_stats: PredictionStats):
    return pred_stats.fp / (pred_stats.fp + pred_stats.tn)


@njit
def detection_rate(pred_stats: PredictionStats):
    return pred_stats.tp / (pred_stats.tp + pred_stats.fn)


def main():
    d, f = 0.99, 0.3
    D, F = 1., 1.
    D_target, F_target = 0.99**5, 0.3**5

    feature_modules = fe.feature_instantiator(14, 'all')
    
    train_labels = 'ALGO'
    val_labels = 'ALGO'
    val_features = 'ALGO'
    train_features = 'ALGO'
    pool = []
    F_prev = F

    node_classifiers = []
    while F_curr > F_target and len(pool) != 0:
        F_curr = F_prev
        weights = initialize_sample_weights(train_labels)
        weak_collection = []
        while F_curr > f*F_prev:

            ####### TRAIN DATA #########
            potentials = compute_potentials(weights, train_labels)
            feature_idx, theta, r = best_weak_ranking(train_features, potentials)
            alpha = get_alpha(r)
            predictions = np.where(train_features[:, feature_idx] > theta, 1, 0)
            weights = update_weights(predictions, train_labels, weights, alpha)

            # Remove feature for next feature selection
            train_features = np.delete(train_features, feature_idx, axis=1)

            # Store the weak classifier
            weak_collection.append(
                WeakClassifier(
                    threshold=theta, alpha=alpha, feature_indx=feature_idx,
                    classifier=feature_modules[feature_idx]
                )
            )

            ####### VAL DATA #########
            val_prediction = run_strong_classifier(
                val_features, weak_collection, compute_features=False
            )
            theta_val, D_curr, F_curr = get_optimal_theta_val(val_prediction, val_labels)
        
        # Store the strong node classifier
        strong_classifier = StrongClassifier(theta_val, weak_collection)
        node_classifiers.append(strong_classifier)
        
        # easy negatives removal
        if F_curr > F_target:
            # Load train data again
            train_features, train_labels = remove_easy_negatives(
                train_features, train_labels, strong_classifier
            )
            val_features, val_labels = remove_easy_negatives(
                val_features, val_labels, strong_classifier
            )
            pool_features, pool_labels = remove_easy_negatives(
                pool_features, pool_labels, strong_classifier
            )

# TODO: Add stats and model logging

if __name__ == '__main__':
    main()