import numpy as np
from sklearn.metrics import (
    roc_curve, roc_auc_score, average_precision_score, f1_score, confusion_matrix
)


def sensivity_specifity_cutoff(y_true: np.ndarray, y_score: np.ndarray):
    '''Finds data-driven cut-off for classification
    Cut-off is determied using Youden's index defined as sensitivity + specificity - 1.
    Args:
      y_true (np.ndarray): True binary labels.
      y_score (np.ndarray): Target scores.
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]


def get_metrics(labels: np.ndarray, preds: np.ndarray):
    """From the predictions and labels arrays get a full stack of metrics
    Args:
        labels (np.ndarray)
        preds (np.ndarray)
    Returns:
        (dict) with values for:
            auroc, f1_score, accuracy, precision,
            sensitivity, specificity, threshold
    """
    th = sensivity_specifity_cutoff(labels, preds)
    bin_preds = np.where(preds > th, True, False)
    tn, fp, fn, tp = confusion_matrix(labels, bin_preds).ravel()
    return {'auroc': roc_auc_score(labels, preds),
            'avgpr': average_precision_score(labels, preds),
            'f1_score': f1_score(labels, bin_preds),
            'accuracy': (tp+tn)/(tp+tn+fp+fn),
            'precision': tp/(tp+fp),
            'sensitivity': tp/(tp+fn),
            'specificity': tn/(tn+fp),
            'threshold': th}


def tensorboard_logs(writer, epoch_loss, epoch, metrics, phase, it=False):
    """Logs a set of metrics and usefull values in a tensorboard session"""
    it = '_its' if it else ''
    writer.add_scalar(f"Loss/{phase}{it}", epoch_loss, epoch)
    writer.add_scalar(f"Accuracy/{phase}{it}", metrics['accuracy'], epoch)
    writer.add_scalar(f"F1_score/{phase}{it}", metrics['f1_score'], epoch)
    writer.add_scalar(f"Auroc/{phase}{it}", metrics['auroc'], epoch)
    writer.add_scalar(f"AvgPR/{phase}{it}", metrics['avgpr'], epoch)
    writer.add_scalar(f"Sensitivity/{phase}{it}", metrics['sensitivity'], epoch)
    writer.add_scalar(f"Specificity/{phase}{it}", metrics['specificity'], epoch)
    writer.add_scalar(f"Precision/{phase}{it}", metrics['precision'], epoch)
