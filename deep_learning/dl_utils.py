import sys
from deep_learning.classification_models.models.resnet_based_classifier import ResNetBased
from deep_learning.classification_models.models.base_classifier import CNNClasssifier
from transformers import SwinForImageClassification
from sklearn.metrics import (
    roc_curve, roc_auc_score, average_precision_score, f1_score, confusion_matrix
)
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
import torchvision
import torch.nn as nn
import numpy as np
import torch
from pathlib import Path
thispath = Path.cwd().resolve()
sys.path.insert(0, str(thispath.parent))


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
    writer.add_scalar(f"Sensitivity/{phase}{it}",
                      metrics['sensitivity'], epoch)
    writer.add_scalar(f"Specificity/{phase}{it}",
                      metrics['specificity'], epoch)
    writer.add_scalar(f"Precision/{phase}{it}", metrics['precision'], epoch)


def get_model_from_checkpoint(model_ckpt: dict, freezed: bool = True):
    """Uses the config file inside the checkpoint to create the model acordingly and
    loads the state dict"""
    cfg = model_ckpt['configuration']
    if cfg['model']['backbone'] == 'swin_transformer':
        model = SwinForImageClassification.from_pretrained(
            'microsoft/swin-tiny-patch4-window7-224',
            num_labels=1,
            ignore_mismatched_sizes=True)
    elif cfg['model']['backbone'] == 'net2':
        use_middle_act = \
            cfg['model']['use_middle_activation'] if (
                'use_middle_activation' in cfg['model'].keys()) else True
        if 'bloc_act' in cfg['model'].keys() and (cfg['model']['bloc_act'] is not None):
            block_act = getattr(nn, cfg['model']['bloc_act'])
        else:
            block_act = None

        model = ResNetBased(
            block=cfg['model']['block'],
            replace_stride_with_dilation=cfg['model']['replace_stride_with_dilation'],
            inplanes=cfg['model']['inplanes'],
            act_fn=getattr(nn, cfg['model']['activation']),
            downsample_blocks=cfg['model']['n_downsamples'],
            fc_dims=cfg['model']['fc_dims'],
            dropout=cfg['model']['dropout'],
            use_middle_act=use_middle_act,
            block_act=block_act
        )
    else:
        model = CNNClasssifier(
            activation=getattr(nn, cfg['model']['activation'])(),
            dropout=cfg['model']['dropout'],
            fc_dims=cfg['model']['fc_dims'],
            freeze_weights=freezed,
            backbone=cfg['model']['backbone'],
            pretrained=cfg['model']['pretrained'],
        )
        model = model.model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(model_ckpt['model_state_dict'])
    if freezed:
        for param in model.parameters():
            param.requires_grad = False
    return model


def get_detection_model_from_checkpoint(model_ckpt: dict, freezed: bool = True):
    """Uses the config file inside the checkpoint to create the model acordingly and
    loads the state dict"""
    cfg = model_ckpt['configuration']

    if cfg['model']['checkpoint_path'] is not None:
        backbone_ckpt = torch.load(cfg['model']['checkpoint_path'])
        model = get_model_from_checkpoint(backbone_ckpt, freezed)
    else:
        model = CNNClasssifier(
            activation=getattr(nn, cfg['model']['activation'])(),
            dropout=cfg['model']['dropout'],
            fc_dims=cfg['model']['fc_dims'],
            freeze_weights=freezed,
            backbone=cfg['model']['backbone'],
            pretrained=cfg['model']['pretrained'],
        )
        model = model.model

    # delete the last fc layers depending on the backbone
    if 'densenet' in cfg['model']['backbone']:
        modules = list(model.children())[:-1]
    else:
        modules = list(model.children())[:-2]

    last_submodule_childs = list(modules[-1][-1].children())
    for i in range(len(last_submodule_childs)-1, -1, -1):
        if hasattr(last_submodule_childs[i], 'out_channels'):
            out_channels = last_submodule_childs[i].out_channels
            break

    # densenet blocks don't have out_channels attribute
    if 'densenet121' in cfg['model']['backbone']:
        out_channels = 1024

    model_backbone = nn.Sequential(*modules)
    model_backbone.out_channels = out_channels
    anchor_generator = AnchorGenerator(
        sizes=(cfg['model']['anchor_sizes'],),
        aspect_ratios=(cfg['model']['anchor_ratios'],))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], output_size=7, sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(
        model_backbone,
        num_classes=cfg['model']['num_classes'],
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        image_mean=[0., 0., 0.],
        image_std=[1., 1., 1.],
        box_score_thresh=1e-4
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(model_ckpt['model_state_dict'])
    if freezed:
        for param in model.parameters():
            param.requires_grad = False

    return model
