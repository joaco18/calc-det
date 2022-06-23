from pathlib import Path
thispath = Path.cwd().resolve()
import sys; sys.path.insert(0, str(thispath.parent))

import cv2
import torch
import logging

import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
from typing import Callable, List, Optional
from functools import partial

import general_utils.utils as utils
from deep_learning.dataset.dataset import ImgCropsDataset
from deep_learning.dl_utils import get_model_from_checkpoint


best_models = {
    16: {
        'checkpoint_path':
            thispath.parent/'deep_learning/classification_models/checkpoints/16_net2_07.pt',
        'threshold': 0.36629703640937805,
        'pred_kind': 'score',
        'norm_kind': 'avg',
        'post_proc': True,
        'batch_size': 2048,
        'k_size': 9,
        'patch_size': 16,
        'stride': 8,
        'nms': True,
        'iou_threshold': 1,
        'normalization': 'z_score',
        'results_path': thispath.parent.parent / 'detections_dl/16_net2_07'
    },
    32: {
        'checkpoint_path':
            thispath.parent/'deep_learning/classification_models/checkpoints/32_net2_05.pt',
        'threshold': 0.2795366644859314,
        'pred_kind': 'score',
        'norm_kind': 'avg',
        'post_proc': True,
        'batch_size': 1024,
        'k_size': 17,
        'patch_size': 32,
        'stride': 8,
        'nms': True,
        'iou_threshold': 1,
        'normalization': 'z_score',
        'results_path': thispath.parent.parent / 'detections_dl/32_net2_05'
    },
    64: {
        'checkpoint_path':
            thispath.parent/'deep_learning/classification_models/checkpoints/64_net2_03.pt',
        'threshold': 0.31617915630340576,
        'pred_kind': 'score',
        'norm_kind': 'avg',
        'post_proc': True,
        'batch_size': 512,
        'k_size': 13,
        'patch_size': 64,
        'stride': 12,
        'nms': True,
        'iou_threshold': 1,
        'normalization': 'z_score',
        'results_path': thispath.parent.parent / 'detections_dl/64_net2_03'
    },
    224: {
        'checkpoint_path':
            thispath.parent/'deep_learning/classification_models/checkpoints/224_resnet50_05.pt',
        'threshold': 0.7274762988090515,
        'pred_kind': 'score',
        'norm_kind': 'avg',
        'post_proc': True,
        'batch_size': 124,
        'k_size': 17,
        'patch_size': 224,
        'stride': 12,
        'nms': True,
        'iou_threshold': 1,
        'normalization': 'z_score',
        'results_path': thispath.parent.parent / 'detections_dl/224_resnet50_05.pt'
    }
}


def get_detections(
    img: np.ndarray, saliency_map: np.ndarray, bbox_size: int, threshold: float
):
    """Finds peaks in saliency map, and generates the corresponding bbox
    Returns:
        detections (np.ndarray): [x1, x2, y1, y2, score]
    """
    # get local maxima and filter by threshold and filter by breast region
    breast_mask = np.where(img == 0, 0, 1)
    peak_centers = utils.peak_local_max(
        saliency_map, footprint=np.ones((bbox_size, bbox_size)),
        threshold_abs=threshold, additional_mask=breast_mask
    )
    if len(peak_centers) == 0:
        logging.warning(
            "The current configuration led to no detections in the image, returning None")
        return None

    # convert from [row, column] (y,x) to (x,y)
    peak_centers = np.fliplr(peak_centers)
    scores = saliency_map[peak_centers[:, 1], peak_centers[:, 0]]
    patch_coordinates_from_center = partial(
        utils.patch_coordinates_from_center,
        image_shape=breast_mask.shape, patch_size=bbox_size)
    patches_coords = np.array(list(map(patch_coordinates_from_center, peak_centers)))
    return np.concatenate([peak_centers, patches_coords, scores.reshape(-1, 1)], axis=1)


class ClassificationBasedDetector():
    """Obtain detections using a classification model using it in sliding window fashion"""
    def __init__(
        self,
        model: Optional[Callable[..., nn.Module]],
        threshold: float = 0,
        pred_kind: str = 'score',
        norm_kind: str = 'avg',
        post_proc: bool = True,
        k_size: int = 3,
        patch_size: int = 224,
        stride: int = 25,
        min_breast_fraction_patch: int = None,
        batch_size: int = 24,
        bbox_size: int = 14,
        device: str = 'cpu',
        nms: bool = True,
        iou_threshold: float = 1,
        in_multiscale: bool = False,
        normalization: str = 'z_score',
        **kwargs
    ):
        """
        Args:
            model (Callable[nn.Module], optional): Trianed and instantiates
                callable classification model from pytorch.
            threshold (float, optional): Threshold to binarize predictions. Defaults to 0.
            pred_kind (str, optional). Which prediction to use.
                'score': the raw score is used.
                'binary': the score is binarized with the threshold passed.
                Defaults to 'score'.
            norm_kind (str, optional).
                'avg': get the average of the predictions on the overlapping patches.
                'normalize': do a min_max_norm over the complete saliency map.
                Defaults to 'avg'.
            post_proc (bool, optional): Wether to apply post processing. Defaults to True
            patch_size (int, optional): Size of the input to the network.
            stride (int, optional): Stride to use when obtaining patches from the imge.
                Defaults to 25.
            min_breast_fraction_patch (int, optional): Minimum of breast tissue that the patch
                should have in order to be classified. Defaults to None (not used).
            batch_size (int, optional): Number of patches to predict in parallel.
                Defaults to 24.
            bbox_size (int, optional): Size of the detection enclosing. Defaults to 14.
            device (str, optional): Defaults to 'cpu'.
            nms (bool, optional): Whether to perform NMS or not. Defaults to True.
            iou_threshold (float, optional): IoU Threshold to be used in NMS. Defaults to 1.
            in_multiscale (bool, optional): Whether the class is used in multiscale case.
                Defaults to False
            normalization (str, optional): Which normalization to apply to patches.
                Defaults to z_score
        """
        self.model = model
        self.threshold = threshold
        self.post_proc = post_proc
        self.patch_size = patch_size
        self.stride = stride
        self.min_breast_fraction_patch = min_breast_fraction_patch
        self.batch_size = batch_size
        self.bbox_size = bbox_size
        self.device = device
        self.nms = nms
        self.iou_threshold = iou_threshold
        self.pred_kind = pred_kind
        self.norm_kind = norm_kind
        self.k_size = k_size
        self.in_multiscale = in_multiscale
        self.normalization = normalization

    def detect(
        self, img: np.ndarray, raw_saliency_path: Path = None,
        final_saliency_path: Path = None, store: bool = False
    ):
        """Divides the image in patches, runs classification, joints the results and extracts
        detections out of the saliency map, using the configured treshold and NMS.
        Args:
            img (np.ndarray): Full image to process
            raw_saliency_path (Path): Path to store raw saliency map
            final_saliency_path (Path): Path to store normalized postprocessed saliency map
            store (bool): Whether to store the saliency map images.
        Returns:
            detections (np.ndarray): [xc, yc, x1, x2, y1, y2, score]
        """
        self.img = img
        # parallelize inference time
        if (final_saliency_path is None) or (not final_saliency_path.exists()):
            if (raw_saliency_path is None) or (not raw_saliency_path.exists()):
                crops_dataset = ImgCropsDataset(
                    img=self.img,
                    patch_size=self.patch_size,
                    stride=self.stride,
                    min_breast_fraction_patch=self.min_breast_fraction_patch,
                    normalization=self.normalization
                )
                crops_dataloader = DataLoader(
                    crops_dataset, batch_size=self.batch_size, shuffle=False, sampler=None,
                    batch_sampler=None, num_workers=4, pin_memory=True, drop_last=False
                )
                # get saliency map
                self.saliency_map = np.zeros(self.img.shape, dtype='float32')
                counts = np.ones(self.img.shape, dtype='int')
                for batch in tqdm(crops_dataloader, total=len(crops_dataloader)):
                    inputs = batch['img'].to(self.device)
                    with torch.set_grad_enabled(False):
                        outputs = torch.sigmoid(self.model(inputs).detach())
                        outputs = np.asarray(outputs.flatten().cpu())
                        # binarize if necessary
                        if self.pred_kind == 'binary':
                            outputs = np.where(outputs >= self.threshold, 1, 0)
                    for k, location in enumerate(batch['location']):
                        [[x1, y1], [x2, y2]] = location
                        counts[y1:y2, x1:x2] += 1
                        self.saliency_map[y1:y2, x1:x2] += outputs[k]

                # scale information to consider overlapping
                if self.norm_kind == 'avg':
                    self.saliency_map = self.saliency_map/counts
                else:
                    self.saliency_map = utils.min_max_norm(
                        self.saliency_map, 1).astype('float32')

                if (raw_saliency_path is not None) and store:
                    cv2.imwrite(str(raw_saliency_path), self.saliency_map)
            else:
                self.saliency_map = cv2.imread(
                    str(raw_saliency_path), cv2.IMREAD_ANYDEPTH)

            # do post_processing
            if self.post_proc:
                sigma = self.k_size // 3
                self.saliency_map = cv2.GaussianBlur(
                    self.saliency_map, ksize=(self.k_size, self.k_size),
                    sigmaX=sigma, sigmaY=sigma)

            # normalize to [0, 1]
            self.saliency_map = utils.min_max_norm(self.saliency_map, 1).astype('float32')
            # adapt threshold if necesary
            if store:
                cv2.imwrite(str(final_saliency_path), self.saliency_map)
        else:
            self.saliency_map = cv2.imread(str(final_saliency_path), cv2.IMREAD_ANYDEPTH)

        if not self.in_multiscale:
            # extract detections bboxes
            detections = get_detections(
                self.img, self.saliency_map, self.bbox_size, self.threshold)
            # nms
            if self.nms:
                detections = utils.non_max_supression(detections, self.iou_threshold)
            return detections


class MultiScaleClassificationBasedDetector():
    """Obtain detections using several classification models using them in sliding window fashion"""
    def __init__(
        self,
        scales: Optional[List[int]] = [16],
        bbox_size: int = 14,
        threshold: float = 0,
        nms: bool = True,
        iou_threshold: float = 1,
        merge_type: str = 'mean'
    ):
        """
        Args:
            scales (List[int], optional): Scales of pretrained models to use.
                Defaults to [16]
            bbox_size (int, optional): Size of the detection enclosing. Defaults to 14.
            threshold (float, optional): Threshold to binarize predictions. Defaults to 0.
            nms (bool, optional): Whether to do nms or not. Defaults to True
            iou_threshold (float, optional): IoU Threshold to be used in NMS. Defaults to 1.
            merge_type (str, optional): Whether to combine the saliency maps with 'max' or
                'mean' operation. Defaults to 'mean'.
        """
        self.bbox_size = bbox_size
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.nms = nms
        self.merge_type = merge_type

        self.detectors = {}
        self.scales = scales
        for scale in self.scales:
            model_params = best_models[scale]
            model_ckpt = torch.load(model_params['checkpoint_path'])
            model = get_model_from_checkpoint(model_ckpt)
            self.detectors[scale] = \
                ClassificationBasedDetector(model, in_multiscale=True, **model_params)

    def detect(
        self, img: np.ndarray, image_id: str, final_saliency_path: Path = None, store: bool = False
    ):
        self.img = img
        if (final_saliency_path is None) or (not final_saliency_path.exists()):
            saliency_map = np.zeros((len(self.scales), self.img.shape[0], self.img.shape[1]))
            for k, scale in enumerate(self.scales):
                results_path_single = best_models[scale]['results_path']
                results_path_img_single = Path(results_path_single) / f'{image_id}'
                raw_saliency_path_single = \
                    Path(results_path_img_single) / f'{image_id}_raw_sm.tiff'
                final_saliency_path_single = \
                    Path(results_path_img_single) / f'{image_id}_final_sm.tiff'
                self.detectors[scale].detect(
                    self.img, raw_saliency_path_single, final_saliency_path_single, store=False)
                saliency_map[k, :, :] = self.detectors[scale].saliency_map
            if self.merge_type == 'mean':
                self.saliency_map = np.mean(saliency_map, axis=0)
            elif self.merge_type == 'max':
                self.saliency_map = np.max(saliency_map, axis=0)

            if store:
                cv2.imwrite(str(final_saliency_path), self.saliency_map)
        else:
            self.saliency_map = cv2.imread(str(final_saliency_path), cv2.IMREAD_ANYDEPTH)

        detections = get_detections(
            self.img, self.saliency_map, self.bbox_size, self.threshold)
        # nms
        if self.nms:
            detections = utils.non_max_supression(detections, self.iou_threshold)
        return detections
