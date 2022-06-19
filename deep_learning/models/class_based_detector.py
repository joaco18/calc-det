from pathlib import Path
thispath = Path.cwd().resolve()
import sys; sys.path.insert(0, str(thispath.parent))

import cv2
import torch
import torchvision
import logging

import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from functools import partial

import general_utils.utils as utils
from deep_learning.dataset.dataset import ImgCropsDataset


class ClassificationBasedDetector():
    """Obtain detections using a classification model using it in patches"""
    def __init__(
        self,
        model,
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
        iou_threshold: float = 0
    ):
        """
        Args:
            model (_type_): Trained classification model from pytorch.
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
            patch_size (int, optional): Size of the input to the network. Defaults to 224.
            stride (int, optional): Stride to use when obtaining patches from the imge.
                Defaults to 25.
            min_breast_fraction_patch (int, optional): Minimum of breast tissue that the patch
                should have in order to be classified. Defaults to None (not used).
            batch_size (int, optional): Number of patches to predict in parallel.
                Defaults to 24.
            bbox_size (int, optional): Size of the detection enclosing. Defaults to 14.
            device (str, optional): Defaults to 'cpu'.
            nms (bool, optional): Whether to perform NMS or not. Defaults to True.
            iou_threshold (float, optional): IoU Threshold to be used in NMS. Defaults to 0.
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

    def detect(self, img: np.ndarray):
        """Divides the image in patches, runs classification, joints the results and extracts
        detections out of the saliency map, using the configured treshold and NMS.
        Args:
            img (np.ndarray): Full image to process
        Returns:
            detections (np.ndarray): [x1, x2, y1, y2, score]
        """
        # parallelize inference time
        self.img = img
        crops_dataset = ImgCropsDataset(
            img=self.img,
            patch_size=self.patch_size,
            stride=self.stride,
            min_breast_fraction_patch=self.min_breast_fraction_patch
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

        # normalize saliency map
        if self.norm_kind == 'avg':
            self.saliency_map = self.saliency_map/counts
        else:
            self.saliency_map = utils.min_max_norm(self.saliency_map, 1).astype('float32')

        # do post_processing
        sigma = self.k_size // 3
        if self.post_proc:
            self.saliency_map_adj = utils.adjust_gamma_float(self.saliency_map, 0.5)
            self.saliency_map = cv2.GaussianBlur(
                self.saliency_map, ksize=(self.k_size, self.k_size), sigmaX=sigma, sigmaY=sigma)

        # extract detections bboxes
        detections = self.get_detections()
        # nms
        if self.nms:
            detections = self.non_max_supression(detections, self.iou_threshold)
        return detections

    def get_detections(self):
        """Finds peaks in saliency map, and generates the corresponding bbox
        Returns:
            detections (np.ndarray): [x1, x2, y1, y2, score]
        """
        # get local maxima and filter by threshold and filter by breast region
        breast_mask = np.where(self.img == 0, 0, 1)
        peak_centers = utils.peak_local_max(
            self.saliency_map, footprint=np.ones((self.bbox_size, self.bbox_size)),
            threshold_abs=self.threshold, additional_mask=breast_mask
        )
        if len(peak_centers) == 0:
            logging.warning(
                "The current configuration led to no detections in the image, returning None")
            return None
        # convert from [row, column] (y,x) to (x,y)
        peak_centers = np.fliplr(peak_centers)
        scores = self.saliency_map[peak_centers[:, 1], peak_centers[:, 0]]
        patch_coordinates_from_center = partial(
            utils.patch_coordinates_from_center,
            image_shape=breast_mask.shape, patch_size=self.bbox_size)
        patches_coords = np.array(list(map(patch_coordinates_from_center, peak_centers)))
        return np.concatenate([peak_centers, patches_coords, scores.reshape(-1, 1)], axis=1)

    @staticmethod
    def non_max_supression(detections: np.ndarray, iou_threshold: float = 0.):
        """Filters the detections bboxes using NMS.
        Args:
            detections (np.ndarray): [xc, yc, x1, x2, y1, y2, score]
            iou_threshold (float): iou threshold value.
        Returns:
            detections (np.ndarray): [xc, yc, x1, x2, y1, y2, score]
        """
        bboxes = np.asarray(
            [detections[:, 2], detections[:, 4], detections[:, 3], detections[:, 5]]).T

        bboxes = torch.from_numpy(bboxes).to(torch.float)
        scores = torch.from_numpy(detections[:, 6]).to(torch.float)
        indxs = torchvision.ops.nms(bboxes, scores, iou_threshold=iou_threshold)
        return detections[indxs, :]
