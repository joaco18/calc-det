from pathlib import Path
thispath = Path.cwd().resolve()
import sys; sys.path.insert(0, str(thispath.parent))

import cv2
import torch
import torchvision

import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from functools import partial

import general_utils.utils as utils
from deep_learning.dataset.dataset import ImgCropsDataset


class DetectionBasedDetector():
    """Obtain detections using a classification model using it in patches"""
    def __init__(
        self,
        model,
        patch_size: int = 224,
        stride: int = 200,
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
        self.patch_size = patch_size
        self.stride = stride
        self.min_breast_fraction_patch = min_breast_fraction_patch
        self.batch_size = batch_size
        self.bbox_size = bbox_size
        self.device = device
        self.nms = nms
        self.iou_threshold = iou_threshold


    def detect(self, img: np.ndarray):
        """Divides the image in patches, runs detection and applied over it NMS.
        Args:
            img (np.ndarray): Full image to process
        Returns:
            detections (np.ndarray): [x1, x2, y1, y2, score]
        """
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
        
        predictions = []
        for batch in crops_dataloader:
            inputs = batch['img'].to(self.device)
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                # outputs = np.asarray(outputs.flatten().cpu())
                predictions.extend(outputs)

        img_crops_locs = [x['location'] for x in crops_dataset]
        
        # go from patch coordinate frame to full image coordinate frame and 
        # match a score to every bbox
        detections = np.concatenate([self.rescale_prediced_bboxes(x,y) for x,y in zip(predictions, img_crops_locs) if self.rescale_prediced_bboxes(x,y)])

        # nms
        if self.nms:
            detections = self.non_max_supression(detections)
        return detections

    @staticmethod
    def rescale_prediced_bboxes(bbox_pred, patch_coords):
        new_boxes_wradius = [[int(x[0] + patch_coords[0][0]), int(x[2] + patch_coords[0][0]), int(x[1] + patch_coords[0][1]), int(x[3] + patch_coords[0][1]), bbox_pred['scores'][xidx].cpu()] for xidx, x in enumerate(bbox_pred['boxes']) if len(bbox_pred['boxes'])]
        return new_boxes_wradius
    
    @staticmethod
    def non_max_supression(detections: np.ndarray):
        """Filters the detections bboxes using NMS.
        Args:
            detections (np.ndarray): [x1, x2, y1, y2, score]
        Returns:
            detections (np.ndarray): [x1, x2, y1, y2, score]
        """
        bboxes = np.asarray(
            [detections[:, 0], detections[:, 2], detections[:, 1], detections[:, 3]]).T

        bboxes = torch.from_numpy(bboxes).to(torch.float)
        scores = torch.from_numpy(detections[:, 4]).to(torch.float)
        indxs = torchvision.ops.nms(bboxes, scores, iou_threshold=0)
        return detections[indxs, :]