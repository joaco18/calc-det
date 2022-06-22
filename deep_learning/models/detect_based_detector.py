from pathlib import Path

thispath = Path.cwd().resolve()
import sys; sys.path.insert(0, str(thispath.parent))

import torch

import deep_learning.dl_utils as dl_utils
import numpy as np

from torch.utils.data import DataLoader
from deep_learning.dataset.dataset import ImgCropsDataset
import general_utils.utils as utils

class DetectionBasedDetector():
    """Obtain detections using a classification model using it in patches"""
    def __init__(
        self,
        model_chkp,
        patch_size: int = 224,
        stride: int = 200,
        min_breast_fraction_patch: int = None,
        batch_size: int = 24,
        bbox_size: int = 14,
        device: str = 'cpu',
        iou_threshold: float = 0.5,
        normalization: str = 'z_score'
    ):
        """
        Args:
            model_chkp (dict): Trained FasterRCNN model checkpoint dict.
                (loaded for the best model with the name  in format *{best_metric_name}.pt)
            patch_size (int, optional): Size of the input to the network. Defaults to 224.
            stride (int, optional): Stride to use when obtaining patches from the imge.
                Defaults to 25.
            min_breast_fraction_patch (int, optional): Minimum of breast tissue that the patch
                should have in order to be classified. Defaults to None (not used).
            batch_size (int, optional): Number of patches to predict in parallel.
                Defaults to 24.
            bbox_size (int, optional): Size of the detection enclosing. Defaults to 14.
            device (str, optional): Defaults to 'cpu'.
            iou_threshold (float, optional): IoU Threshold to be used in NMS. Defaults to 0.5
            normalization (str, optional): Which normalization to apply to patches.
                Defaults to z_score
        """
        self.model_chkp = model_chkp
        self.model = dl_utils.get_detection_model_from_checkpoint(model_chkp, True).eval()
        self.patch_size = patch_size
        self.stride = stride
        self.min_breast_fraction_patch = min_breast_fraction_patch
        self.batch_size = batch_size
        self.bbox_size = bbox_size
        self.device = device
        self.iou_threshold = iou_threshold
        self.normalization = normalization

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
            min_breast_fraction_patch=self.min_breast_fraction_patch,
            normalization=self.normalization
        )
        crops_dataloader = DataLoader(
            crops_dataset, batch_size=self.batch_size, shuffle=False, sampler=None,
            batch_sampler=None, num_workers=4, pin_memory=True, drop_last=False
        )

        predictions = []
        img_crops_locs = []
        for batch in crops_dataloader:
            inputs = batch['img'].to(self.device)
            img_crops_locs.extend(list(batch['location'].numpy()))
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                predictions.extend(outputs)

        # go from patch coordinate frame to full image coordinate frame and
        # match a score to every bbox
        detections = np.concatenate(
            [self.rescale_prediced_bboxes(x, y) for x, y in zip(predictions, img_crops_locs)
                if self.rescale_prediced_bboxes(x, y)])

        # include the centers in the predictions to later compute metrics
        centers = utils.get_center_bboxes([((x[0], x[2]), (x[1], x[3])) for x in detections])
        detections = np.concatenate([centers, detections], axis=1)

        # perform NMS to avoid duplicated detections over the overlapped regions
        detections = dl_utils.non_max_supression(detections, 0.5)

        return detections

    @staticmethod
    def rescale_prediced_bboxes(bbox_pred, patch_coords):
        new_boxes_wradius = [[
            int(x[0] + patch_coords[0][0]), int(x[2] + patch_coords[0][0]),
            int(x[1] + patch_coords[0][1]), int(x[3] + patch_coords[0][1]),
            bbox_pred['scores'][xidx].cpu()
        ] for xidx, x in enumerate(bbox_pred['boxes']) if len(bbox_pred['boxes'])]
        return new_boxes_wradius
