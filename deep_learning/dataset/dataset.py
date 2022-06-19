from pathlib import Path
from re import X
thispath = Path.cwd().resolve()
import sys; sys.path.insert(0, str(thispath.parent))

import cv2
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple

from database.dataset import INBreast_Dataset
import general_utils.utils as utils
from database.roi_extraction import slice_image, padd_image, view_as_windows

datapath = thispath.parent / "data" / "INbreast Release 1.0"


class INBreast_Dataset_pytorch(INBreast_Dataset):
    def __init__(
        self, imgpath: Path = datapath/'AllPNGs',
        mask_path: Path = datapath/'AllMasks',
        patch_images_path: Path = None,
        dfpath: Path = datapath,
        lesion_types: List[str] = ['calcification', 'cluster'],
        seed: int = 0,
        partitions: List[str] = ['train', 'validation', 'test'],
        extract_patches: bool = True,
        delete_previous: bool = True,
        extract_patches_method: str = 'all',
        patch_size: int = 224,
        stride: int = 100,
        min_breast_fraction_roi: float = 0.7,
        n_jobs: int = -1,
        cropped_imgs: bool = True,
        ignore_diameter_px: int = 15,
        neg_to_pos_ratio: int = None,
        balancing_seed: int = 0,
        normalization: str = 'z_score',
        get_lesion_bboxes: bool = False,
        for_detection_net: bool = False,
        **extra
    ):
        super(INBreast_Dataset_pytorch, self).__init__(
            imgpath=imgpath, mask_path=mask_path, dfpath=dfpath, lesion_types=lesion_types,
            partitions=partitions, delete_previous=delete_previous, extract_patches=extract_patches,
            extract_patches_method=extract_patches_method, patch_size=patch_size, stride=stride,
            min_breast_fraction_roi=min_breast_fraction_roi, n_jobs=n_jobs, level='rois',
            cropped_imgs=cropped_imgs, ignore_diameter_px=ignore_diameter_px, seed=seed,
            return_lesions_mask=get_lesion_bboxes, max_lesion_diam_mm=None, use_muscle_mask=False
        )
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.balancing_seed = balancing_seed
        self.total_df = self.df.copy()
        self.normalization = normalization
        self.for_detection_net = for_detection_net

        if patch_images_path is not None:
            self.patch_img_path = patch_images_path/'patches'
            self.patch_mask_path = patch_images_path/'patches_masks'

        if self.for_detection_net:
            self.discard_negative_patches()

    def balance_dataset(self, balancing_seed: int = None):
        n_pos = self.total_df.loc[self.total_df.label == 'abnormal', :].shape[0]
        n_neg = len(self.total_df) - n_pos
        n_to_sample = n_pos * self.neg_to_pos_ratio
        if n_to_sample > n_neg:
            n_to_sample = n_neg
        if balancing_seed is None:
            balancing_seed = self.balancing_seed
        self.df = pd.concat([
            self.total_df.loc[self.total_df.label == 'abnormal', :],
            self.total_df.loc[self.total_df.label == 'normal', :].sample(
                n=n_to_sample, replace=False, random_state=balancing_seed
            )
        ], ignore_index=True)

    def update_sample_used(self, balancing_seed: int = None):
        if self.neg_to_pos_ratio is not None:
            self.balance_dataset(balancing_seed)

    def discard_negative_patches(self):
        self.df = self.total_df.loc[self.total_df.label == 'abnormal', :].copy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = {}
        sample["label"] = 1 if self.df['label'].iloc[idx] == 'abnormal' else 0

        img_path = self.patch_img_path / self.df['filename'].iloc[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_ANYDEPTH)
        # Convert all images in left oriented ones
        side = self.df['side'].iloc[idx]
        if side == 'R' and self.level == 'image':
            img = cv2.flip(img, 1)

        # Convert into float for better working of pytorch augmentations
        if self.normalization == 'min_max':
            img = utils.min_max_norm(img, 1).astype('float32')
        elif self.normalization == 'z_score':
            img = utils.z_score_norm(img, non_zero_region=True)

        # to RGB
        img = np.expand_dims(img, 0)
        img = np.repeat(img, 3, axis=0)
        sample['img'] = img

        if not self.for_detection_net:
            return sample
        else:
            mask_filename = self.df['mask_filename'].iloc[idx]
            if mask_filename != 'empty_mask':
                mask_filename = self.patch_mask_path / mask_filename
                mask = cv2.imread(str(mask_filename), cv2.IMREAD_ANYDEPTH)
                sample['boxes'] = np.asarray(utils.get_bbox_of_lesions_in_patch(mask))
                # sample['ignored_lesion_bboxes'] = np.asarray(utils.get_bbox_of_lesions_in_patch(
                #     mask, ignored_lesions=True), dtype=object)
                sample['lesion_centers'] = \
                    [utils.get_center_bbox(bbox[0], bbox[1]) for bbox in sample['boxes']]
                # sample['ignored_lesion_centers'] = [
                #     utils.get_center_bbox(bbox[0], bbox[1])
                #     for bbox in sample['ignored_lesion_bboxes']]
                
                sample['boxes'] = [[*tl, *br] for tl, br in  sample['boxes']]
                sample['boxes'] = [self.correct_boxes(b, mask.shape) for b in sample['boxes']]
                sample['boxes'] = torch.as_tensor(sample['boxes'], dtype=torch.float32)
                sample['labels'] = torch.ones((len(sample['boxes']),), dtype=torch.int64)
                
                target = {'boxes':sample['boxes'], 'labels':sample['labels']}
            else:
                sample['boxes'] = []
                # sample['boxes'] = []
                sample['lesion_centers'] = []
                # sample['ignored_lesion_centers'] = []
                sample['labels'] = np.zeros((len(sample['boxes']), 1))
            
            return torch.as_tensor(sample['img']), target
    @staticmethod
    def correct_boxes(box, image_shape, min_box_size=14):
        def correct_boxes_axis(x1, x2, image_shape_ax, min_box_size=min_box_size):
            exp_x = min_box_size - (x2 - x1)
            
            if x1 >= exp_x//2:
                xleft_exp = 0
                x1 = x1 - exp_x//2
            elif x1 < exp_x//2:
                xleft_exp = exp_x//2 - x1
                x1 = 0
            
            x2 = x2 + xleft_exp
            if x2 + exp_x//2 <= image_shape_ax:
                x2 = x2 +  exp_x//2
            elif x2 +  exp_x//2 > image_shape_ax:
                x1 = x1 - (image_shape_ax - x2 +  exp_x//2)
                x2 = image_shape_ax
            return x1, x2
        corr_x1, corr_x2 = correct_boxes_axis(box[0], box[2], image_shape[0])
        corr_y1, corr_y2 = correct_boxes_axis(box[1], box[3], image_shape[1])
        return [corr_x1, corr_y1, corr_x2, corr_y2]
class ImgCropsDataset():
    """Dataset of patches obtained from a single image"""
    def __init__(
        self,
        img: np.ndarray,
        patch_size: int = 224,
        stride: int = 100,
        min_breast_fraction_patch: float = None
    ):
        """
        Args:
            img (np.ndarray): Image to process
            patch_size (int, optional): Defaults to 224.
            stride (int, optional): Defaults to 100.
            min_breast_fraction_patch (float, optional): Minimum of breast tissue that the patch
                should have in order to be classified. Defaults to None.
        """
        # instatiate atributes
        self.patch_size = patch_size
        self.stride = stride
        self.min_breast_frac = min_breast_fraction_patch

        # extract patches equally from image and the mask
        img = padd_image(img, self.patch_size)
        self.image_patches = slice_image(img, window_size=self.patch_size, stride=self.stride)

        # calculate patches coordinates
        bbox_coordinates = []
        row_num, col_num, _, __ = view_as_windows(img, self.patch_size, self.stride).shape
        for col in range(row_num):
            row_idx = [((row * self.stride, col * self.stride),
                        (self.patch_size + row * self.stride,
                        self.patch_size + col * self.stride)) for row in range(col_num)]
            bbox_coordinates.extend(row_idx)
        self.bbox_coordinates = np.array(bbox_coordinates)

        if self.min_breast_frac is not None:
            breast_pixels = np.array([(roi != 0).sum() for roi in self.image_patches])
            breast_fraction = breast_pixels / (self.patch_size*self.patch_size)
            self.breast_fraction_selection = np.where(
                breast_fraction >= self.min_breast_frac, True, False)
            self.image_patches = self.image_patches[self.breast_fraction_selection, :, :]
            self.bbox_coordinates = self.bbox_coordinates[self.breast_fraction_selection, :, :]

    def __len__(self):
        return self.image_patches.shape[0]

    def __getitem__(self, idx):
        img = self.image_patches[idx, :, :]
        if img.any():
            img = utils.min_max_norm(img, 1).astype('float32')
        else:
            img = img.astype('float32')

        # to RGB
        img = np.expand_dims(img, 0)
        img = np.repeat(img, 3, axis=0)
        return {
            'img': img,
            'location': self.bbox_coordinates[idx, :, :],
        }
