from pathlib import Path
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
        crop_size: int = 32,
        center_noise: int = 10,
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
        self.extract_patches_method = extract_patches_method
        self.patch_size = patch_size
        self.crop_size = crop_size
        self.half_crop = int(self.crop_size // 2)
        self.center_noise = center_noise

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

        if self.for_detection_net or (self.extract_patches_method != 'all'):
            mask_filename = self.df['mask_filename'].iloc[idx]
            if mask_filename != 'empty_mask':
                mask_filename = self.patch_mask_path / mask_filename
                mask = cv2.imread(str(mask_filename), cv2.IMREAD_ANYDEPTH)
                sample['lesion_bboxes'] = np.asarray(utils.get_bbox_of_lesions_in_patch(mask))
                sample['lesion_centers'] = np.asarray(
                    [utils.get_center_bbox(bbox[0], bbox[1]) for bbox in sample['lesion_bboxes']])
                sample['labels'] = np.ones((len(sample['lesion_bboxes']), 1))
            else:
                sample['lesion_bboxes'] = []
                sample['lesion_centers'] = []
                sample['labels'] = np.zeros((len(sample['lesion_bboxes']), 1))

            if self.extract_patches_method == 'all':
                return sample['label'], sample['img'], sample['lesion_bboxes'], \
                    sample['lesion_centers'], sample['labels']
            else:
                patch_center = self.patch_size // 2
                if len(sample['lesion_centers']) != 0:
                    distances = np.asarray([
                        abs(center[0] - patch_center) + abs(center[1] - patch_center)
                        for center in sample['lesion_centers']])
                    sample['lesion_center'] = \
                        np.asarray(sample['lesion_centers'][np.argmin(distances)], dtype=int)
                    if sample['label'] == 1:
                        center = sample['lesion_center']
                        offset = self.center_noise // 2
                        center_noise_x = np.random.randint(-offset, offset, dtype=int)
                        center_noise_y = np.random.randint(-offset, offset, dtype=int)
                        center[0], center[1] = center[0]+center_noise_x, center[1]+center_noise_y
                        sample['img'] = sample['img'][
                            :,
                            center[1] - self.half_crop: center[1] + self.crop_size - self.half_crop,
                            center[0] - self.half_crop: center[0] + self.crop_size - self.half_crop]
                else:
                    sample['lesion_center'] = np.asanyarray([-1, -1])
                    center_x = np.random.randint(
                        self.half_crop+1, high=self.patch_size-self.half_crop+1, dtype=int)
                    center_y = np.random.randint(
                        self.half_crop+1, high=self.patch_size-self.half_crop+1, dtype=int)
                    sample['img'] = sample['img'][
                        :,
                        center_y - self.half_crop: center_y + self.crop_size - self.half_crop,
                        center_x - self.half_crop: center_x + self.crop_size - self.half_crop]
                del sample['lesion_bboxes'], sample['lesion_centers'], sample['labels']
        return sample


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
