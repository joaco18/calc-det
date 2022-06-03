from pathlib import Path
thispath = Path.cwd().resolve()
import sys; sys.path.insert(0, str(thispath.parent))

from database.dataset import INBreast_Dataset
import general_utils.utils as utils

import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple

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
        stride: Tuple[int] = 100,
        min_breast_fraction_roi: float = 0.7,
        n_jobs: int = -1,
        cropped_imgs: bool = True,
        ignore_diameter_px: int = 15,
        neg_to_pos_ratio: int = None,
        balancing_seed: int = 0
    ):
        super(INBreast_Dataset_pytorch, self).__init__(
            imgpath=imgpath, mask_path=mask_path, dfpath=dfpath, lesion_types=lesion_types,
            partitions=partitions, delete_previous=delete_previous, extract_patches=extract_patches,
            extract_patches_method=extract_patches_method, patch_size=patch_size, stride=stride,
            min_breast_fraction_roi=min_breast_fraction_roi, n_jobs=n_jobs, level='rois',
            cropped_imgs=cropped_imgs, ignore_diameter_px=ignore_diameter_px, seed=seed,
            return_lesions_mask=False, max_lesion_diam_mm=None, use_muscle_mask=False
        )
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.balancing_seed = balancing_seed
        self.total_df = self.df.copy()

        if patch_images_path is not None:
            self.patch_img_path = patch_images_path/'patches'
            self.patch_mask_path = patch_images_path/'patches_masks'

    def balance_dataset(self, balancing_seed: int = None):
        n_pos = self.total_df.loc[self.df.label == 'abnormal', :].shape[0]
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
        self.balance_dataset(balancing_seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = {}
        sample["label"] = 1 if self.labels[idx] == 'abnormal' else 0

        img_path = self.patch_img_path / self.df['filename'].iloc[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_ANYDEPTH)
        # Convert all images in left oriented ones
        side = self.df['side'].iloc[idx]
        if side == 'R' and self.level == 'image':
            img = cv2.flip(img, 1)
        # Convert into float for better working of pytorch augmentations
        img = utils.min_max_norm(img, 1).astype('float32')

        # to RGB
        img = np.expand_dims(img, 0)
        img = np.repeat(img, 3, axis=0)
        sample['img'] = img

        patch_bbox = self.df['patch_bbox'].iloc[idx]
        if isinstance(patch_bbox, str):
            patch_bbox = utils.load_patch_coords(patch_bbox)
        sample["patch_bbox"] = patch_bbox
        return sample
