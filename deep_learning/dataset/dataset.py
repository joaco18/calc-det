from pathlib import Path
thispath = Path.cwd().resolve()
import sys; sys.path.insert(0, str(thispath.parent.parent))

from database.dataset import INBreast_Dataset
import general_utils.utils as utils

import cv2
import random
import numpy as np
import torch

from typing import List, Tuple
from torchvision import transforms

datapath = thispath.parent.parent / "data" / "INbreast Release 1.0"


class INBreast_Dataset_pytorch(INBreast_Dataset):
    def __init__(
        self, imgpath: Path = datapath/'AllPNGs',
        mask_path: Path = datapath/'AllMasks',
        dfpath: Path = datapath,
        lesion_types: List[str] = ['calcification', 'cluster'],
        transform: List[str] = None,
        data_aug: List[str] = None,
        partitions: List[str] = ['train', 'validation', 'test'],
        extract_patches: bool = True,
        delete_previous: bool = True,
        extract_patches_method: str = 'all',
        patch_size: int = 224,
        stride: Tuple[int] = 100,
        min_breast_fraction_roi: float = 0.7,
        n_jobs: int = -1,
        cropped_imgs: bool = True,
        ignore_diameter_px: int = 15
    ):
        super(INBreast_Dataset_pytorch, self).__init__(
            imgpath=imgpath, mask_path=mask_path, dfpath=dfpath, lesion_types=lesion_types,
            transform=transform, data_aug=data_aug, partitions=partitions,
            delete_previous=delete_previous, extract_patches=extract_patches,
            extract_patches_method=extract_patches_method, patch_size=patch_size,
            stride=stride, min_breast_fraction_roi=min_breast_fraction_roi,
            n_jobs=n_jobs, cropped_imgs=cropped_imgs, ignore_diameter_px=ignore_diameter_px,
            level='rois', return_lesions_mask=False, max_lesion_diam_mm=None, use_muscle_mask=False
        )

    def __getitem__(self, idx):
        sample = {}
        sample["label"] = self.labels[idx]

        img_path = self.patch_img_path / self.df['filename'].iloc[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_ANYDEPTH)
        # Convert all images in left oriented ones
        side = self.df['side'].iloc[idx]
        if side == 'R' and self.level == 'image':
            img = cv2.flip(img, 1)

        img = utils.min_max_norm(img, 255).astype('uint8')
        # Apply transformations
        # Warning: normalization should be indicated as a Transformation
        if self.transform is not None:
            transform_seed = np.random.randint(self.seed)
            random.seed(transform_seed)
            img = self.transform(img)

        # Apply data augmentations
        if self.data_aug is not None:
            transform_seed = np.random.randint(self.seed)
            random.seed(transform_seed)
            img = self.data_aug(img)

        sample['img'] = img

        patch_bbox = self.df['patch_bbox'].iloc[idx]
        if isinstance(patch_bbox, str):
            patch_bbox = utils.load_patch_coords(patch_bbox)
        sample["patch_bbox"] = patch_bbox
        return sample
