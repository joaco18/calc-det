import collections
import cv2
import logging
import pprint
import random

import numpy as np
import pandas as pd

from pathlib import Path
# from preprocessing.generate_roi_patches
# from torchvision import transforms
from typing import List
import utils

thispath = Path(__file__).resolve()
datapath = thispath.parent.parent / "data" / "INbreast Release 1.0"

# this is for caching small things for speed
_cache_dict = {}


class Dataset():
    def __init__(self):
        pass

    def totals(self):
        counts = [
            dict(
                collections.Counter(items[~np.isnan(items)]).most_common()
            ) for items in self.labels.T
        ]
        return dict(zip(self.lesion_type, counts))

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.string()

    def check_paths_exist(self):
        if not self.img_csv_path.exists():
            raise Exception("img_csv_path must be a directory")
        if not self.full_img_path.exists():
            raise Exception("full_img_path must be a file")
        if not self.rois_csv_path.exists():
            raise Exception("rois_csv_path must be a file")
        if not self.rois_img_path.exists():
            raise Exception("rois_img_path must be a file")

    def limit_to_selected_views(self, views):
        # TODO: Correct docstring
        """
        This function is called by subclasses to filter the
        images by view based on the values in .csv ['view']
        """
        if type(views) is not list:
            views = [views]
        if '*' in views:
            views = ["*"]
        self.views = views

        # Missing data is unknown
        self.img_csv.view.fillna("UNKNOWN", inplace=True)
        self.rois_csv.view.fillna("UNKNOWN", inplace=True)

        # Select the view
        if "*" not in views:
            self.img_csv = self.csv[self.csv["view"].isin(self.views)]
            self.rois_csv = self.csv[self.csv["view"].isin(self.views)]


class INBreast_Dataset(Dataset):
    # TODO: Correct docstring
    """INBreast Dataset
    Dataset release website:
    https://drive.google.com/file/d/19n-p9p9C0eCQA1ybm6wkMo-bbeccT_62/view

    Download the images, extract the zip and run *python database/parsing_metadata.py*
    following README.md instructions.
    """
    def __init__(
        self, imgpath: Path = datapath/'AllPNGs',
        mask_path: Path = datapath/'AllMasks',
        csvpath: Path = datapath,
        views: List[str] = ["*"],
        lesion_type: List[str] = ['calcification', 'cluster'],
        transform: List[str] = None,
        data_aug: List[str] = None,
        nrows: int = None,
        seed: int = 0,
        lesions_mask: bool = False,
        level: str = 'image',
        extract_rois: bool = True,
        normalize: str = None
    ):
        # TODO: Add docstring
        super(INBreast_Dataset, self).__init__()

        # Set seed so all runs are the same.
        self.seed = seed
        np.random.seed(self.seed)

        self.full_img_path = imgpath/'full_imgs'
        self.img_csv_path = csvpath/'images_metadata.csv'
        self.full_mask_path = mask_path/'full_imgs'

        # Work on rois csv
        self.rois_img_path = imgpath/'rois'
        self.rois_csv_path = csvpath/'rois_metadata.csv'
        self.rois_mask_path = mask_path/'rois'

        # Configurations
        self.level = level
        self.transform = transform
        self.data_aug = data_aug
        self.lesions_mask = lesions_mask
        self.normalize = normalize
        self.lesion_type = lesion_type

        # Load data
        self.check_paths_exist()
        self.rois_csv = pd.read_csv(self.rois_csv_path, nrows=nrows, index_col=0)
        self.img_csv = pd.read_csv(self.img_csv_path, nrows=nrows, index_col=0)

        # Remove images with view position other than specified
        self.limit_to_selected_views(views)
        self.rois_csv = self.rois_csv.reset_index()
        self.img_csv = self.img_csv.reset_index()

        # Filter dataset based on lesion type
        self.filter_by_lesion()
        self.add_image_label_to_image_csv()

        # TODO: Add Vlad's functions
        # If aren't stored yet, extract all patches and modify rois' csv
        # if extract_rois and level 'rois':
        #     generate_roi_patches(self.full_img_path, self.csv_path, cfg)
        # In my mind this should:
        #   - Get the rois from the lesions from the roi_csv that has already been filtered
        #       to the specific lesion type desired
        #   - Maybe add a filter that from the above rois if a micro-calcification flag is
        #       passed only lesions smaller than 1mm will be cropped as such. The rest of them
        #       can be removed from the csv
        #   - Crop as many normal rois as we want, maybe options: all, same n° as lesions, number
        #   - Add does Rois with the metadata that can be filled to the rois csv and add a column
        #       "label" that will contain "normal" or "lesion"
        # In cfg the specfific configuration should be passed?

        # Get our classes.
        self.csv = self.img_csv if level == 'image' else self.rois_csv
        self.labels = self.csv['label'].values
        # TODO: Define the format of output labels, now strings
        # self.labels = self.labels.astype(np.float32)

    def add_image_label_to_image_csv(self):
        # TODO: Add docstring
        self.img_df['label'] = 'normal'
        pathologic = [False] * self.img_df.shape[0]
        if 'mass' in self.lesion_type:
            pathologic = pathologic | (self.img_df.mass == True)
        if 'calcification' in self.lesion_type:
            pathologic = pathologic | (self.img_df.micros == True)
        if 'distortion' in self.lesion_type:
            pathologic = pathologic | (self.img_df.distortion == True)
        if 'asymmetry' in self.lesion_type:
            pathologic = pathologic | (self.img_df.asymmetry == True)
        self.img_df.loc[pathologic, 'img_label'] = 'abnormal'

    def filter_by_lesion(self):
        # TODO: Add docstring
        self.rois_csv = self.rois_csv.loc[self.rois_csv.lesion_type.isin(self.lesion_type), :]
        images_selection = (
            self.img_csv.img_id.isin(self.rois_csv.img_id.unique()) |
            self.img_csv.img_label == 'normal'
        )
        self.img_csv = self.img_csv.loc[images_selection, :]

    def string(self):
        return \
            f'{self.__class__.__name__ } num_samples={len(self)} \
                views={self.views} data_aug={self.data_aug}'

    def adjust_mask_to_selected_lesions(self, mask, idx):
        # TODO: Add docstring
        rois_from_img = self.roi_csv.img_id == self.csv['img_id'].iloc[idx]
        lesion_idxs = self.roi_csv.loc[rois_from_img, 'index_in_image'].values
        for les_idx in np.unique(mask):
            if les_idx not in lesion_idxs:
                mask[mask == les_idx] = 0
        return mask

    def flip_coordinates(self, idx):
        # TODO: Add docstring
        img_id = self.csv['img_id'].iloc[idx]
        breast_bbox = utils.load_coords(
            self.img_csv.loc[self.img_csv.img_id == img_id, 'breast_bbox']
        )
        bbox_shape = (
            breast_bbox[1][0] - breast_bbox[0][0],
            breast_bbox[1][1] - breast_bbox[0][1]
        )
        centers = self.rois_csv.loc[self.rois_csv.img_id == img_id, 'center_crop'].values
        for k, center in enumerate(centers):
            center = utils.load_point(center, 'int')
            centers[k][1] = bbox_shape[1] - center[1]
        self.rois_csv.loc[self.rois_csv.img_id == img_id, 'center_crop'] = centers

        lesion_bboxs_crop = \
            self.rois_csv.loc[self.rois_csv.img_id == img_id, 'lesion_bbox_crop'].values
        for k, lesion_bbox_crop in enumerate(lesion_bboxs_crop):
            lesion_bbox_crop = utils.load_coords(lesion_bbox_crop, 'int')
            lesion_bboxs_crop[k] = [
                (point[0], bbox_shape[1] - point[1]) for point in lesion_bbox_crop
            ]
        self.rois_csv.loc[self.rois_csv.img_id == img_id, 'lesion_bbox_crop'] = \
            lesion_bboxs_crop

        point_pxs_crop = \
            self.rois_csv.loc[self.rois_csv.img_id == img_id, 'point_px_crop'].values
        for k, point_px_crop in enumerate(point_pxs_crop):
            point_px_crop = utils.load_coords(point_px_crop, 'int')
            point_pxs_crop[k] = [
                (point[0], bbox_shape[1] - point[1]) for point in point_px_crop
            ]
        self.rois_csv.loc[self.rois_csv.img_id == img_id, 'point_px_crop'] = \
            point_pxs_crop

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        # Read image png
        filename = self.csv['filename'].iloc[idx]
        img_path = self.imgpath/filename
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        side = self.csv['side'].iloc[idx]

        # Convert all images in left oriented ones
        if side == 'R':
            img = cv2.flip(img, 1)
            self.flip_coordinates(idx)

        # Normalize
        if self.normalize == 'min_max':
            sample["img"] = utils.min_max_norm(img)
        elif self.normalize == 'z_score':
            sample["img"] = utils.z_score_norm(img)
        else:
            sample["img"] = img

        # If level is image also return bboxes coords for det CNNs and metrics
        if self.bbox and self.level == 'image':
            rois_from_img = self.roi_csv.img_id == self.csv['img_id'].iloc[idx]
            bboxes_coords = self.roi_csv.loc[rois_from_img, 'lesion_bbox_crop'].values
            sample["lesion_bboxes"] = [
                utils.load_coords(bbox) if isinstance(bbox, 'str')
                else bbox for bbox in bboxes_coords
            ]

        # Load lesion mask
        if self.lesions_mask:
            mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
            if self.level == 'image':
                mask = self.adjust_mask_to_selected_lesions(mask, idx)
            sample["lesion_mask"] = np.where(mask != 0, 1, 0)

        # Apply transformations
        if self.transform is not None:
            transform_seed = np.random.randint(self.seed)
            random.seed(transform_seed)
            sample["img"] = self.transform(sample["img"])
            if self.lesions_mask:
                random.seed(transform_seed)
                sample["lesion_mask"] = self.transform(sample["lesion_mask"])

        # Apply data augmentations
        if self.data_aug is not None:
            transform_seed = np.random.randint(self.seed)
            random.seed(transform_seed)
            sample["img"] = self.data_aug(sample["img"])
            if self.lesion_mask:
                for i in sample["lesion_mask"].keys():
                    random.seed(transform_seed)
                    sample["lesion_mask"] = self.data_aug(sample["lesion_mask"])

        return sample
