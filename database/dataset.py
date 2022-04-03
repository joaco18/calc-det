from pathlib import Path
thispath = Path(__file__).resolve()
import sys; sys.path.insert(0, str(thispath.parent))

import collections
import cv2
import logging
import pprint
import random
import utils

import multiprocessing as mp
import numpy as np
import pandas as pd

from pathlib import Path
from roi_extraction import slice_image, padd_image, view_as_windows
# from torchvision import transforms
from typing import List, Tuple
from tqdm import tqdm

thispath = Path(__file__).resolve()
datapath = thispath.parent.parent / "data" / "INbreast Release 1.0"
LESION_TYPES = ['asymmetry', 'calcification', 'cluster', 'distortion', 'mass', 'normal']


class Dataset():
    def __init__(self):
        pass

    def totals(self):
        counts = [
            dict(
                collections.Counter(items[~np.isnan(items)]).most_common()
            ) for items in self.labels.T
        ]
        return dict(zip(self.lesion_types, counts))

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.string()

    def check_paths_exist(self):
        if not self.img_df_path.exists():
            raise Exception(f"{self.img_df_path} could not be found")
        if not self.full_img_path.exists():
            raise Exception(f"{self.full_img_path} could not be found")
        if not self.rois_df_path.exists():
            raise Exception(f"{self.rois_df_path} could not be found")
        if not self.patch_img_path.exists():
            self.patch_img_path.mkdir(parents=True, exist_ok=True)
        if not self.patch_mask_path.exists():
            self.patch_mask_path.mkdir(parents=True, exist_ok=True)


class INBreast_Dataset(Dataset):
    """
    INBreast Dataset
    Dataset release website:
    https://drive.google.com/file/d/19n-p9p9C0eCQA1ybm6wkMo-bbeccT_62/view

    Download the images, extract the zip and run *python database/parsing_metadata.py*
    following README.md instructions.
    """
    def __init__(
        self, imgpath: Path = datapath/'AllPNGs',
        mask_path: Path = datapath/'AllMasks',
        dfpath: Path = datapath,
        views: List[str] = ["*"],
        lesion_types: List[str] = ['calcification', 'cluster'],
        transform: List[str] = None,
        data_aug: List[str] = None,
        nrows: int = None,
        seed: int = 0,
        return_lesions_mask: bool = False,
        level: str = 'image',
        partitions: List[str] = ['train', 'test'],
        max_lesion_size_mm: float = 1.0,
        extract_patches: bool = True,
        extract_patches_method: str = 'all',  # 'centered'
        patch_size: int = 12,
        stride: Tuple[int] = (1, 1),
        min_breast_fraction_roi: float = 0.,
        normalize: str = None,
        n_jobs: int = -1,
    ):
        """
        Constructor of INBreast_Dataset class
        Args:
            imgpath (Path, optional): path to the AllPNGs directory.
                Defaults to datapath/'AllPNGs'.
            mask_path (Path, optional): path to the AllMasks directory.
                Defaults to datapath/'AllMasks'.
            dfpath (Path, optional): path of the directory containing the csvs generated
                with parsing_metadata.py. Defaults to datapath.
            views (List[str], optional): List of views subset of ['CC', 'LMO'].
                Defaults to ["*"], which includes all of them
            lesion_types (List[str], optional): List of lesion types, subset of
                ['asymmetry', 'calcification', 'cluster', 'distortion', 'mass', 'normal'].
            transform (List[str], optional): List of transformations. Defaults to None.
            data_aug (List[str], optional): List of data augmentation procedures.
                Defaults to None.
            nrows (int, optional): List to filter the dataset to a number of examples.
                Defaults to None.
            seed (int, optional): Seed to gurantee reproducibility. Defaults to 0.
            return_lesions_mask (bool, optional): Whether to return the lesion mask for each
                example or not. Defaults to False.
            level (str, optional): Whether to generate a dataset at 'roi' or 'image' level.
                Defaults to 'image'.
            partitions (List[str]): Select predefined sets, subset from ['train', 'test'].
                Defaults to ['train', 'test']
            max_lesion_size_mm (float): Maximum horizontal or vertical diameter allowed for the
                lesion.
            extract_patches (bool, optional): Whether to extract the rois or not. Defaults to True.
            extract_patches_method (str, optional): Which method to use in the rois extraction.
                One of ['all', 'centered']. Defaults to 'all'.
            patch_size (int): size of the roi in pixels. Only used if rois are extracted.
            stride (Tuple[int], optional): If rois are extracted with 'all' method, define the
                stride to use. Defaults to (1, 1).
            min_breast_fraction_roi (float, optional): Minimum percentage of breast to consider
                the roi as a valid example. Defaults to 0.
            normalize (str, optional): ['min]. Defaults to None.
            n_jobs (int, optional): Number of processes to use in parallel operations.
                Defaults to -1.
        """
        super(INBreast_Dataset, self).__init__()

        # Set seed and number of cores to use.
        self.seed = seed
        np.random.seed(self.seed)
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

        self.full_img_path = imgpath/'full_imgs'
        self.img_df_path = dfpath/'images_metadata.csv'
        self.full_mask_path = mask_path/'full_imgs'

        # Work on rois df
        self.patch_img_path = imgpath/'patches'
        self.rois_df_path = dfpath/'rois_metadata.csv'
        self.patch_mask_path = mask_path/'patches_masks'

        # Configurations
        self.level = level
        self.views = views
        self.partitions = partitions
        self.transform = transform
        self.data_aug = data_aug
        self.lesions_mask = return_lesions_mask
        self.normalize = normalize
        self.lesion_types = lesion_types
        self.max_lesion_size_px = int(max_lesion_size_mm / 0.07)

        # Load data
        self.check_paths_exist()
        self.rois_df = pd.read_csv(self.rois_df_path, nrows=nrows, index_col=0)
        self.img_df = pd.read_csv(self.img_df_path, nrows=nrows, index_col=0)

        # Filter dataset based on different criteria
        self.rois_df = self.rois_df.loc[self.rois_df.stored]
        self.filter_by_partition()
        self.filter_by_lesion_size()
        self.limit_to_selected_views()
        self.filter_by_lesion_type()
        self.add_image_label_to_image_df()

        # Get rois df
        if level == 'rois':
            if extract_patches:
                self.patch_size = patch_size
                self.min_breast_frac = min_breast_fraction_roi
                self.stride = stride
                if extract_patches_method == 'all':
                    self.patches_df = self.all_patches_extraction()
                    self.patches_df.to_csv(str(dfpath/'complete_rois_metadata.csv'))
                else:
                    self.patches_df = self.centered_rois_extraction()
                    self.patches_df.to_csv(str(dfpath/'complete_rois_metadata.csv'))
            else:
                self.patches_df = pd.read_csv(
                    str(dfpath/'complete_rois_metadata.csv'), index_col=0
                )

        # Get our classes.
        self.df = self.img_df if level == 'image' else self.patches_df
        self.labels = self.df['label'].values

    def string(self):
        return \
            f'{self.__class__.__name__ } num_samples={len(self)} \
                views={self.views} data_aug={self.data_aug}'

    def limit_to_selected_views(self):
        """
        This method is called to filter the images by view based on the values
        in self.df['view']
        """
        if type(self.views) is not list:
            self.views = [self.views]
        if '*' in self.views:
            self.views = ["*"]

        # Missing data is unknown
        self.img_df['view'].fillna("UNKNOWN", inplace=True)
        self.rois_df['view'].fillna("UNKNOWN", inplace=True)

        # Select the view
        if "*" not in self.views:
            self.img_df = self.df[self.df["view"].isin(self.views)]
            self.rois_df = self.df[self.df["view"].isin(self.views)]

        self.rois_df.reset_index(inplace=True, drop=True)
        self.img_df.reset_index(inplace=True, drop=True)

    def filter_by_partition(self):
        """
        This method is called to filter the images according to the predefined partitions
        in the INBreast Database
        """
        # filter rois df
        self.rois_df = self.rois_df.loc[self.rois_df.partition.isin(self.partitions), :]
        self.rois_df.reset_index(inplace=True, drop=True)
        # filter imgs df
        self.img_df = self.img_df.loc[self.img_df['partition'].isin(self.partitions)]
        self.img_df.reset_index(inplace=True, drop=True)

    def filter_by_lesion_size(self):
        """
        Filters the images according to the diameter
        of the circle enclosing the lesion.
        """
        # filter rois df
        self.rois_df = self.rois_df.loc[self.rois_df.radius <= self.max_lesion_size_px, :]
        self.rois_df.reset_index(inplace=True, drop=True)

    def filter_by_lesion_type(self):
        """
        Filters the images by view based on the values
        in self.df['lesion_type']
        """
        # filter rois df
        self.rois_df = self.rois_df.loc[self.rois_df.lesion_type.isin(self.lesion_types), :]
        self.rois_df.reset_index(inplace=True, drop=True)

        # filter imgs df
        if 'normal' in self.lesion_types:
            images_selection = (
                self.img_df.img_id.isin(self.rois_df.img_id.unique()) |
                self.img_df.img_label == 'normal'
            )
        else:
            images_selection = self.img_df.img_id.isin(self.rois_df.img_id.unique())
        self.img_df = self.img_df.loc[images_selection, :]
        self.img_df.reset_index(inplace=True, drop=True)

    def add_image_label_to_image_df(self):
        """
        Adds an abnormal/normal label to the complete image
        based on the lesion type selection and on the lesions present in the image
        """
        self.img_df['label'] = 'normal'
        pathologic = [False] * self.img_df.shape[0]
        if 'mass' in self.lesion_types:
            pathologic = pathologic | (self.img_df.mass)
        if 'calcification' in self.lesion_types:
            pathologic = pathologic | (self.img_df.micros)
        if 'distortion' in self.lesion_types:
            pathologic = pathologic | (self.img_df.distortion)
        if 'asymmetry' in self.lesion_types:
            pathologic = pathologic | (self.img_df.asymmetry)
        self.img_df.loc[pathologic, 'img_label'] = 'abnormal'

    def all_patches_extraction(self):
        """
        Extracts all possible rois from the images according to the paramenters
        passed to the constructor. It saves the rois and the masks and updates the rois csv.
        The processing is done in parallel to make it faster.
        """
        n_rows = self.img_df.shape[0]
        res = []
        # for i in tqdm(indxs, total=len(indxs)):
        #     res.append(self.extract_patches_from_image(i))
        with mp.Pool(self.n_jobs) as pool:
            for result in tqdm(
                pool.imap(self.extract_patches_from_image, range(n_rows)), total=n_rows
            ):
                res.append(result)
        patches_df = pd.concat(res, ignore_index=True)
        return patches_df.sort_values(by='img_id')

    def extract_patches_from_image(self, idx: int):
        """
        Extracts rois from an image and returns their description for given image and mask.
        Args:
            idx (int): index of the row to read in the images dataframe
        Returns:
            (pd.DataFrame): patches_descr describing each ROI.
        """
        # Read images pngs
        filename = Path(self.img_df['filename'].iloc[idx]).name
        img_path = self.full_img_path / filename
        image = cv2.imread(str(img_path), cv2.IMREAD_ANYDEPTH)
        img_id = self.img_df['img_id'].iloc[idx]
        mask_path = self.full_mask_path / f'{img_id}_lesion_mask.png'
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        side = self.img_df['side'].iloc[idx]
        if side == 'R':
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
            self.flip_coordinates(img_id)

        # Extract patches equally from image and the mask
        image = padd_image(image, self.patch_size)
        mask = padd_image(mask, self.patch_size)
        image_patches = slice_image(image, window_size=self.patch_size, stride=self.stride)
        mask_patches = slice_image(mask, window_size=self.patch_size, stride=self.stride)

        # Filter rois already filtered from the rois_df
        present_indx = self.rois_df.loc[self.rois_df.img_id == img_id, 'index_in_image'].tolist()
        indexes_to_filter = \
            [index for index in np.unique(mask_patches) if index not in present_indx]
        for index in indexes_to_filter:
            mask_patches = np.where(mask_patches == index, 0, mask_patches)

        # count number of pixels per lesion in each roi
        index_freqs = [dict(zip(*np.unique(m, return_counts=True))) for m in mask_patches]
        patches_descr = pd.DataFrame(index_freqs)

        image_patches_df = self.rois_df[self.rois_df.img_id == img_id]

        # mapping indexes in image to type name
        rois_types = pd.Series(
            image_patches_df.lesion_type.values, index=image_patches_df.index_in_image
        ).to_dict()
        rois_types[0] = "normal"
        patches_descr = patches_descr.rename(columns=rois_types)

        # grouping rois with the same type
        patches_descr = patches_descr.groupby(lambda x: x, axis=1).sum() / \
            (image_patches.shape[1]*image_patches.shape[2])

        # select only rois from lesion types selection.
        patches_descr = patches_descr[patches_descr.columns.intersection(self.lesion_types)]

        # standartize df to habe always same number and types of columns
        for lt in LESION_TYPES:
            if lt not in patches_descr.columns:
                patches_descr[lt] = 0

        # Get the percentage of breast in the roi
        breast_pixels = np.array([(roi != 0).sum() for roi in image_patches])
        patches_descr['breast_fraction'] = breast_pixels / \
            (image_patches.shape[1]*image_patches.shape[2])

        # Filter Rois with more background than breast or just bkgrd
        keep_idx = \
            patches_descr.loc[patches_descr.breast_fraction >= self.min_breast_frac].index.tolist()
        patches_descr = patches_descr.iloc[keep_idx]
        image_patches = image_patches[keep_idx, :, :]
        mask_patches = mask_patches[keep_idx, :, :]
        patches_descr.reset_index(inplace=True, drop=True)

        # Generate a binary mask
        mask_patches = np.where(mask_patches != 0, 255, 0)

        # calculating patches coordinates
        bbox_coordinates = []
        row_num, col_num, _, __ = view_as_windows(image, self.patch_size, self.stride).shape
        for col in range(row_num):
            row_idx = [((row * self.stride, col * self.stride),
                        (self.patch_size + row * self.stride,
                        self.patch_size + col * self.stride)) for row in range(col_num)]
            bbox_coordinates.extend(row_idx)
        bbox_coordinates = np.array(bbox_coordinates)
        patches_descr['patch_bbox'] = list(bbox_coordinates[keep_idx, :, :])

        # Save rois and masks
        patch_filenames, patch_mask_filenames = [], []
        for roi_idx in range(image_patches.shape[0]):
            # You won't have empty images in this case due to the min_breast constrain
            roi_name = f'{img_id}_roi_{roi_idx}.png'
            patch_filenames.append(roi_name)
            cv2.imwrite(str(self.patch_img_path/roi_name), image_patches[roi_idx, :, :])

            if mask_patches[roi_idx, :, :].any():  # Empty images cannot be stored
                roi_mask_name = f'{img_id}_roi_{roi_idx}_mask.png'
                patch_mask_filenames.append(roi_mask_name)
                cv2.imwrite(str(self.patch_mask_path/roi_mask_name), mask_patches[roi_idx, :, :])
            else:
                patch_mask_filenames.append('empty_mask')

        # complete dataframe
        patches_descr['filename'] = patch_filenames
        patches_descr['mask_filename'] = patch_mask_filenames
        for column in ['case_id', 'img_id', 'side', 'view', 'acr', 'birads']:
            patches_descr[column] = self.img_df[column].iloc[idx]

        # Add the same label as expected in the constructor
        patches_descr['label'] = False
        for les_type in LESION_TYPES:
            patches_descr['label'] = \
                patches_descr['label'] | np.where(patches_descr[les_type] != 0, True, False)
        patches_descr['label'] = np.where(patches_descr['label'], 'abnormal', 'normal')

        return patches_descr

    def centered_rois_extraction(self):
        """
        Extracts rois of a fixed size centered in each lesion.
        The processing is done in parallel to make it faster.
        """
        n_rows = self.img_df.shape[0]
        res = []
        with mp.Pool(self.n_jobs) as pool:
            for result in tqdm(
                pool.imap(self.extract_centered_rois_from_image, range(n_rows)),
                total=n_rows
            ):
                res.append(result)
        patches_df = pd.concat(res, ignore_index=True)
        return patches_df.sort_values(by='img_id')

    def extract_centered_rois_from_image(self, idx: int):
        """
        Extracts the rois from an image using a bbox centered at each lesion.
        Args:
            idx (int): index of the row to read in the images dataframe
        Returns:
            (pd.DataFrame): patches_descr describing each ROI.
        """
        # Goes image by image extracting all rois if any available
        image_id = self.img_df['img_id'].iloc[idx]
        rois_subset_df = self.rois_df.loc[self.rois_df.img_id == image_id]
        if rois_subset_df.shape[0] == 0:
            rois_subset_df = rois_subset_df.reindex(
                columns=rois_subset_df.columns.tolist() + ['patch_filename', 'patch_mask_filename']
            )
            return rois_subset_df

        # Read images pngs
        filename = Path(self.img_df['filename'].iloc[idx]).name
        img_path = self.full_img_path / filename
        image = cv2.imread(str(img_path), cv2.IMREAD_ANYDEPTH)
        img_id = self.img_df['img_id'].iloc[idx]
        mask_path = self.full_mask_path / f'{img_id}_lesion_mask.png'
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # TODO: ALEX FUNCTIONS
        # Flip the image
        # Extract the roi based on the bbox for each lesion in rois_subset
        # check breast fraction
        # use self.patch_size
        # Save roi and mask
        # Add patch_filename and patch_mask_filename columns to the dataframe
        # Add patch_filename and patch_mask_filename columns to the dataframe
        # Add patch_bbox

        return rois_subset_df

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        # Read image png
        if self.level == 'image':
            filename = Path(self.df['filename'].iloc[idx]).name
            img_path = self.full_img_path / filename
            img = cv2.imread(str(img_path), cv2.IMREAD_ANYDEPTH)
        else:
            img_path = self.patch_img_path / self.df['filename'].iloc[idx]
            img = cv2.imread(str(img_path), cv2.IMREAD_ANYDEPTH)

        # Convert all images in left oriented ones
        side = self.df['side'].iloc[idx]
        img_id = self.df['img_id'].iloc[idx]
        if side == 'R' and self.level == 'image':
            img = cv2.flip(img, 1)
            self.flip_coordinates(img_id)

        # Normalize
        if self.normalize == 'min_max':
            sample["img"] = utils.min_max_norm(img)
        elif self.normalize == 'z_score':
            sample["img"] = utils.z_score_norm(img)
        else:
            sample["img"] = img

        # Return bboxes coords for det CNNs and metrics
        if self.level == 'image':
            rois_from_img = self.rois_df.img_id == self.df['img_id'].iloc[idx]
            bboxes_coords = self.rois_df.loc[rois_from_img, 'lesion_bbox_crop'].values
            sample["lesion_bboxes"] = [
                utils.load_coords(bbox) if isinstance(bbox, str)
                else bbox for bbox in bboxes_coords
            ]
        else:
            sample["patch_bbox"] = [self.df['patch_bbox'].iloc[idx]]

        # Load lesion mask
        if self.lesions_mask:
            if self.level == 'image':
                mask_path = \
                    self.full_mask_path / f'{self.df["img_id"].iloc[idx]}_lesion_mask.png'
                if not mask_path.exists():
                    mask = np.zeros(img.shape)
                else:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    mask = self.adjust_mask_to_selected_lesions(mask, idx)
                if side == 'R':
                    mask = cv2.flip(mask, 1)
            else:
                mask_filename = self.df['mask_filename'].iloc[idx]
                if mask_filename != 'empty_mask':
                    mask_filename = self.patch_mask_path / mask_filename
                    mask = cv2.imread(str(mask_filename), cv2.IMREAD_ANYDEPTH)
                else:
                    mask = np.zeros(img.shape)
            sample["lesion_mask"] = np.where(mask != 0, 255, 0)

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

    def adjust_mask_to_selected_lesions(self, mask: np.ndarray, idx: int):
        """
        Keeps just the lesions remaining after different filterings
        Args:
            mask (np.ndarray): Mask of the different lesions in the image
            idx (int): index in 'df' of the example considered
        Returns:
            (np.ndarray): Corrected mask
        """
        rois_from_img = self.rois_df.img_id == self.df['img_id'].iloc[idx]
        lesion_idxs = self.rois_df.loc[rois_from_img, 'index_in_image'].values
        if len(lesion_idxs) != 0:
            for les_idx in np.unique(mask):
                if les_idx not in lesion_idxs:
                    mask[mask == les_idx] = 0
        else:
            mask = np.zeros(mask.shape)
        return mask

    def flip_coordinates(self, img_id: int):
        """
        If the image is flipped, the coordinates of the points in the df should be flipped
        Specially the bbox, which also need to keep the the upper left
        bottom right convention.
        Args:
            img_id (int): img_id of the example to analyse
        """
        # img_id = self.df['img_id'].iloc[idx]
        breast_bbox = utils.load_coords(
            self.img_df.loc[self.img_df.img_id == img_id, 'breast_bbox'].values[0],
            dtype=int
        )
        bbox_shape = (
            breast_bbox[1][0] - breast_bbox[0][0],
            breast_bbox[1][1] - breast_bbox[0][1]
        )
        centers = self.rois_df.loc[self.rois_df.img_id == img_id, 'center_crop'].tolist()
        if isinstance(centers[0], str):
            centers = [np.array(utils.load_point(point, 'int')) for point in centers]
        else:
            centers = [np.array(point) for point in centers]
        for k, center in enumerate(centers):
            centers[k][0] = bbox_shape[0] - center[0]
        centers = [tuple(center) for center in centers]

        lesion_bboxs_crop = \
            self.rois_df.loc[self.rois_df.img_id == img_id, 'lesion_bbox_crop'].tolist()
        for k, lesion_bbox_crop in enumerate(lesion_bboxs_crop):
            if isinstance(lesion_bbox_crop, str):
                lesion_bbox_crop = utils.load_coords(lesion_bbox_crop, 'int')
            lesion_bbox_crop = [
                (bbox_shape[0] - point[0], point[1]) for point in lesion_bbox_crop
            ]
            lesion_bboxs_crop[k] = [
                (lesion_bbox_crop[1][0], lesion_bbox_crop[0][1]),
                (lesion_bbox_crop[0][0], lesion_bbox_crop[1][1]),
            ]

        point_pxs_crop = self.rois_df.loc[self.rois_df.img_id == img_id, 'point_px_crop'].values
        for k, point_px_crop in enumerate(point_pxs_crop):
            if isinstance(point_px_crop, str):
                point_px_crop = utils.load_coords(point_px_crop, 'int')
            point_pxs_crop[k] = [
                (bbox_shape[0] - point[0], point[1]) for point in point_px_crop
            ]
        for k, row in enumerate(self.rois_df.loc[self.rois_df.img_id == img_id].iterrows()):
            self.rois_df.at[row[0], 'center_crop'] = centers[k]
            self.rois_df.at[row[0], 'lesion_bbox_crop'] = lesion_bboxs_crop[k]
            self.rois_df.at[row[0], 'point_px_crop'] = point_pxs_crop[k]
