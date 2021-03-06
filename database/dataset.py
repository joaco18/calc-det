from pathlib import Path
thispath = Path(__file__).resolve()
import sys; sys.path.insert(0, str(thispath.parent))

import collections
import cv2
import logging
import pprint
import random
import shutil
import general_utils.utils as utils

from functools import partial
import multiprocessing as mp
import numpy as np
import pandas as pd

from pathlib import Path
from roi_extraction import slice_image, padd_image, view_as_windows
from typing import List, Tuple
from tqdm import tqdm

datapath = thispath.parent.parent / "data" / "INbreast Release 1.0"
LESION_TYPES = [
    'asymmetry', 'calcification', 'cluster', 'distortion', 'mass', 'normal', 'ignored_lesion']


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
        keep_just_images_of_lesion_type: List[str] = False,
        seed: int = 0,
        return_lesions_mask: bool = False,
        level: str = 'image',
        partitions: List[str] = ['train', 'validation', 'test'],
        max_lesion_diam_mm: float = 1.0,
        extract_patches: bool = True,
        delete_previous: bool = True,
        extract_patches_method: str = 'all',  # 'centered'
        patch_size: int = 12,
        stride: int = 1,
        min_breast_fraction_roi: float = 0.,
        n_jobs: int = -1,
        cropped_imgs: bool = True,
        use_muscle_mask: bool = False,
        ignore_diameter_px: int = 15,
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
                If None, no filtering is applied
            keep_just_images_of_lesion_type (bool, optoinal): If True, only the images with
                lesions included in lesion types will be retained. Defaults to False, which
                means all images are retained and then just the rois of 'lesion_types' are
                retained.
            seed (int, optional): Seed to gurantee reproducibility. Defaults to 0.
            return_lesions_mask (bool, optional): Whether to return the lesion mask for each
                example or not. Defaults to False.
            level (str, optional): Whether to generate a dataset at 'rois' or 'image' level.
                Defaults to 'image'.
            partitions (List[str]): Select predefined sets, subset from
                ['train', 'validation', 'test']. Defaults to ['train', 'validation', 'test']
            max_lesion_diam_mm (float): Maximum horizontal or vertical diameter allowed for the
                lesion. If None, no filtring is applied
            extract_patches (bool, optional): Whether to extract the rois or not. Defaults to True.
            delete_previous (bool, optional): Whether to remove preexisting files before extracting
                crops. Defaults to True
            extract_patches_method (str, optional): Which method to use in the rois extraction.
                One of ['all', 'centered']. Defaults to 'all'.
            patch_size (int): size of the roi in pixels. Only used if rois are extracted.
            stride (Tuple[int], optional): If rois are extracted with 'all' method, define the
                stride to use. Defaults to 1.
            min_breast_fraction_roi (float, optional): Minimum percentage of breast to consider
                the roi as a valid example. If muscle masks are used, this same criteria will
                apply to the region of muscle. Defaults to 0.
            n_jobs (int, optional): Number of processes to use in parallel operations.
                Defaults to -1
            cropped_imgs (bool): whether the images to read has the breast region cropped.
            use_muscle_mask (bool): whether to use the pectoral muscle masks or not.
                If image level, simply the pectoral mask is returned. If roi level, the
                pectoral mask is used to define the "in breast" patches, considering
                pectoral muscle as if it was background with min_breast_fraction_roi
            ignore_diameter_px (int, optional): Defaults to 15. The maximum diameter to consider
                a labeled mc indeed as one. If set to None, no distintion will be done. If
                a diameter is chosen then:
                At the image level:
                    all rois larger than that threshold will have a label index = -1 in the
                    lesions mask
                At the rois level:
                    A patch having ONLY labels to ignore (all) or centered at them ('centered')
                    will be discarded.
        """
        super(INBreast_Dataset, self).__init__()

        # Set seed and number of cores to use.
        self.seed = seed
        np.random.seed(self.seed)
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

        self.full_img_path = imgpath/'full_imgs'
        self.img_df_path = dfpath/'images_metadata.csv'
        self.full_mask_path = mask_path/'full_imgs'
        self.full_muscle_mask_path = mask_path/'muscle_masks'

        # Work on rois df
        self.patch_img_path = imgpath.parent.parent.parent.parent/'data_rois'/'patches'
        self.rois_df_path = dfpath/'rois_metadata.csv'
        self.patch_mask_path = imgpath.parent.parent.parent.parent/'data_rois'/'patches_masks'

        # Configurations
        self.use_muscle_mask = use_muscle_mask
        self.ignore_diameter_px = ignore_diameter_px
        self.level = level
        self.views = views
        self.partitions = partitions
        self.lesions_mask = return_lesions_mask
        self.lesion_types = lesion_types
        self.keep_just_images_of_lesion_type = keep_just_images_of_lesion_type
        self.max_lesion_diam_px = \
            int(max_lesion_diam_mm / 0.07) if (max_lesion_diam_mm is not None) else None
        self.cropped_imgs = cropped_imgs

        # Load data
        self.check_paths_exist()
        self.rois_df = pd.read_csv(self.rois_df_path, index_col=0)
        self.img_df = pd.read_csv(self.img_df_path, index_col=0)

        # Add validation partition
        self.generate_validation_partition()

        # Filter dataset based on different criteria
        self.filter_excluded_cases()
        self.filter_stored_rois()
        self.filter_by_partition()
        if self.max_lesion_diam_px is not None:
            self.filter_by_lesion_size()
        self.limit_to_selected_views()
        if self.lesion_types is not None:
            self.filter_by_lesion_type()
        if self.ignore_diameter_px is not None:
            self.identify_lesions_to_ignore()
        self.add_image_label_to_image_df()
        self.flip_coordinates()

        # Get rois df
        if level == 'rois':
            if extract_patches:
                if delete_previous:
                    if self.patch_img_path.exists():
                        shutil.rmtree(self.patch_img_path)
                        self.patch_img_path.mkdir(exist_ok=True, parents=True)
                    if self.patch_mask_path.exists():
                        shutil.rmtree(self.patch_mask_path)
                        self.patch_mask_path.mkdir(exist_ok=True, parents=True)

                self.patch_size = patch_size
                self.min_breast_frac = min_breast_fraction_roi
                self.stride = stride
                if extract_patches_method == 'all':
                    self.patches_df = self.all_patches_extraction()
                else:
                    if self.max_lesion_diam_px is not None:
                        assert self.patch_size >= self.max_lesion_diam_px, \
                            'The largest lesion selected doesn\' fit inside the patch ' \
                            'size selected.\n Please modify it or use \'all\' extraction method.'
                    selection = self.rois_df.lesion_type != 'ignored_lesion'
                    max_size_in_db = int(
                        np.ceil(self.rois_df.loc[selection, 'radius'].max()))
                    assert self.patch_size >= max_size_in_db, \
                        f'The largest lesion present in the dataset ({max_size_in_db}) doesn\'t' \
                        f' fit inside the patch size selected.\n Please modify it or use' \
                        f' \'all\' extraction method.'

                    self.patches_df = self.centered_patches_extraction()
                for partition in self.partitions:
                    self.patches_df.loc[self.patches_df.partition == partition].to_csv(
                        str(dfpath/f'complete_rois_metadata_{partition}.csv'))
            else:
                patches_df = []
                for partition in self.partitions:
                    csv_path = dfpath / f'complete_rois_metadata_{partition}.csv'
                    assert (csv_path).exists(), \
                        f'file {csv_path} doesn\'t exist, change ' \
                        f'\'extract_patches\' to True and run again'
                    patches_df.append(pd.read_csv(
                        str(dfpath/f'complete_rois_metadata_{partition}.csv'), index_col=0))
                self.patches_df = pd.concat(patches_df, ignore_index=True)

        # Get our classes.
        self.df = self.img_df if level == 'image' else self.patches_df
        self.labels = self.df['label'].values

    def string(self):
        return \
            f'{self.__class__.__name__ } num_samples={len(self)} views={self.views}'

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

    def filter_excluded_cases(self):
        # Filter out cases with offset in labels
        with open(str(thispath.parent.parent / 'data/abnormal_images.txt'), 'r') as f:
            abnormal_images_ids = [int(img_id.strip()) for img_id in f.readlines()]

        rois2drop = self.rois_df.index[self.rois_df.img_id.isin(abnormal_images_ids)]
        self.rois_df.drop(index=rois2drop)
        self.rois_df.reset_index(inplace=True, drop=True)

        imgs2drop = self.img_df.index[self.img_df.img_id.isin(abnormal_images_ids)]
        self.img_df.drop(index=imgs2drop)
        self.img_df.reset_index(inplace=True, drop=True)

    def filter_stored_rois(self):
        self.rois_df = self.rois_df.loc[self.rois_df.stored]
        self.rois_df.reset_index(drop=True, inplace=True)

    def generate_validation_partition(self):
        """Add validation partition"""
        train_cases = self.rois_df.loc[self.rois_df.partition == 'train', 'case_id'].unique()
        val_size = int(len(train_cases) * 0.3)
        val_selection = np.random.choice(train_cases, size=val_size, replace=False)

        # Add partition to rois df
        self.rois_df.loc[self.rois_df.case_id.isin(val_selection), 'partition'] = 'validation'
        # filter imgs df
        self.img_df.loc[self.img_df.case_id.isin(val_selection), 'partition'] = 'validation'

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
        self.rois_df = self.rois_df.loc[2 * self.rois_df.radius <= self.max_lesion_diam_px, :]
        self.rois_df.reset_index(inplace=True, drop=True)

    def identify_lesions_to_ignore(self):
        """
        Change the lesion type to 'ignored_lesion' if is bigger than the desired threshold
        """
        self.rois_df.loc[2 * self.rois_df.radius >= self.ignore_diameter_px, 'lesion_type'] = \
            'ignored_lesion'

    def filter_by_lesion_type(self):
        """
        Filters the images by view based on the values
        in self.df['lesion_type']
        """
        # filter rois df
        self.rois_df['lesion_type'] = self.rois_df['lesion_type'].fillna(value='calcification')
        self.rois_df = self.rois_df.loc[self.rois_df.lesion_type.isin(self.lesion_types), :]
        self.rois_df.reset_index(inplace=True, drop=True)
        # filter imgs df
        if self.keep_just_images_of_lesion_type:
            if 'normal' in self.lesion_types:
                images_selection = (
                    self.img_df.img_id.isin(self.rois_df.img_id.unique()) |
                    (self.img_df.img_label == 'normal')
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
        self.img_df.loc[self.img_df.img_id.isin(self.rois_df.img_id.unique()), 'label'] = 'abnormal'

    def all_patches_extraction(self):
        """
        Extracts all possible rois from the images according to the paramenters
        passed to the constructor. It saves the rois and the masks and updates the rois csv.
        The processing is done in parallel to make it faster.
        """
        n_rows = self.img_df.shape[0]
        logging.info('Start extracting patches')
        res = []
        # for i in tqdm(range(n_rows), total=n_rows):            # Kept for easy debbuging
        #     res.append(self.extract_patches_from_image(i))
        with mp.Pool(self.n_jobs) as pool:
            for result in tqdm(
                pool.imap(self.extract_patches_from_image, range(n_rows)), total=n_rows
            ):
                res.append(result)
        patches_df = pd.concat(res, ignore_index=True)
        return patches_df.sort_values(by='img_id')

    def extract_patches_from_image(self, idx: int, save_lesions: bool = True):
        """
        Extracts rois from an image and returns their description for given image and mask.
        Args:
            idx (int): index of the row to read in the images dataframe
            save_lesions (boll) Whether to save or not the lesion patches. Defaults to True
        Returns:
            (pd.DataFrame): patches_descr describing each ROI.
        """
        # Read images pngs
        filename = Path(self.img_df['filename'].iloc[idx]).name
        img_path = self.full_img_path / filename
        image = cv2.imread(str(img_path), cv2.IMREAD_ANYDEPTH)
        img_id = self.img_df['img_id'].iloc[idx]
        partition = self.img_df['partition'].iloc[idx]
        mask_path = self.full_mask_path / f'{img_id}_lesion_mask.png'
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(image.shape)
        if self.use_muscle_mask:
            muscle_mask_path = self.full_muscle_mask_path / f'{img_id}_pectoral_muscle_mask.png'
            if muscle_mask_path.exists():
                muscle_mask = cv2.imread(str(muscle_mask_path), cv2.IMREAD_GRAYSCALE)
            else:
                muscle_mask = np.zeros(image.shape)
        side = self.img_df['side'].iloc[idx]
        if side == 'R':
            image = cv2.flip(image, 1)
            if mask.any():
                mask = cv2.flip(mask, 1)
            if self.use_muscle_mask and muscle_mask.any():
                muscle_mask = cv2.flip(muscle_mask, 1)

        # Extract patches equally from image and the mask
        image = padd_image(image, self.patch_size)
        mask = padd_image(mask, self.patch_size)
        image_patches = slice_image(image, window_size=self.patch_size, stride=self.stride)
        mask_patches = slice_image(mask, window_size=self.patch_size, stride=self.stride)

        # Filter rois already filtered from the rois_df
        # Identify and remove patches of undesired lesion types
        present_indx = self.rois_df.loc[self.rois_df.img_id == img_id, 'index_in_image'].values
        indexes_to_filter = \
            [index for index in np.unique(mask_patches) if index not in present_indx]
        for index in indexes_to_filter:
            mask_patches = np.where(mask_patches == index, 0, mask_patches)

        # Identify labels to ignore, and relabel them with -1 index
        if self.ignore_diameter_px is not None:
            indexes_to_ignore = self.rois_df.loc[
                (self.rois_df.img_id == img_id) & (self.rois_df.lesion_type == 'ignored_lesion'),
                'index_in_image'].values
            if len(indexes_to_ignore) != 0:
                for index in indexes_to_ignore:
                    mask_patches = np.where(mask_patches == index, -1, mask_patches)

        # count number of pixels per lesion in each roi
        index_freqs = [dict(zip(*np.unique(m, return_counts=True))) for m in mask_patches]
        patches_descr = pd.DataFrame(index_freqs)

        image_patches_df = self.rois_df[self.rois_df.img_id == img_id]

        # mapping indexes in image to type name
        rois_types = pd.Series(
            image_patches_df.lesion_type.values, index=image_patches_df.index_in_image
        ).to_dict()
        rois_types[0] = "normal"
        if self.ignore_diameter_px is not None:
            rois_types[-1] = "ignored_lesion"
        patches_descr = patches_descr.rename(columns=rois_types)

        # grouping rois with the same type
        patches_descr = patches_descr.groupby(lambda x: x, axis=1).sum()

        # select only rois from lesion types selection.
        patches_descr = patches_descr[patches_descr.columns.intersection(
            self.lesion_types+['ignored_lesion'])]

        # standartize df to habe always same number and types of columns
        for lt in LESION_TYPES:
            if (lt not in patches_descr.columns) and (lt != 'normal'):
                patches_descr[lt] = 0

        # Filter patches with ONLY labels to ignore
        if self.ignore_diameter_px is not None:
            patches_with_labels = np.unique(np.where(mask_patches > 0)[0])
            patches_to_ignore = np.unique(np.where(mask_patches < 0)[0])
            patch_idx_to_ignore = [
                idx for idx in patches_to_ignore if idx not in patches_with_labels]

        # Get the percentage of breast in the roi. If the muscle mask is used this considers
        # brest the non-muscle region inside the breast
        if self.use_muscle_mask:
            muscle_mask = padd_image(muscle_mask, self.patch_size)
            breast_patches = slice_image(
                np.where(muscle_mask == 0, image, 0),
                window_size=self.patch_size, stride=self.stride)
            breast_pixels = np.array([(roi != 0).sum() for roi in breast_patches])
            del breast_patches
        else:
            breast_pixels = np.array([(roi != 0).sum() for roi in image_patches])
        patches_descr['breast_fraction'] = breast_pixels / \
            (image_patches.shape[1]*image_patches.shape[2])

        # Filter Rois with more background than breast or just bkgrd
        keep_idx = \
            patches_descr.loc[patches_descr.breast_fraction >= self.min_breast_frac].index.tolist()
        if self.ignore_diameter_px is not None:
            keep_idx = [idx for idx in keep_idx if idx not in patch_idx_to_ignore]
        patches_descr = patches_descr.iloc[keep_idx]
        image_patches = image_patches[keep_idx, :, :]
        mask_patches = mask_patches[keep_idx, :, :]
        patches_descr.reset_index(inplace=True, drop=True)

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
            if mask_patches[roi_idx, :, :].any() and not save_lesions:
                # This condition is for the usage of this method in the
                # centered_patch_cropping were not normal cases need not to be saved
                patch_filenames.append('roi_not_saved')
                patch_mask_filenames.append('empty_mask')
                continue
            roi_name = f'{img_id}_roi_{roi_idx}.png'
            patch_filenames.append(f'{img_id}/{roi_name}')
            temp = image_patches[roi_idx, :, :]
            (self.patch_img_path/str(img_id)).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.patch_img_path/str(img_id)/roi_name), temp)

            if mask_patches[roi_idx, :, :].any():  # Empty images cannot be stored
                roi_mask_name = f'{img_id}_roi_{roi_idx}_mask.png'
                patch_mask_filenames.append(f'{img_id}/{roi_mask_name}')
                (self.patch_mask_path/str(img_id)).mkdir(parents=True, exist_ok=True)
                patch_mask_filename = str(self.patch_mask_path/str(img_id)/roi_mask_name)
                cv2.imwrite(patch_mask_filename, mask_patches[roi_idx, :, :])
            else:
                patch_mask_filenames.append('empty_mask')

        # complete dataframe
        patches_descr['filename'] = patch_filenames
        patches_descr['partition'] = partition
        patches_descr['mask_filename'] = patch_mask_filenames
        for column in ['case_id', 'img_id', 'side', 'view', 'acr', 'birads']:
            patches_descr[column] = self.img_df[column].iloc[idx]

        # Add the same label as expected in the constructor
        patches_descr['label'] = False
        for les_type in LESION_TYPES:
            if les_type == 'normal':
                continue
            patches_descr['label'] = \
                patches_descr['label'] | np.where(patches_descr[les_type] != 0, True, False)
            patches_descr[les_type] = np.where(patches_descr[les_type] != 0, True, False)
        patches_descr['label'] = np.where(patches_descr['label'], 'abnormal', 'normal')
        return patches_descr

    def centered_patches_extraction(self):
        """
        Extracts rois of a fixed size centered in each lesion.
        The processing is done in parallel to make it faster.
        """
        # Get lesion patches
        n_rows = self.img_df.shape[0]
        res = []
        logging.info('Extracting centered lesion patches...\n')
        # for i in tqdm(range(n_rows), total=n_rows):            # Kept for easy debbuging
        #     res.append(self.extract_centered_patches_from_image(i))
        with mp.Pool(self.n_jobs) as pool:
            for result in tqdm(
                pool.imap(self.extract_centered_patches_from_image, range(n_rows)),
                total=n_rows
            ):
                res.append(result)
        patches_df = pd.concat(res, ignore_index=True)

        # Get normal patches
        logging.info('Extracting normal patches...\n')
        res = []
        # for i in tqdm(range(n_rows), total=n_rows):            # Kept for easy debbuging
        #     res.append(self.extract_patches_from_image(i, save_lesions=False))
        partial_func = partial(self.extract_patches_from_image, save_lesions=False)
        with mp.Pool(self.n_jobs) as pool:
            for result in tqdm(
                pool.imap(partial_func, range(n_rows)),
                total=n_rows
            ):
                res.append(result)
        normal_patches_df = pd.concat(res, ignore_index=True)
        normal_patches_df = normal_patches_df.loc[normal_patches_df.label == 'normal']

        # Concatenate dfs
        patches_df = pd.concat([patches_df, normal_patches_df], ignore_index=True)
        return patches_df.sort_values(by='img_id')

    def extract_centered_patches_from_image(self, idx: int):
        """
        Extracts the rois from an image using a bbox centered at each lesion.
        Args:
            idx (int): index of the row to read in the images dataframe
        Returns:
            (pd.DataFrame): patches_descr describing each ROI.
        """
        # Get the columns of the returned df
        columns_of_interest = ['case_id', 'img_id', 'side', 'view', 'acr', 'birads', 'partition']
        column_names = [lt for lt in LESION_TYPES if lt != 'normal'] + \
            ['breast_fraction', 'patch_bbox', 'filename', 'mask_filename'] + \
            columns_of_interest + ['label']

        img_id = self.img_df['img_id'].iloc[idx]
        rois_subset_df = self.rois_df.loc[self.rois_df.img_id == img_id]
        # If no rois in the image (normal cases)
        if rois_subset_df.shape[0] == 0:
            patches_descr = pd.DataFrame(columns=column_names)
            return patches_descr

        # Read images pngs
        filename = Path(self.img_df['filename'].iloc[idx]).name
        img_path = self.full_img_path / filename
        image = cv2.imread(str(img_path), cv2.IMREAD_ANYDEPTH)
        image_size = image.shape
        mask_path = self.full_mask_path / f'{img_id}_lesion_mask.png'
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(image.shape)

        if self.use_muscle_mask:
            muscle_mask_path = self.full_muscle_mask_path / f'{img_id}_pectoral_muscle_mask.png'
            if muscle_mask_path.exists():
                muscle_mask = cv2.imread(str(muscle_mask_path), cv2.IMREAD_GRAYSCALE)
            else:
                muscle_mask = np.zeros(image.shape)
        side = self.img_df['side'].iloc[idx]
        if side == 'R':
            image = cv2.flip(image, 1)
            if mask.any():
                mask = cv2.flip(mask, 1)
            if self.use_muscle_mask and muscle_mask.any():
                muscle_mask = cv2.flip(muscle_mask, 1)

        # Filter rois already filtered from the rois_df
        present_indx = self.rois_df.loc[self.rois_df.img_id == img_id, 'index_in_image'].values
        indexes_to_filter = \
            [index for index in np.unique(mask) if index not in present_indx]
        for index in indexes_to_filter:
            mask = np.where(mask == index, 0, mask)

        # Identify labels to ignore, and relabel them with -1 index
        if self.ignore_diameter_px is not None:
            indexes_to_ignore = self.rois_df.loc[
                (self.rois_df.img_id == img_id) & (self.rois_df.lesion_type == 'ignored_lesion'),
                'index_in_image'].values
            if len(indexes_to_ignore) != 0:
                for index in indexes_to_ignore:
                    mask = np.where(mask == index, 0, mask)

        # Accumulator for the output df
        patches_descr = []

        for k, (index, roi) in enumerate(rois_subset_df.iterrows()):
            if roi['lesion_type'] == 'ignored_lesion':
                continue
            patches_descr_row = []
            center_column_name = 'center_crop' if self.cropped_imgs else 'center'
            center = roi[center_column_name]
            if isinstance(center, str):
                roi_center = utils.load_point(center)
            else:
                roi_center = center

            # Get the coordinates of the patch centered in the lesion
            patch_x1, patch_x2, patch_y1, patch_y2, image, mask = \
                utils.patch_coordinates_from_center_w_padding(
                    roi_center, image_size, self.patch_size, image=image, mask=mask)

            # Crop the patch
            image_patch = image[patch_y1:patch_y2, patch_x1:patch_x2]
            mask_patch = mask[patch_y1:patch_y2, patch_x1:patch_x2]
            if self.use_muscle_mask:
                muscle_mask_patch = muscle_mask[patch_y1:patch_y2, patch_x1:patch_x2]
                muscle_mask_patch = np.where(muscle_mask_patch > 0, 0, 1)
                if muscle_mask.shape != image_patch.shape:
                    temp = muscle_mask_patch.copy()
                    muscle_mask_patch = np.zeros_like(image_patch)
                    muscle_mask_patch[0:temp.shape[0], 0:temp.shape[1]] = temp
                    del temp
                breast_fraction = (
                    (((image_patch * muscle_mask_patch) != 0).sum()) /
                    (image_patch.shape[0] * image_patch.shape[1]))
            else:
                breast_fraction = \
                    ((image_patch != 0).sum()) / (image_patch.shape[0] * image_patch.shape[1])
            if breast_fraction < self.min_breast_frac:
                continue

            # Save patches and masks
            patch_filename = f'{img_id}_les_patch_{k}.png'
            (self.patch_img_path/str(img_id)).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.patch_img_path/str(img_id)/patch_filename), image_patch)
            if mask_patch.any():  # Empty images cannot be stored
                (self.patch_mask_path/str(img_id)).mkdir(parents=True, exist_ok=True)
                patch_mask_name = f'{str(img_id)}/{img_id}_les_patch_{k}_mask.png'
                cv2.imwrite(str(self.patch_mask_path/patch_mask_name), mask_patch)
            else:
                patch_mask_name = 'empty_mask'

            # Complete row of the dataframe
            for lt in LESION_TYPES:
                if lt == 'normal':
                    continue
                if (lt == roi['lesion_type']):
                    patches_descr_row.append(True)
                else:
                    patches_descr_row.append(False)

            patch_bbox = np.array([[patch_x1, patch_y1], [patch_x2, patch_y2]])
            patches_descr_row.extend(
                [breast_fraction, patch_bbox, f'{img_id}/{patch_filename}', patch_mask_name]
            )
            for element in columns_of_interest:
                patches_descr_row.append(self.img_df[element].iloc[idx])
            patches_descr_row.append('abnormal')

            # Append row to the dataframe content
            patches_descr.append(patches_descr_row)
        return pd.DataFrame(data=patches_descr, columns=column_names)

    def get_normal_imgs_ids(self):
        return self.img_df.loc[self.img_df.img_label == 'normal', 'img_id'].unique()

    def get_free_of_selected_lesions_imgs_ids(self):
        img_ids_with_rois = self.rois_df.img_id.unique()
        return [img_id for img_id in self.img_df.img_id.unique() if img_id not in img_ids_with_rois]

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
        if self.ignore_diameter_px is not None:
            lesion_idxs_to_ignore = self.rois_df.loc[
                rois_from_img & (self.rois_df.lesion_type == 'ignored_lesion'),
                'index_in_image'].values

        if len(lesion_idxs) != 0:
            for les_idx in np.unique(mask):
                if les_idx not in lesion_idxs:
                    mask[mask == les_idx] = 0
                elif (self.ignore_diameter_px is not None) and (les_idx in lesion_idxs_to_ignore):
                    mask[mask == les_idx] = -1
        else:
            mask = np.zeros(mask.shape)
        return mask

    def flip_coordinates(self):
        """
        If the image is flipped, the coordinates of the points in the df should be flipped
        Specially the bbox, which also need to keep the the upper left
        bottom right convention.
        """
        if self.cropped_imgs:
            center_tag = 'center_crop'
            lesion_tag = 'lesion_bbox_crop'
            poit_tag = 'point_px_crop'
        else:
            center_tag = 'center'
            lesion_tag = 'lesion_bbox'
            poit_tag = 'point_px'
        for img_id in self.rois_df.img_id.unique():
            side = self.img_df.loc[self.img_df.img_id == img_id, 'side'].values[0]
            if side != 'R':
                continue
            if self.cropped_imgs:
                breast_bbox = self.img_df.loc[self.img_df.img_id == img_id, 'breast_bbox'].values[0]
                if isinstance(breast_bbox, str):
                    breast_bbox = utils.load_coords(
                        self.img_df.loc[self.img_df.img_id == img_id, 'breast_bbox'].values[0]
                    )
                breast_bbox_shape = (
                    breast_bbox[1][0] - breast_bbox[0][0],
                    breast_bbox[1][1] - breast_bbox[0][1]
                )
            else:
                image_size = utils.load_point(
                    self.img_df.loc[self.img_df.img_id == img_id, 'img_size'].values[0]
                )
                breast_bbox_shape = (image_size[1], image_size[0])

            centers = self.rois_df.loc[self.rois_df.img_id == img_id, center_tag].tolist()
            if isinstance(centers[0], str):
                centers = [np.array(utils.load_point(point)) for point in centers]
            else:
                centers = [np.array(point) for point in centers]
            for k, center in enumerate(centers):
                centers[k][0] = breast_bbox_shape[0] - center[0]
            centers = [tuple(center) for center in centers]

            lesion_bboxs_crop = \
                self.rois_df.loc[self.rois_df.img_id == img_id, lesion_tag].tolist()
            for k, lesion_bbox_crop in enumerate(lesion_bboxs_crop):
                if isinstance(lesion_bbox_crop, str):
                    lesion_bbox_crop = utils.load_coords(lesion_bbox_crop)
                lesion_bbox_crop = [
                    (breast_bbox_shape[0] - point[0], point[1]) for point in lesion_bbox_crop
                ]
                lesion_bboxs_crop[k] = [
                    (lesion_bbox_crop[1][0], lesion_bbox_crop[0][1]),
                    (lesion_bbox_crop[0][0], lesion_bbox_crop[1][1]),
                ]

            point_pxs_crop = self.rois_df.loc[self.rois_df.img_id == img_id, poit_tag].values
            for k, point_px_crop in enumerate(point_pxs_crop):
                if isinstance(point_px_crop, str):
                    point_px_crop = utils.load_coords(point_px_crop)
                point_pxs_crop[k] = [
                    (breast_bbox_shape[0] - point[0], point[1]) for point in point_px_crop
                ]

            for k, row in enumerate(self.rois_df.loc[self.rois_df.img_id == img_id].iterrows()):
                self.rois_df.at[row[0], center_tag] = centers[k]
                self.rois_df.at[row[0], lesion_tag] = lesion_bboxs_crop[k]
                self.rois_df.at[row[0], poit_tag] = point_pxs_crop[k]

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
        sample['img_id'] = img_id
        if side == 'R' and self.level == 'image':
            img = cv2.flip(img, 1)
        sample['img'] = img

        # Return bboxes coords for det CNNs and metrics
        if self.level == 'image':
            rois_from_img = \
                (self.rois_df.img_id == img_id) & (self.rois_df.lesion_type != 'ignored_lesion')
            lesion_tag = 'lesion_bbox_crop' if self.cropped_imgs else 'lesion_bbox'
            bboxes_coords = self.rois_df.loc[rois_from_img, lesion_tag].values
            sample["lesion_bboxes"] = [
                utils.load_coords(bbox) if isinstance(bbox, str)
                else bbox for bbox in bboxes_coords
            ]
            sample['radiuses'] = self.rois_df.loc[rois_from_img, 'radius'].values
            if self.ignore_diameter_px is not None:
                ignored_rois_from_img = \
                    (self.rois_df.img_id == img_id) & (self.rois_df.lesion_type == 'ignored_lesion')
                bboxes_coords = self.rois_df.loc[ignored_rois_from_img, lesion_tag].values
                sample["ignored_lesion_bboxes"] = [
                    utils.load_coords(bbox) if isinstance(bbox, str)
                    else bbox for bbox in bboxes_coords
                ]
                sample['ignored_lesion_radiuses'] = self.rois_df.loc[
                    ignored_rois_from_img, 'radius'].values
        else:
            patch_bbox = self.df['patch_bbox'].iloc[idx]
            if isinstance(patch_bbox, str):
                patch_bbox = utils.load_patch_coords(patch_bbox)
            sample["patch_bbox"] = patch_bbox

        # Load lesion mask
        if self.lesions_mask:
            if self.level == 'image':
                mask_path = \
                    self.full_mask_path / f'{self.df["img_id"].iloc[idx]}_lesion_mask.png'
                if not mask_path.exists():
                    mask = np.zeros(img.shape)
                else:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_ANYDEPTH)
                    mask = mask.astype(np.int16)
                    mask = self.adjust_mask_to_selected_lesions(mask, idx)
                    if side == 'R':
                        mask = cv2.flip(mask, 1)
            else:
                mask_filename = self.df['mask_filename'].iloc[idx]
                if mask_filename != 'empty_mask':
                    mask_filename = self.patch_mask_path / mask_filename
                    mask = cv2.imread(str(mask_filename), cv2.IMREAD_ANYDEPTH)
                    sample["lesion_bboxes"] = np.asarray(utils.get_bbox_of_lesions_in_patch(mask))
                    sample["ignored_lesion_bboxes"] = np.asarray(utils.get_bbox_of_lesions_in_patch(
                        mask, ignored_lesions=True))
                    sample['lesion_centers'] = \
                        [utils.get_center_bbox(bbox) for bbox in sample["lesion_bboxes"]]
                    sample['ignored_lesion_centers'] = \
                        [utils.get_center_bbox(bbox) for bbox in sample["ignored_lesion_bboxes"]]
                else:
                    mask = np.zeros(img.shape)
                    sample["lesion_bboxes"] = []
                    sample["ignored_lesion_bboxes"] = []
                    sample['lesion_centers'] = []
                    sample['ignored_lesion_centers'] = []

            # Consider the cases with lesions inside lesions
            holes = mask.astype('float32').copy()
            cv2.floodFill(holes, None, (0, 0), newVal=1)
            holes = np.where(holes == 0, 255, 0)
            sample['lesion_mask'] = mask + holes.astype('uint8')

        if self.use_muscle_mask:
            if self.level == 'image':
                muscle_mask_path = self.full_muscle_mask_path / \
                    f'{self.df["img_id"].iloc[idx]}_pectoral_muscle_mask.png'
                if not muscle_mask_path.exists():
                    muscle_mask = np.zeros(img.shape)
                else:
                    muscle_mask = cv2.imread(str(muscle_mask_path), cv2.IMREAD_GRAYSCALE)
                    if side == 'R':
                        muscle_mask = cv2.flip(muscle_mask, 1)
                sample['muscle_mask'] = muscle_mask

        sample['side'] = side
        return sample
