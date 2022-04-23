import cv2
import numpy as np
import pandas as pd


class ImgDataset:
    def __init__(
        self, pos_df: pd.DataFrame, neg_df: pd.DataFrame,
        kr: float = 5, normalize: bool = False, seed: int = 42
    ):
        self.pos_df = pos_df
        self.neg_df = neg_df
        self.kr = kr
        self.nomalize = normalize
        self.seed = seed

        self.train_discarded_files = []
        self.val_discarded_files = []

        self.train_pos_cases_list = self.pos_df['filename'].sample(
            frac=0.5, replace=False, random_state=self.seed
        )
        self.train_pos_cases_list = self.train_pos_cases_list.tolist()
        condition = ~self.pos_df.filename.isin(self.train_pos_cases_list)
        self.val_pos_cases_list = self.pos_df.loc[condition, 'filename'].tolist()

        self.n_neg = len(self.train_pos_cases_list) * self.kr
        self.used_neg_files = []

    def open_img(self, filename: str):
        # filename = '/home/jseia/Desktop/ml-dl/data_rois/'+filename
        return cv2.imread(filename, cv2.IMREAD_ANYDEPTH)

    def open_img_normalize(self, filename: str, mean: float, std: float):
        # TODO: add defaults mean and std
        return (cv2.imread(filename, cv2.IMREAD_ANYDEPTH) - mean) / std

    def load_imgs(self, pos_img_files: list, neg_img_files: list):
        # Imgs/features # TODO: Define if images or features
        xs = []
        # TODO: parallelize dataloading with multithread
        if self.normalize:
            xs.extend([self.open_img_normalize(f) for f in pos_img_files])
            xs.extend([self.open_img_normalize(f) for f in neg_img_files])
        else:
            xs.extend([self.open_img(f) for f in pos_img_files])
            xs.extend([self.open_img(f) for f in neg_img_files])

        # Labels
        ys = np.zeros((len(pos_img_files)+len(neg_img_files)))
        ys[:len(pos_img_files)] = 1
        return np.array(xs), ys

    def get_train_batch(self, discarded_files: list = None):
        # Define the number of files to sample
        n_sample = self.n_neg if discarded_files is None else len(discarded_files)

        # Discard rejected files
        self.train_neg_cases_list = \
            [fn for fn in self.train_neg_cases_list if fn not in discarded_files]

        # Define the cases from which to sample and sample
        condition = ~self.neg_df.filename.isin(self.used_neg_files),
        sample = self.neg_df.loc[condition, 'filename'].sample(
            n=n_sample, replace=False, random_state=self.seed
        )
        sample = sample.tolist()
        self.train_neg_cases_list.extend(sample)
        train_files_list = self.train_pos_cases_list + self.train_neg_cases_list

        # Keep track of history
        self.used_neg_files.extend(sample)

        # Load the images/features # TODO: Define if images or features
        xs, ys = self.load_imgs(self.train_pos_cases_list, self.train_neg_cases_list)

        return train_files_list, xs, ys

    def get_val_batch(self, discarded_files: list = None):
        # Define the number of files to sample
        n_sample = self.n_neg if discarded_files is None else len(discarded_files)

        # Discard rejected files
        self.val_neg_cases_list = \
            [fn for fn in self.val_neg_cases_list if fn not in discarded_files]

        # Define the cases from which to sample and sample
        condition = ~self.neg_df.filename.isin(self.used_neg_files),
        sample = self.neg_df.loc[condition, 'filename'].sample(
            n=n_sample, replace=False, random_state=self.seed
        )
        sample = sample.tolist()
        self.val_neg_cases_list.extend(sample)
        val_files_list = self.val_pos_cases_list + self.val_neg_cases_list

        # Keep track of history
        self.used_neg_files.extend(sample)

        # Load the images/features  # TODO: Define if images or features
        xs, ys = self.load_imgs(self.val_pos_cases_list, self.val_neg_cases_list)

        return val_files_list, xs, ys


class FeaturesDataset:
    def __init__(
        self, pos_df: pd.DataFrame, neg_df: pd.DataFrame,
        kr: float = 5, normalize: bool = False, seed: int = 42
    ):
        self.pos_df = pos_df
        self.neg_df = neg_df
        self.kr = kr
        self.nomalize = normalize
        self.seed = seed

        self.train_discarded_files = []
        self.val_discarded_files = []

        self.train_pos_cases_list = self.pos_df['filename'].sample(
            frac=0.5, replace=False, random_state=self.seed
        )
        self.train_pos_cases_list = self.train_pos_cases_list.tolist()
        condition = ~self.pos_df.filename.isin(self.train_pos_cases_list)
        self.val_pos_cases_list = self.pos_df.loc[condition, 'filename'].tolist()

        self.n_neg = len(self.train_pos_cases_list) * self.kr
        self.used_neg_files = []