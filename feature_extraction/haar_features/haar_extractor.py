import sys; sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import general_utils.utils as utils
import feature_extraction.haar_features.haar_modules as hm
import numpy as np
import typing as tp
from dask import delayed
from skimage.feature import haar_like_feature


WINDOW_SIZE = 14
Size = tp.NamedTuple('Size', [('height', int), ('width', int)])
RotSize = tp.NamedTuple('RotSize', [('dx', int), ('dy', int), ('z', int)])
Location = tp.NamedTuple('Location', [('top', int), ('left', int)])


def possible_position(size: int, window_size: int = WINDOW_SIZE) -> tp.Iterable[int]:
    return range(0, window_size - size + 1)


def possible_locations(base_shape: Size, window_size: int = WINDOW_SIZE) -> tp.Iterable[Location]:
    return (Location(left=x, top=y)
            for x in possible_position(base_shape.width, window_size)
            for y in possible_position(base_shape.height, window_size))


def possible_shapes(base_shape: Size, window_size: int = WINDOW_SIZE) -> tp.Iterable[Size]:
    base_height = base_shape.height
    base_width = base_shape.width
    return (Size(height=height, width=width)
            for width in range(base_width, window_size + 1, base_width)
            for height in range(base_height, window_size + 1, base_height))


def possible_locations_rot(
    base_shape: RotSize, window_size: int = WINDOW_SIZE
) -> tp.Iterable[Location]:
    """Gets all the possible tl coordinates of the features in the window"""
    return (Location(left=x, top=y)
            for x in range(0, window_size + 1)
            for y in range(0, window_size + 1))


def possible_shapes_rot(window_size: int = WINDOW_SIZE) -> tp.Iterable[RotSize]:
    """Gets all the possible combinations of shearing fitting inside the window"""
    z = 14
    return (RotSize(dx=dx, dy=dy, z=z)
            for dx in range(1, z + 1)
            for dy in range(1, z + 1))


def all_possible_variations(feat: hm.Feature, window_size: int):
    """Instantiates all the possible features fitting in the window size"""
    return (feat(location.left, location.top, shape.dx, shape.dy, shape.z, window_size)
            for shape in possible_shapes_rot(window_size)
            for location in possible_locations_rot(shape, window_size))


def feature_instantiator(
    window_size: int = WINDOW_SIZE, mode: str = 'all',
    horizontal_feature_types: list = None, rotated_feature_types: list = None
):
    features = []
    if mode != 'rot':
        if horizontal_feature_types is None:
            horizontal_feature_types = [
                (hm.Feature2h, 1, 2), (hm.Feature2v, 2, 1), (hm.Feature3h, 1, 3),
                (hm.Feature3v, 3, 1), (hm.Feature4h, 1, 4), (hm.Feature4v, 4, 1),
                (hm.Feature2h2v, 2, 2), (hm.Feature3h3v, 3, 3)
            ]
        features = []
        for feat in horizontal_feature_types:
            features.extend(
                list(feat[0](location.left, location.top, shape.width, shape.height)
                     for shape in possible_shapes(Size(height=feat[1], width=feat[2]), window_size)
                     for location in possible_locations(shape, window_size))
            )
    features_rot = []
    if mode != 'hor':
        if rotated_feature_types is None:
            rotated_feature_types = [
                hm.Feature2hRot, hm.Feature2vRot, hm.Feature3hRot, hm.Feature3vRot,
                hm.Feature4hRot, hm.Feature4vRot, hm.Feature2h2vRot, hm.Feature3h3vRot
            ]
        features_rot = []
        for feat in rotated_feature_types:
            # Keep only the plaussible features from all the candidate ones inside the window
            features_rot.extend(
                [feat for feat in all_possible_variations(feat, window_size) if feat.plausible])

    return features + features_rot


@delayed
def extract_haar_feature_image_skimage(img, feature_type=None, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = utils.integral_img(img)[1:, 1:]
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)


class HaarFeatureExtractor:
    def __init__(
        self, patch_size: int = WINDOW_SIZE,
        horizontal: bool = True, rot: bool = True,
        horizontal_feature_types: list = None, rotated_feature_types: list = None,
        horizontal_features_selection: list = None, rotated_features_selection: list = None
    ):
        self.hor = horizontal
        self.rot = rot
        if horizontal:
            if horizontal_features_selection is not None:
                self.features_h = horizontal_features_selection
            else:
                self.features_h = \
                    feature_instantiator(patch_size, 'hor', horizontal_feature_types)
        else:
            self.features_h = []
        if rot:
            if rotated_features_selection is not None:
                self.features_r = rotated_features_selection
            else:
                self.features_r = feature_instantiator(
                    patch_size, 'rot', rotated_feature_types=rotated_feature_types)
        else:
            self.features_r = []
        self.patch_size = patch_size

    def extract_features(
        self, image: np.ndarray, locations: np.ndarray,
        integral_image: np.ndarray = None, diagintegral_image: np.ndarray = None
    ):
        # Get the integral image
        if integral_image is None:
            self.integral_image = utils.integral_img(image)
            self.diagintegral_image = utils.diagonal_integral_img(image)
        else:
            self.integral_image = integral_image
            self.diagintegral_image = diagintegral_image

        image_features = np.empty((len(locations), len(self.features_h)+len(self.features_r)))

        for j, location in enumerate(locations):   # tqdm(, total=len(locations)):
            # Get the patch arround center
            x1, _, y1, _ = utils.patch_coordinates_from_center(
                center=(location[0], location[1]), image_shape=image.shape,
                patch_size=self.patch_size)

            # # Generate the horizontal features
            features_values = np.empty(len(self.features_h) + len(self.features_r))
            if self.hor:
                for i, feature in enumerate(self.features_h):
                    points_coords_y = (y1-1) + feature.coords_y
                    points_coords_x = (x1-1) + feature.coords_x
                    features_values[i] = np.dot(
                        self.integral_image[points_coords_y, points_coords_x], feature.coeffs)

            # Generate the rotated features
            if self.rot:
                for i, feature in enumerate(self.features_r):
                    points_coords_y = (y1-1) + feature.coords_y
                    points_coords_x = (x1-1) + feature.coords_x
                    features_values[i] = np.dot(
                        self.diagintegral_image[points_coords_y, points_coords_x], feature.coeffs)
            image_features[j, :] = features_values
        return image_features

    def extract_features_from_crop(self, img):
        if self.hor:
            self.integral_image = utils.integral_img(img)
        if self.rot:
            self.diagintegral_image = utils.diagonal_integral_img(img)

        h_features_values = np.empty(len(self.features_h))
        if self.hor:
            for i, feature in enumerate(self.features_h):
                h_features_values[i] = np.dot(
                    self.integral_image[feature.coords_y, feature.coords_x],
                    feature.coeffs)

        # Generate the rotated features
        r_features_values = np.empty(len(self.features_r))
        if self.rot:
            for i, feature in enumerate(self.features_r):
                r_features_values[i] = np.dot(
                    self.diagintegral_image[feature.coords_y, feature.coords_x],
                    feature.coeffs)
        return np.concatenate([h_features_values, r_features_values])
