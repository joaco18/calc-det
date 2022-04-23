from numba import njit
import numpy as np
import typing as tp
import feature_modules as fm


WINDOW_SIZE = 14
Size = tp.NamedTuple('Size', [('height', int), ('width', int)])
RotSize = tp.NamedTuple('RotSize', [('dx', int), ('dy', int), ('z', int)])
Location = tp.NamedTuple('Location', [('top', int), ('left', int)])


@njit
def to_integral(img_arr):
    shape = img_arr.shape
    row_sum = np.zeros(shape)
    int_img = np.zeros((shape[0] + 1, shape[1] + 1))
    for x in range(shape[1]):
        for y in range(shape[0]):
            row_sum[y, x] = row_sum[y-1, x] + img_arr[y, x]
            int_img[y+1, x+1] = int_img[y+1, x-1+1] + row_sum[y, x]
    return int_img  #.astype(np.int8)


@njit
def to_diag_integral(img_arr):
    shape = img_arr.shape
    diag_int_img = np.zeros((shape[0] + 3, shape[1] + 3))
    img_arr_ = np.zeros((shape[0] + 1, shape[1] + 2))
    img_arr_[1:, 1:-1] = img_arr
    for y in range(shape[0]):
        for x in range(img_arr_.shape[1]-1):
            diag_int_img[y+2, x+2] = \
                diag_int_img[y+1, x+1] + diag_int_img[y+1, x+3] - \
                diag_int_img[y, x+2] + img_arr_[y+1, x+1] + \
                img_arr_[y, x+1]
            diag_int_img[y+2, 1] = diag_int_img[y+1, 2]
            diag_int_img[y+2, -1] = diag_int_img[y+1, -2]
    return diag_int_img[1:-1, 1:-1]  #.astype(np.int8)


def possible_position(
    size: int, window_size: int = WINDOW_SIZE
) -> tp.Iterable[int]:
    return range(0, window_size - size + 1)


def possible_locations(
    base_shape: Size, window_size: int = WINDOW_SIZE
) -> tp.Iterable[Location]:
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
    return (Location(left=x, top=y)
            for x in possible_position(base_shape.z, window_size)
            for y in possible_position(base_shape.z, window_size))


def possible_shapes_rot(
    base_shape: Size, window_size: int = WINDOW_SIZE
) -> tp.Iterable[RotSize]:
    base_z = base_shape.height
    return (RotSize(dx=dx, dy=dy, z=z)
            for z in range(base_z, window_size + 1, base_z)
            for dx in range(1, z)
            for dy in range(1, z))


def feature_instantiator(window_size: int = WINDOW_SIZE, mode: str = 'all'):
    features = []
    if mode != 'rot':
        features_types = [
            (fm.Feature2h, 1, 2), (fm.Feature2v, 2, 1), (fm.Feature3h, 1, 3),
            (fm.Feature3v, 3, 1), (fm.Feature4h, 1, 4), (fm.Feature4v, 4, 1),
            (fm.Feature2h2v, 2, 2), (fm.Feature3h3v, 3, 3)
        ]
        features = []
        for feat in features_types:
            features.extend(
                list(feat[0](location.left, location.top, shape.width, shape.height)
                     for shape in possible_shapes(Size(height=feat[1], width=feat[2]), window_size)
                     for location in possible_locations(shape, window_size))
            )
    features_rot = []
    if mode != 'hor':
        def base_iterator(feat: fm.FeatureRot):
            return (feat(location.left, location.top, shape.dx, shape.dy, shape.z)
                    for shape in possible_shapes_rot(Size(height=1, width=1), window_size)
                    for location in possible_locations_rot(shape, window_size))

        rot_features_types = [
            fm.Feature2hRot, fm.Feature2vRot, fm.Feature3hRot, fm.Feature3vRot,
            fm.Feature4hRot, fm.Feature4vRot, fm.Feature2h2vRot, fm.Feature3h3vRot
        ]
        features_rot = []
        for feat in rot_features_types:
            feature_rot = (map(lambda feat: feat if feat.plausible else None, base_iterator(feat)))
            features_rot.extend([feat for feat in feature_rot if feat is not None])

    return features + features_rot


def main():
    print(len(feature_instantiator(14)))


if __name__ == '__main__':
    main()
