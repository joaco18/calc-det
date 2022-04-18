from numba import jit
import numpy as np


@jit(nopython=True)
def to_integral(img_arr):
    shape = img_arr.shape
    row_sum = np.zeros(shape)
    int_img = np.zeros((shape[0] + 1, shape[1] + 1))
    for x in range(shape[1]):
        for y in range(shape[0]):
            row_sum[y, x] = row_sum[y-1, x] + img_arr[y, x]
            int_img[y+1, x+1] = int_img[y+1, x-1+1] + row_sum[y, x]
    return int_img.astype(np.int8)


@jit(nopython=True)
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
    # diag_int_img [:, 1] = 0
    # diag_int_img [:, -1] = 0
    return diag_int_img[:-1, 1:].astype(np.int8)



class Box:
    # Class `Box` that allows to determine the integral of an image region by
    # means of the integral image.
    def __init__(self, x: int, y: int, width: int, height: int):
        self.coords_x = [x, x + width, x,          x + width]
        self.coords_y = [y, y,         y + height, y + height]
        self.coeffs = [1, -1,        -1,         1]

    def __call__(self, integral_image: np.ndarray) -> float:
        return np.dot(integral_image[self.coords_y, self.coords_x], self.coeffs)


class Feature:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __call__(self, integral_image: np.ndarray) -> float:
        try:
            return np.dot(integral_image[self.coords_y, self.coords_x], self.coeffs)
        except IndexError as e:
            raise IndexError(str(e) + ' in ' + str(self))

    def __repr__(self):
        return f'{self.__class__.__name__}(x={self.x}, ' \
            f'y={self.y}, width={self.width}, height={self.height})'


class Feature2h(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        hw = width // 2
        self.coords_x = [x,      x + hw,     x,          x + hw,
                         x + hw, x + width,  x + hw,     x + width]
        self.coords_y = [y,      y,          y + height, y + height,
                         y,      y,          y + height, y + height]
        self.coeffs = [1,     -1,         -1,          1,
                       -1,     1,          1,         -1]


class Feature2v(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        hh = height // 2
        self.coords_x = [x,      x + width,  x,          x + width,
                         x,      x + width,  x,          x + width]
        self.coords_y = [y,      y,          y + hh,     y + hh,
                         y + hh, y + hh,     y + height, y + height]
        self.coeffs = [1,     -1,         -1,          1,
                       -1,     1,          1,         -1]


class Feature3h(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        tw = width // 3
        self.coords_x = [x,        x + tw,    x,          x + tw,
                         x + tw,   x + 2*tw,  x + tw,     x + 2*tw,
                         x + 2*tw, x + width, x + 2*tw,   x + width]
        self.coords_y = [y,        y,         y + height, y + height,
                         y,        y,         y + height, y + height,
                         y,        y,         y + height, y + height]
        self.coeffs = [1,      -1,        -1,          1,
                       -1,       1,         1,         -1,
                       1,      -1,        -1,          1]


class Feature3v(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        th = height // 3
        self.coords_x = [x,        x + width,  x,          x + width,
                         x,        x + width,  x,          x + width,
                         x,        x + width,  x,          x + width]
        self.coords_y = [y,        y,          y + th,     y + th,
                         y + th,   y + th,     y + 2*th,   y + 2*th,
                         y + 2*th, y + 2*th,   y + height, y + height]
        self.coeffs = [1,       -1,        -1,          1,
                       -1,        1,         1,         -1,
                       1,       -1,        -1,          1]


class Feature4h(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        tw = width // 4
        self.coords_x = [x,        x + tw,    x,          x + tw,
                         x + tw,   x + 3*tw,  x + tw,     x + 3*tw,
                         x + 3*tw, x + width, x + 3*tw,   x + width]
        self.coords_y = [y,        y,         y + height, y + height,
                         y,        y,         y + height, y + height,
                         y,        y,         y + height, y + height]
        self.coeffs = [1,      -1,        -1,          1,
                       -1,       1,         1,         -1,
                       1,      -1,        -1,          1]


class Feature4v(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        th = height // 4
        self.coords_x = [x,        x + width,  x,          x + width,
                         x,        x + width,  x,          x + width,
                         x,        x + width,  x,          x + width]
        self.coords_y = [y,        y,          y + th,     y + th,
                         y + th,   y + th,     y + 3*th,   y + 3*th,
                         y + 3*th, y + 3*th,   y + height, y + height]
        self.coeffs = [1,       -1,        -1,          1,
                       -1,        1,         1,         -1,
                       1,       -1,        -1,          1]


class FeatureChecker(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        hw = width // 2
        hh = height // 2
        self.coords_x = [x,      x + hw,     x,          x + hw,
                         x + hw, x + width,  x + hw,     x + width,
                         x,      x + hw,     x,          x + hw,
                         x + hw, x + width,  x + hw,     x + width]
        self.coords_y = [y,      y,          y + hh,     y + hh,
                         y,      y,          y + hh,     y + hh,
                         y + hh, y + hh,     y + height, y + height,
                         y + hh, y + hh,     y + height, y + height]
        self.coeffs = [1,     -1,         -1,          1,
                       -1,     1,          1,         -1,
                       -1,     1,          1,         -1,
                       1,    -1,         -1,          1]


class FeatureCenter(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        tw = width // 3
        th = height // 3

        self.coords_x = [x,        x + width,  x,          x + width,
                         x,        x + tw,     x,          x + tw,
                         x + tw,   x + 2 * tw, x + tw,     x + 2 * tw,
                         x + 2*tw, x + width,  x + 2*tw,   x + width,
                         x,        x + width,  x,          x + width]
        self.coords_y = [y,        y,          y + th,     y + th,
                         y + th,   y + th,     y + 2*th,   y + 2*th,
                         y + th,   y + th,     y + 2*th,   y + 2*th,
                         y + th,   y + th,     y + 2*th,   y + 2*th,
                         y + 2*th, y + 2*th,   y + height, y + height]
        self.coeffs = [1,     -1,        -1,        1,
                       1,     -1,        -1,        1,
                       -1,      1,         1,       -1,
                       1,     -1,        -1,        1,
                       1,     -1,        -1,        1]


class Diamond:
    # Class `Diamond` that allows to determine the integral of an image region by
    # means of the integral image.
    def __init__(self, x: int, y: int, z: int, r: float):
        w = int(z * r)  # right half
        x = x + 1
        y = y + 2
        self.coords_x = np.array([x + w,      x,              x + z,      x + (z - w)])
        self.coords_y = np.array([y + z - 1,  y + z - w - 1,   y + w - 1,  y - 1])
        self.coords_x[[1, 3]] = self.coords_x[[1, 3]] - 1
        self.coords_y[[1, 3]] = self.coords_y[[1, 3]] - 1
        self.coeffs = [1, -1, -1, 1]

    def __call__(self, integral_image: np.ndarray) -> float:
        # print(integral_image[self.coords_y, self.coords_x])
        return np.dot(integral_image[self.coords_y, self.coords_x], self.coeffs)


class FeatureRot:
    def __init__(self, x: int, y: int, z: int, r: float):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        # w = int(z * self.r)
        # Default feature is Diamond
        # x = x + 1
        # y = y + 2
        # self.coords_x = np.array([x + w,      x ,              x + z,      x + (z - w)])
        # self.coords_y = np.array([y + z - 1,  y + z - w - 1,   y + w - 1,  y - 1])
        # self.coords_x[[1, 3]] = self.coords_x[[1, 3]] - 1
        # self.coords_y[[1, 3]] = self.coords_y[[1, 3]] - 1
        # self.coeffs = [1, -1, -1, 1]

    def __call__(self, integral_image: np.ndarray) -> float:
        try:
            print(integral_image[self.coords_y, self.coords_x])
            return np.dot(integral_image[self.coords_y, self.coords_x], self.coeffs)
        except IndexError as e:
            raise IndexError(str(e) + ' in ' + str(self))

    def __repr__(self):
        return f'{self.__class__.__name__}(x={self.x}, ' \
            f'y={self.y}, side_size={self.z}, ratio={self.r})'
