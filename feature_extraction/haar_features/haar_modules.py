import numpy as np


class Box:
    # Class `Box` that allows to determine the integral of an image region by
    # means of the integral image.
    def __init__(self, x: int, y: int, width: int, height: int, ws: int):
        self.coords_x = np.asarray([x, x + width, x,          x + width])
        self.coords_y = np.asarray([y, y,         y + height, y + height])
        self.coeffs = np.asarray([1, -1,        -1,         1])

    def __call__(self, integral_image: np.ndarray) -> float:
        return np.dot(integral_image[self.coords_y, self.coords_x], self.coeffs)


class Feature:
    def __init__(
        self, x: int, y: int, width: int, height: int,
        z: int = None, ws: int = None
    ):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.z = z
        self.ws = ws

    def __call__(self, integral_image: np.ndarray) -> float:
        try:
            return np.dot(integral_image[self.coords_y, self.coords_x], self.coeffs)
        except IndexError as e:
            raise IndexError(str(e) + ' in ' + str(self))

    def __repr__(self):
        if self.z is None:
            return f'{self.__class__.__name__}(x={self.x}, ' \
                f'y={self.y}, width={self.width}, height={self.height})'
        return f'{self.__class__.__name__}(x={self.x}, ' \
            f'y={self.y}, dx={self.width}, dy={self.height}, z={self.z})'


class Feature2h(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        hw = width // 2
        self.type = 'Feature2h'
        self.coords_x = np.asarray(
            [x,      x + width,      x,           x + width,
             x + hw, x + width,      x + hw,     x + width])

        self.coords_y = np.asarray(
            [y,      y,          y + height, y + height,
             y,      y,          y + height, y + height])

        self.coeffs = np.asarray(
            [1,     -1,         -1,          1,
             -2,     2,          2,         -2])


class Feature2v(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        hh = height // 2
        self.type = 'Feature2v'
        self.coords_x = np.asarray(
            [x,      x + width,  x,              x + width,
             x,      x + width,  x,              x + width])

        self.coords_y = np.asarray(
            [y,      y,          y + height,     y + height,
             y + hh, y + hh,     y + height,     y + height])

        self.coeffs = np.asarray(
            [1,     -1,         -1,          1,
             -2,     2,          2,         -2])


class Feature3h(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        tw = width // 3
        self.type = 'Feature3h'
        self.coords_x = np.asarray(
            [x,        x + width,    x,          x + width,
             x + tw,   x + 2*tw,     x + tw,     x + 2*tw])

        self.coords_y = np.asarray(
            [y,        y,         y + height, y + height,
             y,        y,         y + height, y + height])

        self.coeffs = np.asarray(
            [-1,       1,         1,         -1,
             2,       -2,        -2,          2])


class Feature3v(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        th = height // 3
        self.type = 'Feature3v'
        self.coords_x = np.asarray(
            [x,        x + width,  x,          x + width,
             x,        x + width,  x,          x + width])

        self.coords_y = np.asarray(
            [y,        y,          y + height, y + height,
             y + th,   y + th,     y + 2*th,   y + 2*th])

        self.coeffs = np.asarray(
            [-1,        1,         1,         -1,
             2,        -2,        -2,          2])


class Feature4h(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        tw = width // 4
        self.type = 'Feature4h'
        self.coords_x = np.asarray(
            [x,        x + width,    x,          x + width,
             x + tw,   x + 3*tw,     x + tw,     x + 3*tw])

        self.coords_y = np.asarray(
            [y,        y,         y + height,    y + height,
             y,        y,         y + height,    y + height])

        self.coeffs = np.asarray(
            [-1,       1,         1,         -1,
             2,       -2,        -2,          2])


class Feature4v(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        th = height // 4
        self.type = 'Feature4v'
        self.coords_x = np.asarray(
            [x,        x + width,  x,          x + width,
             x,        x + width,  x,          x + width])

        self.coords_y = np.asarray(
            [y,        y,          y + height,     y + height,
             y + th,   y + th,     y + 3*th,   y + 3*th])

        self.coeffs = np.asarray(
            [-1,        1,         1,         -1,
             2,        -2,        -2,          2])


class Feature2h2v(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        hw = width // 2
        hh = height // 2
        self.type = 'Feature2h2v'
        self.coords_x = np.asarray(
            [x,         x + width,     x,             x + width,
             x + hw,    x + width,     x + hw,        x + width,
             x,         x + hw,        x,             x + hw])

        self.coords_y = np.asarray(
            [y,         y,             y + height,     y + height,
             y,         y,             y + hh,         y + hh,
             y + hh,    y + hh,        y + height,     y + height])

        self.coeffs = np.asarray(
            [1,      -1,       -1,        1,
             -2,      2,        2,       -2,
             -2,      2,        2,       -2])


class Feature3h3v(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        tw = width // 3
        th = height // 3
        self.type = 'Feature3h3v'
        self.coords_x = np.asarray(
            [x,        x + width,    x,          x + width,
             x + tw,   x + 2 * tw,   x + tw,     x + 2 * tw])

        self.coords_y = np.asarray(
            [y,        y,          y + height, y + height,
             y + th,   y + th,     y + 2*th,   y + 2*th])

        self.coeffs = np.asarray(
            [-1,      1,         1,       -1,
             2,     -2,        -2,        2])


class Diamond:
    # Class `Diamond` that allows to determine the integral of an image region by
    # means of the integral image.
    # p0, p1, p2, p3 --> according to https://docs.opencv.org/3.4/integral.png
    def __init__(self, x: int, y: int, width: int, height: int, z: int, ws: int):
        self.plausible = \
            ((x - height) >= 0) and ((x + width) < ws) and ((y + width + height) < ws)

        self.coords_x = np.asarray([x,  x + width,  x - height,  x + width - height])
        self.coords_y = np.asarray([y,  y + width,  y + height,  y + width + height])
        self.coeffs = np.asarray([1, -1, -1, 1])

    def __call__(self, integral_image: np.ndarray) -> float:
        return np.dot(integral_image[self.coords_y, self.coords_x], self.coeffs)


class Feature2hRot(Feature):
    def __init__(self, x: int, y: int, dx: int, dy: int, z: int, ws: int):
        super().__init__(x, y, dx, dy, z, ws)
        self.type = 'Feature2hRot'
        self.plausible = \
            ((x - dy) >= 0) and ((x + 2*dx) < ws) and ((y + 2*dx + dy) < ws)

        self.coords_x = np.asarray(
            [x,  x + dx*2,  x - dy,  x + dx*2 - dy,
             x,  x + dx,  x - dy,  x + dx - dy])

        self.coords_y = np.asarray(
            [y,  y + dx*2,  y + dy,  y + dx*2 + dy,
             y,  y + dx,  y + dy,  y + dx + dy])

        self.coeffs = np.asarray(
            [-1,      1,          1,         -1,
             2,      -2,         -2,         2])


class Feature2vRot(Feature):
    def __init__(self, x: int, y: int, dx: int, dy: int, z: int, ws: int):
        super().__init__(x, y, dx, dy, z, ws)
        self.type = 'Feature2vRot'
        self.plausible = \
            ((x - 2*dy) >= 0) and ((x + dx) < ws) and ((y + dx + 2*dy) < ws)

        self.coords_x = np.asarray(
            [x, x + dx, x - 2*dy, x + dx - 2*dy,
             x, x + dx, x - dy,   x + dx - dy])

        self.coords_y = np.asarray(
            [y, y + dx, y + 2*dy, y + dx + 2*dy,
             y, y + dx, y + dy,   y + dx + dy])

        self.coeffs = np.asarray(
            [-1,     1,          1,         -1,
             2,     -2,          -2,         2])


class Feature3hRot(Feature):
    def __init__(self, x: int, y: int, dx: int, dy: int, z: int, ws: int):
        super().__init__(x, y, dx, dy, z, ws)
        self.type = 'Feature3hRot'
        self.plausible = \
            ((x - dy) >= 0) and ((x + 3*dx) < ws) and ((y + 3*dx + dy) < ws)

        self.coords_x = np.asarray(
            [x,       x + dx*3,  x - dy,  x + dx*3 - dy,
             x + dx,  x + 2*dx,  x + dx - dy,  x + 2*dx - dy])

        self.coords_y = np.asarray(
            [y,       y + dx*3,     y + dy,      y + dx*3 + dy,
             y + dx,  y + 2*dx,     y + dx + dy,  y + 2*dx + dy])

        self.coeffs = np.asarray(
            [-1,      1,          1,         -1,
             2,      -2,         -2,         2])


class Feature3vRot(Feature):
    def __init__(self, x: int, y: int, dx: int, dy: int, z: int, ws: int):
        super().__init__(x, y, dx, dy, z, ws)
        self.type = 'Feature3vRot'
        self.plausible = \
            ((x - 3*dy) >= 0) and ((x + dx) < ws) and ((y + dx + 3*dy) < ws)

        self.coords_x = np.asarray(
            [x,      x + dx,      x - 3*dy,   x + dx - 3*dy,
             x - dy, x - dy + dx, x - 2*dy,   x + dx - 2*dy])

        self.coords_y = np.asarray(
            [y,      y + dx,      y + 3*dy,   y + dx + 3*dy,
             y + dy, y + dy + dx, y + 2*dy,   y + dx + 2*dy])

        self.coeffs = np.asarray(
            [-1,     1,          1,         -1,
             2,     -2,          -2,         2])


class Feature4hRot(Feature):
    def __init__(self, x: int, y: int, dx: int, dy: int, z: int, ws: int):
        super().__init__(x, y, dx, dy, z, ws)
        self.type = 'Feature4hRot'
        self.plausible = \
            ((x - dy) >= 0) and ((x + 4*dx) < ws) and ((y + 4*dx + dy) < ws)

        self.coords_x = np.asarray(
            [x,       x + dx*4,  x - dy,  x + dx*4 - dy,
             x + dx,  x + 3*dx,  x + dx - dy,  x + 3*dx - dy])

        self.coords_y = np.asarray(
            [y,       y + dx*4,     y + dy,      y + dx*4 + dy,
             y + dx,  y + 3*dx,     y + dx + dy,  y + 3*dx + dy])

        self.coeffs = np.asarray(
            [-1,      1,          1,         -1,
             2,      -2,         -2,         2])


class Feature4vRot(Feature):
    def __init__(self, x: int, y: int, dx: int, dy: int, z: int, ws: int):
        super().__init__(x, y, dx, dy, z, ws)
        self.type = 'Feature4vRot'
        self.plausible = \
            ((x - 4*dy) >= 0) and ((x + dx) < ws) and ((y + dx + 4*dy) < ws)

        self.coords_x = np.asarray(
            [x,      x + dx,      x - 4*dy,   x + dx - 4*dy,
             x - dy, x - dy + dx, x - 3*dy,   x + dx - 3*dy])

        self.coords_y = np.asarray(
            [y,      y + dx,      y + 4*dy,   y + dx + 4*dy,
             y + dy, y + dy + dx, y + 3*dy,   y + dx + 3*dy])

        self.coeffs = np.asarray(
            [-1,     1,          1,         -1,
             2,     -2,          -2,         2])


class Feature2h2vRot(Feature):
    def __init__(self, x: int, y: int, dx: int, dy: int, z: int, ws: int):
        super().__init__(x, y, dx, dy, z, ws)
        self.type = 'Feature2h2vRot'
        self.plausible = \
            ((x - 2*dy) >= 0) and ((x + 2*dx) < ws) and ((y + 2*dx + 2*dy) < ws)

        self.coords_x = np.asarray(
            [x,              x + 2*dx,          x - 2*dy,        x + 2*dx - 2*dy,
             x,              x + dx,            x - dy,          x + dx - dy,
             x + dx - dy,    x + 2*dx - dy,     x + dx - 2*dy,   x + 2*dx - 2*dy])

        self.coords_y = np.asarray(
            [y,              y + 2*dx,        y + 2*dy,        y + 2*dx + 2*dy,
             y,              y + dx,          y + dy,          y + dx + dy,
             y + dx + dy,    y + dy + 2*dx,   y + 2*dy + dx,   y + 2*dx + 2*dy])

        self.coeffs = np.asarray(
            [-1,  1,  1, -1,
             2, -2, -2,  2,
             2, -2, -2,  2])


class Feature3h3vRot(Feature):
    def __init__(self, x: int, y: int, dx: int, dy: int, z: int, ws: int):
        super().__init__(x, y, dx, dy, z, ws)
        self.type = 'Feature3h3vRot'
        self.plausible = \
            ((x - 3*dy) >= 0) and ((x + 3*dx) < ws) and ((y + 3*dx + 3*dy) < ws)

        self.coords_x = np.asarray(
            [x,             x + 3*dx,          x - 3*dy,           x + 3*dx - 3*dy,
             x + dx - dy,   x + 2*dx - dy,     x + dx - 2*dy,      x + 2*dx - 2*dy])

        self.coords_y = np.asarray(
            [y,            y + 3*dx,       y + 3*dy,        y + 3*dx + 3*dy,
             y + dx + dy,  y + 2*dx + dy,  y + dx + 2*dy,   y + 2*dx + 2*dy])

        self.coeffs = np.asarray(
            [-1,  1,  1, -1,
             2, -2, -2, 2])
