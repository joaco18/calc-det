import feature_extraction as fe
import numpy as np

test_image = np.array([
    [5, 2, 3, 4, 1],
    [1, 5, 4, 2, 3],
    [2, 2, 1, 3, 4],
    [3, 5, 6, 4, 5],
    [4, 1, 3, 2, 6]])
test_integral = fe.to_integral(test_image)


def test_Box(x, y, w, h):
    expected = np.sum(test_image[y:h+y, x:w+x])
    test_box = fe.Box(x, y, w, h)
    actual = test_box(test_integral)
    assert expected == actual, f'test_Box failed: {expected} == {actual}'


def test_Feature2h():
    expected = fe.Box(0, 1, 2, 3)(test_integral) - fe.Box(2, 1, 2, 3)(test_integral)
    actual = fe.Feature2h(0, 1, 4, 3)(test_integral)
    assert expected == actual, f'test Features2h failed: {expected} == {actual}'


def test_Feature2v():
    expected = fe.Box(0, 1, 4, 2)(test_integral) - fe.Box(0, 3, 4, 2)(test_integral)
    actual = fe.Feature2v(0, 1, 4, 4)(test_integral)
    assert expected == actual, f'test Features2h failed: {expected} == {actual}'


def test_Feature3h():
    expected = fe.Box(0, 0, 1, 2)(test_integral) \
        - fe.Box(1, 0, 1, 2)(test_integral) + fe.Box(2, 0, 1, 2)(test_integral)
    actual = fe.Feature3h(0, 0, 3, 2)(test_integral)
    assert expected == actual, f'test Features3h failed: {expected} == {actual}'


def test_Feature3v():
    expected = fe.Box(0, 0, 2, 1)(test_integral) \
        - fe.Box(0, 1, 2, 1)(test_integral) + fe.Box(0, 2, 2, 1)(test_integral)
    actual = fe.Feature3v(0, 0, 2, 3)(test_integral)
    assert expected == actual, f'test Features3h failed: {expected} == {actual}'


def test_Feature4h():
    expected = fe.Box(0, 0, 1, 2)(test_integral) \
        - fe.Box(1, 0, 2, 2)(test_integral) + fe.Box(3, 0, 1, 2)(test_integral)
    actual = fe.Feature4h(0, 0, 4, 2)(test_integral)
    assert expected == actual, f'test Features3h failed: {expected} == {actual}'


def test_Feature4v():
    expected = fe.Box(0, 0, 2, 1)(test_integral) \
        - fe.Box(0, 1, 2, 2)(test_integral) + fe.Box(0, 3, 2, 1)(test_integral)
    actual = fe.Feature4v(0, 0, 2, 4)(test_integral)
    assert expected == actual, f'test Features3h failed: {expected} == {actual}'


def test_FeatureChecker():
    expected = fe.Box(0, 0, 1, 1)(test_integral) - fe.Box(0, 1, 1, 1)(test_integral) \
        - fe.Box(1, 0, 1, 1)(test_integral) + fe.Box(1, 1, 1, 1)(test_integral)
    actual = fe.FeatureChecker(0, 0, 2, 2)(test_integral)
    assert expected == actual, f'test Features3h failed: {expected} == {actual}'


def test_FeatureCenter():
    expected = fe.Box(0, 0, 3, 1)(test_integral) + fe.Box(0, 1, 1, 1)(test_integral) \
        - fe.Box(1, 1, 1, 1)(test_integral) + fe.Box(2, 1, 1, 1)(test_integral) \
        + fe.Box(0, 2, 3, 1)(test_integral)
    actual = fe.FeatureCenter(0, 0, 3, 3)(test_integral)
    assert expected == actual, f'test fe.Features3h failed: {expected} == {actual}'


def test_Diamond():
    expected = 2+3+4+1+5+4+2+3+2+2+1+3+4+5+6+4+3
    actual = fe.Diamond(0, 0, 5, 0.5)(test_integral)
    assert expected == actual, f'test_Box failed: {expected} == {actual}'


def test_FeatureRot():
    expected = 2+3+4+1+5+4+2+3+2+2+1+3+4+5+6+4+3
    actual = fe.FeatureRot(0, 0, 5, 0.5)(test_integral)
    assert expected == actual, f'test Features2h failed: {expected} == {actual}'
