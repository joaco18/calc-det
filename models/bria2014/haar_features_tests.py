import haar_modules as hm
import numpy as np


test_image = np.array([
    [5, 2, 3, 4, 1, 4, 1, 4, 1, 1, 4, 1],
    [1, 5, 4, 2, 3, 2, 3, 2, 3, 3, 2, 3],
    [2, 2, 1, 3, 4, 3, 4, 3, 4, 4, 3, 4],
    [3, 5, 6, 4, 5, 4, 5, 4, 5, 5, 4, 5],
    [4, 1, 3, 2, 6, 2, 6, 2, 6, 6, 2, 6],
    [3, 5, 6, 4, 5, 4, 5, 4, 5, 5, 4, 5],
    [4, 1, 3, 2, 6, 2, 6, 2, 6, 6, 2, 6],
    [3, 5, 6, 4, 5, 4, 5, 4, 5, 5, 4, 5],
    [4, 1, 3, 2, 6, 2, 6, 2, 6, 6, 2, 6],
    [4, 1, 3, 2, 6, 2, 6, 2, 6, 6, 2, 6],
    [3, 5, 6, 4, 5, 4, 5, 4, 5, 5, 4, 5],
    [4, 1, 3, 2, 6, 2, 6, 2, 6, 6, 2, 6]]
)
test_integral = hm.to_integral(test_image)
test_integral_dig = hm.to_diag_integral(test_image)


def test_Box(x, y, w, h):
    expected = np.sum(test_image[y:h+y, x:w+x])
    test_box = hm.Box(x, y, w, h)
    actual = test_box(test_integral)
    assert expected == actual, f'test_Box failed: {expected} == {actual}'
    return True


def test_Feature2h():
    expected = hm.Box(0, 1, 2, 3)(test_integral) - hm.Box(2, 1, 2, 3)(test_integral)
    actual = hm.Feature2h(0, 1, 4, 3)(test_integral)
    assert expected == actual, f'test Features2h failed: {expected} == {actual}'
    return True


def test_Feature2v():
    expected = hm.Box(0, 1, 4, 2)(test_integral) - hm.Box(0, 3, 4, 2)(test_integral)
    actual = hm.Feature2v(0, 1, 4, 4)(test_integral)
    assert expected == actual, f'test Features2h failed: {expected} == {actual}'
    return True


def test_Feature3h():
    expected = -hm.Box(0, 0, 1, 2)(test_integral) \
        + hm.Box(1, 0, 1, 2)(test_integral) - hm.Box(2, 0, 1, 2)(test_integral)
    actual = hm.Feature3h(0, 0, 3, 2)(test_integral)
    assert expected == actual, f'test Features3h failed: {expected} == {actual}'
    return True


def test_Feature3v():
    expected = - hm.Box(0, 0, 2, 1)(test_integral) \
        + hm.Box(0, 1, 2, 1)(test_integral) - hm.Box(0, 2, 2, 1)(test_integral)
    actual = hm.Feature3v(0, 0, 2, 3)(test_integral)
    assert expected == actual, f'test Features3v failed: {expected} == {actual}'
    return True


def test_Feature4h():
    expected = - hm.Box(0, 0, 1, 2)(test_integral) \
        + hm.Box(1, 0, 2, 2)(test_integral) - hm.Box(3, 0, 1, 2)(test_integral)
    actual = hm.Feature4h(0, 0, 4, 2)(test_integral)
    assert expected == actual, f'test Features4h failed: {expected} == {actual}'
    return True


def test_Feature4v():
    expected = - hm.Box(0, 0, 2, 1)(test_integral) \
        + hm.Box(0, 1, 2, 2)(test_integral) - hm.Box(0, 3, 2, 1)(test_integral)
    actual = hm.Feature4v(0, 0, 2, 4)(test_integral)
    assert expected == actual, f'test Features4v failed: {expected} == {actual}'
    return True


def test_Feature2h2v():
    expected = + hm.Box(0, 0, 1, 1)(test_integral) - hm.Box(0, 1, 1, 1)(test_integral) \
        - hm.Box(1, 0, 1, 1)(test_integral) + hm.Box(1, 1, 1, 1)(test_integral)
    actual = hm.Feature2h2v(0, 0, 2, 2)(test_integral)
    assert expected == actual, f'test Features2h2v failed: {expected} == {actual}'
    return True


def test_Feature3h3v():
    expected = - hm.Box(0, 0, 3, 1)(test_integral) - hm.Box(0, 1, 1, 1)(test_integral) \
        + hm.Box(1, 1, 1, 1)(test_integral) - hm.Box(2, 1, 1, 1)(test_integral) \
        - hm.Box(0, 2, 3, 1)(test_integral)
    actual = hm.Feature3h3v(0, 0, 3, 3)(test_integral)
    assert expected == actual, f'test hm.Features3h3v failed: {expected} == {actual}'
    return True


def test_Diamond():
    expected = 3+4+5+2+2+3
    actual = hm.Diamond(3, 0, 1, 3, 12)(test_integral_dig)
    assert expected == actual, f'test_Diamond failed: {expected} == {actual}'
    return True


def test_Feature2hRot():
    expected = 2 * hm.Diamond(3, 0, 1, 3, 12)(test_integral_dig) - \
        hm.Diamond(3, 0, 2, 3, 12)(test_integral_dig)
    actual = hm.Feature2hRot(3, 0, 1, 3, 12)(test_integral_dig)
    assert expected == actual, f'test_Feature2hRot failed: {expected} == {actual}'
    return True


def test_Feature2vRot():
    expected = 2 * hm.Diamond(2, 0, 2, 1, 12)(test_integral_dig) - \
        hm.Diamond(2, 0, 2, 2, 12)(test_integral_dig)
    actual = hm.Feature2vRot(2, 0, 2, 1, 12)(test_integral_dig)
    assert expected == actual, f'test_Feature2vRot failed: {expected} == {actual}'
    return True


def test_Feature3hRot():
    expected = 2 * hm.Diamond(5, 1, 1, 3, 12)(test_integral_dig) - \
        hm.Diamond(4, 0, 3, 3, 12)(test_integral_dig)
    actual = hm.Feature3hRot(4, 0, 1, 3, 12)(test_integral_dig)
    assert expected == actual, f'test_Feature3hRot failed: {expected} == {actual}'
    return True


def test_Feature3vRot():
    expected = 2 * hm.Diamond(3, 1, 3, 1, 12)(test_integral_dig) - \
        hm.Diamond(4, 0, 3, 3, 12)(test_integral_dig)
    actual = hm.Feature3vRot(4, 0, 3, 1, 12)(test_integral_dig)
    assert expected == actual, f'test_Feature3vRot failed: {expected} == {actual}'
    return True


def test_Feature4hRot():
    expected = 2 * hm.Diamond(5, 1, 2, 3, 12)(test_integral_dig) - \
        hm.Diamond(4, 0, 4, 3, 12)(test_integral_dig)
    actual = hm.Feature4hRot(4, 0, 1, 3, 12)(test_integral_dig)
    assert expected == actual, f'test_Feature4hRot failed: {expected} == {actual}'
    return True


def test_Feature4vRot():
    expected = 2 * hm.Diamond(3, 1, 3, 2, 12)(test_integral_dig) - \
        hm.Diamond(4, 0, 3, 4, 12)(test_integral_dig)
    actual = hm.Feature4vRot(4, 0, 3, 1, 12)(test_integral_dig)
    assert expected == actual, f'test_Feature4vRot failed: {expected} == {actual}'
    return True


def test_Feature2h2vRot():
    expected = 2 * hm.Diamond(4, 0, 2, 2, 12)(test_integral_dig) + \
        2 * hm.Diamond(4, 4, 2, 2, 12)(test_integral_dig) - \
        hm.Diamond(4, 0, 4, 4, 12)(test_integral_dig)
    actual = hm.Feature2h2vRot(4, 0, 2, 2, 12)(test_integral_dig)
    assert expected == actual, f'test_Feature2h2vRot failed: {expected} == {actual}'
    return True


def test_Feature3h3vRot():
    expected = 2 * hm.Diamond(3, 1, 1, 1, 12)(test_integral_dig) - \
        hm.Diamond(4, 0, 3, 3, 12)(test_integral_dig)
    actual = hm.Feature3h3vRot(4, 0, 1, 1, 12)(test_integral_dig)
    assert expected == actual, f'test_Feature3h3vRot failed: {expected} == {actual}'
    return True


def main():
    print(f'Test Box passed? {test_Box(0, 0, 3, 3)}')
    print(f'Test Feature2h passed? {test_Feature2h()}')
    print(f'Test Feature2v passed? {test_Feature2v()}')
    print(f'Test Feature3h passed? {test_Feature3h()}')
    print(f'Test Feature3v passed? {test_Feature3v()}')
    print(f'Test Feature4h passed? {test_Feature4h()}')
    print(f'Test Feature4v passed? {test_Feature4v()}')
    print(f'Test Feature2h2v passed? {test_Feature2h2v()}')
    print(f'Test Feature3h3v passed? {test_Feature3h3v()}')
    print(f'Test Diamond passed? {test_Diamond()}')
    print(f'Test Feature2hRot passed? {test_Feature2hRot()}')
    print(f'Test Feature2vRot passed? {test_Feature2vRot()}')
    print(f'Test Feature3hRot passed? {test_Feature3hRot()}')
    print(f'Test Feature3vRot passed? {test_Feature3vRot()}')
    print(f'Test Feature4hRot passed? {test_Feature4hRot()}')
    print(f'Test Feature4vRot passed? {test_Feature4vRot()}')
    print(f'Test Feature2h2vRot passed? {test_Feature2h2vRot()}')
    print(f'Test Feature3h3vRot passed? {test_Feature3h3vRot()}')
    print('\nALL TESTS PASSED GO GRAB A BEER TO CELEBRATE!')


if __name__ == '__main__':
    main()
