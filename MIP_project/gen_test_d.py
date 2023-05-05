import numpy as np
from fea_grid.fea_grid import bi_quad_basis_fns


def gen_test_d_c(res=100):
    """
    generate D for a test element on the cartesian plane with x, y in [-1, 1]

    :param res: resolution for midpoint rule in integration

    :return:
    """

    bfns, dbdxes, dbdyes = bi_quad_basis_fns()
    mp_bounds = (-1 + (1/res), 1 - (1/res))
    x = np.linspace(*mp_bounds, res)
    y = np.linspace(*mp_bounds, res)
    xg, yg = np.meshgrid(x, y)
    w = 4 / (res**2)

    D = np.zeros((9, 9))

    for i in range(9):
        for j in range(i, 9):
            fn = np.vectorize(lambda x, y: dbdxes[i](x, y) * dbdxes[j](x, y) + dbdyes[i](x, y) * dbdyes[j](x, y))
            # integrate with midpoint quadrature
            quad = w * fn(xg, yg)
            D[i, j] = quad.sum()
            D[j, i] = quad.sum()
    return D


def gen_test_d_polar(res=100):
    """
    generate D for a test element on the polar plane with r in [1, 2] and theta in [0, pi/2]
    Use midpoint quadrature to estimate the D matrix for a 3x3 bi-quadratic element from
    the physical element in the polar space

    :param res: resolution for midpoint rule in integration

    :return D: matrix from quadrature estimation
    """

    bfns, dbdxes, dbdyes = bi_quad_basis_fns()
    r0 = 1
    r1 = 2
    t0 = 0
    t1 = np.pi / 2
    r = (np.linspace(r0, r1, res + 1) + (.5 / res))[:-1]
    t = (np.linspace(t0, t1, res + 1) + (.5 / res))[:-1]
    rg, tg = np.meshgrid(r, t)
    w = (r1 - r0) * (t1 - t0) / (res**2)

    D = np.zeros((9, 9))

    for i in range(9):
        for j in range(i, 9):
            fn = np.vectorize(lambda t, r: (dbdxes[i](t, r) * dbdxes[j](t, r) + dbdyes[i](t, r) * dbdyes[j](t, r)) * r)
            # integrate with midpoint quadrature
            quad = w * fn(tg, rg)
            D[i, j] = quad.sum()
            D[j, i] = quad.sum()
    return D


def main():
    print(gen_test_d_polar(100))


if __name__ == '__main__':
    main()