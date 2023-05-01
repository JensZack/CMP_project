"""
Contains tests for the heat diffusion solver,
Including MMS tests on the taurus grid
"""
from material_formats import GridTaurus
import matplotlib.pyplot as plt


def test_linear_mms(plot=False):
    grid = GridTaurus(2, 10, 20, 20)
    f = lambda x, y: 3 * (x - 2) + .5 * (y + 1)

    for node in grid.nodes:
        node.z = f(node.x, node.y)

    if plot:
        grid.plot()
        plt.show()


def test_quad_mms(plot=False):
    grid = GridTaurus(2, 10, 20, 20)
    f = lambda x, y: 3 * (x - 2)**2 + .5 * (y ** 2 + 3 * y + 1)

    for node in grid.nodes:
        node.z = f(node.x, node.y)

    if plot:
        grid.plot()
        plt.show()

