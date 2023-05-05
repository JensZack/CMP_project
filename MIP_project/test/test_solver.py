"""
Contains tests for the heat diffusion solver,
Including MMS tests on the taurus grid
"""
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from pylatex import Document, Section, Math, Figure
from pylatex.utils import NoEscape
from fea_grid import MMS, GridTorus, gauss_points5, Node, Element
from gen_test_d import gen_test_d_c, gen_test_d_polar



def gen_mms_cubic():
    """
    generate the MMS object for x^3 + y^3
    :return:
    """
    mms = MMS(
        lambda x, y: x ** 3 + y ** 3,
        lambda x, y: 6 * x + 6 * y,
        "f(x, y) = x^3 + y^3, l(x, y) = 6(x + y)",
        "cubic_mms"
    )
    return mms


def gen_mms_quad():
    """
    generate the MMS object for x^2 + y^2
    :return:
    """
    mms = MMS(
        lambda x, y: x ** 2 + y ** 2,
        lambda x, y: 4,
        "f(x, y) = x^2 + y^2, l(x, y) = 0",
        "quadratic_mms"
    )
    return mms


def gen_mms_linear():
    """
    generate the MMS object for x + y
    :return:
    """
    mms = MMS(
        lambda x, y: x + y,
        lambda x, y: 0,
        "f(x, y) = x + y, l(x, y) = 0",
        "linear_mms"
    )
    return mms


def mms_latex_result_file(mms_list, l2_means):
    doc = Document()

    latex_output = 'results/mms_report'
    for mms, l2_mean in zip(mms_list, l2_means):
        img_path = pathlib.Path(f'results/{mms.name}.png')
        with doc.create(Section(mms.name)):
            doc.append("FEA results for the mms function and laplacian:")
            doc.append(Math(data=[NoEscape(mms.description)]))
            with doc.create(Figure(position='h!')) as mms_plot:
                mms_plot.add_image(str(img_path.absolute()), width=NoEscape(r'\textwidth'))
            doc.append(NoEscape(r"The $\frac{L2 norm}{n-nodes}$ measure is " + f"{l2_mean}"))
        doc.append(NoEscape(r'\newpage'))
    doc.generate_tex(latex_output)
    doc.dumps()


def compare_grid_to_mms(grid: GridTorus, mms: MMS):
    x, y, z = [], [], []
    for node in grid.nodes:
        x.append(node.x)
        y.append(node.y)
        z.append(node.z)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    fn = np.vectorize(mms.fn)
    fnz = fn(x, y)

    diff = fnz - z
    return diff, fnz, z


def test_mms():
    """
    Iterates over list of test mms functions and compares the results of GridTorus with the mms function.
    Once the evaluation is completed this test will add each mms plot as well as l2-norm/n-nodes into the
    latex doc for the MIP project
    """
    mms_list = [
        gen_mms_linear(),
        gen_mms_quad(),
        gen_mms_cubic(),
    ]
    diffs = []
    l2_means = []
    for mms in mms_list:
        grid = GridTorus(4, 8, 10, 10)
        grid.gen_linear_system(mms.fn, mms=mms)
        grid.solve_linear_system()
        grid.plot(mms=mms, plot_type='nodes')
        plt.savefig(f'results/{mms.name}.png')
        diff, fnz, z = compare_grid_to_mms(grid, mms)
        diffs.append(diff)

        # print the l2 norm of the error
        l2 = np.linalg.norm(diff)
        l2_avg = l2 / grid.n_nodes
        l2_means.append(l2_avg)
    mms_latex_result_file(mms_list, l2_means)

    for diff in diffs:
        assert np.allclose(diff, atol=1e-4)


def test_gauss_quad():
    """
    Using the gauss points and weights in GridTorus, compute quadratures that should be exact or close
    and compare with analytical integral solutions. Uses default absolute tolerance of 1e-5
    """

    gx, gy, gw = gauss_points5()

    fn1 = lambda x, y: x ** 3 + 2 * x ** 2 + 4 * x + y ** 3 + 5
    sol1 = 68 / 3

    fn2 = lambda x, y: 3 * x ** 2 * y ** 2
    sol2 = 4/3

    fn3 = lambda x, y: 2
    sol3 = 8

    for fn, sol in zip([fn1, fn2, fn3], [sol1, sol2, sol3]):
        quad = 0.
        for x, y, w in zip(gx, gy, gw):
            quad += w * fn(x, y)
        assert np.isclose(quad, sol)


def test_element_cartesian():
    """
    create an Element in cartesian space and use the GridTorus bi_quad_me to generate the bi quadratic
    D matrix on the master element, compare the generated D to a reference D that is estimated with
    midpoint rule
    """
    nodes = [
        Node(-1, -1, -1, x=-1, y=-1, on_boundary=False),
        Node(-1, -1, -1, x=0, y=-1, on_boundary=False),
        Node(-1, -1, -1, x=1, y=-1, on_boundary=False),
        Node(-1, -1, -1, x=-1, y=0, on_boundary=False),
        Node(-1, -1, -1, x=0, y=0, on_boundary=False),
        Node(-1, -1, -1, x=1, y=0, on_boundary=False),
        Node(-1, -1, -1, x=-1, y=1, on_boundary=False),
        Node(-1, -1, -1, x=0, y=1, on_boundary=False),
        Node(-1, -1, -1, x=1, y=1, on_boundary=False),
    ]
    element = Element(0, nodes)
    k = lambda x, y: 1
    GridTorus.bi_quad_me(element, k)

    ref_D = gen_test_d_c(200)

    assert np.allclose(element.D, ref_D, atol=1e-3)


def test_element_cartesian_shift():
    """ """
    nodes = [
        Node(-1, -1, -1, x=2, y=4, on_boundary=False),
        Node(-1, -1, -1, x=3, y=4, on_boundary=False),
        Node(-1, -1, -1, x=4, y=4, on_boundary=False),
        Node(-1, -1, -1, x=2, y=5, on_boundary=False),
        Node(-1, -1, -1, x=3, y=5, on_boundary=False),
        Node(-1, -1, -1, x=4, y=5, on_boundary=False),
        Node(-1, -1, -1, x=2, y=6, on_boundary=False),
        Node(-1, -1, -1, x=3, y=6, on_boundary=False),
        Node(-1, -1, -1, x=4, y=6, on_boundary=False),
    ]
    element_offset = Element(0, nodes)
    k = lambda x, y: 1
    GridTorus.bi_quad_me(element_offset, k)

    ref_D = gen_test_d_c(200)

    assert np.allclose(element_offset.D, ref_D, atol=1e-3)


def test_element_cartesian_stretch():
    """
    """
    nodes = [
        Node(-1, -1, -1, x=0, y=0, on_boundary=False),
        Node(-1, -1, -1, x=2, y=0, on_boundary=False),
        Node(-1, -1, -1, x=4, y=0, on_boundary=False),
        Node(-1, -1, -1, x=0, y=3, on_boundary=False),
        Node(-1, -1, -1, x=2, y=3, on_boundary=False),
        Node(-1, -1, -1, x=4, y=3, on_boundary=False),
        Node(-1, -1, -1, x=0, y=6, on_boundary=False),
        Node(-1, -1, -1, x=2, y=6, on_boundary=False),
        Node(-1, -1, -1, x=4, y=6, on_boundary=False),
    ]
    element_stretch = Element(0, nodes)
    k = lambda x, y: 1
    GridTorus.bi_quad_me(element_stretch, k)

    ref_D = gen_test_d_c(200)

    # I would assume that the D matrix for the stretched element should be 6 * ref_D

    assert np.allclose(element_stretch.D / 24, ref_D, atol=1e-3)


def test_element_polar():
    grid = GridTorus(1, 2, 4, 1)
    element = grid.elements[0]
    k = lambda x, y: 1
    GridTorus.bi_quad_me(element, k)

    ref_D = gen_test_d_polar(500)
    diff = ref_D - element.D

    assert np.allclose(diff, 0, atol=1e-2)
