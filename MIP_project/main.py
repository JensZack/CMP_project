import fea_grid
import matplotlib.pyplot as plt
from test.test_solver import gen_mms_quad, gen_mms_cubic


def main():
    grid = fea_grid.GridTorus(2, 10, 15, 15)
    mms = gen_mms_quad()
    # fn = lambda x, y: x**3
    grid.gen_linear_system(bc_fn=mms.fn, mms=mms)
    grid.solve_linear_system()
    grid.plot(plot_type='nodes', mms=mms)  # plot_type='mesh', mms=mms)
    plt.show()


if __name__ == '__main__':
    main()
