import fea_grid
import matplotlib.pyplot as plt
from fea_grid.test import gen_mms_quad


def main():
    grid = fea_grid.GridTorus(2, 10, 15, 15)
    mms = gen_mms_quad()
    grid.gen_linear_system(bc_fn=mms.fn, mms=mms)
    grid.solve_linear_system()
    grid.plot(plot_type='nodes')
    plt.show()


if __name__ == '__main__':
    main()
