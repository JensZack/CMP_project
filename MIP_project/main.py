import material_formats
import matplotlib.pyplot as plt
from test.test_solver import test_linear_mms, test_quad_mms


def main():
    grid = material_formats.GridTaurus(2, 10, 20, 15)
    # grid.plot()
    # plt.show()
    # test_quad_mms(True)


if __name__ == '__main__':
    main()
