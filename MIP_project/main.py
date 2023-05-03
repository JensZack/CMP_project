import material_formats
from test.test_solver import gen_mms_cubic


def main():
    grid = material_formats.GridTaurus(2, 10, 3, 3)
    # bc_fn = lambda x, y: (x ** 2 + y ** 2) / 8
    # k_fn = lambda x, y: 100 if np.sqrt(x ** 2 + y ** 2) > 6 else 1
    mms = gen_mms_cubic()
    grid.gen_linear_system(mms.fn, mms=mms)
    grid.solve_linear_system()
    grid.plot(plot_type='nodes', fn=mms.fn)


if __name__ == '__main__':
    main()
