"""
Class and functions to model materials used in modeling heat flux
"""
import functools
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Optional
from scipy.sparse import csr_matrix


@dataclass
class Material:
    """
    Class to hold information for a given material
    """
    name: str
    k: float  # W/m*K: thermal conductivity
    layer_width: float  # m: width of a single layer of the given material
    density: float  # g/m^3
    thermal_expansion: float  # m/K
    price: float  # $/m^3


class Grid1d:
    """
    1D grid where the first and last cell have BC's applied on their boundaries

    """

    def __init__(self, n_cells: int, bc_type: int = 0, bc: tuple = (0, 0)):
        """
        Assuming a length of 1
        :param n_cells:
        :param bc_type:
            0 - dirchlet
            1 - derivative
        :param bc
            values applied at each boundary
        """
        self.n_cells = n_cells
        self.bc_type = bc_type
        self.bc = bc

        self.grid = np.zeros(n_cells)
        self.dx = 1 / n_cells
        self.laplacian_matrix: Optional[csr_matrix] = None

    def initialize_laplacian(self):
        row_idx = []
        col_idx = []
        vals = []

        # initialize diagonal
        row_idx += list(np.arange(self.n_cells))
        col_idx += list(np.arange(self.n_cells))
        vals += list(np.ones(self.n_cells) * -2)

        # initialize left neighbors
        row_idx += list(np.arange(self.n_cells - 1) + 1)
        col_idx += list(np.arange(self.n_cells - 1))
        vals += list(np.ones(self.n_cells - 1))

        # initialize right neighbors
        row_idx += list(np.arange(self.n_cells - 1))
        col_idx += list(np.arange(self.n_cells - 1) + 1)
        vals += list(np.ones(self.n_cells - 1))

        self.laplacian_matrix = csr_matrix((vals, (row_idx, col_idx)), dtype=int)
        print('dbg')

    def solve_laplacian(self):
        """ """


@dataclass
class Node:
    r_idx: int
    c_idx: int
    n_circum: int = 0
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None

    @property
    def nid(self):
        return self.c_idx + (self.r_idx * self.n_circum)

    @property
    def r(self):
        return np.sqrt(self.x ** 2 + self.y ** 2)

    @property
    def theta(self):
        if self.x == 0:
            return np.sign(self.y)
        return 0

    @property
    def r_neighbor(self):
        return ((self.c_idx + 1) % self.n_circum) + self.r_idx * self.n_circum

    @property
    def l_neighbor(self):
        return ((self.c_idx - 1) % self.n_circum) + self.r_idx * self.n_circum

    @property
    def u_neighbor(self):
        return self.c_idx + (self.r_idx + 1) * self.n_circum

    @property
    def d_neighbor(self):
        return self.c_idx + (self.r_idx - 1) * self.n_circum


@dataclass
class Element:
    id: int
    nodes: list[Node] # nodes should be added counter-clockwise

    def add_grid_lines(self, ax):
        for i in range(4):
            ax.plot([self.nodes[i].x, self.nodes[(i + 1) % 4].x],
                    [self.nodes[i].y, self.nodes[(i + 1) % 4].y],
                    [self.nodes[i].z, self.nodes[(i + 1) % 4].z],
                    color='r')


def rotation_mat(theta):
    """
    :param theta: rotation parameter in radians
    :return:
    """
    return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])


class GridTaurus:

    def __init__(
        self,
        inner_radius,
        outer_radius,
        n_circum,
        n_radial
    ):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.n_circum = n_circum
        self.n_radial = n_radial
        self.nodes = []
        self.n_nodes = n_radial * n_circum
        for r in range(n_radial):
            for c in range(n_circum):
                t = (c / n_circum) * 2 * np.pi
                s = ((outer_radius - inner_radius) * (r / (n_radial - 1))) + inner_radius
                vec = rotation_mat(t) @ np.array([0, 1]).T * s
                self.nodes.append(Node(r, c, self.n_circum, vec[0], vec[1], 0))

        self.bi_quad_me(0, 0)
        print('dbg')

        # self.elements22 = []  # 2x2 elements of the torus
        # for idx, n in enumerate(self.nodes):
        #     if n.id > (n_radial - 1) * n_circum:
        #         # don't make elements for nodes on the outer circumference
        #         # change this when switching from bi linear to bi quadratic
        #         continue

        #     n1 = idx
        #     n2 = idx + 1 if (idx + 1) % n_circum != 0 else idx - n_circum + 1
        #     n3 = n2 + n_circum
        #     n4 = idx + n_circum

        #     self.elements22.append(
        #         Element(
        #             id=idx + 1,
        #             nodes=[
        #                 self.nodes[n1],
        #                 self.nodes[n2],
        #                 self.nodes[n3],
        #                 self.nodes[n4],
        #             ]
        #     ))

    def solve_fe(self, bc_values, df=1):
        """
        bc_values must have length n_circum for dirchlet boundary condition
        the master element is a square with coord basis x-y
        the pd is in theta-r
        :param bc_values:
        :param df: the power of the quadratic test and basis functions, 1 = bi-linear, 2 = bi-quadratic
        :return:
        """

        # steps?
        # map from pd to me and back

        # using square master element

    def bi_quad_me(self, nid, k):
        """
        iterate over nodes starting at 0, going to nodes that are 2 from the outer boundary

        master element:
        ---------------
           7 | 8 | 9
           ---------
           4 | 5 | 6
           ---------
           1 | 2 | 3
        ---------------
        :param nid: node id in p-d to align with ME node 1
        :return:
        i: list of basis fn numbers
        j: list of basis fn numbers
        c: list of d_i,j values
        """

        # basis fns:
        b1 = lambda xe, ye: (1/4) * xe * (xe - 1) * ye * (ye - 1)
        b2 = lambda xe, ye: (-1/2) * (xe + 1) * (xe - 1) * ye * (ye - 1)
        b3 = lambda xe, ye: (1/4) * (xe + 1) * xe * ye * (ye - 1)
        b4 = lambda xe, ye: (-1/2) * xe * (xe - 1) * (ye + 1) * (ye - 1)
        b5 = lambda xe, ye: (xe + 1) * (xe - 1) * (ye + 1) * (ye - 1)
        b6 = lambda xe, ye: (-1/2) * (xe + 1) * xe * (ye + 1) * (ye - 1)
        b7 = lambda xe, ye: (1/4) * xe * (xe - 1) * ye * (ye + 1)
        b8 = lambda xe, ye: (-1/2) * (xe + 1) * (xe - 1) * ye * (ye + 1)
        b9 = lambda xe, ye: (1/4) * (xe + 1) * xe * ye * (ye + 1)

        db1_dxe = lambda xe, ye: (1/4) * (2 * xe - 1) * ye * (ye - 1)
        db2_dxe = lambda xe, ye: (-1/2) * (2 * xe) * ye * (ye - 1)
        db3_dxe = lambda xe, ye: (1/4) * (2 * xe + 1) * ye * (ye - 1)
        db4_dxe = lambda xe, ye: (-1/2) * (2 * xe - 1) * (ye + 1) * (ye - 1)
        db5_dxe = lambda xe, ye: (2 * xe) * (ye + 1) * (ye - 1)
        db6_dxe = lambda xe, ye: (-1/2) * (2 * xe + 1) * (ye + 1) * (ye - 1)
        db7_dxe = lambda xe, ye: (1/4) * (2 * xe - 1) * ye * (ye + 1)
        db8_dxe = lambda xe, ye: (-1/2) * (2 * xe) * ye * (ye + 1)
        db9_dxe = lambda xe, ye: (1/4) * (2 * xe + 1) * ye * (ye + 1)

        db1_dye = lambda xe, ye: (1/4) * xe * (xe - 1) * (2 * ye - 1)
        db2_dye = lambda xe, ye: (-1/2) * (xe + 1) * (xe - 1) * (2 * ye - 1)
        db3_dye = lambda xe, ye: (1/4) * (xe + 1) * xe * ye * (2 * ye - 1)
        db4_dye = lambda xe, ye: (-1/2) * xe * (xe - 1) * (2 * ye)
        db5_dye = lambda xe, ye: (xe + 1) * (xe - 1) * (2 * ye)
        db6_dye = lambda xe, ye: (-1/2) * (xe + 1) * xe * (2 * ye)
        db7_dye = lambda xe, ye: (1/4) * xe * (xe - 1) * (2 * ye + 1)
        db8_dye = lambda xe, ye: (-1/2) * (xe + 1) * (xe - 1) * (2 * ye + 1)
        db9_dye = lambda xe, ye: (1/4) * (xe + 1) * xe * (2 * ye + 1)

        # collect nodes, raise error if nid is too far out on the mesh
        x = []
        y = []
        quad = []
        n1 = self.nodes[nid]
        if n1.r_idx >= self.n_radial - 2:
            logging.warning('No Master Element available')
            return x, y, quad

        n2 = self.nodes[n1.r_neighbor]
        n3 = self.nodes[n2.r_neighbor]
        n4 = self.nodes[n1.u_neighbor]
        n5 = self.nodes[n2.u_neighbor]
        n6 = self.nodes[n3.u_neighbor]
        n7 = self.nodes[n4.u_neighbor]
        n8 = self.nodes[n5.u_neighbor]
        n9 = self.nodes[n6.u_neighbor]

        nodes = [n1, n2, n3, n4, n5, n6, n7, n8, n9]
        basis_fns = [b1, b2, b3, b4, b5, b6, b7, b8, b9]
        dbdxe_fns = [db1_dxe, db2_dxe, db3_dxe, db4_dxe, db5_dxe, db6_dxe, db7_dxe, db8_dxe, db9_dxe]
        dbdye_fns = [db1_dye, db2_dye, db3_dye, db4_dye, db5_dye, db6_dye, db7_dye, db8_dye, db9_dye]
        e_idxs = list(range(9))

        dtheta_dx = (n3.theta - n1.theta) / 2
        dr_dy = (n7.r - n1.r) / 2
        det_j = np.abs(dtheta_dx * dr_dy)

        dx_dtheta = np.zeros(25)
        dx_dr = np.zeros(25)
        dy_dtheta = np.zeros(25)
        dy_dr = np.zeros(25)
        rphys = np.zeros(25)
        tphys = np.zeros(25)

        # compute these partial derivatives for all gauss points
        gx, gy, gw = self.gauss_points5

        for idx, (gxi, gyi) in enumerate(zip(gx, gy)):
            for db_dy, db_dx, bfn, node in zip(dbdye_fns, dbdxe_fns, basis_fns, nodes):
                dx_dtheta[idx] += node.r * db_dy(gxi, gyi)
                dx_dr[idx] += node.theta * db_dy(gxi, gyi)
                dy_dtheta[idx] += node.r * db_dx(gxi, gyi)
                dy_dr[idx] += node.theta * db_dx(gxi, gyi)
                rphys[idx] += node.r * bfn(gxi, gyi)
                tphys[idx] += node.theta * bfn(gxi, gyi)

        dx_dtheta /= det_j
        dx_dr /= -det_j
        dy_dtheta /= -det_j
        dy_dr /= det_j

        D = np.zeros((9, 9))  # matrix storing all of the dwi*dwj quadrature results

        for idx1, node1, basis1, dbxe1, dbye1 in zip(e_idxs, nodes, basis_fns, dbdxe_fns, dbdye_fns):
            for idx2, node2, basis2, dbxe2, dbye2 in zip(e_idxs[idx1:], nodes[idx1:], basis_fns[idx1:], dbdxe_fns[idx1:], dbdye_fns[idx1:]):
                d = 0
                for gidx, (gxi, gyi, gwi) in enumerate(zip(gx, gy, gw)):
                    db_dr1 = dbxe1(gxi, gyi) * dx_dr[gidx] + dbye1(gxi, gyi) * dy_dr[gidx]
                    db_dt1 = dbxe1(gxi, gyi) * dx_dtheta[gidx] + dbye1(gxi, gyi) * dy_dtheta[gidx]
                    db_dr2 = dbxe2(gxi, gyi) * dx_dr[gidx] + dbye2(gxi, gyi) * dy_dr[gidx]
                    db_dt2 = dbxe2(gxi, gyi) * dx_dtheta[gidx] + dbye2(gxi, gyi) * dy_dtheta[gidx]
                    d += gwi * det_j * (db_dr1 * db_dr2 + db_dt1 * db_dt2)
                D[idx1, idx2] = d
                D[idx2, idx1] = d
        print('dbg')

    def plot(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(
            [n.x for n in self.nodes],
            [n.y for n in self.nodes],
            [n.z for n in self.nodes],
        )
        for element in self.elements22:
            element.add_grid_lines(ax)

    def id_map(self, r, s):
        """
        :param r: increment around the circle angle starts at 0
        :param s: increment distance from inner radius starts at 0
        :return: int: id starts at 1
        """
        return r + (s * self.n_circum)

    @functools.cached_property
    def gauss_points5(self):
        x1 = -(1/3) * np.sqrt(5 + 2 * np.sqrt(10/7))
        x2 = -(1/3) * np.sqrt(5 - 2 * np.sqrt(10/7))
        x3 = 0
        x4 = -x2
        x5 = -x1

        w1 = (322 - 13 * np.sqrt(70)) / 900
        w2 = (322 + 13 * np.sqrt(70)) / 900
        w3 = 128/225
        w4 = -w2
        w5 = -w1

        x = np.array([x1, x2, x3, x4, x5] * 5)
        y = np.array([x1] * 5 + [x2] * 5 + [x3] * 5 + [x4] * 5 + [x5] * 5)
        w = np.array([w1*w1, w1*w2, w1*w3, w1*w4, w1*w5,
                      w2*w1, w2*w2, w2*w3, w2*w4, w2*w5,
                      w3*w1, w3*w2, w3*w3, w3*w4, w3*w5,
                      w4*w1, w4*w2, w4*w3, w4*w4, w4*w5,
                      w5*w1, w5*w2, w5*w3, w5*w4, w5*w5])
        return x, y, w


