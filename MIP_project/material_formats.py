"""
Class and functions to model materials used in modeling heat flux
"""
import functools
from dataclasses import dataclass
import numpy as np
import scipy.sparse.linalg as spl
import matplotlib.pyplot as plt
import logging
from typing import Optional

from scipy.sparse import csr_matrix


LOGGER = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s:%(levelname)s - %(message)s')
LOGGER.setLevel('DEBUG')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
LOGGER.addHandler(sh)


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


@functools.lru_cache
def p_to_c(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


@functools.lru_cache
def c_to_p(x, y):
    r = np.sqrt(x**2 + y**2)
    if x == 0:
        theta = np.pi / 2 if y > 0 else np.pi * (3 / 2)
        return r, theta
    else:
        theta = np.arctan(y / x)
        if x < 0:
            theta += np.pi

        if theta < 0:
            theta += 2 * np.pi
        return r, theta


@dataclass
class Node:
    r_idx: int
    c_idx: int
    n_circum: int = 0
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    on_boundary: bool = False

    @property
    def nid(self):
        return int(self.c_idx + (self.r_idx * self.n_circum))

    @functools.cached_property
    def r(self):
        r, _ = c_to_p(self.x, self.y)
        return r

    @functools.cached_property
    def theta(self):
        _, t = c_to_p(self.x, self.y)
        return t

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


def basis_fns():
    b1 = lambda xe, ye: (1 / 4) * xe * (xe - 1) * ye * (ye - 1)
    b2 = lambda xe, ye: (-1 / 2) * (xe + 1) * (xe - 1) * ye * (ye - 1)
    b3 = lambda xe, ye: (1 / 4) * (xe + 1) * xe * ye * (ye - 1)
    b4 = lambda xe, ye: (-1 / 2) * xe * (xe - 1) * (ye + 1) * (ye - 1)
    b5 = lambda xe, ye: (xe + 1) * (xe - 1) * (ye + 1) * (ye - 1)
    b6 = lambda xe, ye: (-1 / 2) * (xe + 1) * xe * (ye + 1) * (ye - 1)
    b7 = lambda xe, ye: (1 / 4) * xe * (xe - 1) * ye * (ye + 1)
    b8 = lambda xe, ye: (-1 / 2) * (xe + 1) * (xe - 1) * ye * (ye + 1)
    b9 = lambda xe, ye: (1 / 4) * (xe + 1) * xe * ye * (ye + 1)

    return [b1, b2, b3, b4, b5, b6, b7, b8, b9]


@dataclass
class Element:
    id: int
    nodes: list[Node]
    D: np.ndarray = np.zeros((9, 9))
    b: np.ndarray = np.zeros(9)
    w: np.ndarray = np.zeros(9)

    def add_grid_lines(self, ax):
        for i in range(4):
            ax.plot([self.nodes[i].x, self.nodes[(i + 1) % 9].x],
                    [self.nodes[i].y, self.nodes[(i + 1) % 9].y],
                    [self.nodes[i].z, self.nodes[(i + 1) % 9].z],
                    color='r')

    def get_w(self, w):
        """
        given the global w, populate local w
        :param w:
        :return:
        """
        nids = [n.nid for n in self.nodes]
        self.w = w[nids]

    @functools.cached_property
    def r_bounds(self):
        return self.nodes[0].r, self.nodes[6].r

    @functools.cached_property
    def t_bounds(self):
        if self.nodes[2].theta == 0:
            return self.nodes[0].theta, 2 * np.pi
        return self.nodes[0].theta, self.nodes[2].theta

    @staticmethod
    def p_to_c(r, theta):
        v = np.array([p_to_c(ri, ti) for ri, ti in zip(r, theta)])
        return v[:, 0], v[:, 1]

    def plot(self):
        """
        plot a 9x9 mesh for the 3x3 element
        :return:
        """
        """
        bfns = basis_fns()

        res = 9
        thetas = np.linspace(self.t_bounds[0], self.t_bounds[1], res)
        rs = np.linspace(self.r_bounds[1], self.r_bounds[0], res)
        rgrid, tgrid = np.meshgrid(rs, thetas)
        rvec = rgrid.flatten()
        tvec = tgrid.flatten()

        x, y = self.p_to_c(rvec, tvec)
        z = np.zeros(res ** 2)

        for idx, (ri, ti) in enumerate(zip(rvec, tvec)):
            for wi, bfn in zip(self.w, bfns):
                xb = 2 * (ti - (self.t_bounds[1] + self.t_bounds[0])/2) / (self.t_bounds[1] - self.t_bounds[0])
                yb = 2 * (ri - (self.r_bounds[1] + self.r_bounds[0])/2) / (self.r_bounds[1] - self.r_bounds[0])
                z[idx] += wi * bfn(xb, yb)
        """
        rs = [n.r for n in self.nodes]
        ts = [n.theta for n in self.nodes]
        z = self.w
        x, y = self.p_to_c(rs, ts)

        return x, y, z


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
        ne_circum,
        ne_radial
    ):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.n_circum = ne_circum * 2
        self.n_radial = ne_radial * 2 + 1
        self.ne_circum = ne_circum
        self.ne_radial = ne_radial
        self.nodes = []
        self.elements = []
        self.n_elements = ne_radial * ne_circum
        self.n_nodes = self.n_radial * self.n_circum
        self.boundary_node_ids = []

        LOGGER.info("Generating torus mesh")

        for ri in range(self.n_radial):
            for ci in range(self.n_circum):
                theta = (ci / self.n_circum) * 2 * np.pi
                r = ((outer_radius - inner_radius) * (ri / (self.n_radial - 1))) + inner_radius
                x, y = p_to_c(r, theta)
                node = Node(ri, ci, self.n_circum, x, y, 0)
                if ri == 0 or ri == self.n_radial - 1:
                    node.on_boundary = True
                    self.boundary_node_ids.append(node.nid)
                self.nodes.append(node)

        e_id = 0
        for re in range(ne_radial):
            for ce in range(ne_circum):
                node_idx = self.id_map(ce*2, re*2)
                n1 = self.nodes[node_idx]
                n2 = self.nodes[n1.r_neighbor]
                n3 = self.nodes[n2.r_neighbor]
                n4 = self.nodes[n1.u_neighbor]
                n5 = self.nodes[n2.u_neighbor]
                n6 = self.nodes[n3.u_neighbor]
                n7 = self.nodes[n4.u_neighbor]
                n8 = self.nodes[n5.u_neighbor]
                n9 = self.nodes[n6.u_neighbor]

                self.elements.append(
                    Element(e_id, [n1, n2, n3, n4, n5, n6, n7, n8, n9])
                )
                e_id += 1

        bc_fn = lambda ri, ti: ri
        D, b = self.gen_linear_system(bc_fn)

        LOGGER.info('solving linear system')

        # w, error = spl.gmres(D, b, maxiter=10000)  # noqa

        # if error > 0:
        #     LOGGER.warning(f'No convergence in {error} iterations of gmres')

        w = spl.spsolve(D, b)

        # get w back into elements
        for eidx, element in enumerate(self.elements):
            self.elements[eidx].get_w(w)

        LOGGER.info('plotting elements')
        x, y, z = np.zeros(0), np.zeros(0), np.zeros(0)

        for element in self.elements:
            xe, ye, ze = element.plot()
            x = np.concatenate([x, xe])
            y = np.concatenate([y, ye])
            z = np.concatenate([z, ze])

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # ax.set_title(f"Element: {element.id}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.scatter(x, y, z)
        plt.show()


    def gen_linear_system(self, bc_fn):
        """
        bc_values must have length n_circum for dirchlet boundary condition
        the master element is a square with coord basis x-y
        the pd is in theta-r
        :param bc_fn: function(r, theta) to apply on boundary
        :return:
        """

        i = []
        j = []
        d = []
        b = np.zeros(self.n_nodes)

        LOGGER.info("Computing element matrix")
        for element in self.elements:
            self.bi_quad_me(element)
            for di in range(9):
                if element.nodes[di].on_boundary:
                    continue
                for dj in range(9):
                    i.append(element.nodes[di].nid)
                    j.append(element.nodes[dj].nid)
                    d.append(element.D[di][dj])

                b[element.nodes[di].nid] += element.b[di]

        LOGGER.info('applying boundary conditions')

        for nid in self.boundary_node_ids:
            node = self.nodes[nid]
            i.append(node.nid)
            j.append(node.nid)
            d.append(1)
            b[node.nid] = bc_fn(node.r, node.theta)

        D = csr_matrix((np.array(d), (np.array(i), np.array(j))), shape=(self.n_nodes, self.n_nodes))
        return D, b

    def bi_quad_me(self, element: Element, k=lambda r, t: 1, mms_lap=lambda r, t: 0):
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
        :param element: element containing all nodes in the element
        :param k: heat diffusion constant across the torus, function of r
        :param mms_lap: method of manufacture solutions laplacian functino
        :return:
        i: list of basis fn numbers
        j: list of basis fn numbers
        c: list of d_i,j values
        """
        basis_fns, dbdxe_fns, dbdye_fns = self.bi_quad_basis_fns
        nodes = element.nodes
        e_idxs = list(range(9))

        # dtheta_dx = (nodes[2].theta - nodes[0].theta) / 2
        # dr_dy = (nodes[6].r - nodes[0].r) / 2
        # det_j = np.abs(dtheta_dx * dr_dy)
        # det_j = lambda r: 1/r
        det_j = np.zeros(25)

        dtheta_dx = np.zeros(25)
        dtheta_dy = np.zeros(25)
        dr_dx = np.zeros(25)
        dr_dy = np.zeros(25)

        # dx_dtheta = np.zeros(25)
        # dx_dr = np.zeros(25)
        # dy_dtheta = np.zeros(25)
        # dy_dr = np.zeros(25)

        rphys = np.zeros(25)
        tphys = np.zeros(25)

        # compute these partial derivatives for all gauss points
        gx, gy, gw = self.gauss_points5

        for idx, (gxi, gyi) in enumerate(zip(gx, gy)):
            for db_dy, db_dx, bfn, node in zip(dbdye_fns, dbdxe_fns, basis_fns, nodes):
                rphys[idx] += node.r * bfn(gxi, gyi)
                tphys[idx] += node.theta * bfn(gxi, gyi)

                dtheta_dx[idx] += node.theta * db_dx(gxi, gyi)
                dtheta_dy[idx] += node.theta * db_dy(gxi, gyi)
                dr_dx[idx] += node.r * db_dx(gxi, gyi)
                dr_dy[idx] += node.r * db_dy(gxi, gyi)

                # dx_dtheta[idx] += node.r * db_dy(gxi, gyi) / det_j(np.sqrt((node.x + gxi) ** 2 + (node.y + gyi) ** 2))
                # dx_dr[idx] += node.theta * db_dy(gxi, gyi) / -det_j(np.sqrt((node.x + gxi) ** 2 + (node.y + gyi) ** 2))
                # dy_dtheta[idx] += node.r * db_dx(gxi, gyi) / -det_j(np.sqrt((node.x + gxi) ** 2 + (node.y + gyi) ** 2))
                # dy_dr[idx] += node.theta * db_dx(gxi, gyi) / det_j(np.sqrt((node.x + gxi) ** 2 + (node.y + gyi) ** 2))

        det_j = dtheta_dx * dr_dy - dtheta_dy * dr_dx
        dx_dtheta = dr_dy / det_j
        dx_dr = -dtheta_dy / det_j
        dy_dtheta = -dr_dx / det_j
        dy_dr = dtheta_dx / det_j

        kphys = np.array([k(r, t) for r, t in zip(rphys, tphys)])

        for idx1, node1, basis1, dbxe1, dbye1 in zip(e_idxs, nodes, basis_fns, dbdxe_fns, dbdye_fns):
            for idx2, node2, basis2, dbxe2, dbye2 in zip(e_idxs[idx1:], nodes[idx1:], basis_fns[idx1:], dbdxe_fns[idx1:], dbdye_fns[idx1:]):
                d = 0
                for gidx, (gxi, gyi, gwi) in enumerate(zip(gx, gy, gw)):
                    db_dr1 = dbxe1(gxi, gyi) * dx_dr[gidx] + dbye1(gxi, gyi) * dy_dr[gidx]
                    db_dt1 = dbxe1(gxi, gyi) * dx_dtheta[gidx] + dbye1(gxi, gyi) * dy_dtheta[gidx]
                    db_dr2 = dbxe2(gxi, gyi) * dx_dr[gidx] + dbye2(gxi, gyi) * dy_dr[gidx]
                    db_dt2 = dbxe2(gxi, gyi) * dx_dtheta[gidx] + dbye2(gxi, gyi) * dy_dtheta[gidx]
                    d += kphys[gidx] * gwi * det_j[gidx] * (db_dr1 * db_dr2 + db_dt1 * db_dt2)
                element.D[idx1, idx2] = d
                element.D[idx2, idx1] = d

            for gidx, (gxi, gyi, gwi) in enumerate(zip(gx, gy, gw)):
                element.b[idx1] -= gwi * det_j[gidx] * basis1(gxi, gyi) * mms_lap(rphys[gidx], tphys[gidx])

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

    @functools.cached_property
    def bi_quad_basis_fns(self):

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

        basis_fns = [b1, b2, b3, b4, b5, b6, b7, b8, b9]
        dbdxe_fns = [db1_dxe, db2_dxe, db3_dxe, db4_dxe, db5_dxe, db6_dxe, db7_dxe, db8_dxe, db9_dxe]
        dbdye_fns = [db1_dye, db2_dye, db3_dye, db4_dye, db5_dye, db6_dye, db7_dye, db8_dye, db9_dye]

        return basis_fns, dbdxe_fns, dbdye_fns
