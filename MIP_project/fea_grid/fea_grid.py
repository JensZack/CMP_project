"""
Class and functions to model materials used in modeling heat flux
"""
import functools
from dataclasses import dataclass
import numpy as np
import scipy.sparse.linalg as spl
import matplotlib.pyplot as plt
import logging
from typing import Optional, Callable

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


@dataclass
class MMS:
    fn: Callable
    lap: Callable
    description: str = ""
    name: Optional[str] = None


@functools.lru_cache
def p_to_c(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


@functools.lru_cache
def c_to_p(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
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


@functools.lru_cache
def bi_quad_basis_fns():
    l1 = np.vectorize(lambda xe: (1 / 2) * xe * (xe - 1))
    l2 = np.vectorize(lambda xe: -(xe - 1) * (xe + 1))
    l3 = np.vectorize(lambda xe: (1 / 2) * xe * (xe + 1))

    dl1 = np.vectorize(lambda xe: xe - .5)
    dl2 = np.vectorize(lambda xe: -2 * xe)
    dl3 = np.vectorize(lambda xe: xe + .5)

    # basis fns:
    b1 = lambda xe, ye: l1(xe) * l1(ye)
    b2 = lambda xe, ye: l2(xe) * l1(ye)
    b3 = lambda xe, ye: l3(xe) * l1(ye)
    b4 = lambda xe, ye: l1(xe) * l2(ye)
    b5 = lambda xe, ye: l2(xe) * l2(ye)
    b6 = lambda xe, ye: l3(xe) * l2(ye)
    b7 = lambda xe, ye: l1(xe) * l3(ye)
    b8 = lambda xe, ye: l2(xe) * l3(ye)
    b9 = lambda xe, ye: l3(xe) * l3(ye)

    db1_dxe = lambda xe, ye: dl1(xe) * l1(ye)
    db2_dxe = lambda xe, ye: dl2(xe) * l1(ye)
    db3_dxe = lambda xe, ye: dl3(xe) * l1(ye)
    db4_dxe = lambda xe, ye: dl1(xe) * l2(ye)
    db5_dxe = lambda xe, ye: dl2(xe) * l2(ye)
    db6_dxe = lambda xe, ye: dl3(xe) * l2(ye)
    db7_dxe = lambda xe, ye: dl1(xe) * l3(ye)
    db8_dxe = lambda xe, ye: dl2(xe) * l3(ye)
    db9_dxe = lambda xe, ye: dl3(xe) * l3(ye)

    db1_dye = lambda xe, ye: l1(xe) * dl1(ye)
    db2_dye = lambda xe, ye: l2(xe) * dl1(ye)
    db3_dye = lambda xe, ye: l3(xe) * dl1(ye)
    db4_dye = lambda xe, ye: l1(xe) * dl2(ye)
    db5_dye = lambda xe, ye: l2(xe) * dl2(ye)
    db6_dye = lambda xe, ye: l3(xe) * dl2(ye)
    db7_dye = lambda xe, ye: l1(xe) * dl3(ye)
    db8_dye = lambda xe, ye: l2(xe) * dl3(ye)
    db9_dye = lambda xe, ye: l3(xe) * dl3(ye)

    basis_fns = [b1, b2, b3, b4, b5, b6, b7, b8, b9]
    dbdxe_fns = [db1_dxe, db2_dxe, db3_dxe, db4_dxe, db5_dxe, db6_dxe, db7_dxe, db8_dxe, db9_dxe]
    dbdye_fns = [db1_dye, db2_dye, db3_dye, db4_dye, db5_dye, db6_dye, db7_dye, db8_dye, db9_dye]

    return basis_fns, dbdxe_fns, dbdye_fns


@functools.lru_cache
def gauss_points5():
    x1 = -(1 / 3) * np.sqrt(5 + 2 * np.sqrt(10 / 7))
    x2 = -(1 / 3) * np.sqrt(5 - 2 * np.sqrt(10 / 7))
    x3 = 0
    x4 = -x2
    x5 = -x1

    w1 = (322 - 13 * np.sqrt(70)) / 900
    w2 = (322 + 13 * np.sqrt(70)) / 900
    w3 = 128 / 225
    w4 = w2
    w5 = w1

    x = np.array([x1, x2, x3, x4, x5] * 5)
    y = np.array([x1] * 5 + [x2] * 5 + [x3] * 5 + [x4] * 5 + [x5] * 5)
    w = np.array([w1 * w1, w1 * w2, w1 * w3, w1 * w4, w1 * w5,
                  w2 * w1, w2 * w2, w2 * w3, w2 * w4, w2 * w5,
                  w3 * w1, w3 * w2, w3 * w3, w3 * w4, w3 * w5,
                  w4 * w1, w4 * w2, w4 * w3, w4 * w4, w4 * w5,
                  w5 * w1, w5 * w2, w5 * w3, w5 * w4, w5 * w5])

    return x, y, w


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


@dataclass
class Element:
    """
    Class for holding all nodes in a given 3x3 element

    This class also contains plotting functionality as well as
    the ability to evaulate the basis functions for a point in the element

    :param id: id for the element in the mesh, starts at 0
    :param nodes: list of Nodes that constitute the element
    :param D: Element local D matrix, defaults to np.zeros
    :param b: Element local b array, defaults to np.zeros
    :param w: Weights assigned to the basis functinos, defaults to np.zeros
    """
    id: int
    nodes: list[Node]
    D: np.ndarray = np.zeros((9, 9))
    b: np.ndarray = np.zeros(9)
    w: np.ndarray = np.zeros(9)

    @functools.cached_property
    def nids(self):
        return [n.nid for n in self.nodes]

    def get_w(self, w):
        """
        given the global w, populate local w

        :param w: np.ndarray with a size of n_nodes
        """
        nids = [n.nid for n in self.nodes]
        self.w = w[nids]

        for wn, node in zip(self.w, self.nodes):
            node.z = wn

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

    def plot_nodes(self):
        """
        generate the x, y, z coordinates for all of the nodes in the element

        :return x:
        :return y:
        :return z:
        """
        rs = [n.r for n in self.nodes]
        ts = [n.theta for n in self.nodes]
        z = self.w
        x, y = self.p_to_c(rs, ts)

        return x, y, z

    def plot_mesh(self, res=9):
        """
        plot the element with a mesh of values interpolating the computed basis function

        :param res: the resolution for generating the mesh on an element, if res=9, there will be 81 points per element

        :return x:
        :return y:
        :return z:
        """

        bfns, _, _ = bi_quad_basis_fns()

        thetas = np.linspace(self.t_bounds[0], self.t_bounds[1], res)
        rs = np.linspace(self.r_bounds[1], self.r_bounds[0], res)
        rgrid, tgrid = np.meshgrid(rs, thetas)
        rvec = rgrid.flatten()
        tvec = tgrid.flatten()

        x, y = self.p_to_c(rvec, tvec)
        z = np.zeros(res ** 2)

        for idx, (ri, ti) in enumerate(zip(rvec, tvec)):
            for wi, bfn in zip(self.w, bfns):
                xb = 2 * (ti - (self.t_bounds[1] + self.t_bounds[0]) / 2) / (self.t_bounds[1] - self.t_bounds[0])
                yb = 2 * (ri - (self.r_bounds[1] + self.r_bounds[0]) / 2) / (self.r_bounds[1] - self.r_bounds[0])
                z[idx] += wi * bfn(xb, yb)
        return x, y, z

    def eval(self, r: np.ndarray, t: np.ndarray):
        """
        evaluate a point on the element given the global r and theta values

        :param r: global r
        :param t: global theta

        :return z: sum of basis functions evaluated at r, theta, times their respective weights
        """
        z = np.zeros(r.size)
        bfns, _, _ = bi_quad_basis_fns()
        for idx, (ri, ti) in enumerate(zip(r, t)):
            for wi, bfn in zip(self.w, bfns):
                xb = 2 * (ti - (self.t_bounds[1] + self.t_bounds[0]) / 2) / (self.t_bounds[1] - self.t_bounds[0])
                yb = 2 * (ri - (self.r_bounds[1] + self.r_bounds[0]) / 2) / (self.r_bounds[1] - self.r_bounds[0])
                z[idx] += wi * bfn(xb, yb)
        return z


def rotation_mat(theta):
    """
    :param theta: rotation parameter in radians
    :return:
    """
    return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])


class GridTorus:
    def __init__(
            self,
            inner_radius,
            outer_radius,
            ne_circum,
            ne_radial
    ):
        """
        Class to generate a torus mesh to compute a finite element solution to Laplace's equation

        :param inner_radius:
        :param outer_radius:
        :param ne_circum: number of elements around each circle, these elements shift with theta
        :param ne_radial: number of elements along the radius of the torus
        """
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
                node_idx = self.id_map(ce * 2, re * 2)
                n1 = self.nodes[node_idx]
                n2 = self.nodes[n1.r_neighbor]
                n3 = self.nodes[n2.r_neighbor]
                n4 = self.nodes[n1.u_neighbor]
                n5 = self.nodes[n2.u_neighbor]
                n6 = self.nodes[n3.u_neighbor]
                n7 = self.nodes[n4.u_neighbor]
                n8 = self.nodes[n5.u_neighbor]
                n9 = self.nodes[n6.u_neighbor]

                element = Element(e_id, [n1, n2, n3, n4, n5, n6, n7, n8, n9])

                self.elements.append(element)
                e_id += 1

        self.D = csr_matrix(([], ([], [])), shape=(self.n_nodes, self.n_nodes))
        self.b = np.zeros(self.n_nodes)

    def gen_linear_system(self, bc_fn, k_fn=None, mms=None):
        """
        Iterating over all elements in the mesh, generate the local D and b
        matrix and vector for each element using the k function and mms.lap if mms
        is supplied.

        Once these local D and b variables are populated, incorporate them into the global
        sparse D matrix and b vector, then apply boundary conditions to the nodes on the inside and outside
        of the torus.

        :param bc_fn: function(x, y) to apply on boundary
        :param k_fn: function(x, y) for the thermal diffusitivity
        :param mms: MMS object containing a laplacian to apply to the b vector
        """

        b = np.zeros(self.n_nodes)
        D = csr_matrix(([], ([], [])), shape=(self.n_nodes, self.n_nodes))

        # default k_fn to k(x, y) = 1
        if k_fn is None:
            k_fn = lambda x, y: 1

        LOGGER.info("Computing element matrix")
        for element in self.elements:
            self.bi_quad_me(element, k=k_fn, mms=mms)
            i = []
            j = []
            d = []
            for di in range(9):
                if element.nodes[di].on_boundary:
                    continue
                for dj in range(9):
                    i.append(element.nodes[di].nid)
                    j.append(element.nodes[dj].nid)
                    d.append(element.D[di][dj])

                b[element.nodes[di].nid] += element.b[di]
            Dtmp = csr_matrix((d, (i, j)), shape=(self.n_nodes, self.n_nodes))
            D += Dtmp

        LOGGER.info('applying boundary conditions')

        i = []
        j = []
        d = []
        for nid in self.boundary_node_ids:
            node = self.nodes[nid]
            i.append(node.nid)
            j.append(node.nid)
            d.append(1)
            b[node.nid] = bc_fn(node.x, node.y)

        Dtmp = csr_matrix((d, (i, j)), shape=(self.n_nodes, self.n_nodes))
        D += Dtmp
        self.D = D
        self.b = b

    @staticmethod
    def bi_quad_me(element: Element, k, mms=None):
        """
        For an element of the fea mesh, generate the D matrix and the b vector
        given k and possibly a function for the mms laplacian. This method will
        save the D and b results into the element, so it returns nothing

        xi, yi are the coordinate names in the master element
        x, y are the coordinates from the physical domain

        master element layout:
           7 | 8 | 9 \n
           4 | 5 | 6 \n
           1 | 2 | 3 \n

        :param element: element containing all nodes in the element
        :param k: heat diffusion constant across the torus, function of x, y
        :param mms: method of manufacture object containing laplacian funtion
        """
        basis_fns, dbdxe_fns, dbdye_fns = bi_quad_basis_fns()
        nodes = element.nodes
        e_idxs = list(range(9))

        dx_dxi = np.zeros(25)
        dx_dyi = np.zeros(25)
        dy_dxi = np.zeros(25)
        dy_dyi = np.zeros(25)

        yphys = np.zeros(25)
        xphys = np.zeros(25)

        # compute these partial derivatives for all gauss points
        gx, gy, gw = gauss_points5()
        gidx = np.arange(25)

        for gidx, (gxi, gyi) in enumerate(zip(gx, gy)):
            for db_dy, db_dx, bfn, node in zip(dbdye_fns, dbdxe_fns, basis_fns, nodes):
                yphys[gidx] += node.y * bfn(gxi, gyi)
                xphys[gidx] += node.x * bfn(gxi, gyi)

                dx_dxi[gidx] += node.x * db_dx(gxi, gyi)
                dx_dyi[gidx] += node.x * db_dy(gxi, gyi)
                dy_dxi[gidx] += node.y * db_dx(gxi, gyi)
                dy_dyi[gidx] += node.y * db_dy(gxi, gyi)

        det_j = dx_dxi * dy_dyi - dx_dyi * dy_dxi
        dxi_dx = dy_dyi / det_j
        dxi_dy = -dx_dyi / det_j
        dyi_dx = -dy_dxi / det_j
        dyi_dy = dx_dxi / det_j

        kphys = np.array([k(x, y) for x, y in zip(xphys, yphys)])

        for idx1, node1, basis1, dbxe1, dbye1 in zip(e_idxs, nodes, basis_fns, dbdxe_fns, dbdye_fns):
            for idx2, node2, basis2, dbxe2, dbye2 in zip(e_idxs[idx1:], nodes[idx1:], basis_fns[idx1:],
                                                         dbdxe_fns[idx1:], dbdye_fns[idx1:]):
                d = 0
                for gidx, (gxi, gyi, gwi) in enumerate(zip(gx, gy, gw)):
                    db_dy1 = dbxe1(gxi, gyi) * dxi_dy[gidx] + dbye1(gxi, gyi) * dyi_dy[gidx]
                    db_dx1 = dbxe1(gxi, gyi) * dxi_dx[gidx] + dbye1(gxi, gyi) * dyi_dx[gidx]
                    db_dy2 = dbxe2(gxi, gyi) * dxi_dy[gidx] + dbye2(gxi, gyi) * dyi_dy[gidx]
                    db_dx2 = dbxe2(gxi, gyi) * dxi_dx[gidx] + dbye2(gxi, gyi) * dyi_dx[gidx]
                    d += kphys[gidx] * gwi * det_j[gidx] * (db_dy1 * db_dy2 + db_dx1 * db_dx2)
                element.D[idx1, idx2] = d
                element.D[idx2, idx1] = d

            if mms is not None:
                for gidx, (gxi, gyi, gwi) in enumerate(zip(gx, gy, gw)):
                    element.b[idx1] -= gwi * det_j[gidx] * basis1(gxi, gyi) * mms.fn(xphys[gidx], yphys[gidx])


    def solve_linear_system(self):
        """
        After the method gen_linear_system has run, the D and b variables are populated with a sparse D matrix
        and a dense b vector.

        Use scipy.sparse.linalg.gmres to solve the system of equations and then save the resulting values
        for w into their respective Elements, Nodes

        This method will log a warning if the gmres algorithm does not converge
        """
        LOGGER.info('solving linear system')
        max_iter = 1e5
        w, error = spl.gmres(self.D, self.b, maxiter=max_iter)  # noqa
        if error > 0:
            LOGGER.warning(f'No convergence in gmres after {error} iterations out of {max_iter}')

        # get w back into elements
        for eidx, element in enumerate(self.elements):
            self.elements[eidx].get_w(w)

    def plot(self, plot_type='mesh', mms: Optional[MMS] = None, res=9):
        """
        Generate a plot of the fea solution. Selecting from two types of plots
         * mesh: plot a mesh of interpolated points over each element
         * nodes: plot only the weights at each node, and plot the grid connecting nodes

        If mms is passed it will also plot the correct mms solution in the first figure. It will also
        create a second figure plotting the difference between the mms solution and the fea solution

        :param plot_type: type of plot to generate, options: 'mesh', 'nodes'
        :param mms: An MMS object that contains the mms function and a description of the function
        :param res: the resolution to plot is plot_type is mesh
        """
        LOGGER.info('plotting elements')
        if plot_type == 'mesh':
            x, y, z = np.zeros(0), np.zeros(0), np.zeros(0)
        elif plot_type == 'nodes':
            x, y, z = np.zeros(self.n_nodes), np.zeros(self.n_nodes), np.zeros(self.n_nodes)
        else:
            raise NotImplementedError(f"Invalid plot type {plot_type}, select from ['mesh', 'nodes']")

        for element in self.elements:
            if plot_type == 'mesh':
                xe, ye, ze = element.plot_mesh(res=res)
                x = np.concatenate([x, xe])
                y = np.concatenate([y, ye])
                z = np.concatenate([z, ze])
            else:
                xe, ye, ze = element.plot_nodes()
                x[element.nids] = xe
                y[element.nids] = ye
                z[element.nids] = ze

        if mms is not None:
            fig, axs = plt.subplots(ncols=2, subplot_kw={"projection": "3d"})
            ax0 = axs[0]
            ax1 = axs[1]
        else:
            fig, ax0 = plt.subplots(subplot_kw={"projection": "3d"})
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")
        ax0.set_zlabel("z")
        ax0.scatter(x, y, z, color='r', label='FEA solution')
        if plot_type == 'nodes':
            self._add_mesh_lines(ax0, color='r')

        if mms is not None:
            f = np.vectorize(mms.fn)
            fz = f(x, y)
            ax0.scatter(x, y, fz, color='b', label='MMS function')
            if plot_type == 'nodes':
                self._add_mesh_lines(ax0, z=fz, color='b')

            diff = z - fz
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1.set_zlabel("z")
            ax1.scatter(x, y, diff, color='r', label='MMS - FEA error')
            if plot_type == 'nodes':
                self._add_mesh_lines(ax1, z=diff, color='r')
            ax1.legend()
        ax0.legend()
        fig.show()

    def _add_mesh_lines(self, ax, z=None, **kwargs):
        if z is None:
            for node in self.nodes:
                for nid in [node.l_neighbor, node.u_neighbor, node.r_neighbor, node.d_neighbor]:
                    if nid not in range(self.n_nodes):
                        continue
                    n = self.nodes[nid]
                    ax.plot([node.x, n.x],
                            [node.y, n.y],
                            [node.z, n.z],
                            **kwargs)
        else:
            for node in self.nodes:
                for nid in [node.l_neighbor, node.u_neighbor, node.r_neighbor, node.d_neighbor]:
                    if nid not in range(self.n_nodes):
                        continue
                    n = self.nodes[nid]
                    ax.plot([node.x, n.x],
                            [node.y, n.y],
                            [z[node.nid], z[n.nid]],
                            **kwargs)

    def id_map(self, r, s):
        """
        :param r: increment around the circle angle starts at 0
        :param s: increment distance from inner radius starts at 0

        :return: int: id starts at 0
        """
        return r + (s * self.n_circum)

