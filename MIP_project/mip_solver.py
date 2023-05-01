from material_formats import Material
import numpy as np


class MIPNode:

    def __init__(self):



class MIPSolver:

    def __init__(self, width: float, materials: list[Material], repeating_layers: int = 5):
        self.width = width
        self.materials = materials
        self.repeating_layers = repeating_layers
        self.y = np.zeros((len(materials), repeating_layers), dtype=float)
        # self.graph ?

    def bound(self):
        """
        find the lower bound of the relaxed problem where self.y is allowed to take on float values
        :return:
        """

    def branch(self):
        """
        branch from the relaxed solution to a solution where y_{i, j} is bounded above by floor and below
        by ceil
        :return:
        """

    def _expansion_constraint_eq(self, layers):
        """
        constraining the layers set to have a maximum thermal expansion * overall width
        LINEAR CONSTRAINT
        """
        thermal_expansion_constraint = 1e-5  # meters per meter width total per Kelvin
        total_thermal_expansion = 0
        total_width = 0
        for idx, layer in enumerate(layers):
            material_width = self.materials[idx].layer_width * layer
            total_width += material_width
            total_thermal_expansion += material_width * self.materials[idx].thermal_expansion

        thermal_expansion = total_thermal_expansion / self.width
        return thermal_expansion < thermal_expansion_constraint


    def _price_constraint_eq(self, layers):
        """ """

    def _width_constraint_eq(self, layers):
        """ """

    def _mass_constraint_eq(self, layers):
        """ """

    def _cost_eq(self, layers):
        """
        The function being optimized over, in this case, minimizing heat flux through set of layers of materials
        :param layers:
        :return:
        """
