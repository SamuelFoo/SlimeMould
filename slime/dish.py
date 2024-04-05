import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from ..slime.cell import Cell
from ..slime.diffusion_params import DEFAULT_DIFFUSION_PARAMS, DiffusionParams
from ..slime.food import FoodCell
from ..slime.mould import Mould


class Dish:
    def __init__(
        self,
        dish_shape: tuple,
        foods: pd.DataFrame,
        start_loc: tuple,
        mould_shape: tuple,
        init_mould_coverage: float,
        decay: float,
        diffusion_params: DiffusionParams = DEFAULT_DIFFUSION_PARAMS,
    ):
        self.lattice = self.initialise_dish(dish_shape)
        self.dish_size = dish_shape[0] * dish_shape[1]
        self.all_foods = {}
        self.all_foods_coords = []
        self.food_positions = {}
        self.food_graph = nx.Graph()
        self.initialise_food(foods)
        self.mould = self.initialise_slime_mould(
            self,
            start_loc,
            mould_shape,
            init_mould_coverage,
            decay,
            diffusion_params=diffusion_params,
        )

    @staticmethod
    def initialise_dish(dish_shape):
        """
        initialise the dish lattice
        :param dish_shape: the shape of the dish lattice
        :return: dish lattice
        """
        lattice = np.empty(dish_shape, object)
        for i in np.ndindex(dish_shape):
            lattice[i] = Cell()
        return lattice

    def initialise_food(self, foods):
        """
        Adds food cells in a square with length size
        """
        self.food_positions_array = np.array([foods["x"], foods["y"]]).T

        for i, station in foods.iterrows():
            coord = (station["x"], station["y"])
            value = station["value"]

            self.food_positions[i] = coord

            for x in range(value // 2):
                for y in range(value // 2):
                    food_coord = (coord[0] - x, coord[1] - y)
                    food = FoodCell(food_id=i, food_coord=food_coord)
                    self.lattice[food_coord] = food

                    # add food coord
                    self.all_foods_coords.append(food_coord)

                    # add all foods
                    if i not in self.all_foods:
                        self.all_foods[i] = [food]
                    else:
                        self.all_foods[i].append(food)

        self.food_graph.add_nodes_from(self.food_positions)

    @staticmethod
    def initialise_slime_mould(
        dish,
        start_loc,
        mould_shape,
        init_mould_coverage,
        decay,
        diffusion_params: DiffusionParams,
    ):
        """
        initialise the mould
        """
        return Mould(
            dish,
            start_loc,
            mould_shape,
            init_mould_coverage,
            decay,
            diffusion_params=diffusion_params,
        )

    @staticmethod
    def pheromones(lattice):
        """
        Returns a lattice of just the pheromones to draw the graph
        """
        pheromones = np.zeros_like(lattice, dtype=float)
        for i in np.ndindex(lattice.shape):
            pheromones[i] = lattice[i].pheromone
        return pheromones

    def run(self, frames):
        data = []
        for frame in range(frames):
            print(f"Running {frame} out of {frames}")
            self.mould.evolve()
            data.append(self.pheromones(self.lattice).T.copy())

        return data

    def get_all_foods_coords(self):
        return self.all_foods_coords

    def get_all_foods(self):
        return self.all_foods

    def get_lattice(self):
        return self.lattice

    def set_lattice(self, idx, obj):
        self.lattice[idx] = obj

    def get_food_position(self, food_id):
        return self.food_positions[food_id]

    def get_food_positions(self, food_ids):
        return self.food_positions_array[np.array(list(food_ids), dtype=np.int64)]

    def add_food_edge(self, source, target):
        self.food_graph.add_edge(source, target)

    def get_food_graph(self):
        return self.food_graph

    def get_dish_size(self):
        return self.dish_size
