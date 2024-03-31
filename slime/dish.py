import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from ..slime.cell import Cell
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
    ):
        self.lattice = self.initialise_dish(dish_shape)
        self.dish_size = dish_shape[0] * dish_shape[1]
        self.all_foods = {}
        self.all_foods_idx = []
        self.food_positions = {}
        self.food_graph = nx.Graph()
        self.initialise_food(foods)
        self.mould = self.initialise_slime_mould(
            self, start_loc, mould_shape, init_mould_coverage, decay
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
            idx = (station["x"], station["y"])
            value = station["value"]

            self.food_positions[i] = idx

            for x in range(value // 2):
                for y in range(value // 2):
                    food_idx = (idx[0] - x, idx[1] - y)
                    food = FoodCell(food_id=i, food_idx=food_idx)
                    self.lattice[food_idx] = food

                    # add food idx
                    self.all_foods_idx.append(food_idx)

                    # add all foods
                    if i not in self.all_foods:
                        self.all_foods[i] = [food]
                    else:
                        self.all_foods[i].append(food)

        self.food_graph.add_nodes_from(self.food_positions)

    @staticmethod
    def initialise_slime_mould(
        dish, start_loc, mould_shape, init_mould_coverage, decay
    ):
        """
        initialise the mould
        """
        return Mould(dish, start_loc, mould_shape, init_mould_coverage, decay)

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

    def get_all_foods_idx(self):
        return self.all_foods_idx

    def get_all_foods(self):
        return self.all_foods

    def get_lattice(self):
        return self.lattice

    def set_lattice(self, idx, obj):
        self.lattice[idx] = obj

    def get_food_nodes(self):
        return self.food_graph.nodes()

    def get_food_position(self, food_id):
        return self.food_positions[food_id]

    def get_food_positions(self, food_idxs):
        return self.food_positions_array[np.array(list(food_idxs)).astype(np.int64)]

    def add_food_edge(self, source, target):
        self.food_graph.add_edge(source, target)

    def get_food_graph(self):
        return self.food_graph

    def get_dish_size(self):
        return self.dish_size
