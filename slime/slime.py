import math
from collections import deque

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

from ..slime.cell import Cell
from ..slime.diffusion_params import DiffusionParams


def get_neighbours(idx):
    """
    get the tuple index of the neighbours of the input slime cell
    :param idx: the tuple index of the given slime cell
    :return: the tuple index of the neighbours
    """
    return [
        (idx[0] - 1, idx[1] - 1),  # up   1, left  1
        (idx[0] - 1, idx[1]),  # up   1,
        (idx[0] - 1, idx[1] + 1),  # up   1, right 1
        (idx[0], idx[1] - 1),  # , left  1
        (idx[0], idx[1] + 1),  # , right 1
        (idx[0] + 1, idx[1] - 1),  # down 1, left  1
        (idx[0] + 1, idx[1]),  # down 1,
        (idx[0] + 1, idx[1] + 1),  # down 1, right 1
    ]


class SlimeCell(Cell):

    def __init__(
        self,
        coord: tuple,
        pheromone: float,
        mould,
        dish,
        is_capital,
        diffusion_params: DiffusionParams,
    ):
        super().__init__(pheromone=pheromone, cell_type=1)

        self.coord = coord
        self.pheromone = pheromone
        self.max_ph = 4
        self.is_capital = is_capital
        self.reached_food_id = None
        self.mould = mould
        self.dish = dish
        self.food_path = []

        # Diffusion Parameters
        self.diffusion_threshold = diffusion_params.diffusion_threshold
        self.diffusion_decay_rate = diffusion_params.diffusion_decay_rate
        self.distance_for_diffusion_threshold = (
            diffusion_params.distance_for_diffusion_threshold
        )
        self.moving_threshold = diffusion_params.moving_threshold
        self.max_ph_threshold = diffusion_params.max_ph_threshold
        self.max_ph_increase_step = diffusion_params.max_ph_increase_step

        # (food_id, food_coord)
        self.step_food = None
        self.curr_target = None

    def find_nearest_food(self, food_ids):
        """
        find the nearest food of the slime cell
        :param food_ids: a list of food ids
        :return: min_i: the nearest food id, min_dist: distance
        """
        if len(food_ids) == 0:
            min_dist = -1
            min_i = 0
            return min_i, min_dist

        dists = cdist(self.dish.get_food_positions(food_ids), [np.array(self.coord)])
        argmin = dists.argmin()
        min_i = list(food_ids)[argmin]
        min_dist = dists[argmin][0]

        return min_i, min_dist

    def set_reached_food_path(self):
        """
        setup the food path the slime cell will go through
        """
        # target food (food_id, food_coord)
        target_food_id = self.mould.get_current_target()[0]
        self.curr_target = target_food_id

        mould_reached_food_ids = self.mould.get_reached_food_ids()

        if len(mould_reached_food_ids) == 0:
            self.food_path.append(self.mould.get_current_target()[0])

        else:
            # find the nearest food
            min_i = self.find_nearest_food(food_ids=self.mould.get_reached_food_ids())[
                0
            ]

            nearest_target = self.mould.get_nearest_connected_target()
            if nearest_target != -1:
                # find the shortest path from the nearest food to the target food
                self.food_path = nx.shortest_path(
                    G=self.dish.get_food_graph(), source=min_i, target=nearest_target
                )
            else:
                self.food_path.append(min_i)
            self.food_path.append(target_food_id)
        step_food_id = self.food_path.pop(0)
        self.step_food = (step_food_id, self.dish.get_food_position(step_food_id))

    def reset_step_food(self):
        """
        The step food is the food in the food path the slime cell will go next.
        reset the step food of the current slime cell
        """
        # reached target
        if (
            self.reached_food_id == self.curr_target
            and self.curr_target == self.mould.get_current_target()[0]
        ):
            return

        if self.step_food is None or len(self.food_path) == 0:
            self.set_reached_food_path()
        else:
            # reached step food
            step_food_coord = self.dish.get_food_position(self.step_food[0])
            if math.dist(step_food_coord, self.coord) < 3:
                step_food_id = self.food_path.pop(0)
                self.step_food = (
                    step_food_id,
                    self.dish.get_food_position(step_food_id),
                )

    def get_diffusion_direction(self):
        """
        set up the next main diffusion direction based on the location of the step food
        """
        if self.reached_food_id == self.step_food:
            self.reset_step_food()

        food_coord = np.array(self.step_food[1])
        coord = np.array(self.coord)

        return np.sign(food_coord - coord).astype(np.int64)

    @staticmethod
    def check_boundary(idx, lattice_shape):
        """
        check if the neighbour cell reached the boundary of the lattice
        :param idx: the location of the neighbour cell
        :param lattice_shape: the shape of the lattice
        :return: false if reach boundary, else true
        """
        if (
            idx[0] >= lattice_shape[0]
            or idx[0] <= 0
            or idx[1] >= lattice_shape[1]
            or idx[1] <= 0
        ):
            return False
        return True

    # diffusion
    def diffusion(self, lattice, decay):
        """
        the slime cell will check all the 8 neighbours and perform diffusion based on conditions
        1. perform diffusion on the next main diffusion direction first
        2. perform diffusion on the rest cells if not reach DIFFUSION THRESHOLD
        3. Take different diffusion strategies when the neighbour cell is empty, slime, food, boundary
        4. the pheromone of the slime cell will decrease after the diffusion is performed each time.
        :param lattice: dish lattice
        :param decay: the rate for decreasing the pheromone of the slime cell
        """
        new_idx = tuple(self.get_diffusion_direction() + self.coord)
        neighbours = get_neighbours(self.coord)

        # make sure the first neighbour is the next step
        neighbours.remove(new_idx)
        neighbours = deque(neighbours)
        neighbours.appendleft(new_idx)

        while neighbours:
            neigh = neighbours.popleft()
            # continue if the neighbor is out of boundary
            if not self.check_boundary(neigh, lattice.shape):
                continue
            neigh_cell = lattice[neigh]
            # neighbour cell is an empty cell
            if neigh_cell.get_cell_type() == 0:

                # next main diffusion place is an empty cell
                if neigh == new_idx and self.pheromone > self.moving_threshold:
                    # self.mould.update_slime_cell(new_idx, self)
                    self.mould.slime_cell_generator(
                        coord=neigh,
                        pheromone=self.pheromone,
                        decay=decay,
                        is_capital=self.is_capital,
                    )
                    self.pheromone *= 1 - self.diffusion_decay_rate * decay

                    self.is_capital = False
                    continue

                # neighbour cell is a random diffusion cell
                if (
                    self.pheromone > self.diffusion_threshold
                    and self.find_nearest_food(self.mould.get_reached_food_ids())[1]
                    < self.distance_for_diffusion_threshold
                ):
                    self.mould.slime_cell_generator(
                        coord=neigh,
                        pheromone=self.pheromone / self.diffusion_decay_rate,
                        decay=decay,
                    )
                    self.pheromone *= 1 - (2 * self.diffusion_decay_rate * decay)

            # neighbor is a slime
            elif neigh_cell.get_cell_type() == 1:

                # next main diffusion place is a slime cell
                if neigh == new_idx and self.pheromone > self.moving_threshold:

                    # Neighbour's pheromone increases.
                    neigh_increase_ph = (
                        neigh_cell.pheromone
                        + self.pheromone / self.diffusion_decay_rate
                    )

                    # Neighbour's pheromone is bounded by the max value.
                    neigh_cell.pheromone = min(neigh_cell.max_ph, neigh_increase_ph)

                    self.pheromone /= self.diffusion_decay_rate

                    self.mould.update_slime_cell(new_idx, neigh_cell)

                # neighbor bigger than self
                # increase self-pheromone when find neighbor nearby
                if (
                    neigh_cell.pheromone > self.pheromone
                    and self.max_ph < self.max_ph_threshold
                ):
                    self.max_ph += self.max_ph_increase_step
                    self.pheromone += neigh_cell.pheromone / 10

            # add pheromone if current cell find food nearby
            elif neigh_cell.get_cell_type() == 2:

                # eat food
                self.pheromone = 7
                self.max_ph = 7
                self.reached_food_id = neigh_cell.get_food_id()
                new_food_id = neigh_cell.get_food_id()
                if new_food_id not in self.mould.get_reached_food_ids():
                    self.mould.update_food_connection(new_food_id)
                    self.mould.get_reached_food_ids().add(new_food_id)

        self.mould.update_slime_cell(self.coord, self)

    def step(self, lattice, decay):
        """
        The sensory stage: all slime cells adjust their direction based on the food
        The diffusion stage: all pheromones undergo diffusion
        """
        self.diffusion(lattice, decay)

    def get_coord(self):
        return self.coord

    def get_pheromone(self):
        return self.pheromone

    def set_pheromone(self, ph):
        self.pheromone = ph

    def get_reached_food_id(self):
        return self.reached_food_id

    def remove_capital(self):
        self.is_capital = False
