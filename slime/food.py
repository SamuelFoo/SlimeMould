from ..slime.cell import Cell


class FoodCell(Cell):

    def __init__(self, food_id: int, food_coord: tuple):
        """
        food_id: id no. of food in dataframe.
        food_coord: coord of food in dish.
        """
        super().__init__(pheromone=10.0, cell_type=2)
        self.food_id = food_id
        self.food_coord = food_coord
        self.pheromone = 10.0

    def get_food_coord(self):
        return self.food_coord

    def get_food_id(self):
        return self.food_id
