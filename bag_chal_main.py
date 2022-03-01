import numpy as np
import random

"""list of all available moves"""
grid_matrix = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
"""places that are currently holding a tiger or a sheep"""
board = np.zeros((3, 3))
"""lists of goats and their respective coordinates"""
goats = []
goat_coord = []

#def probability_matrix_calculation(x,y):


class TIGER:
    def __init__(self, init_x, init_y):
        self.pos_x = init_x
        self.pos_y = init_y
        board[self.pos_x, self.pos_y] = 1

    def return_position(self):
        return self.pos_x, self.pos_y

    """this function checks whether the tiger move is legal and makes the move. works for both normal moves and eating """

    def move(self, dx, dy, mission):
        if mission == "move":
            """numbers can't be bigger than 1 in each direction"""
            constraint_1 = abs(int(dx) * int(dy)) == 1
            constraint_2 = abs(int(dx) * int(dy)) == 0
            """numbers can't be bigger than 2 in each direction"""
        elif mission == "eat":
            constraint_1 = abs(int(dx) * int(dy)) == 4
            constraint_2 = abs(int(dx) * int(dy)) == 0
        if constraint_1 or constraint_2:
            """check whether the move is inside the grid"""
            if [self.pos_x + dx, self.pos_y + dy] not in grid_matrix:
                return None
            else:
                """check that the spot we want to move to is free"""
                if board[self.pos_x + dx, self.pos_y + dy] != 0:
                    return None
                else:
                    """execute the move"""
                    board[self.pos_x, self.pos_y] = 0
                    self.pos_x += dx
                    self.pos_y += dy
                    board[self.pos_x, self.pos_y] = 1
        else:
            return None
        return self.pos_x, self.pos_y

    """check if there are any goats nearby and, if yes,
     specify the direction in which we want the tiger to jump in order to eat the goat"""

    def scan_for_food(self):
        attack_directions = []
        for goat in goat_coord:
            goat_x, goat_y = goat
            if abs(self.pos_x - goat_x) <= 1 and abs(self.pos_y - goat_y) <= 1:
                vector = 2 * (goat_x - self.pos_x), 2 * (goat_y - self.pos_y)
                attack_directions.append(vector)
        return attack_directions

    """state all the moves the tiger can move to.
   the square the tiger is currently at in -1,
    unreachable squares are 0 and the possible squares are random between 0 and 1.
    T"""

    def probabilities_matrix(self):
        self.probability_matrix = np.zeros((3, 3))
        self.probability_matrix[self.pos_x, self.pos_y] = -1
        if [self.pos_x + 1, self.pos_y + 1] in grid_matrix and board[self.pos_x + 1, self.pos_y + 1] == 0:
            self.probability_matrix[self.pos_x + 1, self.pos_y + 1] = random.random()
        if [self.pos_x - 1, self.pos_y - 1] in grid_matrix and board[self.pos_x - 1, self.pos_y - 1] == 0:
            self.probability_matrix[self.pos_x - 1, self.pos_y - 1] = random.random()
        if [self.pos_x, self.pos_y + 1] in grid_matrix and board[self.pos_x, self.pos_y + 1] == 0:
            self.probability_matrix[self.pos_x, self.pos_y + 1] = random.random()
        if [self.pos_x + 1, self.pos_y] in grid_matrix and board[self.pos_x + 1, self.pos_y] == 0:
            self.probability_matrix[self.pos_x + 1, self.pos_y] = random.random()
        if [self.pos_x, self.pos_y - 1] in grid_matrix and board[self.pos_x, self.pos_y - 1] == 0:
            self.probability_matrix[self.pos_x, self.pos_y - 1] = random.random()
        if [self.pos_x - 1, self.pos_y] in grid_matrix and board[self.pos_x - 1, self.pos_y] == 0:
            self.probability_matrix[self.pos_x - 1, self.pos_y] = random.random()
        if [self.pos_x + 1, self.pos_y - 1] in grid_matrix and board[self.pos_x + 1, self.pos_y - 1] == 0:
            self.probability_matrix[self.pos_x + 1, self.pos_y - 1] = random.random()
        if [self.pos_x - 1, self.pos_y + 1] in grid_matrix and board[self.pos_x - 1, self.pos_y + 1] == 0:
            self.probability_matrix[self.pos_x - 1, self.pos_y + 1] = random.random()
        directions = self.scan_for_food()
        for vector in directions:
            if [self.pos_x + vector[0], self.pos_y + vector[1]] in grid_matrix and board[self.pos_x + vector[0],
                                                                                         self.pos_y + vector[1]] == 0:
                self.probability_matrix[self.pos_x + vector[0], self.pos_y + vector[1]] = random.randrange(1, 3)
        return self.probability_matrix


"""Goat class is identical to tiger in lots of aspects, just simpler"""


class GOAT:
    def __init__(self, init_x, init_y):
        self.pos_x = init_x
        self.pos_y = init_y
        if board[self.pos_x, self.pos_y] == 0:
            board[self.pos_x, self.pos_y] = 2

    def return_position(self):
        return self.pos_x, self.pos_y

    def probabilities_matrix(self):
        self.probability_matrix = np.zeros((3, 3))
        self.probability_matrix[self.pos_x, self.pos_y] = -1
        if [self.pos_x + 1, self.pos_y + 1] in grid_matrix and board[self.pos_x + 1, self.pos_y + 1] == 0:
            self.probability_matrix[self.pos_x + 1, self.pos_y + 1] = random.random()
        if [self.pos_x - 1, self.pos_y - 1] in grid_matrix and board[self.pos_x - 1, self.pos_y - 1] == 0:
            self.probability_matrix[self.pos_x - 1, self.pos_y - 1] = random.random()
        if [self.pos_x, self.pos_y + 1] in grid_matrix and board[self.pos_x, self.pos_y + 1] == 0:
            self.probability_matrix[self.pos_x, self.pos_y + 1] = random.random()
        if [self.pos_x + 1, self.pos_y] in grid_matrix and board[self.pos_x + 1, self.pos_y] == 0:
            self.probability_matrix[self.pos_x + 1, self.pos_y] = random.random()
        if [self.pos_x, self.pos_y - 1] in grid_matrix and board[self.pos_x, self.pos_y - 1] == 0:
            self.probability_matrix[self.pos_x, self.pos_y - 1] = random.random()
        if [self.pos_x - 1, self.pos_y] in grid_matrix and board[self.pos_x - 1, self.pos_y] == 0:
            self.probability_matrix[self.pos_x - 1, self.pos_y] = random.random()
        if [self.pos_x + 1, self.pos_y - 1] in grid_matrix and board[self.pos_x + 1, self.pos_y - 1] == 0:
            self.probability_matrix[self.pos_x + 1, self.pos_y - 1] = random.random()
        if [self.pos_x - 1, self.pos_y + 1] in grid_matrix and board[self.pos_x - 1, self.pos_y + 1] == 0:
            self.probability_matrix[self.pos_x - 1, self.pos_y + 1] = random.random()
        return self.probability_matrix

    def move(self, dx, dy):
        if abs(int(dx) * int(dy)) == 1 or abs(int(dx) * int(dy)) == 0:
            if [self.pos_x + dx, self.pos_y + dy] not in grid_matrix:
                return None
            else:
                if board[self.pos_x + dx, self.pos_y + dy] != 0:
                    return None
                else:
                    board[self.pos_x, self.pos_y] = 0
                    self.pos_x += dx
                    self.pos_y += dy
                    board[self.pos_x, self.pos_y] = 2
        else:
            return None
        return self.pos_x, self.pos_y


def placing_the_goat():
    placement_matrix = np.zeros((3, 3))
    for i in range(0, 2):
        for j in range(0, 2):
            if board[i, j] == 0:
                placement_matrix[i, j] = random.random()
            else:
                placement_matrix[i, j] = 0
    highest_value = np.amax(placement_matrix)
    index_x, index_y = np.where(placement_matrix == highest_value)
    return index_x[0], index_y[0]


class TIGER_AI():
    def __init__(self, Tiger):
        self.Tiger = Tiger

    def eat(self, goat_x, goat_y):
        board[goat_x, goat_y] = 0
        index = goat_coord.index((goat_x, goat_y))
        del goat_coord[index]
        del goats[index]

    def make_a_move(self):
        possible_moves = self.Tiger.probabilities_matrix()
        highest_value = np.amax(possible_moves)
        index_x, index_y = np.where(possible_moves == highest_value)
        dx, dy = index_x[0] - self.Tiger.return_position()[0], index_y[0] - self.Tiger.return_position()[1]
        if highest_value < 1:
            self.Tiger.move(dx, dy, "move")
        else:
            self.Tiger.move(dx, dy, "eat")
            self.eat(self.Tiger.return_position()[0] - int(0.5 * dx), self.Tiger.return_position()[1] - int(0.5 * dy))


class GOAT_AI():
    def __init__(self, max_number_of_goats):
        self.max_number_of_goats = max_number_of_goats
        self.number_of_goats_on_the_board = 0

    def where_to_place_a_goat(self):
        placement_matrix = np.zeros((3, 3))
        for i in range(0, 2):
            for j in range(0, 2):
                if board[i, j] == 0:
                    placement_matrix[i, j] = random.random()
                else:
                    placement_matrix[i, j] = 0
        highest_value = np.amax(placement_matrix)
        index_x, index_y = np.where(placement_matrix == highest_value)
        return index_x[0], index_y[0]


def run_environment(episodes):
    tiger = TIGER(2, 2)
    episode = 0
    for episode in range(episodes):
        tiger_ai = TIGER_AI(tiger)
        goat_x, goat_y = placing_the_goat()
        goat = GOAT(goat_x, goat_y)
        goat_coord.append(goat.return_position())
        goats.append(goat)
        print(board)
        tiger_ai.make_a_move()
        print(board)


run_environment(5)
