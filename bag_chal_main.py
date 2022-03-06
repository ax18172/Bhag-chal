import numpy as np
import random

"""list of all available moves"""
grid_matrix = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
"""places that are currently holding a tiger or a sheep"""
board = np.zeros((3, 3))
"""lists of goats and their respective coordinates"""
goats = []
goat_coord = []
max_number_of_goats_on_the_board = 7
goats_to_win_the_game = 2
tiger_score = 0
goat_score = 0
eaten_goats = 0
memory = []


def probability_matrix_calculation(pos_x, pos_y):
    probability_matrix = np.zeros((3, 3))
    probability_matrix[pos_x, pos_y] = -1
    if [pos_x + 1, pos_y + 1] in grid_matrix and board[pos_x + 1, pos_y + 1] == 0:
        probability_matrix[pos_x + 1, pos_y + 1] = random.random()
    if [pos_x - 1, pos_y - 1] in grid_matrix and board[pos_x - 1, pos_y - 1] == 0:
        probability_matrix[pos_x - 1, pos_y - 1] = random.random()
    if [pos_x, pos_y + 1] in grid_matrix and board[pos_x, pos_y + 1] == 0:
        probability_matrix[pos_x, pos_y + 1] = random.random()
    if [pos_x + 1, pos_y] in grid_matrix and board[pos_x + 1, pos_y] == 0:
        probability_matrix[pos_x + 1, pos_y] = random.random()
    if [pos_x, pos_y - 1] in grid_matrix and board[pos_x, pos_y - 1] == 0:
        probability_matrix[pos_x, pos_y - 1] = random.random()
    if [pos_x - 1, pos_y] in grid_matrix and board[pos_x - 1, pos_y] == 0:
        probability_matrix[pos_x - 1, pos_y] = random.random()
    if [pos_x + 1, pos_y - 1] in grid_matrix and board[pos_x + 1, pos_y - 1] == 0:
        probability_matrix[pos_x + 1, pos_y - 1] = random.random()
    if [pos_x - 1, pos_y + 1] in grid_matrix and board[pos_x - 1, pos_y + 1] == 0:
        probability_matrix[pos_x - 1, pos_y + 1] = random.random()
    return probability_matrix


def move(pos_x, pos_y, dx, dy, mission, animal):
    constraint_1 = 0
    constraint_2 = 0
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
        if [pos_x + dx, pos_y + dy] not in grid_matrix:
            # print("not in grid matrix")
            return None
        else:
            """check that the spot we want to move to is free"""
            if board[pos_x + dx, pos_y + dy] != 0:
                # print("not available")
                # print(pos_x, pos_y)
                # print(dx, dy)
                # print(board[pos_x + dx, pos_y + dy])
                return None
            else:
                """execute the move"""
                board[pos_x, pos_y] = 0
                pos_x += dx
                pos_y += dy
                if animal == "tiger":
                    board[pos_x, pos_y] = 1
                elif animal == "goat":
                    board[pos_x, pos_y] = 2

    else:
        return None
    return pos_x, pos_y


class TIGER:
    def __init__(self, init_x, init_y):
        self.pos_x = init_x
        self.pos_y = init_y
        board[self.pos_x, self.pos_y] = 1

    def return_position(self):
        return self.pos_x, self.pos_y

    """this function checks whether the tiger move is legal and makes the move. works for both normal moves and eating """

    def move_tiger(self, dx, dy, mission):
        self.pos_x, self.pos_y = move(self.pos_x, self.pos_y, dx, dy, mission, "tiger")
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
    """

    def probabilities_matrix(self):
        probability_matrix = probability_matrix_calculation(self.pos_x, self.pos_y)
        directions = self.scan_for_food()
        for vector in directions:
            if [self.pos_x + vector[0], self.pos_y + vector[1]] in grid_matrix and board[self.pos_x + vector[0],
                                                                                         self.pos_y + vector[1]] == 0:
                probability_matrix[self.pos_x + vector[0], self.pos_y + vector[1]] = random.random()
        return probability_matrix


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
        probability_matrix = probability_matrix_calculation(self.pos_x, self.pos_y)
        return probability_matrix

    def move_goat(self, dx, dy):
        self.pos_x, self.pos_y = move(self.pos_x, self.pos_y, dx, dy, "move", "goat")
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
    def __init__(self, tiger):
        self.Tiger = tiger
        self.killed_goats = 0

    def eat(self, goat_x, goat_y):
        board[goat_x, goat_y] = 0
        index = goat_coord.index((goat_x, goat_y))
        del goat_coord[index]
        del goats[index]

    def make_a_move(self, dx_nn, dy_nn, neural_network_inputs):
        if neural_network_inputs:
            dx, dy = dx_nn, dy_nn
        else:
            possible_moves = self.Tiger.probabilities_matrix()
            highest_value = np.amax(possible_moves)
            index_x, index_y = np.where(possible_moves == highest_value)
            dx, dy = index_x[0] - self.Tiger.return_position()[0], index_y[0] - self.Tiger.return_position()[1]
        if abs(dx) + abs(dy) <= 2:
            self.Tiger.move_tiger(dx, dy, "move")
        else:
            self.Tiger.move_tiger(dx, dy, "eat")
            self.eat(self.Tiger.return_position()[0] - int(0.5 * dx),
                     self.Tiger.return_position()[1] - int(0.5 * dy))
            self.killed_goats += 1
        return dx, dy

    def return_killed_goats(self):
        return self.killed_goats

    def return_tiger_position(self):
        return self.Tiger.return_position()

    def return_tiger_probability_matrix(self):
        return self.Tiger.probabilities_matrix()


class GOAT_AI():
    def __init__(self, max_number_of_goats):
        self.max_number_of_goats = max_number_of_goats
        self.number_of_goats_on_the_board = 0

    def where_to_place_a_goat(self, index_x_nn, index_y_nn, neural_network_inputs):
        if neural_network_inputs:
            index_x, index_y = index_x_nn, index_y_nn
        else:
            placement_matrix = np.zeros((3, 3))
            for i in range(0, 3):
                for j in range(0, 3):
                    if board[i, j] == 0:
                        placement_matrix[i, j] = random.random()
                    else:
                        placement_matrix[i, j] = 0
            highest_value = np.amax(placement_matrix)
            index_x, index_y = np.where(placement_matrix == highest_value)
            index_x, index_y = index_x[0], index_y[0]
        return index_x, index_y

    def placing_a_goat(self):
        goat_x, goat_y = self.where_to_place_a_goat(None, None, False)
        goat = GOAT(goat_x, goat_y)
        goat_coord.append(goat.return_position())
        goats.append(goat)
        self.number_of_goats_on_the_board += 1

    def picking_a_goat_to_move(self):
        location_matrix = np.zeros((3, 3))
        for i in range(0, 3):
            for j in range(0, 3):
                if board[i, j] == 2:
                    goat = goats[goat_coord.index((i, j))]
                    movement_matrix = goat.probabilities_matrix()
                    # print("mm", movement_matrix)
                    test_matrix = np.zeros((3, 3))
                    test_matrix[i, j] = -1
                    if np.all(movement_matrix == test_matrix):
                        location_matrix[i, j] = 0
                    else:
                        location_matrix[i, j] = random.random()
        # print("lm", location_matrix)
        highest_value = np.amax(location_matrix)
        index_x, index_y = np.where(location_matrix == highest_value)
        goat = goats[goat_coord.index((index_x[0], index_y[0]))]
        return goat

    def make_a_move(self, goat):
        possible_moves = goat.probabilities_matrix()
        # print("actual mm", possible_moves)
        highest_value = np.amax(possible_moves)
        index_x, index_y = np.where(possible_moves == highest_value)
        # print("where it should go", (index_x, index_y))
        position_in_list = goat_coord.index((goat.return_position()[0], goat.return_position()[1]))
        goat_coord[position_in_list] = (index_x[0], index_y[0])
        dx, dy = index_x[0] - goat.return_position()[0], index_y[0] - goat.return_position()[1]
        goat.move_goat(dx, dy)
        return dx, dy

    def return_action(self, move_type):
        if move_type == "placement":
            index_x, index_y = self.where_to_place_a_goat(None, None, False)
            return 0, index_x, index_y, 0, 0,
        if move_type == "movement":
            goat = self.picking_a_goat_to_move()
            index_x, index_y = goat.return_position()
            dx, dy = self.make_a_move(goat)
            return 1, index_x, index_y, dx, dy


def score__and_game_check(tiger_ai):
    global tiger_score
    global goat_score
    global eaten_goats
    newly_eaten_goats = tiger_ai.return_killed_goats()
    if newly_eaten_goats != eaten_goats:
        tiger_score += 10
        goat_score -= 10
        eaten_goats = newly_eaten_goats
    movement_matrix = tiger_ai.return_tiger_probability_matrix()
    index_x, index_y = tiger_ai.return_tiger_position()
    test_matrix = np.zeros((3, 3))
    test_matrix[index_x, index_y] = -1
    if np.all(movement_matrix == test_matrix):
        tiger_score -= 1000
        goat_score += 1000
        return False
    elif eaten_goats >= goats_to_win_the_game:
        tiger_score += 1000
        goat_score -= 1000
        return False
    else:
        return True


def run_environment(episodes, neural_network_inputs, tiger_dx, tiger_dy):
    tiger = TIGER(2, 2)
    episode = 0
    tiger_ai = TIGER_AI(tiger)
    goat_ai = GOAT_AI(max_number_of_goats_on_the_board)
    avialable_goats = max_number_of_goats_on_the_board
    for episode in range(episodes):
        if episode < max_number_of_goats_on_the_board:
            goat_ai.placing_a_goat()
            state = board
            current_state = state.copy()
            if neural_network_inputs:
                action_tiger = tiger_ai.make_a_move(tiger_dx, tiger_dy, True)
            else:
                action_tiger = tiger_ai.make_a_move(None, None, False)
            avialable_goats = avialable_goats - tiger_ai.return_killed_goats()
            state = board
            next_state = state.copy()
            episode += 1
            play = score__and_game_check(tiger_ai)
            tiger_reward = tiger_score
            # goat_reward = goat_score
            done = False
            if not play:
                done = True
                memory.append((current_state, action_tiger, tiger_reward, next_state, done))
                break
            memory.append((current_state, action_tiger, tiger_reward, next_state, done))
        else:
            goat = goat_ai.picking_a_goat_to_move()
            goat_ai.make_a_move(goat)
            state = board
            current_state = state.copy()
            if neural_network_inputs:
                action_tiger = tiger_ai.make_a_move(tiger_dx, tiger_dy, True)
            else:
                action_tiger = tiger_ai.make_a_move(None, None, False)
            state = board
            next_state = state.copy()
            play = score__and_game_check(tiger_ai)
            tiger_reward = tiger_score
            done = False
            episode += 1
            if not play:
                done = True
                memory.append((current_state, action_tiger, tiger_reward, next_state, done))
                break
            else:
                memory.append((current_state, action_tiger, tiger_reward, next_state, done))



