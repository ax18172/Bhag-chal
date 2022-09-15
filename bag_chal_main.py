import numpy as np
import random

possible_moves = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1), (0, 2), (2, 0), (0, -2),
                  (-2, 0), (2, 2), (2, -2),
                  (-2, 2), (-2, -2)]

grid_matrix = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
max_number_of_goats_on_the_board = 7
goats_to_win_the_game = 2
memory = []


def probability_matrix_calculation(pos_x, pos_y, board):
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


def move(pos_x, pos_y, dx, dy, mission, animal, board):
    constraint_1 = 0
    constraint_2 = 0
    move_is_made = False
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
        if [pos_x + dx, pos_y + dy] in grid_matrix and board[pos_x + dx, pos_y + dy] == 0:
            board[pos_x, pos_y] = 0
            pos_x += dx
            pos_y += dy
            if animal == "tiger":
                board[pos_x, pos_y] = 1
            elif animal == "goat":
                board[pos_x, pos_y] = 2
            move_is_made = True
    return pos_x, pos_y, move_is_made


class TIGER:
    def __init__(self, init_x, init_y, board):
        self.pos_x = init_x
        self.pos_y = init_y
        board[self.pos_x, self.pos_y] = 1

    def return_position(self):
        return self.pos_x, self.pos_y

    """this function checks whether the tiger move is legal and makes the move. works for both normal moves and eating """

    def move_tiger(self, dx, dy, mission, board):
        self.pos_x, self.pos_y, move_is_made = move(self.pos_x, self.pos_y, dx, dy, mission, "tiger", board)
        return self.pos_x, self.pos_y, move_is_made

    """check if there are any goats nearby and, if yes,
     specify the direction in which we want the tiger to jump in order to eat the goat"""

    def scan_for_food(self, goat_coord):
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

    def probabilities_matrix(self, board, goat_coord):
        probability_matrix = probability_matrix_calculation(self.pos_x, self.pos_y, board)
        test_probability_matrix = probability_matrix.copy()
        directions = self.scan_for_food(goat_coord)
        for vector in directions:
            if [self.pos_x + vector[0], self.pos_y + vector[1]] in grid_matrix and board[self.pos_x + vector[0],
                                                                                         self.pos_y + vector[1]] == 0:
                probability_matrix[self.pos_x + vector[0], self.pos_y + vector[1]] = random.random()
                test_probability_matrix[self.pos_x + vector[0], self.pos_y + vector[1]] = random.randint(1, 3)
        return probability_matrix, test_probability_matrix


class GOAT:
    def __init__(self, init_x, init_y, board):
        self.pos_x = init_x
        self.pos_y = init_y
        if board[self.pos_x, self.pos_y] == 0:
            board[self.pos_x, self.pos_y] = 2

    def return_position(self):
        return self.pos_x, self.pos_y

    def probabilities_matrix(self, board):
        probability_matrix = probability_matrix_calculation(self.pos_x, self.pos_y, board)
        return probability_matrix

    def move_goat(self, dx, dy, board):
        self.pos_x, self.pos_y, move_is_made = move(self.pos_x, self.pos_y, dx, dy, "move", "goat", board)
        return self.pos_x, self.pos_y


class TIGER_AI():
    def __init__(self, tiger):
        self.Tiger = tiger
        self.killed_goats = 0

    def eat(self, goat_x, goat_y, board, goat_coord, goats):
        goat_present = False
        if (goat_x, goat_y) in goat_coord:
            board[goat_x, goat_y] = 0
            index = goat_coord.index((goat_x, goat_y))
            del goat_coord[index]
            del goats[index]
            goat_present = True
        return goat_present

    def make_a_move(self, q_values, board, goat_coord, goats):
        current_state = board.copy()
        probability_matrix_tiger, test_probability_matrix = self.Tiger.probabilities_matrix(board, goat_coord)
        if q_values != []:
            q_values = np.reshape(q_values, (3, 3))
            probability_matrix_tiger = np.reshape(probability_matrix_tiger, (9,))
            for probability in probability_matrix_tiger:
                probability_index = np.where(probability_matrix_tiger == probability)
                if probability != 0 and probability > 0:
                    probability_matrix_tiger[probability_index] = 1
                else:
                    probability_matrix_tiger[probability_index] = 0
            probability_matrix_tiger = np.reshape(probability_matrix_tiger, (3, 3))
            probability_matrix_tiger = np.multiply(q_values, probability_matrix_tiger)
        highest_value = np.max(probability_matrix_tiger[np.nonzero(probability_matrix_tiger)])
        index_x, index_y = np.where(probability_matrix_tiger == highest_value)
        dx, dy = index_x[0] - self.Tiger.return_position()[0], index_y[0] - self.Tiger.return_position()[1]
        if abs(dx) < 2 and abs(dy) < 2:
            self.Tiger.move_tiger(dx, dy, "move", board)
        else:
            self.Tiger.move_tiger(dx, dy, "eat", board)
            self.eat(self.Tiger.return_position()[0] - int(0.5 * dx),
                     self.Tiger.return_position()[1] - int(0.5 * dy), board, goat_coord, goats)
            self.killed_goats += 1
        next_state = board.copy()
        return current_state, next_state, goat_coord, goats, self.Tiger.return_position(), test_probability_matrix

    def return_killed_goats(self):
        return self.killed_goats

    def return_tiger_position(self):
        return self.Tiger.return_position()

    def return_tiger_probability_matrix(self, board, goat_coord):
        return self.Tiger.probabilities_matrix(board, goat_coord)


"""def goat_move(board, goat_ai, goats, type_of_act, goat_coord):
    if type_of_act == "placing":
        goat_ai.placing_a_goat(board, goat_coord, goats)
    elif type_of_act == "moving":
        goat = goat_ai.picking_a_goat_to_move(board, goats, goat_coord)
        goat_ai.make_a_move(goat, board, goat_coord)
    return board, goats, goat_coord"""


class GOAT_AI:
    def __init__(self, board_dimension):
        self.board_dimension = board_dimension

    def picking_a_goat_to_move(self, goats, goat_coord, input):
        index_x, index_y = input[0], input[1]
        goat = goats[goat_coord.index((index_x, index_y))]
        return goat

    def make_a_move(self, goat, board, goat_coord, input):
        index_x, index_y = input[0], input[1]
        position_in_list = goat_coord.index((goat.return_position()[0], goat.return_position()[1]))
        goat_coord[position_in_list] = (index_x[0], index_y[0])
        dx, dy = index_x[0] - goat.return_position()[0], index_y[0] - goat.return_position()[1]
        goat.move_goat(dx, dy, board)

    def pd_ai(self, probabilities, board, available_goats,goat_coord,goats):
        if available_goats > 0:
            placement_options = probabilities[:self.board_dimension ** 2]
            placement_options = np.reshape(placement_options, (self.board_dimension, self.board_dimension))
            for i in range(0, self.board_dimension):
                for j in range(0, self.board_dimension):
                    if board[i, j] != 0:
                        placement_options[i, j] = 0
            max_value = np.amax(placement_options)
            goat_x, goat_y = np.where(placement_options == max_value)
            goat = GOAT(goat_x[0], goat_y[0], board)
            goat_coord.append(goat.return_position())
            goats.append(goat)
            available_goats -= 1
        else:
            goat_options = probabilities[self.board_dimension ** 2:]
            goat_options = np.reshape(goat_options,
                                      (self.board_dimension ** 2, self.board_dimension ** 2))
            auxiliary_matrix = np.reshape(goat_options[0, :], (self.board_dimension, self.board_dimension))
            for i in range(0, self.board_dimension):
                for j in range(0, self.board_dimension):
                    if board[i, j] != 2:
                        auxiliary_matrix[i, j] = 0
                    else:
                        goat = goats[goat_coord.index((i, j))]
                        movement_matrix = goat.probabilities_matrix(board)
                        test_matrix = np.zeros((self.board_dimension, self.board_dimension))
                        test_matrix[i, j] = -1
                        if np.all(movement_matrix == test_matrix):
                            auxiliary_matrix[i, j] = 0
            goat_options[0, :] = np.reshape(auxiliary_matrix, (1, self.board_dimension ** 2))
            for index in range(goat_options.shape[0]):
                if goat_options[0, index] == 0:
                    goat_options[:, index] = np.zeros((self.board_dimension ** 2))
                else:
                    auxiliary_value = goat_options[0, index]
                    goat_index_x, goat_index_y = np.where(auxiliary_matrix == auxiliary_value)
                    goat = self.picking_a_goat_to_move(goats, goat_coord, [goat_index_x, goat_index_y])
                    movement_options = np.reshape(goat_options[:, index], (self.board_dimension, self.board_dimension))
                    movement_matrix = goat.probabilities_matrix(board)
                    for i in range(0, self.board_dimension):
                        for j in range(0, self.board_dimension):
                            if movement_matrix[i, j] != 0 and movement_matrix[i, j] != -1:
                                movement_matrix[i, j] = 1
                            else:
                                movement_matrix[i, j] = 0
                    movement_options = np.multiply(movement_options, movement_matrix)
                    goat_options[:, index] = np.reshape(movement_options, ((self.board_dimension ** 2)))
            max_value = np.amax(goat_options)
            pos_x, pos_y = np.where(goat_options == max_value)
            pos_x, pos_y = pos_x[0], pos_y[0]
            auxiliary_matrix = np.reshape(goat_options[pos_x, :], (self.board_dimension, self.board_dimension))
            goat_index_x, goat_index_y = np.where(auxiliary_matrix == max_value)
            goat = self.picking_a_goat_to_move(goats, goat_coord, [goat_index_x, goat_index_y])
            movement_options = np.reshape(goat_options[:, pos_y], (self.board_dimension, self.board_dimension))
            new_pos_x, new_pos_y = np.where(movement_options == max_value)
            self.make_a_move(goat, board, goat_coord, [new_pos_x, new_pos_y])
        index= np.where(probabilities == max_value)
        return board, index, available_goats,goat_coord,goats


def tiger_score_check(tiger_ai, eaten_goats):
    newly_eaten_goats = tiger_ai.return_killed_goats()
    if newly_eaten_goats != eaten_goats:
        tiger_reward = 1
        eaten_goats = newly_eaten_goats
        goat_reward = -5
    else:
        tiger_reward = 0
        goat_reward = 0
    if eaten_goats >= goats_to_win_the_game:
        tiger_reward = 10
        goat_reward = -10
        return True, tiger_reward, eaten_goats, goat_reward
    else:
        return False, tiger_reward, eaten_goats, goat_reward


def goat_score_check(tiger_ai, board, goat_coord):
    tiger_reward = 0
    goat_reward = 0
    movement_matrix = tiger_ai.return_tiger_probability_matrix(board, goat_coord)
    index_x, index_y = tiger_ai.return_tiger_position()
    test_matrix = np.zeros((3, 3))
    test_matrix[index_x, index_y] = -1
    if np.all(movement_matrix == test_matrix):
        tiger_reward = -10
        goat_reward = 10
        return True, tiger_reward, goat_reward
    else:
        return False, tiger_reward, goat_reward


def run_episode(board_dimension, max_number_of_goats, tiger_initial_pos, maximum_number_of_timesteps, agent):
    board = np.zeros((board_dimension, board_dimension))
    tiger = TIGER(tiger_initial_pos[0], tiger_initial_pos[1], board)
    tiger_ai = TIGER_AI(tiger)
    goat_coord = []
    goats = []
    eaten_goats = 0
    available_goats = max_number_of_goats
    goat_ai = GOAT_AI(board_dimension)
    for timestep in range(maximum_number_of_timesteps+1):
        probabilities = agent.choose_action(board)
        board, index, available_goats,goat_coord,goats = goat_ai.pd_ai(probabilities, board, available_goats,goat_coord,goats)
        print("goats moved\n", board)
        action_goat = np.zeros((1,board_dimension ** 4+board_dimension**2))
        action_goat[0, index] = 1
        done, tiger_reward, goat_reward_1 = goat_score_check(tiger_ai, board, goat_coord)
        if not done:
            eaten_goats = eaten_goats
            current_state, next_state, goat_coord, goats, action, test_probability_matrix = tiger_ai.make_a_move([],
                                                                                                                 board,
                                                                                                                 goat_coord,
                                                                                                                 goats)
            done, tiger_reward, eaten_goats, goat_reward_2 = tiger_score_check(tiger_ai, eaten_goats)
            goat_reward = goat_reward_1 + goat_reward_2
            print("random tiger moved\n", board)
        else:
            goat_reward = goat_reward_1
            state = board.copy()
            agent.store_transition(np.reshape(state, (1, board_dimension ** 2)), action_goat, goat_reward)
            break
        state = board.copy()
        agent.store_transition(np.reshape(state,(1,board_dimension**2)),action_goat,goat_reward)
        if done:
            break







