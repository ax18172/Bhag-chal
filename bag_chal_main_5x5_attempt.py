import numpy as np
import random

"""list of all available moves"""
grid_matrix = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [2, 0], [2, 1], [2, 2],[2, 3], [2, 4], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4]]
"""places that are currently holding a tiger or a sheep"""
board = np.zeros((5, 5))
"""lists of goats and their respective coordinates"""
goats = []
tigers = []
goat_coord = []
tiger_coord = []
max_number_of_goats_on_the_board = 20
goats_to_win_the_game = 20
tiger_score = 0
goat_score = 0
eaten_goats = 0
memory = []
"""list of cells on the board where only possible movements are horizontal and vertical, not diagonal"""
restricted_cells = [[1, 0], [3, 0], [0, 1], [2, 1], [4, 1], [1, 2], [3, 2], [0, 3], [2, 3], [4, 3], [1, 4], [3, 4]]

def probability_matrix_calculation(pos_x, pos_y):
    probability_matrix = np.zeros((5, 5))
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
    if [pos_x, pos_y] in restricted_cells and [pos_x + 1, pos_y + 1] in grid_matrix:
        probability_matrix[pos_x + 1, pos_y + 1] = 0
    if [pos_x, pos_y] in restricted_cells and [pos_x - 1, pos_y - 1] in grid_matrix:
        probability_matrix[pos_x - 1, pos_y - 1] = 0
    if [pos_x, pos_y] in restricted_cells and [pos_x + 1, pos_y - 1] in grid_matrix:
        probability_matrix[pos_x + 1, pos_y - 1] = 0
    if [pos_x, pos_y] in restricted_cells and [pos_x - 1, pos_y + 1] in grid_matrix:
        probability_matrix[pos_x - 1, pos_y + 1] = 0
    # print ("probability_matrix = ", probability_matrix)
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
        if [pos_x + dx, pos_y + dy] not in grid_matrix:
            return None
        else:
            """check that the spot we want to move to is free"""
            if board[pos_x + dx, pos_y + dy] != 0:
                return None
            elif [pos_x, pos_y] in restricted_cells and mission == 'move':
                if (abs(int(dx)) + abs(int(dy))) == 2:
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
            
            elif [pos_x, pos_y] in restricted_cells and mission == 'eat':
                if (abs(int(dx)) + abs(int(dy))) == 4:
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
                    pass
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
                if [self.pos_x, self.pos_y] not in restricted_cells: #and [goat_x, goat_y] not in restricted_cells:
                    vector = 2 * (goat_x - self.pos_x), 2 * (goat_y - self.pos_y)
                    attack_directions.append(vector)
                elif [self.pos_x, self.pos_y] in restricted_cells and [goat_x, goat_y] not in restricted_cells:
                    vector = 2 * (goat_x - self.pos_x), 2 * (goat_y - self.pos_y)
                    attack_directions.append(vector)
                else:
                    pass
        return attack_directions
    
    
    """state all the moves the tiger can move to.
   the square the tiger is currently at in -1,
    unreachable squares are 0 and the possible squares are random between 0 and 1.
    """
    #def return_tiger_probability_matrix(self):
    #    return self.probabilities_matrix()
    
    #def return_tiger_position(self):
    #    return self.return_position()

    def probabilities_matrix(self):
        probability_matrix = probability_matrix_calculation(self.pos_x, self.pos_y)
        directions = self.scan_for_food()
        for vector in directions:
            if [self.pos_x + vector[0], self.pos_y + vector[1]] in grid_matrix and board[self.pos_x + vector[0],
                                                                                         self.pos_y + vector[1]] == 0:
                probability_matrix[self.pos_x + vector[0], self.pos_y + vector[1]] = random.randint(1, 3)
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

def placing_the_tigers(x, y):
    #placement_matrix = np.zeros((5, 5))
    board[x, y] = 2
    
def placing_the_goat():
    placement_matrix = np.zeros((5, 5))
    for i in range(0, 5):
        for j in range(0, 5):
            if board[i, j] == 0:
                placement_matrix[i, j] = random.random()
            else:
                placement_matrix[i, j] = 0
    highest_value = np.amax(placement_matrix)
    index_x, index_y = np.where(placement_matrix == highest_value)
    return index_x[0], index_y[0]


class TIGER_AI:
    def __init__(self):#, tiger):
        self.killed_goats = 0

    def eat(self, goat_x, goat_y):
        board[goat_x, goat_y] = 0
        index = goat_coord.index((goat_x, goat_y))
        del goat_coord[index]
        del goats[index]
    
    def where_to_place_a_tiger(self, index_x_nn, index_y_nn, neural_network_inputs):
        if neural_network_inputs:
            index_x, index_y = index_x_nn, index_y_nn
        else:
            placement_matrix = np.zeros((5, 5))
            for i in range(0, 5):
                for j in range(0, 5):
                    if board[i, j] == 0:
                        placement_matrix[i, j] = random.random()
                    else:
                        placement_matrix[i, j] = 0
            highest_value = np.amax(placement_matrix)
            index_x, index_y = np.where(placement_matrix == highest_value)
            index_x, index_y = index_x[0], index_y[0]
        return index_x, index_y
    
#     def placing_a_tiger(self):
#         tiger_x, tiger_y = self#self.where_to_place_a_tiger(None, None, False)
#         tiger = TIGER(tiger_x, tiger_y)
#         tiger_coord.append(tiger.return_position())
#         tigers.append(tiger)
# #        self.number_of_tigers_on_the_board += 1
    def placing_a_tiger(self):
        tiger_x, tiger_y = (0,0)#self.where_to_place_a_tiger(None, None, False)
        tiger = TIGER(tiger_x, tiger_y)
        tiger_coord.append(tiger.return_position())
        tigers.append(tiger)
        tiger_x, tiger_y = (0,4)#self.where_to_place_a_tiger(None, None, False)
        tiger = TIGER(tiger_x, tiger_y)
        tiger_coord.append(tiger.return_position())
        tigers.append(tiger)
        tiger_x, tiger_y = (4,0)#self.where_to_place_a_tiger(None, None, False)
        tiger = TIGER(tiger_x, tiger_y)
        tiger_coord.append(tiger.return_position())
        tigers.append(tiger)
        tiger_x, tiger_y = (4,4)#self.where_to_place_a_tiger(None, None, False)
        tiger = TIGER(tiger_x, tiger_y)
        tiger_coord.append(tiger.return_position())
        tigers.append(tiger)
        
    def picking_a_tiger_to_move(self):
        location_matrix = np.zeros((5, 5))
        for i in range(0, 5):
            for j in range(0, 5):
                if board[i, j] == 1:
                    tiger = tigers[tiger_coord.index((i, j))]
                    movement_matrix = tiger.probabilities_matrix()
                    #print (movement_matrix)
                    # print("mm", movement_matrix)
                    test_matrix = np.zeros((5, 5))
                    test_matrix[i, j] = -1
                    if np.all(movement_matrix == test_matrix):
                        location_matrix[i, j] = 0
                    else:
                        location_matrix[i, j] = random.random()
        # print("lm", location_matrix)
        highest_value = np.amax(location_matrix)
        index_x, index_y = np.where(location_matrix == highest_value)
        tiger = tigers[tiger_coord.index((index_x[0], index_y[0]))]
        return tiger    
        
    def make_a_move(self, tiger, dx_nn, dy_nn, neural_network_inputs):
        if neural_network_inputs:
            dx, dy = dx_nn, dy_nn
        else:
            possible_moves = self.tiger.probabilities_matrix()
            highest_value = np.amax(possible_moves)
            index_x, index_y = np.where(possible_moves == highest_value)
            position_in_list = tiger_coord.index((tiger.return_position()[0], tiger.return_position()[1]))
            tiger_coord[position_in_list] = (index_x[0], index_y[0])
            dx, dy = index_x[0] - self.tiger.return_position()[0], index_y[0] - self.tiger.return_position()[1]
        if abs(dx) < 2 and abs(dy) < 2:
            self.tiger.move_tiger(dx, dy, "move")
        else:
            self.tiger.move_tiger(dx, dy, "eat")
            self.eat(self.tiger.return_position()[0] - int(0.5 * dx),
                     self.tiger.return_position()[1] - int(0.5 * dy))
            self.killed_goats += 1
        return dx, dy
    
    def return_action(self):
        tiger = self.picking_a_tiger_to_move(self)
        index_x, index_y = tiger.return_position()
        dx, dy = self.make_a_move(tiger)
        return 1, index_x, index_y, dx, dy
        
    def return_killed_goats(self):
        return self.killed_goats

    def return_tiger_position(self,tiger):
        self.tiger = tiger
        return self.tiger.return_position()

    def return_tiger_probability_matrix(self,tiger):
        self.tiger = tiger
        return self.tiger.probabilities_matrix()


class GOAT_AI():
    def __init__(self, max_number_of_goats):
        self.max_number_of_goats = max_number_of_goats
        self.number_of_goats_on_the_board = 0

    def where_to_place_a_goat(self, index_x_nn, index_y_nn, neural_network_inputs):
        if neural_network_inputs:
            index_x, index_y = index_x_nn, index_y_nn
        else:
            placement_matrix = np.zeros((5, 5))
            for i in range(0, 5):
                for j in range(0, 5):
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
        location_matrix = np.zeros((5, 5))
        for i in range(0, 5):
            for j in range(0, 5):
                if board[i, j] == 2:
                    goat = goats[goat_coord.index((i, j))]
                    movement_matrix = goat.probabilities_matrix()
                    # print("mm", movement_matrix)
                    test_matrix = np.zeros((5, 5))
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


def tiger_score_check(tiger_ai):
    global tiger_score
    global eaten_goats
    global goat_score
    newly_eaten_goats = tiger_ai.return_killed_goats()
    tiger_reward = 0
    goat_reward = 0
    if newly_eaten_goats != eaten_goats:
        tiger_reward = 10
        goat_reward = - 10
        eaten_goats = newly_eaten_goats
    if eaten_goats >= goats_to_win_the_game:
        tiger_reward = 1000
        goat_reward = - 1000
        return False, tiger_reward, goat_reward
    else:
        return True, tiger_reward, goat_reward


def goat_score_check(tiger_ai,tiger):
    tiger_reward = 0
    goat_reward = 0
    movement_matrix = tiger_ai.return_tiger_probability_matrix(tiger)
    index_x, index_y = tiger_ai.return_tiger_position(tiger)
    test_matrix = np.zeros((5, 5))
    test_matrix[index_x, index_y] = -1
    if np.all(movement_matrix == test_matrix):
        tiger_reward = -1000
        goat_reward = 1000
        return False, tiger_reward, goat_reward
    else:
        return True, tiger_reward, goat_reward

#TIGER_AI.placing_a_tiger([0,0])
#TIGER_AI.placing_a_tiger([4,0])
#TIGER_AI.placing_a_tiger([0,4])
#TIGER_AI.placing_a_tiger([4,4])
#TIGER_AI(TIGER(0,0))
#TIGER_AI(TIGER(0,4))
#TIGER_AI(TIGER(4,0))
#TIGER_AI(TIGER(4,4))
#tiger = TIGER(4, 4)
#tiger = tiger_list
def run_environment(episodes, neural_network_inputs, tiger_dx, tiger_dy):#, tiger):
    
    #tiger_ai = TIGER_AI()#TIGER(0,0))
    
    goat_ai = GOAT_AI(max_number_of_goats_on_the_board)
    avialable_goats = max_number_of_goats_on_the_board
    tiger_ai = TIGER_AI()
    tiger_ai.placing_a_tiger()
#    tiger_ai.placing_a_tiger([4,0])
#    tiger_ai.placing_a_tiger([0,4])
#    tiger_ai.placing_a_tiger([4,4])
    #tiger_ai = TIGER_AI()
    for episode in range(episodes):
        if episode <= max_number_of_goats_on_the_board:
            goat_ai.placing_a_goat()
            state = board
            print(board)
            current_state = state.copy()
            done = False
            tiger = tiger_ai.picking_a_tiger_to_move()
            play, tiger_reward, goat_reward = goat_score_check(tiger_ai,tiger)
            if not play:
                done = True
                memory.append((current_state, np.array([0, 0]), tiger_reward, current_state, done))
                break
            if neural_network_inputs:
                
                tiger = tiger_ai.picking_a_tiger_to_move()#TIGER(0,0))
                
                action_tiger = tiger_ai.make_a_move(tiger_dx, tiger_dy, True)
            else:
                
                tiger = tiger_ai.picking_a_tiger_to_move()#TIGER(0,0))
                
                action_tiger = tiger_ai.make_a_move(tiger, None, None, False)
            avialable_goats = avialable_goats - tiger_ai.return_killed_goats()
            state = board
            print(board)
            next_state = state.copy()
            tiger_reward = tiger_score
            play, tiger_reward, goat_reward = tiger_score_check(tiger_ai)
            #goat_reward = goat_score
            if not play:
                done = True
                memory.append((current_state, action_tiger, tiger_reward, next_state, done))
                break
            memory.append((current_state, action_tiger, tiger_reward, next_state, done))
        else:
            goat = goat_ai.picking_a_goat_to_move()
            goat_ai.make_a_move(goat)
            print(board)
            state = board
            current_state = state.copy()
            done = False
            play, tiger_reward, goat_reward = goat_score_check(tiger_ai)
            if not play:
                done = True
                memory.append((current_state, np.array([0, 0]), tiger_reward, current_state, done))
                break
            if neural_network_inputs:
                tiger = TIGER_AI.picking_a_tiger_to_move()
                action_tiger = tiger_ai.make_a_move(tiger_dx, tiger_dy, True)
            else:
                tiger = tiger_ai.picking_a_tiger_to_move()#TIGER(0,0))#.picking_a_tiger_to_move()
                action_tiger = tiger_ai.make_a_move(tiger,None, None, False)
            state = board
            print(board)
            next_state = state.copy()
            play, tiger_reward, goat_reward = tiger_score_check(tiger_ai)
            #tiger_reward = tiger_score
            if not play:
                done = True
                memory.append((current_state, action_tiger, tiger_reward, next_state, done))
                break
            else:
                memory.append((current_state, action_tiger, tiger_reward, next_state, done))


run_environment(1000, False, None, None)#, tiger)
##print(memory)

