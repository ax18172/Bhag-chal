from bag_chal_main import TIGER, GOAT, eat, grid_matrix, board
import numpy as np


# def new_move(tiger_i,tiger_j):


def minimax(depth, tiger_x, tiger_y, maximising_player):
    tiger = TIGER(tiger_x, tiger_y)
    possible_moves = tiger.probabilities_matrix()

    if maximising_player:
        bestval = -1e6
        for spot in possible_moves:
            if spot != 0:
                index = np.where(board == spot)
                tiger.move(index[0] - tiger.return_position()[0], index[1] - tiger.return_position()[1],
                           "move")
                """score_update
                value = minimax(depth+1,tiger.return_position()[0],tiger.return_position()[1],False)
                bestval = max(bestval,value)
                return bestval"""
    else:
        bestval = 1e6
        for spot in possible_moves:
            if spot != 0:
                index = np.where(board == spot)
                tiger.move(index[0] - tiger.return_position()[0], index[1] - tiger.return_position()[1],
                           "move")
                """score_update
                value = minimax(depth+1,tiger.return_position()[0],tiger.return_position()[1],True)
                bestval = min(bestval,value)
                return bestval"""



    """while move < depth:
        possible_moves = tiger.probabilities_matrix()
        for spot in possible_moves:
            if spot != 0:
                index = np.where(taken_spots == spot)
                tiger.move(index[0] - tiger.return_position()[0], index[1] - tiger.return_position()[1],
                           "move")"""
