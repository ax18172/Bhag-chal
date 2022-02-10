import pygame
import numpy as np
import random

grid_matrix = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
taken_spots = np.zeros((3, 3))
goat_coord = []


# avilable_moves = [[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]


class TIGER:
    def __init__(self, init_x, init_y):
        self.pos_x = init_x
        self.pos_y = init_y
        taken_spots[self.pos_x, self.pos_y] = 1
        self.probability_matrix = np.zeros((3, 3))

    def return_position(self):
        return self.pos_x, self.pos_y

    def move(self, dx, dy):
        if abs(int(dx) * int(dy)) == 1 or abs(int(dx) * int(dy)) == 0:
            if [self.pos_x + dx, self.pos_y + dy] not in grid_matrix:
                pass
            else:
                if taken_spots[self.pos_x + dx, self.pos_y + dy] != 0:
                    pass
                else:
                    taken_spots[self.pos_x, self.pos_y] = 0
                    self.pos_x += dx
                    self.pos_y += dy
                    taken_spots[self.pos_x, self.pos_y] = 1
        else:
            pass
        return self.pos_x, self.pos_y

    def scan_for_food(self):
        possible_targets = []
        attack_direction = []
        for goat in goat_coord:
            goat_x, goat_y = goat
            if abs(self.pos_x - goat_x) <= 1 and abs(self.pos_y - goat_y) <= 1:
                vector = goat_x - self.pos_x, goat_y - self.pos_y
                possible_targets.append((goat_x, goat_y))
                attack_direction.append(2 * vector)
        return attack_direction, possible_targets

    def probability_matrix(self):


class GOAT:
    def __init__(self, init_x, init_y, deployment):
        self.pos_x = init_x
        self.pos_y = init_y
        self.deployment = deployment
        if taken_spots[self.pos_x, self.pos_y] == 0:
            taken_spots[self.pos_x, self.pos_y] = 2

    def return_position(self):
        return self.pos_x, self.pos_y

    def move(self, dx, dy):
        if abs(int(dx) * int(dy)) == 1 or abs(int(dx) * int(dy)) == 0:
            if self.deployment:
                if [self.pos_x + dx, self.pos_y + dy] not in grid_matrix:
                    pass
                else:
                    if taken_spots[self.pos_x + dx, self.pos_y + dy] != 0:
                        pass
                    else:
                        taken_spots[self.pos_x, self.pos_y] = 0
                        self.pos_x += dx
                        self.pos_y += dy
                        taken_spots[self.pos_x, self.pos_y] = 2
        else:
            pass
        return self.pos_x, self.pos_y

    def eaten(self, consumed):
        if consumed:
            self.deployment = False
            self.pos_x = -10
            self.pos_y = -10

def eat(Tiger):
    direction, prey = tiger.scan_for_food()
    for vector in direction:
        move = tiger.move(vector[0], vector[1])
        if move is not None:
           

def main():
    tiger = TIGER(2, 2)
    coord_x = int(input("enter goat x coordinates:"))
    coord_y = int(input("enter goat y coordinates:"))
    goat1 = GOAT(coord_x, coord_y, True)
    goat_coord.append(goat1.return_position())



# main()
tiger = TIGER(1, 1)
# goat = GOAT(1, 1, True)
# matrix = tiger.move_probabilities(1, 1)
# new_coord = tiger.eat(0, -1)
# new_coord1 = goat.move(1, 0)
print(matrix)
