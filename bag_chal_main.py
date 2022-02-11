import pygame
import numpy as np
import random

grid_matrix = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
taken_spots = np.zeros((3, 3))
goat_coord = []
goats = []


# avilable_moves = [[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]


class TIGER:
    def __init__(self, init_x, init_y):
        self.pos_x = init_x
        self.pos_y = init_y
        taken_spots[self.pos_x, self.pos_y] = 1
        self.probability_matrix = np.zeros((3, 3))

    def return_position(self):
        return self.pos_x, self.pos_y

    def move(self, dx, dy, mission):
        if mission == "move":
            constraint_1 = abs(int(dx) * int(dy)) == 1
            constraint_2 = abs(int(dx) * int(dy)) == 0
        elif mission == "eat":
            constraint_1 = abs(int(dx) * int(dy)) == 4
            constraint_2 = abs(int(dx) * int(dy)) == 0
        if constraint_1 or constraint_2:
            if [self.pos_x + dx, self.pos_y + dy] not in grid_matrix:
                return None
            else:
                if taken_spots[self.pos_x + dx, self.pos_y + dy] != 0:
                    return None
                else:
                    taken_spots[self.pos_x, self.pos_y] = 0
                    self.pos_x += dx
                    self.pos_y += dy
                    taken_spots[self.pos_x, self.pos_y] = 1
        else:
            return None
        return self.pos_x, self.pos_y

    def scan_for_food(self):
        attack_direction = []
        for goat in goat_coord:
            goat_x, goat_y = goat
            if abs(self.pos_x - goat_x) <= 1 and abs(self.pos_y - goat_y) <= 1:
                vector = 2 * (goat_x - self.pos_x), 2 * (goat_y - self.pos_y)
                attack_direction.append(vector)
        return attack_direction

    def probabilities_matrix(self):
        self.probability_matrix[self.pos_x, self.pos_y] = 1
        if [self.pos_x + 1, self.pos_y + 1] in grid_matrix and taken_spots[self.pos_x + 1, self.pos_y + 1] == 0:
            self.probability_matrix[self.pos_x + 1, self.pos_y + 1] = random.random()
        if [self.pos_x - 1, self.pos_y - 1] in grid_matrix and taken_spots[self.pos_x - 1, self.pos_y - 1] == 0:
            self.probability_matrix[self.pos_x - 1, self.pos_y - 1] = random.random()
        if [self.pos_x, self.pos_y + 1] in grid_matrix and taken_spots[self.pos_x, self.pos_y + 1] == 0:
            self.probability_matrix[self.pos_x, self.pos_y + 1] = random.random()
        if [self.pos_x + 1, self.pos_y] in grid_matrix and taken_spots[self.pos_x + 1, self.pos_y] == 0:
            self.probability_matrix[self.pos_x + 1, self.pos_y] = random.random()
        if [self.pos_x, self.pos_y - 1] in grid_matrix and taken_spots[self.pos_x, self.pos_y - 1] == 0:
            self.probability_matrix[self.pos_x, self.pos_y - 1] = random.random()
        if [self.pos_x - 1, self.pos_y] in grid_matrix and taken_spots[self.pos_x - 1, self.pos_y] == 0:
            self.probability_matrix[self.pos_x - 1, self.pos_y] = random.random()
        if [self.pos_x + 1, self.pos_y - 1] in grid_matrix and taken_spots[self.pos_x + 1, self.pos_y - 1] == 0:
            self.probability_matrix[self.pos_x + 1, self.pos_y - 1] = random.random()
        if [self.pos_x - 1, self.pos_y + 1] in grid_matrix and taken_spots[self.pos_x - 1, self.pos_y + 1] == 0:
            self.probability_matrix[self.pos_x - 1, self.pos_y + 1] = random.random()
        return self.probability_matrix


class GOAT:
    def __init__(self, init_x, init_y):
        self.pos_x = init_x
        self.pos_y = init_y
        if taken_spots[self.pos_x, self.pos_y] == 0:
            taken_spots[self.pos_x, self.pos_y] = 2

    def return_position(self):
        return self.pos_x, self.pos_y

    def move(self, dx, dy):
        if abs(int(dx) * int(dy)) == 1 or abs(int(dx) * int(dy)) == 0:
            if [self.pos_x + dx, self.pos_y + dy] not in grid_matrix:
                return None
            else:
                if taken_spots[self.pos_x + dx, self.pos_y + dy] != 0:
                    return None
                else:
                    taken_spots[self.pos_x, self.pos_y] = 0
                    self.pos_x += dx
                    self.pos_y += dy
                    taken_spots[self.pos_x, self.pos_y] = 2
        else:
            return None
        return self.pos_x, self.pos_y


def eat(tiger):
    direction = tiger.scan_for_food()
    for vector in direction:
        move = tiger.move(vector[0], vector[1], "eat")
        if move is not None:
            del goats[direction.index(vector)]
            goat_x, goat_y = goat_coord[direction.index(vector)]
            taken_spots[goat_x, goat_y] = 0
        break


def main():
    tiger = TIGER(2, 2)
    coord_x = int(input("enter goat x coordinates:"))
    coord_y = int(input("enter goat y coordinates:"))
    goat1 = GOAT(coord_x, coord_y)
    print(taken_spots)
    goat_coord.append(goat1.return_position())
    goats.append(goat1)
    eat(tiger)
    print(taken_spots)


main()
