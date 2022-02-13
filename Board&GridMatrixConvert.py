import sys
import pygame

# import pygame.display

pygame.init()
n = 500
m = 50
dim = 5
width, height = n, n
screensize = n + 2 * m, n + 2 * m
screen = pygame.display.set_mode(screensize)
pygame.display.flip()
running = True

black = (47, 17, 6)
green = (71, 165, 87)
gold = (219, 207, 32)


def drawBoard():
    for i in range(0, int(dim * n / (dim - 1)), int(n / (dim - 1))):
        pygame.draw.line(screen, black, (i + m, m), (i + m, n + m), 5)  # vertical lines
        pygame.draw.line(screen, black, (m, i + m), (n + m, i + m), 5)  # horizontal lines
    for i in range(0, int(n / 2), int(n / (dim - 1))):
        pygame.draw.line(screen, black, (m + 2 * i, m), (m, m + 2 * i), 3)
        pygame.draw.line(screen, black, (m + 2 * i, m), (m + n, m + n - 2 * i), 3)
        pygame.draw.line(screen, black, (m, 2 * i + m), (m + n - 2 * i, n + m), 3)
        pygame.draw.line(screen, black, (m + 2 * i, n + m), (m + n, 2 * i + m), 3)

    for i in range(0, n, int(n / (dim - 1))):
        for j in range(0, n, int(n / (dim - 1))):
            pygame.draw.circle(screen, gold, (2 * i + m, 2 * j + m), 20)
            pygame.draw.circle(screen, gold, (2 * i + int(n / (dim - 1)) + m, 2 * j + m), 10)
            pygame.draw.circle(screen, gold, (2 * i + m, 2 * j + int(n / (dim - 1)) + m), 10)
            pygame.draw.circle(screen, gold, (2 * i + int(n / (dim - 1)) + m, 2 * j + int(n / (dim - 1)) + m), 15)

            pygame.draw.circle(screen, black, (2 * i + m, 2 * j + m), 20, 4)
            pygame.draw.circle(screen, black, (2 * i + int(n / (dim - 1)) + m, 2 * j + m), 10, 3)
            pygame.draw.circle(screen, black, (2 * i + m, 2 * j + int(n / (dim - 1)) + m), 10, 3)
            pygame.draw.circle(screen, black, (2 * i + int(n / (dim - 1)) + m, 2 * j + int(n / (dim - 1)) + m), 15, 3)

import numpy as np

grid_matrix = []
for i in range(dim):
    for j in range(dim):
        grid_matrix.append([i, j])

board_array = np.zeros(2 * dim ** 2).reshape(dim ** 2, 2)
board_matrix = board_array.tolist()
for i in range(len(grid_matrix)):
    for j in range(2):
        board_matrix[i][j] = m + n * grid_matrix[i][j] / (dim - 1)

while running:
    screen.fill(green)
    drawBoard()
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

print(grid_matrix)
print(board_matrix)
