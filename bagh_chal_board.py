import sys
import pygame

# import pygame.display

pygame.init()
n = 600
m = 80
width, height = n, n
screensize = n + 2 * m, n + 2 * m
screen = pygame.display.set_mode(screensize)
pygame.display.flip()
running = True

black = (47, 17, 6)
green = (71, 165, 87)
gold = (219, 207, 32)


def drawBoard():
    for i in range(0, int(5 * n / 4), int(n / 4)):
        pygame.draw.line(screen, black, (i + m, m), (i + m, n + m), 5)  # vertical lines
        pygame.draw.line(screen, black, (m, i + m), (n + m, i + m), 5)  # horizontal lines

        pygame.draw.line(screen, black, (2 * i + m, m), (n + m, n - 2 * i + m), 2)
        pygame.draw.line(screen, black, (m, 2 * i + m), (n - 2 * i + m, n + m), 2)
    pygame.draw.line(screen, black, (int(n / 2) + m, m), (m, int(n / 2) + m), 2)
    pygame.draw.line(screen, black, (int(n) + m, m), (m, int(n) + m), 2)
    pygame.draw.line(screen, black, (int(n) + m, int(n / 2) + m), (int(n / 2) + m, n + m), 2)

    for i in range(0, n, int(n / 4)):
        for j in range(0, n, int(n / 4)):
            pygame.draw.circle(screen, gold, (2 * i + m, 2 * j + m), 20)
            pygame.draw.circle(screen, gold, (2 * i + int(n / 4) + m, 2 * j + m), 10)
            pygame.draw.circle(screen, gold, (2 * i + m, 2 * j + int(n / 4) + m), 10)
            pygame.draw.circle(screen, gold, (2 * i + int(n / 4) + m, 2 * j + int(n / 4) + m), 15)

            pygame.draw.circle(screen, black, (2 * i + m, 2 * j + m), 20, 4)
            pygame.draw.circle(screen, black, (2 * i + int(n / 4) + m, 2 * j + m), 10, 3)
            pygame.draw.circle(screen, black, (2 * i + m, 2 * j + int(n / 4) + m), 10, 3)
            pygame.draw.circle(screen, black, (2 * i + int(n / 4) + m, 2 * j + int(n / 4) + m), 15, 3)


while running:
    screen.fill(green)
    drawBoard()
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
