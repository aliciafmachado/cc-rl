import matplotlib.pyplot as plt
import numpy as np
import os
import sys
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame

class Renderer:
    WIDTH = 1024
    HEIGHT = 800
    BAR_HEIGHT = 100
    FONT_HEIGHT = 30
    RADIUS = 10
    FPS = 1
    ROOT_COORDS = [WIDTH / 2, RADIUS]
    colors = { 'background': (255, 255, 255), 'black': (0, 0, 0), 'line': (0, 0, 0), 'highlight': (255, 0, 0) }

    def __init__(self, mode, path, probabilities):
        assert(mode == 'draw' or mode == 'print')

        self.mode = mode
        self.path = path
        self.probabilities = probabilities
        self.coords = Renderer.ROOT_COORDS

        if mode == 'draw':
            pygame.init()
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', Renderer.FONT_HEIGHT)
            self.display = pygame.display.set_mode((Renderer.WIDTH, Renderer.HEIGHT), 0, 32)
            self.display.fill(Renderer.colors['background'])
            self.clock = pygame.time.Clock()

    def render(self, current_estimator):
        if self.mode == 'print':
            self.__render_print(current_estimator)
        else:
            self.__render_draw(current_estimator)
    
    def reset(self):
        if self.mode == 'print':
            print(' Proba: {:.4f}'.format(np.prod(self.probabilities)))
        else: 
            self.coords = Renderer.ROOT_COORDS
            plt.show()

    def __render_print(self, current_estimator):
        if self.path[current_estimator] == -1:
            print('L', end='')
        else:
            print('R', end='')
    
    def __render_draw(self, current_estimator):
        pygame.draw.circle(self.display, Renderer.colors['line'], self.coords, Renderer.RADIUS)
        next_coords = self.__next_coords(current_estimator)
        pygame.draw.line(self.display, Renderer.colors['line'], self.coords, next_coords)
        pygame.draw.circle(self.display, Renderer.colors['highlight'], next_coords, Renderer.RADIUS)
        self.coords = next_coords

        cur_p = np.prod(self.probabilities[:current_estimator])
        text = self.font.render('Current reward: ' + str(cur_p) + ' Best reward:', False, Renderer.colors['black'])
        rect = pygame.Rect((0, Renderer.HEIGHT - Renderer.FONT_HEIGHT), (Renderer.WIDTH, Renderer.HEIGHT))
        pygame.draw.rect(self.display, self.colors['background'], rect)
        self.display.blit(text, (0, Renderer.HEIGHT - Renderer.FONT_HEIGHT))

        pygame.display.update()
        self.clock.tick(Renderer.FPS)

    def __next_coords(self, current_estimator):
        next_coords = [0, 0]
        next_coords[1] = self.coords[1] + (Renderer.HEIGHT - Renderer.BAR_HEIGHT) / len(self.path)

        wstep = Renderer.WIDTH / (2 ** (current_estimator + 2))
        if self.path[current_estimator] == -1:
            next_coords[0] = self.coords[0] + wstep
        else:
            next_coords[0] = self.coords[0] - wstep
        
        return next_coords
