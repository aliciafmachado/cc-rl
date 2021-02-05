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
    FPS = 10
    ROOT_COORDS = [WIDTH / 2, RADIUS]
    colors = { 'background': (255, 255, 255), 'black': (0, 0, 0), 'line': (0, 0, 0), 'highlight': (255, 0, 0) }

    def __init__(self, mode, n_labels):
        assert(mode == 'draw' or mode == 'print')

        self.mode = mode
        self.coords = Renderer.ROOT_COORDS
        self.cur_reward = 1.
        self.best_reward = 0.
        self.cur_depth = 0
        self.depth = n_labels

        if mode == 'draw':
            pygame.init()
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', Renderer.FONT_HEIGHT)
            self.display = pygame.display.set_mode((Renderer.WIDTH, Renderer.HEIGHT), 0, 32)
            self.display.fill(Renderer.colors['background'])
            self.clock = pygame.time.Clock()

    def render(self, action, probability):
        self.cur_reward *= probability
        if self.mode == 'print':
            self.__render_print(action, probability)
        else:
            self.__render_draw(action, probability)
        self.cur_depth += 1
    
    def reset(self):
        self.best_reward = max(self.cur_reward, self.best_reward)
        
        if self.mode == 'print':
            print(' Reward: {:.4f}'.format(self.cur_reward))
        else:
            self.coords = Renderer.ROOT_COORDS
        
        self.cur_depth = 0
        self.cur_reward = 1

    def __render_print(self, action, probability):
        if action == -1:
            print('L', end='')
        else:
            print('R', end='')
    
    def __render_draw(self, action, probability):
        def draw_branch(last_coords, next_coords):
            pygame.draw.circle(self.display, Renderer.colors['line'], self.coords, Renderer.RADIUS)
            pygame.draw.line(self.display, Renderer.colors['line'], self.coords, next_coords)
            pygame.draw.circle(self.display, Renderer.colors['highlight'], next_coords, Renderer.RADIUS)
        
        def update_bar(cur_reward, best_reward):
            text = 'Current reward: {:.6f} Best reward: {:.6f}'.format(cur_reward, best_reward)
            text_blit = self.font.render(text, False, Renderer.colors['black'])
            rect = pygame.Rect((0, Renderer.HEIGHT - Renderer.FONT_HEIGHT), (Renderer.WIDTH, Renderer.HEIGHT))
            pygame.draw.rect(self.display, self.colors['background'], rect)
            self.display.blit(text_blit, (0, Renderer.HEIGHT - Renderer.FONT_HEIGHT))
        
        next_coords = self.__next_coords(action)
        draw_branch(self.coords, next_coords)
        self.coords = next_coords

        update_bar(self.cur_reward, self.best_reward)

        pygame.display.update()
        self.clock.tick(Renderer.FPS)

    def __next_coords(self, action):
        next_coords = [0, 0]
        next_coords[1] = self.coords[1] + (Renderer.HEIGHT - Renderer.BAR_HEIGHT) / self.depth

        wstep = Renderer.WIDTH / (2 ** (self.cur_depth + 2))
        if action == -1:
            next_coords[0] = self.coords[0] + wstep
        else:
            next_coords[0] = self.coords[0] - wstep
        
        return next_coords
