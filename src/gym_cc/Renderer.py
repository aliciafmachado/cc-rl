import numpy as np
import os
import sys
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame

class Renderer:
    constants = { 'width': 1028, 'height': 800, 'bar_height': 100, 'font_height': 30, 'radius': 10, 'fps': 10, 'wheel_sensibility': 1.25 }
    colors = { 'background': (255, 255, 255), 'black': (0, 0, 0), 'line': (0, 0, 0), 'highlight': (255, 0, 0) }

    def __init__(self, mode, n_labels):
        assert(mode == 'draw' or mode == 'print')

        self.mode = mode
        self.cur_reward = 1.
        self.best_reward = 0.

        if mode == 'draw':
            pygame.init()
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', self.constants['font_height'])
            self.display = pygame.display.set_mode((self.constants['width'], self.constants['height']), pygame.RESIZABLE, 32)
            self.display.fill(Renderer.colors['background'])
            self.clock = pygame.time.Clock()

            self.width = self.constants['width']
            self.height = self.constants['height']
            self.translation = np.array([0, 0], dtype=int)
            self.scale = 1.
            self.panning = False
            self.root = [1., np.array([self.constants['width'] / 2, self.constants['radius']]), [None, None]]
            self.cur_depth = 0
            self.cur_node = self.root
            self.depth = n_labels

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
            self.coords = self.root[1]
        
        self.cur_depth = 0
        self.cur_reward = 1
        self.cur_node = self.root

    def __render_print(self, action, probability):
        if action == -1:
            print('L', end='')
        else:
            print('R', end='')
    
    def __render_draw(self, action, probability):
        def transform(pos):
            return pos * self.scale + self.translation

        def update_tree(action, probability):
            def next_coords(cur_coords):
                next_coords = np.array([0, 0], dtype=int)
                next_coords[1] = cur_coords[1] + (self.constants['height'] - self.constants['bar_height']) / self.depth

                wstep = self.constants['width'] / (2 ** (self.cur_depth + 2))
                if action == 0:
                    next_coords[0] = cur_coords[0] + wstep
                else:
                    next_coords[0] = cur_coords[0] - wstep
                
                return next_coords

            action = 0 if action == -1 else 1

            if self.cur_node[2][action] is None:
                prob2 = self.cur_node[0] * probability
                coords2 = next_coords(self.cur_node[1])
                self.cur_node[2][action] = [prob2, coords2, [None, None]]

            self.cur_reward *= probability
            self.cur_node = self.cur_node[2][action]

        def update_events():
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.panning = True
                    self.mouse_pos = event.pos
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.panning = False
                elif event.type == pygame.MOUSEMOTION:
                    if self.panning:
                        for i in range(2):
                            self.translation[i] += event.pos[i] - self.mouse_pos[i]
                        self.mouse_pos = event.pos
                elif event.type == pygame.MOUSEWHEEL:
                    if event.y == 1:
                        scale2 = self.scale * self.constants['wheel_sensibility']
                    else:
                        scale2 = self.scale / self.constants['wheel_sensibility']
                    self.translation += (np.array(pygame.mouse.get_pos(), dtype=float) * (self.scale - scale2)).astype(int)
                    self.scale = scale2
                elif event.type == pygame.QUIT:
                    exit()
                elif event.type == pygame.WINDOWRESIZED:
                    self.width = event.x
                    self.height = event.y

        def draw():
            self.display.fill(self.colors['background'])

            # Tree
            st = []
            st.append(self.root)
            while len(st) != 0:
                node = st[-1]
                st.pop()

                pygame.draw.circle(self.display, Renderer.colors['line'], transform(node[1]), self.constants['radius'] * self.scale)
                for i in range(2):
                    if node[2][i] is not None:
                        pygame.draw.line(self.display, Renderer.colors['line'], transform(node[1]), transform(node[2][i][1]))
                        st.append(node[2][i])

            # Bottom bar
            text = 'Current reward: {:.6f} Best reward: {:.6f}'.format(self.cur_node[0], self.best_reward)
            text_blit = self.font.render(text, False, Renderer.colors['black'])
            rect = pygame.Rect((0, self.height - self.constants['font_height']), (self.width, self.height))
            pygame.draw.rect(self.display, self.colors['background'], rect)
            self.display.blit(text_blit, (0, self.height - self.constants['font_height'])) 

        update_tree(action, probability)
        update_events()
        draw()

        pygame.display.update()
        self.clock.tick(self.constants['fps'])
