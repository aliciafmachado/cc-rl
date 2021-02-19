import numpy as np
import os
import sys
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
try:
    get_ipython()
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
except NameError:
    pass
import pygame

class Renderer:
    constants = { 'width': -1, 'height': -1, 'margin': -1, 'font_size': 30, 'radius': 10, 'fps': 10, 'wheel_sensibility': 1.25, 'bar_margin': 10 }
    colors = { 'background': (24, 26, 27), 'font': (211, 211, 211), 'black': (0, 0, 0), 'line': (100, 100, 0), 'highlight': (9, 255, 243), 'highlight2': (255, 43, 0) }

    def __init__(self, mode, n_labels):
        assert(mode == 'none' or mode == 'draw' or mode == 'print')

        self.mode = mode
        self.cur_reward = 1.
        self.best_reward = 0.
        self.cur_actions = []
        self.best_actions = []

        if mode == 'draw':
            # Setup pygame
            pygame.init()
            pygame.font.init()
            
            # Setup display
            self.constants['width'] = pygame.display.Info().current_w
            self.constants['height'] = pygame.display.Info().current_h
            self.font = pygame.font.SysFont('helvetica', self.constants['font_size'])
            self.display = pygame.display.set_mode((self.constants['width'], self.constants['height']), pygame.RESIZABLE, 32)
            self.display.fill(Renderer.colors['background'])
            self.node_id_cnt = 1

            self.clock = pygame.time.Clock()
            self.width = self.constants['width']
            self.height = self.constants['height']
            self.depth = n_labels
            self.next_sample()

    def render(self, action, probability):
        self.cur_reward *= probability
        if self.mode == 'print':
            self.__render_print(action, probability)
        elif self.mode == 'draw':
            self.__render_draw(action, probability)
            self.cur_depth += 1
    
    def reset(self):
        if self.cur_reward > self.best_reward and self.cur_reward != 1:
            self.best_reward = self.cur_reward
            self.best_actions = self.cur_actions
        
        if self.mode == 'print':
            print(' Reward: {:.4f}'.format(self.cur_reward))
        elif self.mode == 'draw':
            self.cur_node = self.root
            self.cur_depth = 0
        
        self.cur_reward = 1
        self.cur_actions = []
    
    def next_sample(self):
        self.best_reward = 0.
        self.best_actions = []
        self.cur_reward = 1
        self.cur_actions = []
        self.node_id_cnt = 1

        if self.mode == 'draw':
            self.cur_depth = 0
            self.root = [self.node_id_cnt, 1., np.array([self.constants['width'] / 2, self.constants['radius']]), [None, None]]
            self.node_id_cnt += 1
            self.cur_node = self.root
            self.translation = np.array([self.width * 0.025, self.height * 0.05], dtype=int)
            self.scale = 0.9
            self.panning = False

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
                next_coords[1] = cur_coords[1] + (self.constants['height'] - self.constants['font_size']) / self.depth

                wstep = (self.constants['width']) / (2 ** (self.cur_depth + 2))
                if action == 0:
                    next_coords[0] = cur_coords[0] + wstep
                else:
                    next_coords[0] = cur_coords[0] - wstep
                
                return next_coords

            action = 0 if action == -1 else 1

            if self.cur_node[3][action] is None:
                coords2 = next_coords(self.cur_node[2])
                self.cur_node[3][action] = [self.node_id_cnt, probability, coords2, [None, None]]
                self.node_id_cnt += 1

            self.cur_reward *= probability
            self.cur_actions.append(action)
            self.cur_node = self.cur_node[3][action]

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
                            self.translation[i] += (event.pos[i] - self.mouse_pos[i])
                        self.mouse_pos = event.pos
                elif event.type == pygame.MOUSEWHEEL:
                    if event.y == 1:
                        scale2 = self.scale * self.constants['wheel_sensibility']
                    else:
                        scale2 = self.scale / self.constants['wheel_sensibility']
                    p = (np.array(pygame.mouse.get_pos(), dtype=float) - self.translation) / self.scale
                    self.translation += (p * (self.scale - scale2)).astype(int)
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
            st.append((self.root, 1))
            while len(st) != 0:
                node, best = st[-1]
                st.pop()

                if node[0] == self.cur_node[0]:
                    pygame.draw.circle(self.display, Renderer.colors['highlight'], transform(node[2]), self.constants['radius'] * self.scale)
                elif best:
                    pygame.draw.circle(self.display, Renderer.colors['highlight2'], transform(node[2]), self.constants['radius'] * self.scale)
                else:
                    pygame.draw.circle(self.display, Renderer.colors['line'], transform(node[2]), self.constants['radius'] * self.scale)

                for i in range(2):
                    if node[3][i] is not None:
                        best2 = len(self.best_actions) == 0 or (best and self.best_actions[best-1] == i)

                        p1 = transform(node[2])
                        p2 = transform(node[3][i][2])
                        if best2:
                            pygame.draw.line(self.display, Renderer.colors['highlight2'], p1, p2, 2)
                        else:
                            pygame.draw.line(self.display, Renderer.colors['line'], p1, p2, 2)

                        text = '{:.1f}'.format(round(node[3][i][1], 1))
                        text_blit = self.font.render(text, False, Renderer.colors['font'])
                        self.display.blit(text_blit, (p1 + p2) / 2 - np.array([self.constants['font_size'] * 0.6, self.constants['font_size'] * 0.45]))

                        if best2:
                            st.append((node[3][i], best+1))
                        else:
                            st.append((node[3][i], 0))

            # Bottom bar
            text = 'Current reward: {:.2e}  Best reward: {:.2e}'.format(self.cur_reward, self.best_reward)
            text_blit = self.font.render(text, False, Renderer.colors['font'])
            rect = pygame.Rect((0, self.height - self.constants['font_size']), (self.width, self.height))
            pygame.draw.rect(self.display, self.colors['background'], rect)
            rect = text_blit.get_rect()
            rect.right = self.width - self.constants['bar_margin']
            rect.bottom = self.height - self.constants['bar_margin']
            self.display.blit(text_blit, rect) 

        update_tree(action, probability)
        update_events()
        draw()

        pygame.display.update()
        self.clock.tick(self.constants['fps'])
