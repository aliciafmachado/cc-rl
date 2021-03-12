import numpy as np
import os

from .RendererNode import RendererNode

# Pygame needs adjustment before import if in notebook
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
try:
    get_ipython()
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
except NameError:
    pass
import pygame


class Renderer:
    """
    Tool to display useful environment information. It can print the path and rewards and
    it can also draw the probability tree and show environment steps.
    """
    constants = {'font_size': 30, 'radius': 10, 'fps': 10, 'wheel_sensibility': 1.25,
                 'bar_margin': 10}
    colors = {'background': (24, 26, 27), 'font': (211, 211, 211), 'black': (0, 0, 0),
              'line': (100, 100, 0), 'highlight': (9, 255, 243),
              'highlight2': (255, 43, 0)}

    def __init__(self, mode: str, n_labels: int, loss='exact_match'):
        """
        Depending on mode, it will start a tree in self.root to be shown by pygame.
        @param mode: 'draw', 'print' or None.
        @param n_labels: Number of labels in the dataset, which will be the depth of the
            tree.
        @param loss: 'exact_match' or 'hamming'.
        """
        assert (mode is None or mode == 'draw' or mode == 'print')

        self.__mode = mode
        self.__cur_reward = 0.
        self.__best_reward = -np.inf
        self.__depth = n_labels
        self.__loss = loss

        if mode == 'draw':
            # Setup pygame
            pygame.init()
            pygame.font.init()

            # Setup display
            self.__width = pygame.display.Info().current_w
            self.__height = pygame.display.Info().current_h
            self.__display = pygame.display.set_mode((self.__width, self.__height),
                                                     pygame.RESIZABLE, 32)
            self.__display.fill(Renderer.colors['background'])
            self.__font = pygame.font.SysFont('helvetica', self.constants['font_size'])
            self.__clock = pygame.time.Clock()

            # Setup tree
            self.__root = RendererNode(np.array([0.5, 0]))
            self.__cur_node = self.__root
            self.__cur_actions = []
            self.__best_actions = []

            # Setup view
            self.__translation = np.zeros(2, dtype=int)
            self.__scale = 1.
            self.__is_panning = False

    def step(self, action: int, probability: float):
        """
        Displays environment according to the Renderer mode.
        @param action: Action took in the last step.
        @param probability: Probability of the action took.
        """
        if self.__mode is None:
            return

        if self.__loss == 'exact_match':
            self.__cur_reward += np.log(probability)
        else:
            self.__cur_reward += probability

        if self.__mode == 'print':
            self.__step_print(action)
        elif self.__mode == 'draw':
            self.__step_draw(action, probability)

    def reset(self, label: int = 0):
        """
        Resets the position of the current node in the tree, and updates the best path if
        it has reached the end.
        @param label: Depth of the node that the agent will be in.
        """
        assert label <= self.__depth - 1

        if (self.__mode == 'draw' and self.__cur_node.depth == self.__depth - 1) or \
                (self.__mode == 'print' and label == 0):
            # Update best path
            if self.__cur_reward > self.__best_reward and self.__cur_reward != 0:
                self.__best_reward = self.__cur_reward
                self.__best_actions = self.__cur_actions

            if self.__mode == 'print':
                print(' Reward: {:.4f}'.format(self.__cur_reward))

        self.__cur_reward = 0.

        if self.__mode == 'draw':
            # Walk from root to label
            self.__cur_actions = self.__cur_actions[:label]

            self.__cur_node = self.__root
            for i in range(label):
                self.__cur_node = self.__cur_node[self.__cur_actions[i]]
                self.__cur_reward *= self.__cur_node.p

            self.__draw()

    def next_sample(self):
        """
        Resets everything in the renderer.
        """
        self.__best_reward = 0.
        self.__best_actions = []
        self.__cur_reward = 1
        self.__cur_actions = []

        if self.__mode == 'draw':
            RendererNode.id_counter = 0
            self.__root = RendererNode(np.array([0.5, 0]))
            self.__cur_node = self.__root
            self.__translation = np.zeros(2, dtype=int)
            self.__scale = 1.
            self.__is_panning = False

    @staticmethod
    def __step_print(action: int):
        if action == -1:
            print('L', end='')
        else:
            print('R', end='')

    def __step_draw(self, action: int, probability: float):
        def update_tree():
            a = 0 if action == -1 else 1

            if self.__cur_node[a] is None:
                next_depth = self.__cur_node.depth + 1
                coords2 = self.__cur_node.next_coords(self.__depth, next_depth, a)
                self.__cur_node[a] = RendererNode(coords2, next_depth, probability)

            if self.__loss == 'exact_match':
                self.__cur_reward += np.log(probability)
            else:
                self.__cur_reward += probability
            self.__cur_actions.append(a)
            self.__cur_node = self.__cur_node[a]

        update_tree()
        self.__draw()

    def __draw(self):
        def update_events():
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.__is_panning = True
                    self.mouse_pos = event.pos
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.__is_panning = False
                elif event.type == pygame.MOUSEMOTION:
                    if self.__is_panning:
                        for i in range(2):
                            self.__translation[i] += (event.pos[i] - self.mouse_pos[i])
                        self.mouse_pos = event.pos
                elif event.type == pygame.MOUSEWHEEL:
                    if event.y == 1:
                        scale2 = self.__scale * self.constants['wheel_sensibility']
                    else:
                        scale2 = self.__scale / self.constants['wheel_sensibility']
                    p = (np.array(pygame.mouse.get_pos(),
                                  dtype=float) - self.__translation) / self.__scale
                    self.__translation += (p * (self.__scale - scale2)).astype(int)
                    self.__scale = scale2
                elif event.type == pygame.QUIT:
                    exit()
                elif event.type == pygame.WINDOWRESIZED:
                    self.__width = event.x
                    self.__height = event.y

        def transform(pos):
            f = 0.9
            fm = (1 - f) / 2
            return pos * self.__scale * np.array([f * self.__width, f * self.__height]) +\
                np.array([fm * self.__width, fm * self.__height]) + self.__translation

        def draw():
            self.__display.fill(self.colors['background'])

            # DFS drawing tree
            st = [(self.__root, True)]
            while len(st) != 0:
                node, in_best_path = st[-1]
                st.pop()

                # Draw node
                if node.id == self.__cur_node.id:
                    color = Renderer.colors['highlight']
                elif in_best_path:
                    color = Renderer.colors['highlight2']
                else:
                    color = Renderer.colors['line']

                pygame.draw.circle(self.__display, color, transform(node.coords),
                                   self.constants['radius'] * self.__scale)

                for i in range(2):
                    if node[i] is not None:
                        in_best_path2 = len(self.__best_actions) == 0 or (
                                in_best_path and self.__best_actions[node.depth] == i)

                        # Draw line joining nodes
                        p1 = transform(node.coords)
                        p2 = transform(node[i].coords)
                        if in_best_path2:
                            color = Renderer.colors['highlight2']
                        else:
                            color = Renderer.colors['line']

                        pygame.draw.line(self.__display, color, p1, p2, 2)

                        # Draw probability text
                        text = '{:.1f}'.format(round(node[i].p, 1))
                        text_blit = self.__font.render(text, False,
                                                       Renderer.colors['font'])
                        self.__display.blit(text_blit, (p1 + p2) / 2 - self.constants[
                            'font_size'] * np.array([0.6, 0.45]))

                        # Put node in stack
                        if in_best_path2:
                            st.append((node[i], True))
                        else:
                            st.append((node[i], False))

            # Bottom bar
            text = 'Current reward: {:.2e}  Best reward: {:.2e}'.format(
                self.__cur_reward, self.__best_reward)
            text_blit = self.__font.render(text, False, Renderer.colors['font'])
            rect = pygame.Rect((0, self.__height - self.constants['font_size']),
                               (self.__width, self.__height))
            pygame.draw.rect(self.__display, self.colors['background'], rect)
            rect = text_blit.get_rect()
            rect.right = self.__width - self.constants['bar_margin']
            rect.bottom = self.__height - self.constants['bar_margin']
            self.__display.blit(text_blit, rect)

        update_events()
        draw()

        pygame.display.update()
        self.__clock.tick(self.constants['fps'])
