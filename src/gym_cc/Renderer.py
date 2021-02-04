class Renderer:
    def __init__(self, mode, path_space, probabilities_space):
        assert(mode == 'draw' or mode == 'print')
        if mode == 'draw':
            raise NotImplementedError

        self.mode = mode
        self.path = path_space
        self.probabilities = probabilities_space

    def render(self):
        if self.mode == 'print':
            self.__render_print()
    
    def reset(self):
        pass

    def __render_print(self):
        print('Path:', self.path)
