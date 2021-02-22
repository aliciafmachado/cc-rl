import numpy as np


class RendererNode:
    id_counter = 0

    def __init__(self, coords, depth=0, p=1.):
        self.id = RendererNode.id_counter
        RendererNode.id_counter += 1

        self.p = p
        self.coords = coords
        self.depth = depth

        self.left = None
        self.right = None

    def next_coords(self, depth, next_depth, action):
        hstep = 1 / depth
        wstep = 1 / (2 ** (next_depth + 1))
        return self.coords + np.array([(action * 2 - 1) * wstep, hstep])

    def __getitem__(self, i):
        assert i == 0 or i == 1
        return self.left if i == 0 else self.right

    def __setitem__(self, i, value):
        assert i == 0 or i == 1
        if i == 0:
            self.left = value
        else:
            self.right = value
