import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QModel(nn.Module):
    def __init__(self, tree_height, device):
        super().__init__()
        self.device = device
        self.tree_height = tree_height
        h_size = (2 * tree_height - 1) // 2
        self.input = nn.Linear(2 * tree_height, h_size)
        self.output = nn.Linear(h_size, 1)

        self.to(device)

    def choose_action(self, actions, probabilities, next_p):
        self.eval()
        with torch.no_grad():
            values = [self.forward(
                actions, probabilities, next_p,
                torch.tensor(i * 2 - 1, device=self.device)) for i in range(2)]
        return np.argmax(values)

    def forward(self, actions, probabilities, next_ps, next_actions):
        actions = actions.reshape(-1, self.tree_height - 1)
        probabilities = probabilities.reshape(-1, self.tree_height - 1)
        next_ps = next_ps.reshape(-1, 1)
        next_actions = next_actions.reshape(-1, 1)
        inp = torch.cat((actions, probabilities, next_ps, next_actions), 1)
        h = F.relu(self.input(inp))
        out = self.output(h)
        return out
