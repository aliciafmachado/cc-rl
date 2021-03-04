import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QModel(nn.Module):
    def __init__(self, tree_height, device):
        super().__init__()
        self.device = device
        self.tree_height = tree_height
        h_size = 2 * (tree_height - 1)
        self.h1 = nn.Linear(h_size, h_size // 2)
        self.output = nn.Linear(h_size // 2, 1)

        self.to(device)

    def choose_action(self, actions, probabilities, next_p, depth):
        self.eval()
        actions = actions.reshape(1, -1).to(self.device)
        probabilities = probabilities.reshape(1, -1).to(self.device)
        with torch.no_grad():
            probabilities[0, depth] = next_p

            values = []
            actions[0, depth] = -1
            values.append(self.forward(actions, probabilities))
            actions[0, depth] = 1
            values.append(self.forward(actions, probabilities))
        return 2 * np.argmax(values) - 1

    def forward(self, actions, probabilities):
        inp = torch.cat((actions, probabilities), 1)
        h = F.relu(self.h1(inp))
        out = self.output(h)
        return out
