import torch
import torch.nn as nn
import torch.nn.functional as F


class MCTSModel(nn.Module):
    def __init__(self, tree_height, device):
        super().__init__()
        self.device = device
        self.tree_height = tree_height
        h_size = 2 * (tree_height - 1)
        self.h1 = nn.Linear(h_size, h_size // 2)
        self.output = nn.Linear(h_size // 2, 2)

        self.to(device)

    def forward(self, actions_history, probas_history):
        """
        Takes the current tree state as input and calculates the value and the policy
        :param actions_history: The past actions taken by the agent
        :param probas_history: The past probas encountered before each action
        :returns the value and the policy in this order
        """
        # Concatenating everything for the first layer
        actions_history = actions_history.reshape(-1, self.tree_height - 1)
        probas_history = probas_history.reshape(-1, self.tree_height - 1)
        inp = torch.cat((actions_history, probas_history), 1)

        # Passing state through the network
        h = F.relu(self.h1(inp))
        out = self.output(h)
        split = out.split(1, 1)
        out = torch.cat((split[0], torch.sigmoid(split[1])), dim=1)
        return out
