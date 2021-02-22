import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MCTSModel(nn.Module):
    def __init__(self, tree_height, device):
        super().__init__()
        self.device = device
        self.tree_height = tree_height
        h_size = (2*tree_height-1)//2

        self.input = nn.Linear(2*tree_height-1, h_size)
        self.output = nn.Linear(h_size, 2)

    def forward(self, actions_history, probas_history, next_proba):
        """
        Takes the current tree state as input and calculates the value and the policy
        @param actions_history: The past actions taken by the agent
        @param probas_history: The past probas encountered before each action
        @param next_proba: The probability (1 for right) for the next decision
        @:returns the value and the policy in this order
        """
        # Concatenating everything for the first layer
        actions_history = actions_history.reshape(-1, self.tree_height - 1)
        probas_history = probas_history.reshape(-1, self.tree_height - 1)
        next_proba = next_proba.reshape(-1, 1)
        inp = torch.cat((actions_history, probas_history, next_proba), 1)

        # Passing state through the network
        h = F.relu(self.input(inp))
        out = torch.sigmoid(self.output(h))
        return out
