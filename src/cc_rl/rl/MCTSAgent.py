from cc_rl.rl.Agent import Agent
from cc_rl.rl.MCTSModel import MCTSModel
import numpy as np
import torch

from cc_rl.gym_cc.Env import Env
from cc_rl.rl.Agent import Agent


class MCTSAgent(Agent):
    """
    Reinforcement learning agent that uses a policy and a value network alongside
    MCTS to find the best tree path
    """

    def __init__(self, environment: Env):
        super().__init__(environment)

    def train(self, *args):
        raise NotImplementedError

    def __experience_environment(self, *args):
        raise NotImplementedError

    def __train_once(self, *args):
        raise NotImplementedError
