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

    def __init__(self, environment: Env,
                 c_puct: float = 0.6,
                 mcts_passes: int = 100):
        super().__init__(environment)
        self.model = MCTSModel(environment.classifier_chain.n_labels + 1, self.device)
        self.c_puct = c_puct
        self.mcts_passes = mcts_passes

    def train(self, *args):
        raise NotImplementedError

    def __experience_environment(self, *args):
        raise NotImplementedError

    def __experience_environment_once(self):
        """
        Goes through a path until the end once using MCTS at each step
        :returns:
        """
        # Each path should go until the end
        depth = self.environment.classifier_chain.n_labels

        # Get current state from environment
        next_proba, action_history, proba_history = self.environment.reset()

        # Parameters used for the MCTS
        N = {} # dictionary that stores the number of times we took action a from state s
        Q = {} # action-value function
        P = {}






        # TODO must return the history of actions probas, next probas adn final values and policies
        return None

    def __MCTS_pass(self, initial_depth, total_depth, N, Q):
        for i in range(self.mcts_passes):
            next_proba, action_history, proba_history = self.environment.reset(initial_depth)

            for j in range(total_depth-initial_depth):
                # Getting prediction from the


                # Getting the next state from environment
                next_proba, action_history, proba_history, final_value, end = \
                    self.environment.step(next_action)

    def __train_once(self, *args):
        raise NotImplementedError
