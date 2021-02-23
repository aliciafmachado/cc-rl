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

    def experience_environment_once(self):
        """
        Goes through a path until the end once using MCTS at each step
        :returns:
        """
        # Each path should go until the end
        depth = self.environment.classifier_chain.n_labels

        # Parameters used for the MCTS
        N = {} # dictionary that stores the number of times we took action a from state s
        Q = {} # action-value function
        P = {}


        print(self.__calculate_improved_policy(0, N, Q, P))
        print(self.__calculate_improved_policy(1, N, Q, P))

        # TODO must return the history of actions probas, next probas adn final values and policies
        return None

    def __calculate_improved_policy(self, initial_depth, N, Q, P):
        # TODO add comment

        # Execute MCTS passes to calculate improved policy
        for i in range(self.mcts_passes):
            next_proba, action_history, proba_history = self.environment.reset(initial_depth)

            self.__MCTS(next_proba, action_history, proba_history, 0, False, N, Q, P)

        # Calculate improved policy
        next_proba, action_history, proba_history = self.environment.reset(initial_depth)
        state = tuple(action_history)
        improved_policy = N[(state, 1)] / (N[(state, -1)] + N[(state, 1)])

        return next_proba, action_history, proba_history, improved_policy

    def __MCTS(self, next_proba, action_history, proba_history, final_value, end, N, Q, P):
        # TODO add initial comment

        # The search reached the end
        if end:
            return final_value

        # Getting the state based on the acton history
        state = tuple(action_history)

        # If we reach a new node we should get the value and policy from the network
        if state not in P:
            # Use the model to predic the value and the policy
            pred = self.model(torch.tensor(action_history).float(),
                              torch.tensor(proba_history).float(),
                              torch.tensor(next_proba[1]).float())
            value, policy = pred[0][0].item(), pred[0][1].item()

            # Initialize N, Q and P
            P[state] = policy

            for action in self.environment.action_space:
                state_action = (state, action)
                Q[state_action] = 0
                N[state_action] = 0

            return value

        # If the node is visited we use the criterion from the MCTS to
        # choose the next action
        max_U, best_A = -float("inf"), -1
        tot_N = 0
        for action in self.environment.action_space:
            tot_N += N[(state, action)]

        for action in self.environment.action_space:
            state_action = (state, action)

            # P[state] represents the probability of action -1 from a given state
            if action == -1:
                U = Q[state_action] + self.c_puct * P[state] * np.sqrt(tot_N)/(1+N[state_action])
            elif action == 1:
                U = Q[state_action] + self.c_puct * (1-P[state]) * np.sqrt(tot_N)/(1+N[state_action])

            # Choose action that maximizes U
            if U > max_U:
                best_A = action
                max_U = U

        # Get the next state from the environment
        next_proba, action_history, proba_history, final_value, end = \
            self.environment.step(best_A)

        # Propagate down the search to get the value from the state below
        value = self.__MCTS(next_proba, action_history, proba_history, final_value, end, N, Q, P)

        # Updating Q and N
        state_action = (state, best_A)
        Q[state_action] = (N[state_action] * Q[state_action] + value)/(N[state_action] + 1)
        N[state_action] += 1

        return value

    def __train_once(self, *args):
        raise NotImplementedError
