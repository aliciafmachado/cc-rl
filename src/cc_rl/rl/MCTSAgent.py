from cc_rl.rl.Agent import Agent
from cc_rl.rl.MCTSModel import MCTSModel
import numpy as np
import queue
import torch
from torch import Tensor

from cc_rl.gym_cc.Env import Env
from cc_rl.rl.Agent import Agent
from typing import List, Dict, Callable
from nptyping import NDArray


class MCTSAgent(Agent):
    """
    Reinforcement learning agent that uses a policy and a value network alongside
    MCTS to find the best tree path
    """

    def __init__(self, environment: Env,
                 c_puct: float = 0.6,
                 mcts_passes: int = 10):
        super().__init__(environment)
        self.model = MCTSModel(environment.classifier_chain.n_labels + 1, self.device)
        self.data_loader = None
        # self.dataset = None
        # self.best_path = None
        # self.best_path_reward = 0
        self.c_puct = c_puct
        self.mcts_passes = mcts_passes
        self.__has_trained = False

    def train(self, nb_sim: int, nb_paths: int, epochs: int, batch_size: int = 64,
              learning_rate: float = 1e-3, verbose: bool = False):
        """
        Trains model from the environment given in the constructor, going through the tree
        nb_sim * nb_paths times.
        :param: nb_sim: Number of training loops that will be executed.
        :param: nb_paths: Number of paths explored in each step.
        :param: epochs: Number of epochs in each training step.
        :param: batch_size: Used in training.
        :param: learning_rate: Used in training.
        :param: verbose: Will print train execution if True.
        """

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_mse = torch.nn.MSELoss()
        loss_bce = torch.nn.BCELoss()

        for sim in range(nb_sim):
            self.__experience_environment(nb_paths, batch_size)
            self.__train_once(epochs, optimizer, loss_mse, loss_bce, verbose)

    def predict(self, return_num_nodes: bool = False, return_reward: bool = False,
                mode: str = 'best_visited'):
        """
        Predicts the best path after the training step is done.
        @param return_num_nodes: If true, will also return the total number of predictions
            ran by estimators in the classifier chain in total by the agent.
        @param return_reward: If true, will also return the reward got in this path.
        @param mode: If 'best_visited', will get the path with the best reward found
            during training. If 'final_decision', will go through the tree one last time
            to find the path.
        @return: (np.array) Prediction outputs of shape (n, d2).
                 (int, optional): The average number of visited nodes in the tree search.
        """

        if mode == 'best_visited':
            # path = self.best_path
            # reward = self.best_path_reward
            pass
        elif mode == 'final_decision':
            actions_history = []
            final_values = []
            last_action = self.__experience_environment_once(actions_history, [], [],
                                                             final_values)
            path = actions_history[-1]
            path[-1] = last_action
            reward = final_values[-1]
        else:
            raise ValueError

        path = (path + 1).astype(bool)
        returns = [path]
        if return_reward:
            returns.append(reward)
        # if return_num_nodes:
        #     returns.append(self.n_visited_nodes)
        return tuple(returns)

    def __experience_environment(self, nb_paths: int, batch_size: int):
        """
        In this method the model is used to predict the best path for a total of
        nb_paths paths. For each decision the model takes, the state is recorded.
        The result is then stored in the variable self.data_loader
        :param: nb_paths: Number of paths that must be experiences from top to bottom, i.e.
            number of resets on the environment.
        :param: batch_size: To be used in training.
        """

        # Resetting history
        actions_history = []
        probas_history = []
        improved_policy_history = []
        final_values = []

        for i in range(nb_paths):
            self.__experience_environment_once(actions_history, probas_history,
                                               improved_policy_history, final_values)

        # Updating data loader to train the network
        actions_history = torch.tensor(actions_history).float()
        probas_history = torch.tensor(probas_history).float()
        improved_policy_history = torch.tensor(improved_policy_history).float()
        final_values = torch.tensor(final_values).float()

        # if self.dataset == None:
        #   self.dataset = torch.utils.data.TensorDataset(actions_history, probas_history,
        #                                          final_values)
        # else:
        #   new_data = torch.utils.data.TensorDataset(actions_history, probas_history,
        #                                          final_values)
        #   self.dataset = torch.utils.data.ConcatDataset([self.dataset, new_data])

        dataset = torch.utils.data.TensorDataset(actions_history, probas_history,
                                                 improved_policy_history, final_values)
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                       shuffle=True)

    def __train_once(self, epochs: int, optimizer: torch.optim.Optimizer,
                     loss_mse: Callable[[Tensor, Tensor], Tensor],
                     loss_bce: Callable[[Tensor, Tensor], Tensor], verbose: bool):
        """
        Fits the model with the data that is currently in self.data_loader.
        :param epochs: Used in training.
        :param learning_rate: Used in training.
        :param verbose: Will print train execution if True.
        """
        # Start training
        self.model.train()

        for epoch in range(epochs):
            for i, data in enumerate(self.data_loader):
                # Pass the data to teh GPU if possible
                actions_history, probas_history, improved_policy_history, final_values = \
                    [d.to(self.device) for d in data]

                optimizer.zero_grad()

                # Calculate value and policies for each test case
                out = self.model(actions_history, probas_history)
                predicted_values = out[:, 0]
                predicted_policies = out[:, 1]

                # Apply loss functions
                loss = loss_mse(predicted_values, final_values) + \
                       loss_bce(predicted_policies, improved_policy_history)

                # Brackprop and optimize
                loss.backward()
                optimizer.step()

                if verbose:
                    print('Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                        epoch + 1, epochs, i + 1, len(self.data_loader),
                        loss.item() / self.data_loader.batch_size))

    def __experience_environment_once(self, actions_history: List[NDArray],
                                      probas_history: List[NDArray],
                                      improved_policy_history: List[float],
                                      final_values: List[float]):
        """
        Goes through a path until the end once using MCTS at each step
        :param actions_history: list of action_history vectors used for training
        :param probas_history: list of proba_history vectors used for training
        :param next_probas: list of next_probas used for training
        :param improved_policy_history: list of improved_policies used for training
        :param final_values: list of final values used as oneof the objectives for trainng
        :returns:
        """
        # Each path should go until the end
        depth = self.environment.classifier_chain.n_labels

        # Parameters used for the MCTS
        N = {}  # dictionary that stores the number of times we took action a from state s
        Q = {}  # action-value function
        P = {}  # policy
        final_value = None

        for i in range(depth):
            # Calculating the next policy and getting the next action
            action_history, proba_history, improved_policy = \
                self.__calculate_improved_policy(i, N, Q, P)
            action = np.random.choice([-1, 1], p=[improved_policy, 1 - improved_policy])

            # Saving history
            actions_history += [action_history]
            probas_history += [proba_history]
            improved_policy_history += [improved_policy]

            _, _, _, value, end = self.environment.step(action)
            if end:
                final_value = value

        for i in range(depth):
            final_values += [final_value]

        return action

    def __calculate_improved_policy(self, initial_depth: int,
                                    N: Dict[tuple, int],
                                    Q: Dict[tuple, int],
                                    P: Dict[tuple, int]):
        """
        Calculates the improved policy for a given initial depth
        :param initial_depth:
        :param N: dict to count a given action-value
        :param Q: dict to define the action-value function
        :param P: dict to define the policy for a given state
        :return: variables defining the current state and the improved policy
        """

        # Execute MCTS passes to calculate improved policy
        for i in range(self.mcts_passes):
            next_proba, action_history, proba_history = self.environment.reset(
                initial_depth)

            self.__MCTS(next_proba[0], action_history, proba_history, 0, False, N, Q, P,
                        initial_depth)

        # Calculate improved policy
        next_proba, action_history, proba_history = self.environment.reset(initial_depth)
        proba_history[initial_depth] = next_proba[0]
        state = tuple(action_history)
        improved_policy = N[(state, -1)] / (N[(state, -1)] + N[(state, 1)])

        return action_history, proba_history, improved_policy

    def __MCTS(self, next_proba: float,
               action_history: NDArray,
               proba_history: NDArray,
               final_value: float,
               end: bool,
               N: Dict[tuple, int],
               Q: Dict[tuple, int],
               P: Dict[tuple, int],
               cur_depth: int):
        """
        Recursive function which updates parametres in N, Q and P via MCTS starting from
        a state defined by next_proba, action_history and proba_history
        :param next_proba: state parameter - probability for the next choice
        :param action_history: state parameter - action history to get to the current position
        :param proba_history: state parameter - probability history to get to the current position
        :param final_value: value if we reached the end of the tree
        :param end: bool to identify if we reached the end of the tree
        :param N: dict to count a given action-value
        :param Q: dict to define the action-value function
        :param P: dict to define the policy for a given state
        :return: the value of a given action
        """

        # The search reached the end
        if end:
            return final_value

        # Getting the state based on the acton history
        state = tuple(action_history)

        # If we reach a new node we should get the value and policy from the network
        if state not in P:
            # Use the model to predict the value and the policy
            proba_history[cur_depth] = next_proba
            self.model.eval()
            pred = self.model(torch.tensor(action_history).float(),
                              torch.tensor(proba_history).float())
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
                U = Q[state_action] + self.c_puct * P[state] * np.sqrt(tot_N) / (
                            1 + N[state_action])
            elif action == 1:
                U = Q[state_action] + self.c_puct * (1 - P[state]) * np.sqrt(tot_N) / (
                            1 + N[state_action])

            # Choose action that maximizes U
            if U > max_U:
                best_A = action
                max_U = U

        # Get the next state from the environment
        next_proba, action_history, proba_history, final_value, end = \
            self.environment.step(best_A)

        # Propagate down the search to get the value from the state below
        value = self.__MCTS(next_proba[0], action_history, proba_history, final_value, end,
                            N, Q, P, cur_depth + 1)

        # Updating Q and N
        state_action = (state, best_A)
        Q[state_action] = (N[state_action] * Q[state_action] + value) / (
                    N[state_action] + 1)
        N[state_action] += 1

        return value
