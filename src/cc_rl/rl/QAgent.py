import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from typing import List

from cc_rl.gym_cc.Env import Env
from cc_rl.rl.Agent import Agent
from cc_rl.rl.QModel import QModel


class QAgent(Agent):
    """
    Reinforcement learning agent that uses only Q-learning to find the best tree path.
    """

    def __init__(self, environment: Env):
        super().__init__(environment)
        self.model = QModel(environment.classifier_chain.n_labels + 1, self.device)
        self.data_loader = None
        self.best_path = None
        self.best_path_reward = 0
        self.n_visited_nodes = 0
        self.__has_trained = False

    def train(self, nb_sim: int, nb_paths: int, epochs: int, batch_size: int = 64,
              learning_rate: float = 1e-3, verbose: bool = False):
        """
        Trains model from the environment given in the constructor, going through the tree
        nb_sim * nb_paths times.
        @param nb_sim: Number of training loops that will be executed.
        @param nb_paths: Number of paths explored in each step.
        @param epochs: Number of epochs in each training step.
        @param batch_size: Used in training.
        @param learning_rate: Used in training.
        @param verbose: Will print train execution if True.
        """
        # We do multiple simulations
        for sim in range(nb_sim):
            self.__experience_environment(nb_paths, batch_size)
            self.__train_once(epochs, learning_rate, verbose)

        self.__has_trained = True

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
        assert self.__has_trained

        if mode == 'best_visited':
            path = self.best_path
            reward = self.best_path_reward
        elif mode == 'final_decision':
            actions_history = []
            final_values = []
            self.__experience_environment_once(actions_history, [], [], [], final_values)
            path = actions_history[-1]
            reward = final_values[-1]
        else:
            raise ValueError

        path = (path + 1).astype(bool)
        returns = [path]
        if return_reward:
            returns.append(reward)
        if return_num_nodes:
            returns.append(self.n_visited_nodes)
        return tuple(returns)

    def __experience_environment(self, nb_paths: int, batch_size: int, exploring_p=0.5):
        """
        In this method the model is used to predict the best path for a total of
        nb_paths paths. For each decision the model takes, the state is recorded.
        The result is then stored in the variable self.data_loader
        @param nb_paths: Number of paths that must be experiences from top to bottom, i.e.
            number of resets on the environment.
        @param batch_size: To be used in training.
        @param exploring_p: Probability that, when exploring, the path will be chosen
            randomly instead of predicted by the model.
        """

        # Resetting history
        actions_history = []
        probas_history = []
        next_probas = []
        next_actions = []
        final_values = []

        for i in range(nb_paths):
            self.__experience_environment_once(actions_history, probas_history,
                                               next_actions, next_probas, final_values,
                                               exploring_p=exploring_p)

        # Updating data loader to train the network
        actions_history = torch.tensor(actions_history).float()
        probas_history = torch.tensor(probas_history).float()
        next_actions = torch.tensor(next_actions).float()
        next_probas = torch.tensor(next_probas).float()
        final_values = torch.tensor(final_values).float()

        dataset = torch.utils.data.TensorDataset(actions_history, probas_history,
                                                 next_actions, next_probas, final_values)
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                       shuffle=True)
        # collate_fn=lambda x: default_collate(x).to(self.device))

    def __train_once(self, epochs: int, learning_rate: float, verbose: bool):
        """
        Fits the model with the data that is currently in self.data_loader.
        @param epochs: Used in training.
        @param learning_rate: Used in training.
        @param verbose: Will print train execution if True.
        """
        # Start training
        self.model.train()
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            for i, data in enumerate(self.data_loader):
                actions_history, probas_history, next_probas, next_actions, \
                final_values = [d.to(self.device) for d in data]

                # Calculate Q value for each test case
                predict = self.model(actions_history, probas_history, next_probas,
                                     next_actions).reshape(-1)

                # Apply loss function
                loss = loss_fn(predict, final_values)

                # Brackprop and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose:
                    print('Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                        epoch + 1, epochs, i + 1, len(self.data_loader),
                        loss.item() / self.data_loader.batch_size))

    def __experience_environment_once(self, actions_history: List[torch.Tensor],
                                      probas_history: List[torch.tensor],
                                      next_actions: List[torch.Tensor],
                                      next_probas: List[torch.Tensor],
                                      final_values: List[float],
                                      exploring_p: float = 0.0):
        # Each path should go until the end
        depth = self.environment.classifier_chain.n_labels

        # Get current state from environment
        next_proba, action_history, proba_history = self.environment.reset()

        for j in range(depth):
            # Getting the path information as torch tensors
            action_history = torch.tensor(action_history).float()
            proba_history = torch.tensor(proba_history).float()
            next_proba = torch.tensor(next_proba[0]).float()

            r = np.random.rand()
            if r < exploring_p:
                # Add randomness to make agent explore more
                next_action = np.random.randint(0, 2) * 2 - 1
            else:
                # Choosing the next action using the agent
                next_action = self.model.choose_action(action_history, proba_history,
                                                       next_proba)
                # Converts actions from {0, 1} to {-1, 1}
                next_action = int(2 * next_action - 1)

            next_proba, action_history, proba_history, final_value, end = \
                self.environment.step(next_action)
            self.n_visited_nodes += 1

            # Adding past actions to the history
            actions_history += [action_history]
            probas_history += [proba_history]
            next_probas += [next_proba[0]]
            next_actions += [next_action]

        # Updating the history for the final values
        for j in range(depth):
            final_values += [final_value]

        # Store best path for prediction
        if final_value > self.best_path_reward:
            self.best_path_reward = final_value
            self.best_path = actions_history[-1]
