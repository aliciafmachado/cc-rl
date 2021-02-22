from abc import ABC
import torch
import warnings


class Agent(ABC):
    """
    Abstract class for the reinforcement learning agent
    """
    def __init__(self, environment):
        self.environment = environment
        warnings.filterwarnings('ignore', category=UserWarning)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        warnings.filterwarnings('default', category=UserWarning)

    def train(self, *args):
        raise NotImplementedError

    def __experience_environment(self, *args):
        raise NotImplementedError

    def __train_once(self, *args):
        raise NotImplementedError

    def predict(self, *args):
        """
        Returns the predictions for the multilabel classification problem based on the
        environment.
        """
        raise NotImplementedError
