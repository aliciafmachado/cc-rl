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

    def train(self):
        raise NotImplementedError

    def train_once(self):
        raise NotImplementedError

    def experience_environment(self):
        raise NotImplementedError

    def predict(self):
        """
        Returns the predictions for the multilabel classification problem
        based on the environment
        """
        raise NotImplementedError