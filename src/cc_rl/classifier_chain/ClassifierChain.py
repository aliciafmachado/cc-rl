import numpy as np
import os
import pickle
from sklearn.multioutput import ClassifierChain as skClassifierChain
import warnings

from cc_rl.data.Dataset import Dataset
from cc_rl.utils.LogisticRegressionExtended import LogisticRegressionExtended
from .classical_inference.BeamSearchInferer import BeamSearchInferer
from .classical_inference.EpsilonApproximationInferer import EpsilonApproximationInferer
from .classical_inference.ExhaustiveSearchInferer import ExhaustiveSearchInferer
from .classical_inference.MonteCarloInferer import MonteCarloInferer
from .classical_inference.RandomInferer import RandomInferer
from .RLInferer import RLInferer


class ClassifierChain:
    """Base classifier chain to be used to compare different inference methods.
    """

    def __init__(self, base_estimator: str = 'logistic_regression', order: str = 'random',
                 random_state: int = 0):
        """Default constructor.

        Args:
            base_estimator (str or sklearn.base.BaseEstimator, optional): Base estimator
                for each node of the chain. Defaults to 'logistic_regression'.
            order (str or list, optional): Labels classification order. Defaults to
                'random'.
            random_state (int, optional): Defaults to 0.
        """

        self.__base_estimator = base_estimator
        if base_estimator == 'logistic_regression':
            base_estimator = LogisticRegressionExtended()

        self.cc = skClassifierChain(
            base_estimator=base_estimator, order=order, random_state=random_state)
        self.n_labels = None

    def fit(self, ds: Dataset, from_scratch: bool = False):
        """Fits the base estimators.

        Args:
            ds (Dataset): Dataset to fit the chain in.
            from_scratch (bool, optional): If False, the model will load a pretrained
                chain.
        """

        self.n_labels = ds.train_y.shape[1]
        if not from_scratch:
            path = Dataset.data_path + 'trainer/cc_' + ds.name + '.pkl'
            if os.path.isfile(path):
                file = open(path, 'rb')
                self.cc = pickle.load(file)
                return

        warnings.filterwarnings('ignore')
        self.cc.fit(ds.train_x, ds.train_y)
        warnings.filterwarnings('default')

    def predict(self, ds: Dataset, inference_method: str, return_num_nodes: bool = False,
                return_reward: bool = False, loss='exact_match', **kwargs):
        """Predicts the test's labels using a chosen inference method.

        Args:
            ds (Dataset): Dataset to get the test data from.
            inference_method (str): Inference method to be used in the prediction. One of 
                ['greedy', 'exhaustive_search', 'epsilon_approximation].
            return_num_nodes (bool, optional): If it should return the number of visited 
                tree nodes during the inference process. Defaults to False.
            return_reward (bool, optional): If it should return the final reward of that
                prediction, calculated using the estimators probabilities.
            loss (str, optional): 'exact_match' or 'hamming'.

        Returns:
            np.array: Predicted output of shape (n, d2).
            int (optional): If return_num_nodes, it is the average number of visited nodes
                in the tree search.
        """

        if inference_method == 'greedy':
            # Greedy inference. O(d). Checkout implementation at
            # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/multioutput.py
            pred, num_nodes = self.cc.predict(ds.test_x), len(self.cc.estimators_)

            # FIXME: put this in another class
            # Calculate reward
            if return_reward:
                if loss == 'exact_match':
                    reward = np.ones((len(ds.test_x),), dtype=float)
                else:
                    reward = np.zeros((len(ds.test_x),), dtype=float)
                for i in range(len(self.cc.estimators_)):
                    x_aug = np.hstack((ds.test_x, pred[:, :i]))
                    proba = self.cc.estimators_[i].predict_proba(x_aug)
                    new_proba = np.take_along_axis(
                        proba, pred[:, i].astype(int).reshape(-1, 1), axis=1).flatten()
                    if loss == 'exact_match':
                        reward *= new_proba
                    else:
                        reward += new_proba
                reward = reward.mean()
            else:
                reward = None
        else:
            if inference_method == 'random':
                # Completely random inference.
                inferer = RandomInferer(self.cc, loss, kwargs['n'])
            elif inference_method == 'exhaustive_search':
                # Exhaustive search inference. O(2^d)
                inferer = ExhaustiveSearchInferer(self.cc, loss)
            elif inference_method == 'epsilon_approximation':
                # Epsilon approximation inference. O(d / epsilon)
                inferer = EpsilonApproximationInferer(
                    self.cc, loss, kwargs['epsilon'])
            elif inference_method == 'beam_search':
                # Beam search inference. O(d * b)
                inferer = BeamSearchInferer(
                    self.cc, loss, kwargs['b'])
            elif inference_method == 'monte_carlo':
                # Monte Carlo sampling inferer. O(d * q)
                inferer = MonteCarloInferer(
                    self.cc, loss, kwargs['q'], False)
            elif inference_method == 'efficient_monte_carlo':
                # Efficient Monte Carlo sampling inferer. O(d * q)
                inferer = MonteCarloInferer(
                    self.cc, loss, kwargs['q'], True)
            elif inference_method == 'qlearning' or inference_method == 'mcts':
                batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else None
                learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs else None
                inferer = RLInferer(self, loss,
                                    agent_type=inference_method, nb_sim=kwargs['nb_sim'],
                                    nb_paths=kwargs['nb_paths'], epochs=kwargs['epochs'],
                                    batch_size=batch_size, learning_rate=learning_rate)
            else:
                raise Exception('This inference method does not exist.')

            pred, num_nodes, reward = inferer.infer(ds.test_x)

        returns = [pred]
        if return_num_nodes:
            returns.append(num_nodes)
        if return_reward:
            returns.append(reward)
        return returns
