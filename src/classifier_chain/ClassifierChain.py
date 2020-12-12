import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.multioutput import ClassifierChain as skClassifierChain
from sklearn.utils import check_random_state

from .classical_inference.BeamSearchInferer import BeamSearchInferer
from .classical_inference.EpsilonApproximationInferer import EpsilonApproximationInferer
from .classical_inference.ExhaustiveSearchInferer import ExhaustiveSearchInferer
from .classical_inference.MonteCarloInferer import MonteCarloInferer


class ClassifierChain:
    """Base classifier chain to be used to compare different inference methods.
    """

    def __init__(self, base_estimator='logistic_regression', order='random',
                 random_state=0):
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
            base_estimator = LogisticRegression()

        self.cc = skClassifierChain(
            base_estimator=base_estimator, order=order, random_state=random_state)

    def fit(self, ds, optimization=True):
        """Fits the base estimators.

        Args:
            ds (Dataset): Dataset to fit the chain in.
            optimization (bool): This activates a precalibration step in the base 
                estimators to identify their best parameters. Only implemented if 
                base_estimator = 'logistic_regression'.
        """

        if optimization:
            self.__optimized_fit(ds)
        else:
            self.cc.fit(ds.train_x, ds.train_y)

    def predict(self, ds, inference_method, return_num_nodes=False, **kwargs):
        """Predicts the test's labels using a chosen inference method.

        Args:
            ds (Dataset): Dataset to get the test data from.
            inference_method (str): Inference method to be used in the prediction. One of 
                ['greedy', 'exhaustive_search', 'epsilon_approximation].
            return_num_nodes (bool, optional): If it should return the number of visited 
                tree nodes during the inference process. Defaults to False.

        Returns:
            np.array: Predicted output of shape (n, d2).
            int (optional): If return_num_nodes, it is the average number of visited nodes
                in the tree search.
        """

        if inference_method == 'greedy':
            # Greedy inference. O(d). Checkout implementation at
            # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/multioutput.py
            pred, num_nodes = self.cc.predict(
                ds.test_x), len(self.cc.estimators_)
        else:
            if inference_method == 'exhaustive_search':
                # Exhaustive search inference. O(2^d)
                inferer = ExhaustiveSearchInferer(self.cc)
            elif inference_method == 'epsilon_approximation':
                # Epsilon approximation inference. O(d / epsilon)
                inferer = EpsilonApproximationInferer(
                    self.cc, kwargs['epsilon'])
            elif inference_method == 'beam_search':
                # Beam search inference. O(d * b)
                inferer = BeamSearchInferer(self.cc, kwargs['b'])
            elif inference_method == 'monte_carlo':
                # Monte Carlo sampling inferer. O(d * q)
                inferer = MonteCarloInferer(self.cc, kwargs['q'], False)
            elif inference_method == 'efficient_monte_carlo':
                # Efficient Monte Carlo sampling inferer. O(d * q)
                inferer = MonteCarloInferer(self.cc, kwargs['q'], True)
            else:
                raise Exception('This inference method does not exist.')

            pred, num_nodes = inferer.infer(ds.test_x)

        if return_num_nodes:
            return pred, num_nodes
        else:
            return pred

    def __optimized_fit(self, ds):
        """Calibrates the base estimators parameters and fits them. 

        If base_estimator = 'logistic_regression', it will find the best regularization 
        parameter C for each individual binary regressor by perform a grid search over the
        values [0.001, 0.01, 0.1, 1, 10, 100, 1000] optimizing brier loss. Same strategy 
        used in MENA et al.

        The fit method from sklearn needed to be rewritten because in it the estimators_ 
        variable is reinitialized every time, so putting specific parameters for each base
        estimator isn't possible.

        Args:
            ds (Dataset): Dataset to get the train and test data from.
        """

        n_estimators = ds.train_y.shape[1]
        best_estimators = [None for _ in range(n_estimators)]
        best_score = np.full((n_estimators,), np.inf)

        self.__initialize_order(n_estimators)
        x_aug = np.hstack((ds.train_x, ds.train_y[:, self.cc.order_]))

        if self.__base_estimator == 'logistic_regression':
            for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
                self.cc.estimators_ = [LogisticRegression(
                    C=C) for _ in range(n_estimators)]

                # Fitting them manually to avoid resetting estimators
                for chain_idx, estimator in enumerate(self.cc.estimators_):
                    y = ds.train_y[:, self.cc.order_[chain_idx]]
                    estimator.fit(
                        x_aug[:, :(ds.train_x.shape[1] + chain_idx)], y)

                pred = self.cc.predict(ds.test_x)
                score = np.array([brier_score_loss(ds.test_y[:, i], pred[:, i])
                                  for i in range(n_estimators)])
                score = score[self.cc.order_]

                change = score < best_score
                best_score[change] = score[change]
                for i in range(len(change)):
                    if change[i]:
                        best_estimators[i] = self.cc.estimators_[i]

            self.cc.estimators_ = best_estimators
        else:
            self.cc.fit(ds.train_x, ds.train_y)

    def __initialize_order(self, n_estimators):
        """Initializes order_ variable.

        Copied from 
        https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/multioutput.py.

        Args:
            n_estimators (int): Number of classes in the output.
        """

        self.cc.random_state = check_random_state(self.cc.random_state)
        self.cc.order_ = self.cc.order
        if isinstance(self.cc.order_, tuple):
            self.order_ = np.array(self.order_)

        if self.cc.order_ is None:
            self.cc.order_ = np.array(range(n_estimators))
        elif isinstance(self.cc.order_, str):
            if self.cc.order_ == 'random':
                self.cc.order_ = self.cc.random_state.permutation(n_estimators)
        elif sorted(self.order_) != list(range(n_estimators)):
            raise ValueError("invalid order")
