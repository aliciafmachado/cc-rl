from sklearn.multioutput import ClassifierChain as skClassifierChain
from sklearn.linear_model import LogisticRegression

from .ExhaustiveSearchInferer import ExhaustiveSearchInferer


class ClassifierChain:
    """Base classifier chain to be used to compare different inference methods.
    """

    def __init__(self, base_estimator=LogisticRegression(), order='random', random_state=0):
        """Default constructor.

        Args:
            base_estimator (sklearn.base.BaseEstimator, optional): Base estimator for each 
                node of the chain. Defaults to LogisticRegression().
            order (str or list, optional): Labels classification order. Defaults to
                'random'.
            random_state (int, optional): Defaults to 0.
        """
        self.cc = skClassifierChain(
            base_estimator=base_estimator, order=order, random_state=random_state)

    def fit(self, x, y):
        """Fits the base estimators.

        Args:
            x (np.array): Train data of shape (n x d1).
            y (np.array): Sparse train outputs of shape (n x d2).
        """

        self.cc.fit(x, y)

    def predict(self, x, inference_method, return_num_nodes=False, **kwargs):
        """Predicts the test's labels using a chosen inference method.

        Args:
            x (np.array): Test data of shape (n x d1).
            inference_method (str): Inference method to be used in the prediction. One of 
                ['greedy', 'exhaustive_search'].
            return_num_nodes (bool, optional): If it should return the number of visited 
                tree nodes during the inference process. Defaults to False.

        Returns:
            np.array: Predicted output of shape (n x d2).
        """

        if inference_method == 'greedy':
            # Greedy inference. O(d). Checkout implementation at
            # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/multioutput.py
            pred, num_nodes = self.cc.predict(x), len(self.cc.estimators_)
        elif inference_method == 'exhaustive_search':
            # Exhaustive search inference. O(2^d)
            inferer = ExhaustiveSearchInferer(self.cc)
            pred, num_nodes = inferer.infer(x)
        else:
            raise Exception('This inference method does not exist.')

        if return_num_nodes:
            return pred, num_nodes
        else:
            return pred
