import numpy as np


class BaseInferer:
    """Base abstract class for infererence algorithms.
    """

    def __init__(self, classifier_chain, loss='exact_match'):
        """Default constructor.

        Args:
            classifier_chain (sklearn.multioutput.ClassifierChain): Classifier chain that 
                this inference will be used on.
            loss (str): 'exact_match' or 'hamming', specifying which loss this prediction
                should minimize.
        """

        self.cc = classifier_chain
        assert(loss == 'exact_match' or loss == 'hamming')
        self.loss = loss

    def infer(self, x):
        """Infers a prediction according to the child inferer algorithm.

        Args:
            x (np.array): Prediction data of shape (n, d1).

        Returns:
            np.array: Prediction outputs of shape (n, d2).
            int: The average number of visited nodes in the tree search.
        """

        pred, n_nodes = self._infer(x)
        pred = self.__fix_order(pred)
        return pred, n_nodes

    def _infer(self, x):
        """Virtual method to do inference.

        Args:
            x (np.array): Prediction data of shape (n, d1).

        Raises:
            NotImplementedError: This method is virtual.
        """        

        raise NotImplementedError

    def _new_score(self, past_score, new_proba):
        """Updates the current score in the tree path. This depends on the loss function
        being used: if 'exact_match', this score is the conditional probability and if
        'hamming', it is the sum of probabilities.

        Args:
            past_score (np.array): Scores until this estimator, shape (n,)
            new_proba (np.array): Probabilities on the new estimator prediction, shape
                (n,)

        Returns:
            np.array: Score of this new prediction, shape (n,)
        """

        if self.loss == 'exact_match':
            return past_score * new_proba
        else:
            return past_score + new_proba

    def __fix_order(self, pred):
        """Estimators in classifier chain are not necessarily in the label order. This
        method reorders the prediction to the label order.

        Args:
            pred (np.array): Prediction in the estimators order of shape (n,).

        Returns:
            np.array: Prediction in the correct order of shape (n,).
        """

        inv_order = np.empty_like(self.cc.order_)
        inv_order[self.cc.order_] = np.arange(len(self.cc.order_))
        return pred[:, inv_order]
