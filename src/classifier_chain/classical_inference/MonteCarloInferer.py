import numpy as np

from src.classifier_chain.BaseInferer import BaseInferer


class MonteCarloInferer(BaseInferer):
    """Inferer that sample trees paths randomly, following a distribution function based
    on each estimator. Complexity O(d * q)
    """

    def __init__(self, classifier_chain, loss, q, efficient):
        """Default constructor.

        Args:
            classifier_chain (sklearn.multioutput.ClassifierChain): Classifier chain that 
                this inference will be used on.
            q (int): Number of samples to be tried out in the inference.
            efficient (bool): If this is the efficient inference or not.
        """

        super().__init__(classifier_chain, loss)
        assert(q >= 1)
        assert(isinstance(q, int))
        assert(isinstance(efficient, bool))
        self.q = q
        self.efficient = efficient

    def _infer(self, x):
        """Searches through the tree q times. This is vectorized, so it is done at the
        same time for every row of x. 

        Args:
            x (np.array): Prediction data of shape (n, d1).

        Returns:
            np.array: Prediction outputs of shape (n, d2).
            int: The average number of visited nodes in the tree search.
        """

        # Initialization
        n_nodes = 0
        best_pred = np.empty((len(x), len(self.cc.estimators_)), dtype=bool)
        if self.efficient:
            best_p = np.zeros((len(x),), dtype=float)
        else:
            preds = np.empty((len(x), self.q, len(
                self.cc.estimators_)), dtype=bool)

        # Search one path for each q
        for qi in range(self.q):
            cur_pred = np.empty((len(x), len(self.cc.estimators_)), dtype=bool)
            cur_p = np.ones((len(x),), dtype=float)

            for i in range(len(self.cc.estimators_)):
                x_aug = np.hstack((x, cur_pred[:, :i]))
                proba = self.cc.estimators_[i].predict_proba(x_aug)
                n_nodes += 1

                w = np.random.random(len(x))
                choice = w > proba[:, 0]
                cur_pred[:, i] = choice
                choice = choice.astype(int).reshape(-1, 1)
                cur_p = self._new_score(cur_p, np.take_along_axis(
                    proba, choice, axis=1).flatten())

            # Efficient method takes the one with highest probability, the non-efficient
            # takes the mode
            if self.efficient:
                mask = cur_p > best_p
                best_p[mask] = cur_p[mask]
                best_pred[mask] = cur_pred[mask]
            else:
                preds[:, qi] = cur_pred

        # Take the mode of the samples
        if not self.efficient:
            for i in range(len(x)):
                preds[i, 0, :] = preds[i, 1, :]
                unq = np.unique(preds[i], axis=0, return_counts=True)
                best_pred[i] = unq[0][np.argmax(unq[1])]

        return best_pred, n_nodes
