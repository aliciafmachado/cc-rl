import numpy as np

from cc_rl.classifier_chain.BaseInferer import BaseInferer


class BeamSearchInferer(BaseInferer):
    """Inferer that expands the tree in only selected nodes, limited by at most b at each
    level of the tree. Complexity O(d * b).

    This inferer is equivalent to greedy if b = 1 and equivalent to exhaustive search if
    b = 2^d.
    """

    def __init__(self, classifier_chain, loss, b):
        """Default constructor.

        Args:
            classifier_chain (sklearn.multioutput.ClassifierChain): Classifier chain that 
                this inference will be used on.
            b (int): b parameter for the inferer.
        """

        super().__init__(classifier_chain.order_, loss)
        self.cc = classifier_chain
        assert(b >= 1)
        assert(isinstance(b, int))
        self.b = b

    def _infer(self, x):
        """Searches through the tree in a bfs. This is vectorized, so it is done at the
        same time for every row of x. Because of this, instead of using many priority 
        queues, just a vector is used.

        Args:
            x (np.array): Prediction data of shape (n, d1)

        Returns:
            np.array: Prediction outputs of shape (n, d2).
            int: The average number of visited nodes in the tree search.
        """

        # New beam of size 2*b to put both paths in this vector, then in the end take only
        # the b with highest probability
        beam = np.zeros((len(x), self.b, len(self.cc.estimators_)), dtype=bool)
        beam_p = np.zeros((len(x), self.b), dtype=float)
        new_beam = np.zeros(
            (len(x), 2 * self.b, len(self.cc.estimators_)), dtype=bool)
        new_beam_p = np.zeros((len(x), 2 * self.b), dtype=float)
        n_nodes = 0

        # Initialize algorithm with first estimator
        proba = self.cc.estimators_[0].predict_proba(x)
        n_nodes += 1
        if self.b == 1:
            beam_p[:, 0] = np.max(proba, axis=1)
            beam[:, 0, 0] = np.argmax(proba, axis=1)
            cur_beam_size = 1
        else:
            beam_p[:, :2] = proba
            beam[:, 0, 0] = False
            beam[:, 1, 0] = True
            cur_beam_size = 2

        # Loop through estimators
        for i in range(1, len(self.cc.estimators_)):
            for j in range(cur_beam_size):
                x_aug = np.hstack((x, beam[:, j, :i].reshape(-1, i)))
                proba = self.cc.estimators_[i].predict_proba(x_aug)
                n_nodes += 1

                new_beam_p[:, 2*j:2*j+2] = self._new_score(beam_p[:, j].reshape(-1, 1), proba)
                new_beam[:, 2*j:2*j+2, :i] = beam[:, j, :i].reshape(-1, 1, i)
                new_beam[:, 2*j, i] = False
                new_beam[:, 2*j+1, i] = True

            cur_beam_size = min(2 * cur_beam_size, self.b)
            idx = np.argpartition(
                new_beam_p[:, :2*cur_beam_size], cur_beam_size, axis=1)
            idx = idx[:, cur_beam_size:]
            beam_p = np.take_along_axis(new_beam_p, idx, axis=1)
            beam = np.take_along_axis(
                new_beam, idx.reshape(-1, cur_beam_size, 1), axis=1)

        # Make prediction
        idx = np.argmax(beam_p, axis=1).reshape(-1, 1, 1)
        best_pred = np.take_along_axis(beam, idx, axis=1)
        best_pred = best_pred.reshape(-1, len(self.cc.estimators_))

        return best_pred, n_nodes
