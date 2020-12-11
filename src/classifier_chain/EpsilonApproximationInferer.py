import numpy as np


class EpsilonApproximationInferer:
    """Inferer that expands the tree in only selected nodes, where the joint probability
    is more than epsilon. Complexity O(d / epsilon).

    This inferer is equivalent to greedy if epsilon >= 0.5 and equialent to exhaustive 
    search if epsilon = 0.
    """

    def __init__(self, classifier_chain, epsilon):
        """Default constructor.

        Args:
            classifier_chain (sklearn.multioutput.ClassifierChain): Classifier chain that 
                this inference will be used on.
            epsilon (float): Epsilon parameter for the inferer.
        """

        assert(0 <= epsilon and epsilon <= 0.5)
        self.cc = classifier_chain
        self.epsilon = epsilon

    def infer(self, x):
        """Infers prediction by expanding the search in only selected nodes.

        Args:
            x (np.array): Prediction data of shape (n, d1).

        Returns:
            np.array: Prediction outputs of shape (n, d2).
            int: If return_num_nodes, it is the average number of visited nodes in the
                tree search.
        """

        cur_pred = np.zeros((len(x), len(self.cc.estimators_)), dtype=bool)
        cur_p = np.ones((len(x),))
        mask = np.ones((len(x),), dtype=bool)
        self.__best_pred = np.copy(cur_pred)
        self.__best_p = np.zeros((len(x),))

        self.__n_nodes = 0
        self.__dfs(x, cur_pred, cur_p, 0, mask)

        inv_order = np.empty_like(self.cc.order_)
        inv_order[self.cc.order_] = np.arange(len(self.cc.order_))
        self.__best_pred = self.__best_pred[:, inv_order]

        self.__n_nodes = self.__n_nodes / len(x)

        return self.__best_pred, self.__n_nodes

    def __dfs(self, x, cur_pred, cur_p, i, mask):
        """Searches through the tree in a dfs. This is vectorized, so it is done at the
        same time for every row of x.

        Args:
            x (np.array): Prediction data of shape (n, d1)
            cur_pred (np.array): Current prediction in the recursion of shape (n, d2)
            cur_p (np.array): Current accumulated probability in the recursion of shape 
                (n,)
            i (int): Index of the current estimator in the recursion.
            mask (np.array): Mask with the current rows that are being analyzed. Since in
                this algorithm there is a condition a node must pass to be visited, in
                some rows of the dataset the search may stop at some point while in others
                it will continue.
        """

        if i == len(self.cc.estimators_):
            change = np.copy(mask)
            change[mask] = cur_p > self.__best_p[mask]
            self.__best_pred[change] = cur_pred[change]
            self.__best_p[change] = cur_p[change[mask]]
        else:
            x_aug = np.hstack((x[mask], cur_pred[mask, :i]))
            proba = self.cc.estimators_[i].predict_proba(x_aug)
            self.__n_nodes += np.count_nonzero(mask)

            # The node with highest proba will always be visited
            proba_max = np.max(proba, axis=1)
            proba_max *= cur_p
            k = np.argmax(proba, axis=1)
            cur_pred[mask, i] = k.astype(bool)
            self.__dfs(x, cur_pred, proba_max, i+1, mask)

            # The node with lowest proba has to pass a condition to be visited
            proba_min = np.min(proba, axis=1)
            proba_min *= cur_p
            condition = proba_min >= self.epsilon
            new_mask = np.copy(mask)
            new_mask[new_mask] = condition

            if np.count_nonzero(new_mask) > 0:
                k = np.argmin(proba[condition], axis=1)
                cur_pred[new_mask, i] = k.astype(bool)
                proba_min = proba_min[condition]
                self.__dfs(x, cur_pred, proba_min, i+1, new_mask)
