import numpy as np


class ExhaustiveSearchInferer:
    """Searches all possible paths on the tree to determine the best one by the product of 
    their probabilities. Complexity O(2^d), where d is the number of classes.
    """

    def __init__(self, classifier_chain):
        """Default constructor.

        Args:
            classifier_chain (sklearn.multioutput.ClassifierChain): Classifier chain that 
                this inference will be used on.
        """

        self.cc = classifier_chain

    def infer(self, x):
        """Infers best prediction analyzing all paths in the tree.

        Args:
            x (np.array): Prediction data of shape (n, d1).

        Returns:
            np.array: Prediction outputs of shape (n, d2).
            int: If return_num_nodes, it is the average number of visited nodes in the
                tree search.
        """
        cur_pred = np.zeros((len(x), len(self.cc.estimators_)), dtype=bool)
        cur_p = np.ones((len(x),))
        self.__best_pred = np.copy(cur_pred)
        self.__best_p = np.zeros((len(x),))

        self.__dfs(x, cur_pred, cur_p, 0)

        inv_order = np.empty_like(self.cc.order_)
        inv_order[self.cc.order_] = np.arange(len(self.cc.order_))
        self.__best_pred = self.__best_pred[:, inv_order]

        return self.__best_pred, (1 << len(self.cc.estimators_)) - 1

    def __dfs(self, x, cur_pred, cur_p, i):
        """Searches through the tree exhaustively. This is vectorized, so it is done in
        the same way for every row of x.

        Args:
            x (np.array): Prediction data of shape (n, d1)
            cur_pred (np.array): Current prediction in the recursion of shape (n, d2)
            cur_p (np.array): Current accumulated probability in the recursion of shape 
                (n,)
            i (int): Index of the current estimator in the recursion.
        """

        if i == len(self.cc.estimators_):
            change = cur_p > self.__best_p
            self.__best_pred[change, :] = cur_pred[change, :]
            self.__best_p[change] = cur_p[change]
        else:
            x_aug = np.hstack((x, cur_pred[:, :i]))
            proba = self.cc.estimators_[i].predict_proba(x_aug)

            for k in range(2):
                cur_pred[:, i] = bool(k)
                next_p = cur_p * proba[:, k]
                self.__dfs(x, cur_pred, next_p, i+1)
