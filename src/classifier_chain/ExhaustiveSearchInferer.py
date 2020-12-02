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
            x (np.array): Prediction data of shape (n x d1).

        Returns:
            np.array: Prediction outputs of shape (n x d2).
        """
        cur_pred = np.zeros(
            (x.shape[0], len(self.cc.estimators_)), dtype=bool)
        cur_p = np.ones((x.shape[0],))
        best_pred = np.copy(cur_pred)
        best_p = np.zeros((x.shape[0],))

        self.__expand(x, cur_pred, cur_p, best_pred, best_p, 0)

        inv_order = np.empty_like(self.cc.order_)
        inv_order[self.cc.order_] = np.arange(len(self.cc.order_))
        best_pred = best_pred[:, inv_order]

        return best_pred, (1 << len(self.cc.estimators_)) - 1

    def __expand(self, x, cur_pred, cur_p, best_pred, best_p, i):
        if i == len(self.cc.estimators_):
            change = cur_p > best_p
            best_pred[change, :] = cur_pred[change, :]
            best_p[change] = cur_p[change]
        else:
            x_aug = np.hstack((x, cur_pred[:, :i]))
            proba = self.cc.estimators_[i].predict_proba(x_aug)

            cur_pred[:, i] = False
            self.__expand(x, cur_pred, cur_p *
                          proba[:, 0], best_pred, best_p, i+1)

            cur_pred[:, i] = True
            self.__expand(x, cur_pred, cur_p *
                          proba[:, 1], best_pred, best_p, i+1)
