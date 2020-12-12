import numpy as np


class BaseInferer:
    """Base abstract class for infererence algorithms.
    """

    def __init__(self, classifier_chain):
        """Default constructor.

        Args:
            classifier_chain (sklearn.multioutput.ClassifierChain): Classifier chain that 
                this inference will be used on.
        """

        self.cc = classifier_chain

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
