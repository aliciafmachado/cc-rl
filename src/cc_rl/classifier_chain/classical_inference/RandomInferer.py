from nptyping import NDArray
import numpy as np
from sklearn.multioutput import ClassifierChain

from cc_rl.classifier_chain.BaseInferer import BaseInferer


class RandomInferer(BaseInferer):
    """
    Predicts every label randomly.
    """

    def __init__(self, classifier_chain: ClassifierChain, loss: str, n: int):
        super().__init__(classifier_chain, loss)
        self.cc = classifier_chain
        self.__n = n

    def _infer(self, x: NDArray[float]):
        n_labels = len(self.cc.estimators_)
        cur_pred = np.zeros((len(x), n_labels), dtype=bool)
        best_pred = np.copy(cur_pred)
        best_p = np.zeros((len(x)), dtype=float)

        for k in range(self.__n):
            cur_p = np.ones((len(x)), dtype=float)
            for i in range(n_labels):
                x_aug = np.hstack((x, cur_pred[:, :i]))
                proba = self.cc.estimators_[i].predict_proba(x_aug)
                pred = np.random.randint(0, 2, size=len(x))
                cur_pred[:, i] = pred
                cur_p = self._new_score(cur_p, np.take_along_axis(
                    proba, pred.reshape(-1, 1), axis=1).flatten())

            mask = cur_p > best_p
            best_p[mask] = cur_p[mask]
            best_pred[mask] = cur_pred[mask]

        return best_pred, self.__n * n_labels
