from nptyping import NDArray
import numpy as np
from sklearn.multioutput import ClassifierChain

from cc_rl.classifier_chain.BaseInferer import BaseInferer


class RandomInferer(BaseInferer):
    """
    Predicts every label randomly.
    """

    def __init__(self, classifier_chain: ClassifierChain):
        super().__init__(classifier_chain.order_)
        self.__n_labels = len(classifier_chain.estimators_)

    def _infer(self, x: NDArray[float]):
        return np.random.randint(0, 1, size=(len(x), self.__n_labels), dtype=bool), \
               self.__n_labels
