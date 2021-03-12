import numpy as np
import os
import pickle
from sklearn.metrics import brier_score_loss

from cc_rl.data.Dataset import Dataset
from cc_rl.classifier_chain.ClassifierChain import ClassifierChain
from cc_rl.utils.LogisticRegressionExtended import LogisticRegressionExtended


class Trainer:
    """Performs automatic calibration of the classifier chains internal estimators for
    each dataset and saves it on pickle a file to be loaded later.
    """

    def __init__(self, ds: Dataset, base_estimator: str = 'logistic_regression',
                 random_state: int = 0):
        self.ds = ds
        self.__base_estimator = base_estimator
        self.__random_state = random_state

        self.best_order = None

    def train(self):
        """Optimizes both order and estimator parameters, and saves a pre-trained
        estimator in a file using pickle.
        """

        self.__optimize_order()
        self.__optimize_parameters()

    def __optimize_order(self):
        # FIXME: stop hardcoding this
        orders = {'birds': [6, 1, 9, 17, 3, 0, 7, 8, 2, 5, 13, 16, 10, 4, 15, 11, 12, 18,
                            14],
                  'emotions': [4, 0, 5, 3, 2, 1],
                  'flags': [2, 5, 1, 6, 4, 3, 0],
                  'genbase': [25, 11, 24, 15, 22, 19, 1, 23, 6, 5, 10, 16, 20, 4, 14, 0,
                              9, 8, 26, 2, 17, 12, 13, 3, 7, 21, 18],
                  'image': [3, 4, 2, 0, 1],
                  'scene': [1, 5, 2, 4, 3, 0],
                  'yeast': [9, 8, 12, 11, 3, 1, 6, 4, 7, 5, 2, 10, 0, 13]}
        if self.ds.name in orders:
            self.best_order = orders[self.ds.name]
        else:
            self.best_order = list(range(self.ds.test_y.shape[1]))
            np.random.shuffle(self.best_order)

    def __optimize_parameters(self):
        """Calibrates the base estimators parameters.

        If base_estimator = 'logistic_regression', it will find the best regularization
        parameter C for each individual binary regressor by perform a grid search over the
        values [0.001, 0.01, 0.1, 1, 10, 100, 1000] optimizing brier loss. Same strategy
        used in MENA et al.

        The fit method from sklearn needed to be rewritten because in it the estimators_
        variable is reinitialized every time, so putting specific parameters for each base
        estimator isn't possible.
        """

        cc = ClassifierChain(base_estimator=self.__base_estimator, order=self.best_order,
                             random_state=self.__random_state)

        cc.n_labels = self.ds.train_y.shape[1]
        best_estimators = [None for _ in range(cc.n_labels)]
        best_score = np.full((cc.n_labels,), np.inf)

        cc.cc.order_ = self.best_order

        # FIXME 1: Stop using test_y here, do cv with train instead
        # TODO: Check this out https://www.researchgate.net/publication/220320172_Trust_Region_Newton_Method_for_Logistic_Regression
        if self.__base_estimator == 'logistic_regression':
            x_aug = np.hstack((self.ds.train_x, self.ds.train_y[:, cc.cc.order_]))

            for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
                cc.cc.estimators_ = [LogisticRegressionExtended(
                    C=C, solver='liblinear') for _ in range(cc.n_labels)]

                # Fitting them manually to avoid resetting estimators
                for chain_idx, estimator in enumerate(cc.cc.estimators_):
                    y = self.ds.train_y[:, cc.cc.order_[chain_idx]]
                    estimator.fit(
                        x_aug[:, :(self.ds.train_x.shape[1] + chain_idx)], y)

                pred = cc.cc.predict(self.ds.test_x)
                score = np.array([brier_score_loss(self.ds.test_y[:, i], pred[:, i])
                                  for i in range(cc.n_labels)])
                score = score[cc.cc.order_]

                change = score < best_score
                best_score[change] = score[change]
                for i in range(len(change)):
                    if change[i]:
                        best_estimators[i] = cc.cc.estimators_[i]

            cc.cc.estimators_ = best_estimators
        else:
            cc.cc.fit(self.ds.train_x, self.ds.train_y)

        # Save parameters
        path = Dataset.data_path + 'trainer/'
        if not os.path.exists(path):
            os.mkdir(path)
        filename = 'cc_' + self.ds.name + '.pkl'
        file = open(path + filename, 'wb')
        pickle.dump(cc.cc, file)
        file.close()
