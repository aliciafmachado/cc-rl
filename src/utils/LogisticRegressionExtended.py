import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticRegressionExtended(LogisticRegression):
    """Extends logistic regression to accept single-class predictions. The default
    implementation raises an error when this happens, and it is quite common on classifier
    chains problems.
    """

    def __init__(self, *args, **kwargs):
        self.single_class = None
        super().__init__(*args, **kwargs)

    def fit(self, X, y):
        if len(np.unique(y)) == 1:
            self.single_class = y[0]
        else:
            super().fit(X, y)

    def predict(self, X):
        if self.single_class is None:
            return super().predict(X)
        else:
            return self.single_class * np.ones((len(X),))
    
    def predict_proba(self, X):
        if self.single_class is None:
            return super().predict_proba(X)
        else:
            pred = np.zeros((len(X), 2), dtype=bool)
            if self.single_class:
                pred[:, 1] = True
            else:
                pred[:, 0] = True
            return pred
