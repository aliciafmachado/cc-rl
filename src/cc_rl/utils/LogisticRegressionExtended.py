import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticRegressionExtended(LogisticRegression):
    """Extends logistic regression to accept single-class predictions. The default
    implementation raises an error when this happens, and it is quite common on classifier
    chains problems.
    """

    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True,
                 intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs',
                 max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                 l1_ratio=None):
        self.classes_ = list()
        super().__init__(penalty=penalty, dual=dual, tol=tol, C=C,
                         fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                         class_weight=class_weight, random_state=random_state,
                         solver=solver, max_iter=max_iter, multi_class=multi_class,
                         verbose=verbose, warm_start=warm_start, n_jobs=n_jobs, 
                         l1_ratio=l1_ratio)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) > 1:
            super().fit(X, y)

    def predict(self, X):
        if len(self.classes_) > 1:
            return super().predict(X)
        else:
            return self.classes_[0] * np.ones((len(X),))
    
    def predict_proba(self, X):
        if len(self.classes_) > 1:
            return super().predict_proba(X)
        else:
            pred = np.zeros((len(X), 2), dtype=bool)
            pred[:, self.classes_[0]] = True
            return pred
