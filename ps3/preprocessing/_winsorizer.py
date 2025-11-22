import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile, upper_quantile):
        self.lq = lower_quantile
        self.uq = upper_quantile

    def fit(self, X, y=None):
        self.lower_quantile_ = np.quantile(X, self.lq)
        self.upper_quantile_ = np.quantile(X, self.uq)
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed = np.where(X_transformed < self.lower_quantile_, self.lower_quantile_,
                            np.where(X_transformed > self.upper_quantile_, self.upper_quantile_, X_transformed))
        return X_transformed

if __name__ == "__main__":
    X = np.random.normal(0, 1, 1000)
    winsoriser = Winsorizer(0.1, 0.9)
    X_transformed = winsoriser.fit_transform(X)

    print(winsoriser.lower_quantile_, winsoriser.upper_quantile_)