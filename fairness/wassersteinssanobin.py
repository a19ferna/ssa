
import numpy as np

from statsmodels.distributions.empirical_distribution import ECDF
from fairness.quantile import EQF


class WassersteinNoBin():
    def __init__(self, sigma=0.0001):
        self.sigma = sigma

        self.ecdf = {}
        self.eqf = {}

    def fit(self, y_calib, x_ssa_calib):

        if not isinstance(x_ssa_calib, np.ndarray):
            raise ValueError('x_ssa_calib must be an array')
        if not isinstance(y_calib, np.ndarray):
            raise ValueError('y_calib must be an array')

        if len(x_ssa_calib) != len(y_calib):
            raise ValueError('x_ssa_calib and y should have the same length')

        self.sens_val = list(set(x_ssa_calib))
        sens_loc = {}
        self.weights = {}

        eps = np.random.uniform(-self.sigma, self.sigma, len(y_calib))
        for mod in self.sens_val:
            sens_loc[mod] = np.where(x_ssa_calib == mod)[0]
            # Fit the ecdf and eqf objects
            self.ecdf[mod] = ECDF(y_calib[sens_loc[mod]]+eps[sens_loc[mod]])
            self.eqf[mod] = EQF(y_calib[sens_loc[mod]]+eps[sens_loc[mod]])
            # Calculate probabilities
            self.weights[mod] = len(sens_loc[mod])/len(x_ssa_calib)

    def transform(self, y_test, x_ssa_test):

        if not isinstance(x_ssa_test, np.ndarray):
            raise ValueError('x_ssa_test must be an array')
        if not isinstance(y_test, np.ndarray):
            raise ValueError('y_test must be an array')

        if len(x_ssa_test) != len(y_test):
            raise ValueError('x_ssa_test and y_test should have the same length')

        if len(set(x_ssa_test)) > len(self.sens_val):
            raise ValueError(
                'x_ssa_test should have at most the number of modalities in x_ssa_calib')
        
        sens_val_test = list(set(x_ssa_test))
        if not all(elem in self.sens_val for elem in sens_val_test):
            raise ValueError(
                'Modalities in x_ssa_test should be included in modalities of x_ssa_calib')
        
        sens_loc = {}
        y_fair = np.zeros_like(y_test)
        eps = np.random.uniform(-self.sigma, self.sigma, len(y_test))
        for mod1 in sens_val_test:
            sens_loc[mod1] = np.where(x_ssa_test == mod1)[0]
            for mod2 in sens_val_test:
                y_fair[sens_loc[mod1]] += self.weights[mod2] * \
                    self.eqf[mod2](self.ecdf[mod1](y_test[sens_loc[mod1]]+eps[sens_loc[mod1]]))

        return y_fair