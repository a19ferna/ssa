
import numpy as np

from statsmodels.distributions.empirical_distribution import ECDF
from fairness.quantile import EQF


class Wasserstein():
    def __init__(self, sigma=0.0001):
        self.sigma = sigma

        self.ecdf0 = {}
        self.ecdf1 = {}
        self.eqf0 = {}
        self.eqf1 = {}

    def fit(self, y_calib, x_ssa_calib):

        if not isinstance(x_ssa_calib, np.ndarray):
            raise ValueError("x_ssa_calib must be an array")
        if not isinstance(y_calib, np.ndarray):
            raise ValueError("y_calib must be an array")

        if len(x_ssa_calib) != len(y_calib):
            raise ValueError('x_ssa_calib and y should ')

        if len(set(x_ssa_calib)) != 2:
            raise ValueError(
                'x_ssa_calib should have only 2 modalities')
        else:
            self.sens_val_0, self.sens_val_1 = set(x_ssa_calib)

        iw0 = np.where(x_ssa_calib == self.sens_val_0)[0]
        iw1 = np.where(x_ssa_calib == self.sens_val_1)[0]

        eps = np.random.uniform(-self.sigma, self.sigma, len(y_calib))
        # Fit the ecdf and eqf objects
        self.ecdf0 = ECDF(y_calib[iw0]+eps[iw0])
        self.ecdf1 = ECDF(y_calib[iw1]+eps[iw1])
        self.eqf0 = EQF(y_calib[iw0]+eps[iw0])
        self.eqf1 = EQF(y_calib[iw1]+eps[iw1])

        w0 = len(iw0)/len(x_ssa_calib)
        self.weights = np.array([w0, 1-w0])

    def transform(self, y_test, x_ssa_test):

        if not isinstance(x_ssa_test, np.ndarray):
            raise ValueError("x_ssa_test must be an array")
        if not isinstance(y_test, np.ndarray):
            raise ValueError("y_test must be an array")

        if len(x_ssa_test) != len(y_test):
            raise ValueError('x_ssa_test and y_test should ')

        if len(set(x_ssa_test)) != 2:
            raise ValueError(
                'x_ssa_test should have only 2 modalities')

        if {self.sens_val_0, self.sens_val_1} != set(x_ssa_test):
            raise ValueError(
                'x_ssa_test and x_ssa_calib should have the same modalities')

        iw0 = np.where(x_ssa_test == self.sens_val_0)[0]
        iw1 = np.where(x_ssa_test == self.sens_val_1)[0]

        y_fair0 = np.zeros_like(y_test[iw0])
        y_fair1 = np.zeros_like(y_test[iw1])

        eps = np.random.uniform(-self.sigma, self.sigma,
                                len(y_test))
        y_fair0 += self.weights[0] * \
            self.eqf0(self.ecdf0(y_test[iw0]+eps[iw0]))
        y_fair0 += self.weights[1] * \
            self.eqf1(self.ecdf0(y_test[iw0]+eps[iw0]))
        y_fair1 += self.weights[0] * \
            self.eqf0(self.ecdf1(y_test[iw1]+eps[iw1]))
        y_fair1 += self.weights[1] * \
            self.eqf1(self.ecdf1(y_test[iw1]+eps[iw1]))

        y_fair = np.zeros_like(y_test)
        y_fair[iw0] = y_fair0
        y_fair[iw1] = y_fair1

        return y_fair