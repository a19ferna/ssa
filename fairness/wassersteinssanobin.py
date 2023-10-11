
import numpy as np

from statsmodels.distributions.empirical_distribution import ECDF
from fairness.quantile import EQF


class BaseHelper():
    def __init__(self):
        self.ecdf = {}
        self.eqf = {}

    def _check_shape(y, x_ssa):
        if not isinstance(x_ssa, np.ndarray):
            raise ValueError('x_ssa_calib must be an array')

        if not isinstance(y, np.ndarray):
            raise ValueError('y_calib must be an array')

        if len(x_ssa) != len(y):
            raise ValueError('x_ssa_calib and y should have the same length')

    def _check_mod(sens_val_calib, sens_val_test):
        if sens_val_test > sens_val_calib:
            raise ValueError(
                'x_ssa_test should have at most the number of modalities in x_ssa_calib')

        if not all(elem in sens_val_calib for elem in sens_val_test):
            raise ValueError(
                'Modalities in x_ssa_test should be included in modalities of x_ssa_calib')

    def _get_mod(x_ssa):
        return list(set(x_ssa))

    def _get_loc(self, x_ssa):
        for mod in self._get_mod(x_ssa):
            sens_loc = {}
            sens_loc[mod] = np.where(x_ssa == mod)[0]
            return sens_loc

    def _get_weights(self, x_ssa):
        for mod in self._get_mod(x_ssa):
            # Calculate probabilities
            weights = {}
            weights[mod] = len(self._get_loc[mod])/len(x_ssa)
            return weights

    def _estimate_ecdf_eqf(self, y, sigma):
        eps = np.random.uniform(-sigma, sigma, len(y))
        for mod in self.sens_val:
            # Fit the ecdf and eqf objects
            self.ecdf[mod] = ECDF(y[self._get_loc[mod]] +
                                  eps[self._get_loc[mod]])
            self.eqf[mod] = EQF(y[self._get_loc[mod]]+eps[self._get_loc[mod]])


class WassersteinNoBin(BaseHelper):
    def __init__(self, sigma=0.0001):
        BaseHelper.__init__(self)
        self.sigma = sigma

    def fit(self, y_calib, x_ssa_calib):
        BaseHelper._check_shape(y_calib, x_ssa_calib)

        self.sens_val_calib = BaseHelper._get_mod(x_ssa_calib)
        self.weights = BaseHelper._get_weights(self, x_ssa_calib)
        BaseHelper._estimate_ecdf_eqf(self, y_calib, self.sigma)

    def transform(self, y_test, x_ssa_test):

        BaseHelper._check_shape(y_test, x_ssa_test)
        sens_val_test = BaseHelper._get_mod(x_ssa_test)
        BaseHelper._check_mod(self.sens_val_calib, sens_val_test)

        sens_loc = BaseHelper._get_loc(self, x_ssa_test)
        y_fair = np.zeros_like(y_test)
        eps = np.random.uniform(-self.sigma, self.sigma, len(y_test))
        for mod1 in sens_val_test:
            for mod2 in sens_val_test:
                y_fair[sens_loc[mod1]] += self.weights[mod2] * \
                    self.eqf[mod2](self.ecdf[mod1](
                        y_test[sens_loc[mod1]]+eps[sens_loc[mod1]]))

        return y_fair
