from statsmodels.distributions.empirical_distribution import ECDF
from fairness.quantile import EQF
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score
from itertools import permutations
import numpy as np

from statsmodels.distributions.empirical_distribution import ECDF
from fairness.quantile import EQF


class BaseHelper():
    def __init__(self):
        self.ecdf = {}
        self.eqf = {}

    def _check_shape(y, x_ssa):
        if not isinstance(x_ssa, np.ndarray):
            raise ValueError('x_ssa must be an array')

        if not isinstance(y, np.ndarray):
            raise ValueError('y must be an array')

        if len(x_ssa) != len(y):
            raise ValueError('x_ssa and y should have the same length')

    def _check_mod(sens_val_calib, sens_val_test):
        if sens_val_test > sens_val_calib:
            raise ValueError(
                'x_ssa_test should have at most the number of modalities in x_ssa_calib')

        if not all(elem in sens_val_calib for elem in sens_val_test):
            raise ValueError(
                'Modalities in x_ssa_test should be included in modalities of x_ssa_calib')
    
    def _check_epsilon(epsilon):
        if epsilon<0 or epsilon>1:
            raise ValueError(
                'epsilon must be between 0 and 1')

    def _get_mod(self, x_ssa):
        return list(set(x_ssa))

    def _get_loc(self, x_ssa):
        sens_loc = {}
        for mod in self._get_mod(x_ssa):
            sens_loc[mod] = np.where(x_ssa == mod)[0]
        return sens_loc

    def _get_weights(self, x_ssa):
        sens_loc = self._get_loc(x_ssa)
        weights = {}
        for mod in self._get_mod(x_ssa):
            # Calculate probabilities
            weights[mod] = len(sens_loc[mod])/len(x_ssa)
        return weights

    def _estimate_ecdf_eqf(self, y, x_ssa, sigma):
        sens_loc = self._get_loc(x_ssa)
        eps = np.random.uniform(-sigma, sigma, len(y))
        for mod in self._get_mod(x_ssa):
            # Fit the ecdf and eqf objects
            self.ecdf[mod] = ECDF(y[sens_loc[mod]] +
                                  eps[sens_loc[mod]])
            self.eqf[mod] = EQF(y[sens_loc[mod]]+eps[sens_loc[mod]])


class WassersteinNoBin(BaseHelper):
    def __init__(self, sigma=0.0001):
        BaseHelper.__init__(self)
        self.sigma = sigma
        self.sens_val_calib = None
        self.weights = None

    def fit(self, y_calib, x_ssa_calib):
        BaseHelper._check_shape(y_calib, x_ssa_calib)

        self.sens_val_calib = BaseHelper._get_mod(self, x_ssa_calib)
        self.weights = BaseHelper._get_weights(self, x_ssa_calib)
        BaseHelper._estimate_ecdf_eqf(self, y_calib, x_ssa_calib, self.sigma)

    def transform(self, y_test, x_ssa_test, epsilon=0):

        BaseHelper._check_epsilon(epsilon)
        BaseHelper._check_shape(y_test, x_ssa_test)
        sens_val_test = BaseHelper._get_mod(self, x_ssa_test)
        BaseHelper._check_mod(self.sens_val_calib, sens_val_test)

        sens_loc = BaseHelper._get_loc(self, x_ssa_test)
        y_fair = np.zeros_like(y_test)
        eps = np.random.uniform(-self.sigma, self.sigma, len(y_test))
        for mod1 in sens_val_test:
            for mod2 in sens_val_test:
                y_fair[sens_loc[mod1]] += self.weights[mod2] * \
                    self.eqf[mod2](self.ecdf[mod1](
                        y_test[sens_loc[mod1]]+eps[sens_loc[mod1]]))

        return (1-epsilon)*y_fair + epsilon*y_test


class MultiWasserStein(WassersteinNoBin):
    def __init__(self, sigma=0.0001):
        WassersteinNoBin.__init__(self, sigma=sigma)

        # self.y_fair_calib_all = {}
        self.y_fair_test = {}

        self.sens_val_calib_all = {}
        self.weights_all = {}

        self.eqf_all = {}
        self.ecdf_all = {}
    
    def _check_epsilon_size(self, epsilon, x_ssa_test):
        if len(epsilon) != len(x_ssa_test.T):
            raise ValueError(
                'epsilon must have the same length than the number of sensitive features')

    def fit(self, y_calib, x_ssa_calib):
        for i, sens in enumerate(x_ssa_calib.T):
            # Create an instance of WassersteinNoBin
            wasserstein_instance = WassersteinNoBin(sigma=self.sigma)
            if i == 0:
                y_calib_inter = y_calib
            # Call the fit method from the instance
            wasserstein_instance.fit(y_calib_inter, sens)
            self.sens_val_calib_all[f'sens_var_{i+1}'] = wasserstein_instance.sens_val_calib
            self.weights_all[f'sens_var_{i+1}'] = wasserstein_instance.weights
            self.eqf_all[f'sens_var_{i+1}'] = wasserstein_instance.eqf
            self.ecdf_all[f'sens_var_{i+1}'] = wasserstein_instance.ecdf
            y_calib_inter = wasserstein_instance.transform(y_calib_inter, sens)

    def transform(self, y_test, x_ssa_test, epsilon=None):
        if epsilon == None :
            epsilon = [0]*len(x_ssa_test.T)
        self._check_epsilon_size(epsilon, x_ssa_test)
        for i, sens in enumerate(x_ssa_test.T):
            wasserstein_instance = WassersteinNoBin(sigma=self.sigma)
            if i == 0:
                y_test_inter = y_test
            wasserstein_instance.sens_val_calib = self.sens_val_calib_all[
                f'sens_var_{i+1}']
            wasserstein_instance.weights = self.weights_all[f'sens_var_{i+1}']
            wasserstein_instance.eqf = self.eqf_all[f'sens_var_{i+1}']
            wasserstein_instance.ecdf = self.ecdf_all[f'sens_var_{i+1}']
            y_test_inter = wasserstein_instance.transform(y_test_inter, sens, epsilon[i])
            self.y_fair_test[f'sens_var_{i+1}'] = y_test_inter
        return self.y_fair_test[f'sens_var_{i+1}']
