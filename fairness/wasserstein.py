from statsmodels.distributions.empirical_distribution import ECDF
from fairness.quantile import EQF
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score
from itertools import permutations
import numpy as np

from statsmodels.distributions.empirical_distribution import ECDF
from fairness.quantile import EQF


class BaseHelper():
    """
    Base class providing helper methods for Wasserstein distance-based fairness adjustment.

    Attributes
    ----------
    ecdf : dict
        Dictionary storing ECDF (Empirical Cumulative Distribution Function) objects for each sensitive modality.
    eqf : dict
        Dictionary storing EQF (Empirical Quantile Function) objects for each sensitive modality.

    Methods
    -------
    _check_shape(y, x_ssa)
        Check the shape and data types of input arrays y and x_ssa.
    _check_mod(sens_val_calib, sens_val_test)
        Check if modalities in test data are included in calibration data's modalities.
    _check_epsilon(epsilon)
        Check if epsilon (fairness parameter) is within the valid range [0, 1].
    _get_mod(x_ssa)
        Get unique modalities from the input sensitive attribute array.
    _get_loc(x_ssa)
        Get the indices of occurrences for each modality in the input sensitive attribute array.
    _get_weights(x_ssa)
        Calculate weights (probabilities) for each modality based on their occurrences.
    _estimate_ecdf_eqf(y, x_ssa, sigma)
        Estimate ECDF and EQF for each modality, incorporating random noise within [-sigma, sigma].

    Notes
    -----
    This base class provides essential methods for Wasserstein distance-based fairness adjustment. It includes
    methods for shape validation, modality checks, epsilon validation, modality extraction, localization of
    modalities in the input data, weight calculation, and ECDF/EQF estimation with random noise.
    """

    def __init__(self):
        self.ecdf = {}
        self.eqf = {}

    def _check_shape(y, x_ssa):
        """
        Check the shape and data types of input arrays y and x_ssa.

        Parameters
        ----------
        y : array-like
            Target values of the data.
        x_ssa : array-like
            Input samples representing the sensitive attribute.

        Raises
        ------
        ValueError
            If the input arrays have incorrect shapes or data types.
        """
        if not isinstance(x_ssa, np.ndarray):
            raise ValueError('x_sa must be an array')

        if not isinstance(y, np.ndarray):
            raise ValueError('y must be an array')

        if len(x_ssa) != len(y):
            raise ValueError('x_sa and y should have the same length')

        for el in y:
            if not isinstance(el, float):
                raise ValueError('y should contain only float numbers')

    def _check_mod(sens_val_calib, sens_val_test):
        """
        Check if modalities in test data are included in calibration data's modalities.

        Parameters
        ----------
        sens_val_calib : list
            Modalities from the calibration data.
        sens_val_test : list
            Modalities from the test data.

        Raises
        ------
        ValueError
            If modalities in test data are not present in calibration data.
        """
        if not all(elem in sens_val_calib for elem in sens_val_test):
            raise ValueError(
                'Modalities in x_ssa_test should be included in modalities of x_sa_calib')

    def _check_epsilon(epsilon):
        """
        Check if epsilon (fairness parameter) is within the valid range [0, 1].

        Parameters
        ----------
        epsilon : float
            Fairness parameter controlling the trade-off between fairness and accuracy.

        Raises
        ------
        ValueError
            If epsilon is outside the valid range [0, 1].
        """
        if epsilon < 0 or epsilon > 1:
            raise ValueError(
                'epsilon must be between 0 and 1')

    def _get_mod(self, x_ssa):
        """
        Get unique modalities from the input sensitive attribute array.

        Parameters
        ----------
        x_ssa : array-like, shape (n_samples,)
            Input samples representing the sensitive attributes.

        Returns
        -------
        list
            List of unique modalities present in the input sensitive attribute array.
        """
        return list(set(x_ssa))

    def _get_loc(self, x_ssa):
        """
        Get the indices of occurrences for each modality in the input sensitive attribute array.

        Parameters
        ----------
        x_ssa : array-like, shape (n_samples,)
            Input sample representing the sensitive attribute.

        Returns
        -------
        dict
            Dictionary where keys are modalities and values are arrays containing their indices.
        """
        sens_loc = {}
        for mod in self._get_mod(x_ssa):
            sens_loc[mod] = np.where(x_ssa == mod)[0]
        return sens_loc

    def _get_weights(self, x_ssa):
        """
        Calculate weights (probabilities) for each modality based on their occurrences.

        Parameters
        ----------
        x_ssa : array-like, shape (n_samples,)
            Input samples representing the sensitive attribute.

        Returns
        -------
        dict
            Dictionary where keys are modalities and values are their corresponding weights.
        """
        sens_loc = self._get_loc(x_ssa)
        weights = {}
        for mod in self._get_mod(x_ssa):
            # Calculate probabilities
            weights[mod] = len(sens_loc[mod])/len(x_ssa)
        return weights

    def _estimate_ecdf_eqf(self, y, x_ssa, sigma):
        """
        Estimate ECDF and EQF for each modality, incorporating random noise within [-sigma, sigma].

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target values corresponding to the sensitive attribute array.
        x_ssa : array-like, shape (n_samples,)
            Input samples representing the sensitive attribute.
        sigma : float
            Standard deviation of the random noise added to the data.

        Returns
        -------
        None
        """
        sens_loc = self._get_loc(x_ssa)
        eps = np.random.uniform(-sigma, sigma, len(y))
        for mod in self._get_mod(x_ssa):
            # Fit the ecdf and eqf objects
            self.ecdf[mod] = ECDF(y[sens_loc[mod]] +
                                  eps[sens_loc[mod]])
            self.eqf[mod] = EQF(y[sens_loc[mod]]+eps[sens_loc[mod]])


class Wasserstein(BaseHelper):
    """
    Class implementing Wasserstein distance-based fairness adjustment for binary classification tasks.

    Parameters
    ----------
    sigma : float, optional (default=0.0001)
        Standard deviation of the random noise added during fairness adjustment.

    Attributes
    ----------
    sigma : float
        Standard deviation of the random noise added during fairness adjustment.
    sens_val_calib : dict
        Dictionary storing modality values obtained from calibration data.
    weights : dict
        Dictionary storing weights (probabilities) for each modality based on their occurrences in calibration data.
    ecdf : dict
        Dictionary storing ECDF (Empirical Cumulative Distribution Function) objects for each sensitive modality.
    eqf : dict
        Dictionary storing EQF (Empirical Quantile Function) objects for each sensitive modality.

    Methods
    -------
    fit(y_calib, x_ssa_calib)
        Fit the fairness adjustment model using calibration data.
    transform(y_test, x_ssa_test, epsilon=0)
        Transform test data to enforce fairness using Wasserstein distance.
    """

    def __init__(self, sigma=0.0001):
        BaseHelper.__init__(self)
        self.sigma = sigma
        self.sens_val_calib = None
        self.weights = None

    def fit(self, y_calib, x_ssa_calib):
        """
        Perform fit on the calibration data and save the ECDF, EQF, and weights of the sensitive variable.

        Parameters
        ----------
        y_calib : array-like, shape (n_samples,)
            The calibration labels.

        x_ssa_calib : array-like, shape (n_samples,)
            The calibration samples representing one single sensitive attribute.

        Returns
        -------
        None

        Notes
        -----
        This method computes the ECDF (Empirical Cumulative Distribution Function),
        EQF (Empirical Quantile Function), and weights for the sensitive variable
        based on the provided calibration data. These computed values are used
        during the transformation process to ensure fairness in predictions.

        Examples
        --------
        >>> wasserstein = Wasserstein(sigma=0.001)
        >>> y_calib = [0, 1, 1, 0]
        >>> x_ssa_calib = [[1, 2], [2, 3], [3, 4], [4, 5]]
        >>> wasserstein.fit(y_calib, x_ssa_calib)
        """
        BaseHelper._check_shape(y_calib, x_ssa_calib)

        self.sens_val_calib = BaseHelper._get_mod(self, x_ssa_calib)
        self.weights = BaseHelper._get_weights(self, x_ssa_calib)
        BaseHelper._estimate_ecdf_eqf(self, y_calib, x_ssa_calib, self.sigma)

    def transform(self, y_test, x_ssa_test, epsilon=0):
        """
        Transform the test data to enforce fairness using Wasserstein distance.

        Parameters
        ----------
        y_test : array-like, shape (n_samples,)
            The target values of the test data.

        x_ssa_test : array-like, shape (n_samples,)
            The test samples representing a single sensitive attribute.

        epsilon : float, optional (default=0)
            The fairness parameter controlling the trade-off between fairness and accuracy.
            It represents the fraction of the original predictions retained after fairness adjustment.
            Epsilon should be a value between 0 and 1, where 0 means full fairness and 1 means no fairness constraint.

        Returns
        -------
        y_fair : array-like, shape (n_samples,)
            Fair predictions for the test data after enforcing fairness constraints.

        Notes
        -----
        This method applies Wasserstein distance-based fairness adjustment to the test data
        using the precomputed ECDF (Empirical Cumulative Distribution Function),
        EQF (Empirical Quantile Function), and weights obtained from the calibration data.
        Random noise within the range of [-sigma, sigma] is added to the test data to ensure fairness.
        The parameter epsilon controls the trade-off between fairness and accuracy,
        with 0 enforcing full fairness and 1 retaining the original predictions.

        Examples
        --------
        >>> wasserstein = Wasserstein(sigma=0.001)
        >>> y_test = [0, 1, 1, 0]
        >>> x_ssa_test = [1, 2, 3, 4]
        >>> fair_predictions = wasserstein.transform(y_test, x_ssa_test, epsilon=0.2)
        """

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


class MultiWasserStein(Wasserstein):
    """
    Class extending Wasserstein for multi-sensitive attribute fairness adjustment.

    Parameters
    ----------
    sigma : float, optional (default=0.0001)
        Standard deviation of the random noise added during fairness adjustment.

    Attributes
    ----------
    sigma : float
        Standard deviation of the random noise added during fairness adjustment.
    y_fair_test : dict
        Dictionary storing fair predictions for each sensitive feature.
    sens_val_calib_all : dict
        Dictionary storing modality values obtained from calibration data for all sensitive features.
    weights_all : dict
        Dictionary storing weights (probabilities) for each modality based on their occurrences in calibration data
        for all sensitive features.
    ecdf_all : dict
        Dictionary storing ECDF (Empirical Cumulative Distribution Function) objects for each sensitive modality
        for all sensitive features.
    eqf_all : dict
        Dictionary storing EQF (Empirical Quantile Function) objects for each sensitive modality
        for all sensitive features.

    Methods
    -------
    fit(y_calib, x_sa_calib)
        Fit the multi-sensitive attribute fairness adjustment model using calibration data.
    transform(y_test, x_sa_test, epsilon=None)
        Transform test data to enforce fairness using Wasserstein distance for multiple sensitive attributes.
    get_sequential_fairness()
        Get fair predictions for each sensitive feature, applied step by step.
    """

    def __init__(self, sigma=0.0001):
        """
        Initialize the MultiWasserStein instance.

        Parameters
        ----------
        sigma : float, optional (default=0.0001)
            The standard deviation of the random noise added to the data during transformation.

        Returns
        -------
        None
        """
        Wasserstein.__init__(self, sigma=sigma)

        # self.y_fair_calib_all = {}
        self.y_fair_test = {}

        self.sens_val_calib_all = {}
        self.weights_all = {}

        self.eqf_all = {}
        self.ecdf_all = {}

    def _check_epsilon_size(self, epsilon, x_sa_test):
        """
        Check if the epsilon list matches the number of sensitive features.

        Parameters
        ----------
        epsilon : list, shape (n_sensitive_features,)
            Fairness parameters controlling the trade-off between fairness and accuracy for each sensitive feature.

        x_sa_test : array-like, shape (n_samples, n_sensitive_features)
            Test samples representing multiple sensitive attributes.

        Raises
        ------
        ValueError
            If the length of epsilon does not match the number of sensitive features.
        """

        if len(epsilon) != len(x_sa_test.T):
            raise ValueError(
                'epsilon must have the same length than the number of sensitive features')

    def fit(self, y_calib, x_sa_calib):
        """
        Perform fit on the calibration data and save the ECDF, EQF, and weights for each sensitive variable.

        Parameters
        ----------
        y_calib : array-like, shape (n_samples,)
            The calibration labels.

        x_sa_calib : array-like, shape (n_samples, n_sensitive_features)
            The calibration samples representing multiple sensitive attributes.

        Returns
        -------
        None

        Notes
        -----
        This method computes the ECDF (Empirical Cumulative Distribution Function),
        EQF (Empirical Quantile Function), and weights for each sensitive variable
        based on the provided calibration data. These computed values are used
        during the transformation process to ensure fairness in predictions.
        """
        for i, sens in enumerate(x_sa_calib.T):
            # Create an instance of Wasserstein
            wasserstein_instance = Wasserstein(sigma=self.sigma)
            if i == 0:
                y_calib_inter = y_calib
            # Call the fit method from the instance
            wasserstein_instance.fit(y_calib_inter, sens)
            self.sens_val_calib_all[f'sens_var_{i+1}'] = wasserstein_instance.sens_val_calib
            self.weights_all[f'sens_var_{i+1}'] = wasserstein_instance.weights
            self.eqf_all[f'sens_var_{i+1}'] = wasserstein_instance.eqf
            self.ecdf_all[f'sens_var_{i+1}'] = wasserstein_instance.ecdf
            y_calib_inter = wasserstein_instance.transform(y_calib_inter, sens)

    def transform(self, y_test, x_sa_test, epsilon=None):
        """
        Transform the test data to enforce fairness using Wasserstein distance.

        Parameters
        ----------
        y_test : array-like, shape (n_samples,)
            The target values of the test data.

        x_sa_test : array-like, shape (n_samples, n_sensitive_features)
            The test samples representing multiple sensitive attributes.

        epsilon : list, shape (n_sensitive_features,), optional (default=None)
            The fairness parameters controlling the trade-off between fairness and accuracy
            for each sensitive feature. If None, no fairness constraints are applied.

        Returns
        -------
        y_fair : array-like, shape (n_samples,)
            Fair predictions for the test data after enforcing fairness constraints.

        Notes
        -----
        This method applies Wasserstein distance-based fairness adjustment to the test data
        using the precomputed ECDF (Empirical Cumulative Distribution Function),
        EQF (Empirical Quantile Function), and weights obtained from the calibration data.
        Random noise within the range of [-sigma, sigma] is added to the test data to ensure fairness.
        The parameter epsilon is a list, where each element controls the trade-off between fairness and accuracy
        for the corresponding sensitive feature.

        Examples
        --------
        >>> wasserstein = MultiWasserStein(sigma=0.001)
        >>> y_calib = [0, 1, 1, 0]
        >>> x_sa_calib = np.array([['blue', 2], ['red', 9], ['green', 5], ['green', 9]])
        >>> wasserstein.fit(y_calib, x_sa_calib)
        >>> y_test = [0, 1, 1, 0]
        >>> x_sa_test = [[1, 2], [2, 3], [3, 4], [4, 5]]
        >>> epsilon = [0.1, 0.2]  # Fairness trade-off values for each sensitive feature
        >>> fair_predictions = wasserstein.transform(y_test, x_sa_test, epsilon=epsilon)
        """
        if epsilon == None:
            epsilon = [0]*len(x_sa_test.T)
        self._check_epsilon_size(epsilon, x_sa_test)

        self.y_fair_test['Base model'] = y_test

        for i, sens in enumerate(x_sa_test.T):
            wasserstein_instance = Wasserstein(sigma=self.sigma)
            if i == 0:
                y_test_inter = y_test
            wasserstein_instance.sens_val_calib = self.sens_val_calib_all[
                f'sens_var_{i+1}']
            wasserstein_instance.weights = self.weights_all[f'sens_var_{i+1}']
            wasserstein_instance.eqf = self.eqf_all[f'sens_var_{i+1}']
            wasserstein_instance.ecdf = self.ecdf_all[f'sens_var_{i+1}']
            y_test_inter = wasserstein_instance.transform(
                y_test_inter, sens, epsilon[i])
            self.y_fair_test[f'sens_var_{i+1}'] = y_test_inter
        return self.y_fair_test[f'sens_var_{i+1}']

    def get_sequential_fairness(self):
        """
        Get the dictionary of fair predictions for each sensitive feature, applied step by step.

        Returns
        -------
        dict
            A dictionary where keys represent sensitive features and values are arrays
            containing the fair predictions corresponding to each sensitive feature.
            Each sensitive feature's fairness adjustment is performed sequentially,
            ensuring that each feature is treated fairly relative to the previous ones.

        Notes
        -----
        This method returns fair predictions for each sensitive feature, applying fairness constraints
        sequentially. The first sensitive feature is adjusted for fairness, and then subsequent features
        are adjusted in sequence, ensuring that each feature is treated fairly relative to the previous ones.

        Examples
        --------
        >>> wasserstein = MultiWasserStein(sigma=0.001)
        >>> y_calib = [0, 1, 1, 0]
        >>> x_sa_calib = np.array([['blue', 2], ['red', 9], ['green', 5], ['green', 9]])
        >>> wasserstein.fit(y_calib, x_sa_calib)
        >>> y_test = [0, 1, 1, 0]
        >>> x_sa_test = [[1, 2], [2, 3], [3, 4], [4, 5]]
        >>> epsilon = [0.1, 0.2]  # Fairness trade-off values for each sensitive feature
        >>> fair_predictions = wasserstein.transform(y_test, x_sa_test, epsilon=epsilon)
        >>> sequential_fairness = wasserstein.get_sequential_fairness()
        """
        return self.y_fair_test
