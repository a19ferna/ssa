from scipy.interpolate import interp1d
import numpy as np

#WARNING:You cannot calculate the EQF function of a single value : this means that if only one individual 
# has a specific sensitive value, you cannot use the transform function. 

class EQF:
    """
    Empirical Quantile Function (EQF) Class.

    This class computes and encapsulates the Empirical Quantile Function for a given set of sample data.
    The EQF provides an interpolation of the cumulative distribution function (CDF) based on the input data.

    Parameters:
    sample_data (array-like): A 1-D array or list-like object containing the sample data.

    Attributes:
    interpolater (scipy.interpolate.interp1d): An interpolation function that maps quantiles to values.
    min_val (float): The minimum value in the sample data.
    max_val (float): The maximum value in the sample data.

    Methods:
    __init__(sample_data): Initializes the EQF object by calculating the interpolater, min_val, and max_val.
    _calculate_eqf(sample_data): Private method to calculate interpolater, min_val, and max_val.
    __call__(value_): Callable method to compute the interpolated value for a given quantile.

    Example usage:
    >>> sample_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> eqf = EQF(sample_data)
    >>> print(eqf(0.5))  # Interpolated value at quantile 0.5
    5.5

    Example usage:
    >>> sample_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> eqf = EQF(sample_data)
    >>> print(eqf([0.2, 0.5, 0.8]))  # Interpolated value at quantiles 0.2, 0.5 and 0.8
    [2.8 5.5 8.2]


    Raises:
    ValueError: If the input value_ is outside the range [0, 1].

    Note:
    - The EQF interpolates values within the range [0, 1] representing quantiles.
    - The input sample_data should be a list or array-like containing numerical values.

    """

    def __init__(self,
                 sample_data,
                 ):
        # définition self.interpoler, self.min_val, self.max_val
        self._calculate_eqf(sample_data)

    def _calculate_eqf(self, sample_data):
        """
        Calculate the Empirical Quantile Function for the given sample data.

        Parameters:
        sample_data (array-like): A 1-D array or list-like object containing the sample data.
        """
        sorted_data = np.sort(sample_data)
        linspace = np.linspace(0, 1, num=len(sample_data))
        # fonction d'interpolation
        self.interpolater = interp1d(linspace, sorted_data)
        self.min_val = sorted_data[0]
        self.max_val = sorted_data[-1]

    def __call__(self, value_):
        """
        Compute the interpolated value for a given quantile.

        Parameters:
        value_ (float): Quantile value between 0 and 1.

        Returns:
        float: Interpolated value corresponding to the input quantile.

        Raises:
        ValueError: If the input value_ is outside the range [0, 1].
        """
        try:
            return self.interpolater(value_)
        except ValueError:
            raise ValueError('Error with input value')