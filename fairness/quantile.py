from scipy.interpolate import interp1d
import numpy as np


class EQF:
    def __init__(self,
                 sample_data,
                 ):
        self._calculate_eqf(sample_data) # définition self.interpoler, self.min_val, self.max_val

    def _calculate_eqf(self, sample_data):
        sorted_data = np.sort(sample_data)
        linspace = np.linspace(0, 1, num=len(sample_data))
        self.interpolater = interp1d(linspace, sorted_data) # fonction d'interpolation
        self.min_val = sorted_data[0]
        self.max_val = sorted_data[-1]

    def __call__(self, value_):
        try:
            return self.interpolater(value_)
        except ValueError:
            raise ValueError('Error with input value')