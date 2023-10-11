import numpy as np


def diff_quantile(data1, data2):
    """
    compute the unfairness of two populations
    """
    probs = np.linspace(0, 1, num=100)
    eqf1 = np.quantile(data1, probs)
    eqf2 = np.quantile(data2, probs)
    unfair_value = np.max(np.abs(eqf1-eqf2))
    return unfair_value


def unfairness(y_fair, x_ssa_test):
    sens_val = list(set(x_ssa_test))
    data1 = y_fair
    lst_unfairness = []
    for mod in sens_val:
        data2 = y_fair[x_ssa_test == mod]
        lst_unfairness.append(diff_quantile(data1, data2))
    return max(lst_unfairness)
