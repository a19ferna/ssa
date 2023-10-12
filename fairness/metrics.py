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
    new_list = []
    for sens in x_ssa_test.T:
        sens_val = list(set(sens))
        data1 = y_fair
        lst_unfairness = []
        for mod in sens_val:
            data2 = y_fair[sens == mod]
            lst_unfairness.append(diff_quantile(data1, data2))
        new_list.append(max(lst_unfairness))
    return max(new_list)


def unfairness_multi(y_fair_dict, x_ssa_test):
    unfairness_dict = {}
    for i, y_fair in enumerate(y_fair_dict.values()):
        result = unfairness(y_fair, x_ssa_test)
        unfairness_dict[f'sens_var_{i}'] = result
    return unfairness_dict
