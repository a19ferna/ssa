import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error


def diff_quantile(data1, data2):
    """
    Compute the unfairness between two populations based on their quantile functions.

    Parameters:
    data1 (array-like): The first set of data points.
    data2 (array-like): The second set of data points.

    Returns:
    float: The unfairness value between the two populations.

    Example:
    >>> data1 = np.array([5, 2, 4, 6, 1])
    >>> data2 = np.array([9, 6, 4, 7, 6])
    >>> diff = diff_quantile(data1, data2)
    >>> print(diff)
    3.98
    """
    probs = np.linspace(0, 1, num=100)
    eqf1 = np.quantile(data1, probs)
    eqf2 = np.quantile(data2, probs)
    unfair_value = np.max(np.abs(eqf1-eqf2))
    return unfair_value


def unfairness(y_predict, x_sa_test):
    """
    Compute the unfairness value for a given fair output and multiple sensitive attributes data contening several modalities.

    Parameters:
    y_predict (array-like): Predicted (fair or not) output data.
    x_ssa_test (array-like): Sensitive attribute data.

    Returns:
    float: Unfairness value in the dataset.

    Example:
    >>> y_predict = np.array([5, 0, 6, 7, 9])
    >>> x_sa_test = np.array([[1, 2, 1, 1, 2], [0, 1, 2, 1, 0]]).T
    >>> unf = unfairness(y_predict, x_sa_test)
    >>> print(unf)
    6.0
    """
    new_list = []
    if len(np.shape(x_sa_test)) == 1:
        sens_val = list(set(x_sa_test))
        data1 = y_predict
        lst_unfairness = []
        for mod in sens_val:
            data2 = y_predict[x_sa_test == mod]
            lst_unfairness.append(diff_quantile(data1, data2))
        new_list.append(max(lst_unfairness))
    else :
        for sens in x_sa_test.T:
            sens_val = list(set(sens))
            data1 = y_predict
            lst_unfairness = []
            for mod in sens_val:
                data2 = y_predict[sens == mod]
                lst_unfairness.append(diff_quantile(data1, data2))
            new_list.append(max(lst_unfairness))
    return max(new_list)

def unfairness_multi(y_fair_dict, x_sa_test):
    """
    Compute unfairness values for multiple fair output datasets and multiple sensitive attributes datasets.

    Parameters:
    y_fair_dict (dict): A dictionary where keys represent sensitive features and values are arrays
            containing the fair predictions corresponding to each sensitive feature.
            Each sensitive feature's fairness adjustment is performed sequentially,
            ensuring that each feature is treated fairly relative to the previous ones.
    x_sa_test (array-like): Sensitive attribute data.

    Returns:
    dict: A dictionary containing unfairness values for each level of fairness.

    Example:
    >>> y_fair_dict = {'Base model':np.array([19,39,65]), 'sens_var_1':np.array([22,40,50]), 'sens_var_2':np.array([28,39,42])}
    >>> x_sa_test = np.array([['blue', 2], ['red', 9], ['green', 5]])
    >>> unf_dict = unfairness_multi(y_fair_dict, x_sa_test)
    >>> print(unf_dict)
    """
    unfairness_dict = {}
    for i, y_fair in enumerate(y_fair_dict.values()):
        result = unfairness(y_fair, x_sa_test)
        unfairness_dict[f'sens_var_{i}'] = result
    return unfairness_dict


def unfairness_multi_permutations(permut_y_fair_dict, all_combs_x_sa_test):
    """
    Compute unfairness values for multiple fair output datasets and multiple sensitive attribute datasets.

    Parameters:
    permut_y_fair_dict (dict): A dictionary containing permutations of fair output datasets.
    all_combs_x_sa_test (dict): A dictionary containing combinations of columns permutations for sensitive attribute datasets.

    Returns:
    list: A list of dictionaries containing unfairness values for each permutation of fair output datasets.

    Example:
    >>> permut_y_fair_dict = {(1,2): {'Base model':np.array([19,39,65]), 'sens_var_1':np.array([22,40,50]), 'sens_var_2':np.array([28,39,42])},
                               (2,1): {'Base model':np.array([19,39,65]), 'sens_var_2':np.array([34,39,60]), 'sens_var_1':np.array([28,39,42])}}
    >>> all_combs_x_sa_test = {(1,2): np.array([['blue', 2], ['red', 9], ['green', 5]]),
                               (2,1): np.array([[2, 'blue'], [9, 'red'], [5, 'green']])}
    >>> unfs_list = unfairness_multi_permutations(permut_y_fair_dict, all_combs_x_sa_test)
    >>> print(unfs_list)
    """
    unfs_list = []
    for key in permut_y_fair_dict.keys():
        unfs_list.append(unfairness_multi(
            permut_y_fair_dict[key], np.array(all_combs_x_sa_test[key])))
    return unfs_list


def risk(y_true, y_predict, classif=False):
    """
    Compute the risk value for predicted fair output compared to the true labels.

    Parameters:
    y_true (array-like): True labels or ground truth values.
    y_predict (array-like): Predicted (fair or not) output values.
    classif (bool, optional): If True, assumes classification task and computes accuracy. 
                              If False (default), assumes regression task and computes mean squared error.

    Returns:
    float: The calculated risk value.

    Example:
    >>> y_true = np.array([1, 0, 1, 1, 0])
    >>> y_predict = np.array([0, 1, 1, 1, 0])
    >>> classification_risk = risk(y_true, y_predict, classif=True)
    >>> print(classification_risk)
    0.6

    >>> y_true = [1.2, 2.5, 3.8, 4.0, 5.2]
    >>> y_predict = [1.0, 2.7, 3.5, 4.2, 5.0]
    >>> regression_risk = risk(y_true, y_predict)
    >>> print(regression_risk)
    0.05
    """
    if classif:
        return accuracy_score(y_true, y_predict)
    else:
        return mean_squared_error(y_true, y_predict)


def risk_multi(y_true, y_fair_dict, classif=False):
    """
    Compute the risk values for multiple fair output datasets compared to the true labels.

    Parameters:
    y_true (array-like): True labels or ground truth values.
    y_fair_dict (dict): A dictionary containing sequentally fair output datasets.
    classif (bool, optional): If True, assumes classification task and computes accuracy. 
                              If False (default), assumes regression task and computes mean squared error.

    Returns:
    dict: A dictionary containing risk values for sequentally fair output datasets.

    Example:
    >>> y_true = np.array([15, 38, 68])
    >>> y_fair_dict = {'Base model':np.array([19,39,65]), 'sens_var_1':np.array([22,40,50]), 'sens_var_2':np.array([28,39,42])}
    >>> risk_values = risk_multi(y_true, y_fair_dict, classif=False)
    """
    risk_dict = {}
    for key in y_fair_dict.keys():
        risk_dict[key] = risk(y_true, list(y_fair_dict[key]), classif)
    return risk_dict


def risk_multi_permutations(y_true, permut_y_fair_dict, classif=False):
    """
    Compute the risk values for multiple fair output datasets compared to the true labels, considering permutations.

    Parameters:
    y_true (array-like): True labels or ground truth values.
    permut_y_fair_dict (dict): A dictionary containing permutations of fair output datasets.
    classif (bool, optional): If True, assumes classification task and computes accuracy. 
                              If False (default), assumes regression task and computes mean squared error.

    Returns:
    list: A list of dictionaries containing risk values for each permutation of fair output datasets.

    Example:
    >>> y_true = np.array([15, 38, 68])
    >>> permut_y_fair_dict = {(1,2): {'Base model':np.array([19,39,65]), 'sens_var_1':np.array([22,40,50]), 'sens_var_2':np.array([28,39,42])},
                               (2,1): {'Base model':np.array([19,39,65]), 'sens_var_2':np.array([34,39,60]), 'sens_var_1':np.array([28,39,42])}}
    >>> risk_values = risk_multi_permutations(y_true, permut_y_fair_dict, classif=False)
    """
    risk_list = []
    for key in permut_y_fair_dict.keys():
        risk_list.append(risk_multi(y_true, permut_y_fair_dict[key], classif))
    return risk_list
