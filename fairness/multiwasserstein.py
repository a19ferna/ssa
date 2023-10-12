
from .quantile import EQF
from .metrics_2 import unfairness
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


class MultiWasserStein(WassersteinNoBin):
    def __init__(self, sigma=0.0001):
        WassersteinNoBin.__init__(self, sigma=sigma)

        # self.y_fair_calib_all = {}
        self.y_fair_test = {}

        self.sens_val_calib_all = {}
        self.weights_all = {}

        self.eqf_all = {}
        self.ecdf_all = {}

    def fit(self, y_calib, x_ssa_calib):
        for i, sens in enumerate(x_ssa_calib):
            if i == 0:
                y_calib_inter = y_calib
            WassersteinNoBin.fit(self, y_calib_inter, sens)
            self.sens_val_calib_all[f'sens_var_{i+1}'] = WassersteinNoBin.sens_val_calib
            self.weights_all[f'sens_var_{i+1}'] = WassersteinNoBin.weights
            self.eqf_all[f'sens_var_{i+1}'] = WassersteinNoBin.eqf
            self.ecdf_all[f'sens_var_{i+1}'] = WassersteinNoBin.ecdf
            y_calib_inter = WassersteinNoBin.transform(
                self, y_calib_inter, sens)
            # self.y_fair_calib_all[f'sens_var_{i+1}'] = y_calib_inter

    def transform(self, y_test, x_ssa_test):
        for i, sens in enumerate(x_ssa_test):
            if i == 0:
                y_test_inter = y_test
            WassersteinNoBin.weights = self.weights_all[f'sens_var_{i+1}']
            WassersteinNoBin.eqf = self.eqf_all[f'sens_var_{i+1}']
            WassersteinNoBin.ecdf = self.ecdf_all[f'sens_var_{i+1}']
            y_test_inter = WassersteinNoBin.transform(self, y_test_inter, sens)
            self.y_fair_test[f'sens_var_{i+1}'] = y_test_inter


class MultiWasserStein:
    def __init__(self,
                 method,
                 sensitive_feature_vector,
                 sigma=0.001) -> None:
        self.method = method  # algo ML train
        self.sensitive_feature_vector = sensitive_feature_vector  # nom des variables MSA
        self.sigma = sigma

        self.level = 0  # niveau de fairness (suivant nombre de variables)
        self.weights = {}  # poids pour la formule de y_fair

        self.eqf_dict = {}  # quantiles
        self.ecdf_dict = {}  # fonctions de répartition

        self.calib_pred = {}
        self.pred = {}
        self.pred_modality = {}

        self.fair_calib = {}
        self.fair = {}
        self.fair_modality = {}

    def fit(self,
            X_calib,
            sensitive_name=None,
            sensitive_idx=None,
            sensitive_modal=None,
            objective='regression') -> None:

        if self.level == 0:  # si 1è variable à rendre fair
            if objective == 'regression':
                self.calib_pred[f'level_{self.level}'] = self.method.predict(
                    X_calib)
                prediction_unlabeled = self.calib_pred[f'level_{self.level}']
            elif objective == 'classification':
                self.calib_pred[f'level_{self.level}'] = self.method.predict_proba(
                    X_calib)
                prediction_unlabeled = self.calib_pred[f'level_{self.level}']
        else:  # si déjà une variable a été rendue fair
            # on prend pred_fair étape préc
            prediction_unlabeled = self.fair_calib[f'level_{int(self.level-1)}']
            self.calib_pred[f'level_{self.level}'] = prediction_unlabeled

        sensitive_idx = self._check_args_fit(
            sensitive_name, sensitive_idx, X_calib)
        sensitive_modal = np.sort(np.unique(X_calib[sensitive_name]))
        sensitive_loc = {}
        sensitive_probas = {}

        for mod in sensitive_modal:
            sensitive_loc[mod] = np.where(
                X_calib.iloc[:, sensitive_idx] == mod)[0]
            sensitive_probas[mod] = len(sensitive_loc[mod])/X_calib.shape[0]
        self.weights[sensitive_idx] = np.array(list(sensitive_probas))

        eps = np.random.uniform(-self.sigma,
                                self.sigma,
                                len(prediction_unlabeled))

        # Fit the ecdf and eqf objects
        self.eqf_dict[f'level_{self.level}'] = {}
        for mod in sensitive_modal:
            self.eqf_dict[f'level_{self.level}'][mod] = EQF(
                prediction_unlabeled[sensitive_loc[mod]]+eps[sensitive_loc[mod]])

        self.ecdf_dict[f'level_{self.level}'] = {}
        for mod in sensitive_modal:
            self.ecdf_dict[f'level_{self.level}'][mod] = ECDF(
                prediction_unlabeled[sensitive_loc[mod]]+eps[sensitive_loc[mod]])

        # Run fair prediction on calibration for next level
        self.fair_calib[f'level_{int(self.level)}'] = self.transform(X=X_calib,
                                                                     sensitive_idx=sensitive_idx,
                                                                     mode='calibration')

    def transform(self,
                  X,
                  sensitive_name=None,
                  sensitive_idx=None,
                  mode='evaluation',
                  epsilon=0,
                  objective='regression'):
        sensitive_idx = self._check_args_fit(sensitive_name, sensitive_idx, X)
        # self.epsilon[sensitive_idx] = epsilon

        if self.level == 0:  # si 1ère variable du modèle à rendre fair
            if objective == 'regression':
                prediction = self.method.predict(X)

            elif objective == 'classification':
                prediction = self.method.predict_proba(X)
        else:
            if mode == 'calibration':
                prediction = self.fair_calib[f'level_{int(self.level-1)}']
            elif mode == 'evaluation':
                prediction = self.fair[f'level_{int(self.level-1)}']
            else:
                raise ValueError(
                    'Need to specify either evaluation or calibration')

        # Recalculate weights and split predictions
        sensitive_modal = np.sort(np.unique(X[sensitive_name]))
        sensitive_loc = {}
        pred = {}
        pred_fair = {}

        for mod in sensitive_modal:
            sensitive_loc[mod] = np.where(X.iloc[:, sensitive_idx] == mod)[0]
            pred[mod] = prediction[sensitive_loc[mod]]
            # Initialize
            pred_fair[mod] = np.zeros_like(pred[mod])

        # Calculate
        eps = np.random.uniform(-self.sigma,
                                self.sigma,
                                len(prediction))

        # Run
        for mod1 in sensitive_modal:
            for mod2 in sensitive_modal:
                pred_fair[mod1] += (self.weights[sensitive_idx][mod2] *
                                    self.eqf_dict[f'level_{self.level}'][mod2](self.ecdf_dict[f'level_{self.level}'][mod2](pred[mod2] + eps[sensitive_loc[mod2]])))

        # Recombine
        pred_fair = np.zeros_like(prediction)
        for mod in sensitive_modal:
            pred_fair[sensitive_loc[mod]] = pred_fair[mod]

        if mode == 'evaluation':
            print('saving mods')
            self.pred[f'level_{self.level}'] = prediction
            for mod1 in sensitive_modal:
                name_dict = f'pred_{mod1}'
                dict_name = {}
                for mod2 in sensitive_modal:
                    dict_name[f'level_{self.level}'] = pred[mod]
                self.pred_modality[name_dict] = dict_name

            self.fair[f'level_{self.level}'] = pred_fair
            for mod1 in sensitive_modal:
                name_dict = f'fair_{mod1}'
                dict_name = {}
                for mod2 in sensitive_modal:
                    dict_name[f'level_{self.level}'] = pred_fair[mod]
                self.fair_modality[name_dict] = dict_name

        if mode == 'calibration':
            return pred_fair
        else:
            return (1-epsilon)*pred_fair + epsilon*prediction

    def _check_args_fit(self, sens_name, sens_idx, data):
        if sens_name is None and sens_idx is None:
            raise ValueError('Specify either idx or name')

        if sens_name is not None and sens_idx is not None:
            raise ValueError('Specify either idx or name, not both')

        if sens_name is not None:
            sens_idx = np.where(data.columns == sens_name)[0][0]

        return sens_idx

    # def _calculate_nb_modality(self, sens_name):
    #    nb_modality = data[sens_name].nunique()
    #    return nb_modality

    def get_all_predictions(self, data_dict):

        all_comb = permutations(self.sensitive_feature_vector, len(
            self.sensitive_feature_vector))

        output_dict = {}

        # model_dict = {}
        for base_model in all_comb:
            level = 0
            output_dict[base_model] = {}
            # model_dict[base_model] = MultiWasserStein(method=self.method)

            for feature_ in base_model:
                # model_dict[base_model].fit(X_calib=data_dict['X_calib'],
                #                        sensitive_name=feature_)
                self.fit(X_calib=data_dict['X_calib'],
                         sensitive_name=feature_)
                # response = (model_dict[base_model]
                #                        .transform(X=data_dict['X_test'],
                #                                   sensitive_name=feature_))
                response = self.transform(X=data_dict['X_test'],
                                          sensitive_name=feature_)
                output_dict[base_model][f'level_{level}'] = {}
                output_dict[base_model][f'level_{level}']['prediction'] = response

                sens_counter = 0
                output_dict[base_model][f'level_{level}']['sensitive'] = {}
                for subfeature_ in base_model:
                    sens_tmp = data_dict['X_test'].loc[:, subfeature_]
                    output_dict[base_model][f'level_{level}']['sensitive'][sens_counter] = sens_tmp
                    sens_counter += 1

                # model_dict[base_model].level += 1
                self.level += 1
                level += 1

        return output_dict


def calculate_metrics(output_dict,
                      y_test,
                      objective='regression',
                      threshold=None):

    metrics_dict = {}

    for key, value in output_dict.items():
        metrics_dict[key] = {}

        for level in value.keys():
            metrics_dict[key][level] = {}

            # First MSE
            prediction = output_dict[key][level]['prediction']

            if objective == 'regression':
                metrics_dict[key][level]['mse'] = mean_squared_error(y_test,
                                                                     prediction)
            elif objective == 'classification':
                predictions_ = np.where(prediction > threshold, 1, 0)

                metrics_dict[key][level]['accuracy'] = accuracy_score(y_test,
                                                                      predictions_)
                metrics_dict[key][level]['f1_score'] = f1_score(y_test,
                                                                predictions_)

            unfair_tmp = 0
            for key_2, sens_feature_ in output_dict[key][level]['sensitive'].items():

                id0 = np.where(sens_feature_ == 0)[0]
                id1 = np.where(sens_feature_ == 1)[0]

                pred_0 = prediction[id0]
                pred_1 = prediction[id1]

                unfair_tmp += unfairness(pred_0, pred_1)
                metrics_dict[key][level][f'unfairness_{key_2}'] = unfairness(
                    pred_0, pred_1)

            metrics_dict[key][level]['unfairness'] = unfair_tmp

    return metrics_dict
