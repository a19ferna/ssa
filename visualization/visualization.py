import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from fairness.wasserstein import MultiWasserStein
import itertools

## density functions##


def viz_fairness_distrib(y_fair_test, x_sa_test):
    """
    Visualizes the distribution of predictions based on different sensitive features using kernel density estimates (KDE).

    Parameters:
    - y_fair_test (dict): A dictionary containing sequentally fair output datasets.
    - x_sa_test (array-like (shape (n_samples, n_sensitive_features)) 
                : The test samples representing multiple sensitive attributes.
    
    Returns:
    None

    Raises:
    ValueError: If the input data is not in the expected format.

    Plotting Conventions:
    - The x-axis represents prediction values, and the y-axis represents density.

    Example:
    >>> y_fair_test = {
            'Base model': [prediction_values],
            'sens_var_1': [prediction_values],
            'sens_var_2': [prediction_values],
            ...
        }
    >>> x_sa_test = [[sensitive_features_of_ind_1_values], [sensitive_feature_of_ind_2_values], ...]

    Usage:
    viz_fairness_distrib(y_fair_test, x_sa_test)
    """

    plt.figure(figsize=(12, 9))
    n_a = len(x_sa_test.T)
    n_m = 1

    for key in y_fair_test.keys():
        title = None
        df_test = pd.DataFrame()
        for i, sens in enumerate(x_sa_test.T):
            df_test[f"sensitive_feature_{i+1}"] = sens

        df_test['Prediction'] = y_fair_test[key]
        if key == 'Base model':
            for i in range(len(x_sa_test.T)):
                title = key
                plt.subplot(n_a, n_m + 1, i * (n_m+1) + 1)
                modalities = df_test[f'sensitive_feature_{i+1}'].unique()
                for mod in modalities:
                    subset_data = df_test[df_test[f'sensitive_feature_{i+1}'] == mod]
                    sns.kdeplot(
                        subset_data['Prediction'], label=f'sensitive_feature_{i+1}: {mod}', fill=True, alpha=0.2)
                plt.legend()
                plt.title(title, fontsize=11)

        else:
            for i in range(len(x_sa_test.T)):
                if key == f'sens_var_{i+1}':
                    title = key
                    plt.subplot(n_a, n_m + 1, i * (n_m+1) + 2)
                    modalities = df_test[f'sensitive_feature_{i+1}'].unique()
                    for mod in modalities:
                        subset_data = df_test[df_test[f'sensitive_feature_{i+1}'] == mod]
                        sns.kdeplot(
                            subset_data['Prediction'], label=f'sensitive_feature_{i+1}: {mod}', fill=True, alpha=0.2)
                        plt.legend()
                    plt.title(title, fontsize=11)

    # Set plot labels and title
    plt.xlabel('Prediction')
    plt.ylabel('Density')

## Waterfall##


# Adapted from: https://github.com/microsoft/waterfall_ax/blob/main/waterfall_ax/waterfall_ax.py
class WaterfallChart():
    '''
    This class creates flexible waterfall charts based on matplotlib. 
    The plot_waterfall() function returns an Axes object. So it’s very flexible to use the object outside the class for further editing.
    '''

    def __init__(self, step_values, step_values_exact_fair=None, step_names=[], metric_name='', last_step_label=''):
        '''
        step_values [list]: the cumulative values for each step.
        step_names [list]: (optional) the corresponding labels for each step. Default is []. If [], Labels will be assigned as 'Step_i' based on the order of step_values.
        metric_name [str]: (optional)  the metric label. Default is ''. If '', a label 'Value' will be assigned as metric name.
        last_step_label [str]: (optional) In the data pre-processing, an additional data point will be appended to reflect the final cumulative value. 
                               This is the label for that value. Default is ''. If '', the label will be assigned as 'Final Value'.
        '''
        self.step_values = step_values
        self.step_values_exact_fair = step_values_exact_fair
        self.step_names = step_names if len(step_names) > 0 else [
            'Step_{0}'.format(x+1) for x in range(len(step_values))]
        self.metric_col = metric_name if metric_name != '' else 'Value'
        self.delta_col = 'delta'
        self.base_col = 'base'
        self.last_step_label = last_step_label if last_step_label != '' else 'Final Value'

    def plot_waterfall(self, ax=None, figsize=(10, 5), title='', bar_labels=True,
                       color_kwargs={}, bar_kwargs={}, line_kwargs={}):
        '''
        Generate the waterfall chart and return the Axes object. 
        Parameters:
            ax [Axes]: (optional) existing axes. Default is None. If None, create a new one.
            figsize [tuple]: (optional) figure size. Default is (10, 5).
            title [string]: (optional) title of the Axes. Default is ''.
            bar_labels [bool|list|str]: (optional) what to show as bar labels on the plot. Refer to check_label_type() for details. 
                                        Default is True. If True, the metric values (deltas and final value) will be shown as labels.
            color_kwargs [dict]: (optional) arguments to control the colors of the plot. Refer to get_colors() function for details. Default is {}.
            bar_kwargs [dict]: (optional) arguments to control the bars on the plot. Valid values are any kwargs for matplotlib.pyplot.bar. Default is {}. 
            line_kwargs [dict]: (optional) arguments to control the lines on the plot. Valid values are any kwargs for matplotlib.axes.Axes.plot. Default is {}.
        '''
        # Prep data for plotting
        df_plot = self.prep_plot()
        # Plot
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax = self.plot_bars(ax, df_plot, color_kwargs, bar_kwargs)
        ax = self.plot_link_lines(ax, df_plot, line_kwargs)
        # Label
        if bar_labels:
            ax = self.add_labels(ax, df_plot, bar_labels, color_kwargs)
        # Format
        ax.set_xlim(-0.5, len(df_plot)-0.5)
        ax.set_ylim(0, df_plot[self.metric_col].max()*1.2)
        if self.step_values_exact_fair is not None:
            y_max = max(df_plot[self.metric_col].max()*1.2,
                        self.df_plot_exact_fair[self.metric_col].max()*1.2)
            ax.set_ylim(0, y_max)
        ax.set_ylabel(self.metric_col)
        ax.set_title(title, fontsize=12)
        return ax

    def prep_plot(self):
        '''
        Take the input values and create a dataframe for plotting
        '''
        # Create a table for plotting
        step_deltas = list(pd.Series(self.step_values).diff().fillna(
            pd.Series(self.step_values)).values)
        df_plot = pd.DataFrame(
            [self.step_values+[self.step_values[-1]],
                step_deltas+[self.step_values[-1]]],
            columns=self.step_names + [self.last_step_label],
            index=[self.metric_col, self.delta_col]).transpose()
        # Add base values
        df_plot[self.base_col] = df_plot[self.metric_col].shift(1).fillna(0)
        df_plot[self.base_col] = list(
            df_plot[self.base_col].values[0:-1]) + [0]
        if self.step_values_exact_fair is not None:
            step_deltas_exact_fair = list(pd.Series(self.step_values_exact_fair).diff(
            ).fillna(pd.Series(self.step_values_exact_fair)).values)
            self.df_plot_exact_fair = pd.DataFrame(
                [self.step_values_exact_fair+[self.step_values_exact_fair[-1]],
                    step_deltas_exact_fair+[self.step_values_exact_fair[-1]]],
                columns=self.step_names + [self.last_step_label],
                index=[self.metric_col, self.delta_col]).transpose()
            # Add base values
            self.df_plot_exact_fair[self.base_col] = self.df_plot_exact_fair[self.metric_col].shift(
                1).fillna(0)
            self.df_plot_exact_fair[self.base_col] = list(
                self.df_plot_exact_fair[self.base_col].values[0:-1]) + [0]
        return df_plot

    def plot_bars(self, ax, df_plot, color_kwargs={}, bar_kwargs={}):
        '''
        Plot the bar elements of the waterfall chart 
        Parameters:
            ax [Axes]: existing axes.
            df_plot [DataFrame]: data to plot.
            color_kwargs [dict]: (optional) arguments to control the colors of the plot. Default is {}
            bar_kwargs [dict]: (optional) arguments to control the bars on the plot. Default is {}
        '''
        barcolors, _ = self.create_color_list(df_plot, color_kwargs)
        bar_kwargs['width'] = bar_kwargs.get('width', 0.8)  # 0.6
        # print(df_plot[self.metric_col])
        if self.step_values_exact_fair is not None:
            barcolors_, _ = self.create_color_list(
                self.df_plot_exact_fair, color_kwargs)

            x_idx_top = 1 * ((self.df_plot_exact_fair[self.base_col] > df_plot[self.base_col]) & (
                self.df_plot_exact_fair[self.base_col] > df_plot[self.metric_col]))
            x_idx_top_grey = x_idx_top * \
                (np.array(barcolors_) == color_kwargs['c_bar_neg'])
            x_idx_top_grey = list(map(bool, x_idx_top_grey))

            x_idx_bot = 1 * ((self.df_plot_exact_fair[self.metric_col] < df_plot[self.base_col]) & (
                self.df_plot_exact_fair[self.metric_col] < df_plot[self.metric_col]))
            x_idx_bot_grey = x_idx_bot * \
                (np.array(barcolors_) == color_kwargs['c_bar_pos'])
            x_idx_bot_grey = list(map(bool, x_idx_bot_grey))

            height_color = df_plot[self.delta_col]
            bottom_color = df_plot[self.base_col]
            height_grey = copy.deepcopy(height_color)
            bottom_grey = copy.deepcopy(bottom_color)

            height_grey.loc[x_idx_bot_grey] = self.df_plot_exact_fair[self.metric_col][x_idx_bot_grey] - \
                df_plot[self.metric_col][x_idx_bot_grey]
            bottom_grey.loc[x_idx_bot_grey] = df_plot[self.metric_col][x_idx_bot_grey]
            height_grey.loc[x_idx_top_grey] = df_plot[self.base_col][x_idx_top_grey] - \
                self.df_plot_exact_fair[self.base_col][x_idx_top_grey]
            bottom_grey.loc[x_idx_top_grey] = self.df_plot_exact_fair[self.base_col][x_idx_top_grey]

            height_color_exact = self.df_plot_exact_fair[self.delta_col]
            bottom_color_exact = self.df_plot_exact_fair[self.base_col]

            height_color_exact.loc[x_idx_top_grey] = self.df_plot_exact_fair[self.metric_col][x_idx_top_grey] - \
                df_plot[self.metric_col][x_idx_top_grey]
            bottom_color_exact.loc[x_idx_top_grey] = df_plot[self.metric_col][x_idx_top_grey]
            height_color_exact.loc[x_idx_bot_grey] = df_plot[self.base_col][x_idx_bot_grey] - \
                self.df_plot_exact_fair[self.base_col][x_idx_bot_grey]
            bottom_color_exact.loc[x_idx_bot_grey] = self.df_plot_exact_fair[self.base_col][x_idx_bot_grey]

            ax.bar(x=self.df_plot_exact_fair.index,
                   height=height_color_exact, bottom=bottom_color_exact,
                   hatch='///', alpha=0.6,  # '\\', '|||', 'xxx' # 0.5
                   linewidth=0.8, color='white', edgecolor=barcolors_)  # steelblue

            ax.bar(x=self.df_plot_exact_fair.index,
                   height=height_grey, bottom=bottom_grey,
                   linewidth=0.8, hatch='///', alpha=0.5,  # '\\', '|||', 'xxx' # steelblue
                   color='white', edgecolor='gray')

        ax.bar(x=df_plot.index, height=df_plot[self.delta_col],
               bottom=df_plot[self.base_col], color=barcolors, **bar_kwargs)
        return ax

    def plot_link_lines(self, ax, df_plot, line_kwargs={}):
        '''
        Plot the line elements of the waterfall chart 
        Parameters:
            ax [Axes]: existing axes.
            df_plot [DataFrame]: data to plot.
            line_kwargs [dict]: (optional) arguments to control the lines on the plot. Default is {}
        '''
        # Create lines
        link_lines = df_plot[self.metric_col].repeat(3).shift(2)
        link_lines[1:-1:3] = np.nan
        link_lines = link_lines[1:-1]
        # Default kwargs
        line_kwargs['color'] = line_kwargs.get('color', 'grey')
        line_kwargs['linestyle'] = line_kwargs.get('linestyle', '--')
        # Plot
        ax.plot(link_lines, **line_kwargs)
        return ax

    def add_labels(self, ax, df_plot, bar_labels, color_kwargs={}):
        '''
        Add labels to the waterfall chart.
        Parameters:
            ax [Axes]: existing axes.
            df_plot [DataFrame]: data to plot.
            bar_labels [bool|list|str]: what to show as bar labels on the plot. Refer to check_label_type() for details. 
            color_kwargs [dict]: (optional) arguments to control the colors of the plot. Default is {}
        '''
        _, txtcolors = self.create_color_list(df_plot, color_kwargs)
        label_type = self.check_label_type(bar_labels)
        for i, v in enumerate(df_plot[self.metric_col]):
            if label_type == 'list':
                label = str(round(bar_labels[i], 2))
            elif label_type == 'value':
                label = '{:,}'.format(
                    round(int(df_plot[self.delta_col][i]), 2))
            else:
                label = bar_labels
            ax.text(i, v*1.03, label, color=txtcolors[i],
                    horizontalalignment='center', verticalalignment='baseline')
        return ax

    def create_color_list(self, df_plot, color_kwargs):
        '''
        Create the lists of colors (bar and label) for the corresponding values to plot
        Parameters:
            df_plot [DataFrame]: data to plot.
            color_kwargs [dict]: arguments to control the colors of the plot.
        '''
        c_bar_pos, c_bar_neg, c_bar_start, c_bar_end, c_text_pos, c_text_neg, c_text_start, c_text_end = self.get_colors(
            color_kwargs)
        mid_values = df_plot[self.delta_col][1:-1].values
        barcolors = [c_bar_start] + [c_bar_neg if x <
                                     0 else c_bar_pos for x in mid_values] + [c_bar_end]
        txtcolors = [c_text_start] + [c_text_neg if x <
                                      0 else c_text_pos for x in mid_values] + [c_text_end]
        return barcolors, txtcolors

    @staticmethod
    def get_colors(color_kwargs):
        '''
        Available color controls and their default values.
        '''
        c_bar_pos = color_kwargs.get(
            'c_bar_pos', 'seagreen')  # Bar color for positive deltas
        # Bar color for negative deltas
        c_bar_neg = color_kwargs.get('c_bar_neg', 'salmon')
        # Bar color for the very first bar
        c_bar_start = color_kwargs.get('c_bar_start', 'c')
        # Bar color for the last bar
        c_bar_end = color_kwargs.get('c_bar_end', 'grey')
        # Label text color for positive deltas
        c_text_pos = color_kwargs.get('c_text_pos', 'darkgreen')
        # Label text color for negative deltas
        c_text_neg = color_kwargs.get('c_text_neg', 'maroon')
        # Label text color for the very first bar
        c_text_start = color_kwargs.get('c_text_start', 'black')
        # Label text color for the last bar
        c_text_end = color_kwargs.get('c_text_end', 'black')
        return c_bar_pos, c_bar_neg, c_bar_start, c_bar_end, c_text_pos, c_text_neg, c_text_start, c_text_end

    @staticmethod
    def check_label_type(bar_labels):
        '''
        Check label type. Valid types are:
            list: a list of labels to be shown for each bar.
            bool: whether to show labels or not. If True, the metric values (deltas and final value) will be shown as labels.
            str: a fixed string to be shown as the label for each bar.  
        '''
        if isinstance(bar_labels, list):
            label_type = 'list'
        elif isinstance(bar_labels, bool):
            label_type = 'value'
        elif isinstance(bar_labels, str):
            label_type = 'str'
        else:
            raise ValueError(
                'bar_labels can only be of type bool, string, or list. Please check input type.')
        return label_type


def waterfall_plot(unfs_list_of_dict):
    unfs_list = [list(unfs.values()) for unfs in unfs_list_of_dict]
    unfs_index = len(unfs_list[0])-1
    categories = []
    unfs_name = ["exact"] + ["approximate"]*(len(unfs_list)-1)
    unfs_hash = [None] + [unfs_list[0]]*(len(unfs_list)-1)

    for i in range(unfs_index):
        if i+1 == 1:
            categories.append('($A_{1}$)-Fair')
        else:
            categories.append(f'($A_{{1:{i+1}}}$)-Fair')

    categories_list = [categories]*len(unfs_list)

    fig, ax = plt.subplots(1, len(unfs_name), figsize=(16, 4))
    if len(unfs_name) < 2:
        ax = [ax]
    color_kwargs = {
        'c_bar_pos': 'orange',
        'c_bar_neg': 'forestgreen',  # 'darkgreen', 'darkred'
        'c_bar_start': 'grey',
        'c_bar_end': 'grey',
        'c_text_pos': 'black',
        'c_text_neg': 'white',
        'c_text_start': 'black',
        'c_text_end': 'black'
    }
    bar_kwargs = {'edgecolor': 'black'}
    line_kwargs = {'color': 'grey'}
    for i, unfs in enumerate(unfs_list):
        waterfall = WaterfallChart(
            step_values=unfs,
            step_values_exact_fair=unfs_hash[i],
            step_names=["Base\nmodel"] + categories_list[i],
            metric_name="Unfairness in $A$"+f"$_{unfs_index}$",
            last_step_label="Final\nmodel"
        )
        wf_ax = waterfall.plot_waterfall(
            ax=ax[i],
            title=f"Sequential ({unfs_name[i]}) fairness: " +
            "$\hat{\mathcal{U}}$"+f"$_{unfs_index}$ result",
            bar_labels=unfs + [unfs[-1]],
            bar_kwargs=bar_kwargs,
            line_kwargs=line_kwargs,
            color_kwargs=color_kwargs)

## Arrow Plot ##

def permutations_cols(x_sa):
    """
    Generate permutations of columns in the input array x_sa.

    Parameters:
    - x_sa (array-like): Input array where each column represents a different sensitive feature.

    Returns:
    dict: A dictionary where keys are tuples representing permutations of column indices,
          and values are corresponding permuted arrays of sensitive features.

    Example:
    >>> x_sa = [[1, 2], [3, 4], [5, 6]]
    >>> permutations_cols(x_sa)
    {(0, 1): [[1, 2], [3, 4], [5, 6]], (1, 0): [[3, 4], [1, 2], [5, 6]]}

    Note:
    This function generates all possible permutations of columns and stores them in a dictionary.
    """
    n = len(x_sa[0])
    ind_cols = list(range(n))
    permut_cols = list(itertools.permutations(ind_cols))
    x_sa_with_ind = np.vstack((ind_cols, x_sa))

    dict_all_combs = {}
    for permutation in permut_cols:
        permuted_x_sa = x_sa_with_ind[:, permutation]
        # First row as the key (converted to tuple)
        key = tuple(permuted_x_sa[0])
        # Other rows as values (converted to list)
        values = permuted_x_sa[1:].tolist()
        dict_all_combs[key] = values

    return dict_all_combs

def calculate_perm_wst(y_calib, x_sa_calib, y_test, x_sa_test, epsilon=None):
    """
    Calculate Wasserstein distance for different permutations of sensitive features between calibration and test sets.
    
    Parameters:
    - y_calib (array-like): Calibration set predictions.
    - x_sa_calib (array-like): Calibration set sensitive features.
    - y_test (array-like): Test set predictions.
    - x_sa_test (array-like): Test set sensitive features.
    - epsilon (array-like or None, optional): Fairness constraints. Defaults to None.

    Returns:
    dict: A dictionary where keys are tuples representing permutations of column indices,
          and values are corresponding sequential fairness values for each permutation.

    Example:
    >>> y_calib = [1, 2, 3]
    >>> x_sa_calib = [[1, 2], [3, 4], [5, 6]]
    >>> y_test = [4, 5, 6]
    >>> x_sa_test = [[7, 8], [9, 10], [11, 12]]
    >>> calculate_perm_wst(y_calib, x_sa_calib, y_test, x_sa_test)
    {(0, 1): {'Base model': 0.5, 'sens_var_1': 0.2}, (1, 0): {'Base model': 0.3, 'sens_var_0': 0.6}}

    Note:
    This function calculates Wasserstein distance for different permutations of sensitive features
    between calibration and test sets and stores the sequential fairness values in a dictionary.
    """
    all_perm_calib = permutations_cols(x_sa_calib)
    all_perm_test = permutations_cols(x_sa_test)
    if epsilon != None:
        all_perm_epsilon = permutations_cols(np.array([np.array(epsilon).T]))
        for key in all_perm_epsilon.keys():
            all_perm_epsilon[key] = all_perm_epsilon[key][0]

    store_dict = {}
    for key in all_perm_calib:
        wst = MultiWasserStein()
        wst.fit(y_calib, np.array(all_perm_calib[key]))
        if epsilon == None:
            wst.transform(y_test, np.array(
                all_perm_test[key]))
        else :
            wst.transform(y_test, np.array(
                all_perm_test[key]), all_perm_epsilon[key])
        store_dict[key] = wst.get_sequential_fairness()
        old_keys = list(store_dict[key].keys())
        new_keys = ['Base model'] + [f'sens_var_{k}' for k in key]
        key_mapping = dict(zip(old_keys, new_keys))
        store_dict[key] = {key_mapping[old_key]: value for old_key, value in store_dict[key].items()}
    return store_dict


def arrow_plot(unfs_dict, risks_dict, permutations=False, base_model=True, final_model=True):
    """
    Generates an arrow plot representing the fairness-risk combinations for each level of fairness.

    Parameters:
    - unfs_dict (dict): A dictionary containing unfairness values associated to the sequentally fair output datasets.
    - risks_dict (dict): A dictionary containing risk values associated to the sequentally fair output datasets.
    - permutations (bool, optional): If True, displays permutations of arrows based on input dictionaries.
                                     Defaults to False.
    - base_model (bool, optional): If True, includes the base model arrow. Defaults to True.
    - final_model (bool, optional): If True, includes the final model arrow. Defaults to True.

    Returns:
    None

    Plotting Conventions:
    - Arrows represent different fairness-risk combinations.
    - Axes are labeled for unfairness (x-axis) and risk (y-axis).

    Note:
    - This function uses global variable `ax` for plotting, ensuring compatibility with external code.
    """
    x = []
    y = []
    sens = [0]

    for i, key in enumerate(unfs_dict.keys()):
        x.append(unfs_dict[key])
        if i != 0:
            sens.append(int(key[9:]))

    for key in risks_dict.keys():
        y.append(risks_dict[key])

    global ax

    if not permutations:
        fig, ax = plt.subplots()

    line = ax.plot(x, y, linestyle="--", alpha=0.25, color="grey")[0]

    for i in range(len(sens)):
        if (i == 0) & (base_model):
            line.axes.annotate(f"Base\nmodel", xytext=(
                x[0]+np.min(x)/20, y[0]), xy=(x[0], y[0]), size=10)
            ax.scatter(x[0], y[0], label="Base model", marker="^", s=100)
        elif i == 1:
            label = f"$A_{sens[i]}$-fair"
            line.axes.annotate(label, xytext=(
                x[i]+np.min(x)/20, y[i]), xy=(x[i], y[i]), size=10)
            ax.scatter(x[i], y[i], label=label, marker="+", s=150)
        elif (i == len(x)-1) & (final_model):
            # Define string with underscore.
            label = f"$A_{1}$" + r"$_:$" + f"$_{i}$-fair"
            line.axes.annotate(label, xytext=(
                x[i]+np.min(x)/20, y[i]), xy=(x[i], y[i]), size=10)
            ax.scatter(x[i], y[i], label=label, marker="*", s=150)
        elif (i == 2) & (i < len(x)-1):
            # Define string with underscore.
            label = f"$A_{sens[1]}$" + r"$_,$" + f"$_{sens[i]}$-fair"
            line.axes.annotate(label, xytext=(
                x[i]+np.min(x)/20, y[i]), xy=(x[i], y[i]), size=10)
            ax.scatter(x[i], y[i], label=label, marker="+", s=150)
        else:
            ax.scatter(x[i], y[i], marker="+", s=150, color="grey", alpha=0.4)
    ax.set_xlabel("Unfairness")
    ax.set_ylabel("Risk")
    ax.set_xlim((np.min(x)-np.min(x)/10-np.max(x)/10,
                np.max(x)+np.min(x)/10+np.max(x)/10))
    ax.set_ylim((np.min(y)-np.min(y)/10-np.max(y)/10,
                np.max(y)+np.min(y)/10+np.max(y)/10))
    ax.set_title("Exact fairness")
    ax.legend(loc="best")


def arrow_plot_permutations(unfs_list, risk_list):
    """
    Plot arrows representing the fairness-risk combinations for each level of fairness for all permutations (order of sensitive variables which with fairness is calculate).

    Parameters:
    - unfs_list (list): A list of dictionaries containing unfairness values for each permutation of fair output datasets.
    - risk_list (list): A list of dictionaries containing risk values for each permutation of fair output datasets.

    Returns:
    None

    Plotting Conventions:
    - Arrows represent different fairness-risk combinations for each scenario in the input lists.
    - Axes are labeled for unfairness (x-axis) and risk (y-axis).

    Example Usage:
    >>> arrow_plot_permutations(unfs_list, risk_list)

    Note:
    - This function uses global variable `ax` for plotting, ensuring compatibility with external code.
    """
    global ax
    fig, ax = plt.subplots()
    for i in range(len(unfs_list)):
        if i == 0:
            arrow_plot(unfs_list[i], risk_list[i],
                       permutations=True, final_model=False)
        elif i == len(unfs_list)-1:
            arrow_plot(unfs_list[i], risk_list[i],
                       permutations=True, base_model=False)
        else:
            arrow_plot(unfs_list[i], risk_list[i], permutations=True,
                       base_model=False, final_model=False)
