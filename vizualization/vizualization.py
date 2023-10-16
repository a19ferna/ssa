import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def viz_fairness_distrib(y_fair_test, x_ssa_test):
    plt.figure(figsize=(12, 9))
    n_a = len(x_ssa_test.T)
    n_m = 1

    for key in y_fair_test.keys():
        title = None
        df_test = pd.DataFrame()
        for i, sens in enumerate(x_ssa_test.T):
            df_test[f"sensitive_feature_{i+1}"] = sens

        df_test['Prediction'] = y_fair_test[key]
        if key == 'Base model':
            for i in range(len(x_ssa_test.T)):
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
            for i in range(len(x_ssa_test.T)):
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
