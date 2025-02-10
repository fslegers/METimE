import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats


def load_data(data_set):

    if data_set == "fish":
        df = pd.read_csv('../data/fish_regression_library.csv')
        df = df[df['census'] >= 2000]
        df.reset_index(drop=True, inplace=True)

    if data_set == "birds":
        df = pd.read_csv('../data/birds_regression_library.csv')
        df = df[df['census'] >= 2005]
        df.reset_index(drop=True, inplace=True)

    if data_set == "BCI":
        df = pd.read_csv('../data/BCI_regression_library.csv')
        #print("BCI: Removing first ... censuses from data")
        #df = df[df['census'] >= 5]
        #df.reset_index(drop=True, inplace=True)

    return df


def set_up_library(df, data_set):
    """
    Sets up the library of predictor variables for the bilinear regression.
    :param df: data frame with columns that are used to form the library
    ('species', 'census', 'n_t', 'e_t', 'S_t', 'N_t', 'E_t', and 'n_t+1')
    :param lib: character strings that represent which predictor variables should be included in the regression library.
    Possible terms to include are: 'C', 'n', 'ne', 'e', 'n/N', 'e/E', 'S', 'N', and 'E'
    :return: a dataframe with columns of predictor variables and n_{t + 1}
    """
    if data_set == "fish":
        lib = ['C', 'n', 'n^2', 'n^3', 'N', 'S', 'nN', 'nS', 'n/N']
    else:
        lib = ['C', 'n', 'n^2', 'n^3', 'ne', 'N', 'S', 'E', 'nN', 'nS', 'nE', 'n/N', 'n/S', 'e/E', 'neE']

    # Rename 'm' to 'e' and 'next_m' to 'next_e' if 'm' is present in columns
    if 'm' in df.columns:
        df = df.rename(columns={'m': 'e', 'next_m': 'next_e', 'dm': 'de'})

    df['weight'] = 1 #TODO: what to do with "weight"

    # Mapping of operations for predictor variables
    operations = {
        'e/E': lambda df: df['e'] / df['E_t'],
        'n/N': lambda df: df['n'] / df['N_t'],
        'n/S': lambda df: df['n'] / df['S_t'],
        'eE': lambda df: df['e'] * df['E_t'],
        'eN': lambda df: df['e'] * df['N_t'],
        'nN': lambda df: df['n'] * df['N_t'],
        'ne': lambda df: df['n'] * df['e'],
        'n^2': lambda df: df['n'] ** 2,
        #'n^3': lambda df: df['n'] ** 3,
        'neE': lambda df: df['e'] * df['n'] * df['E_t'],
        'C': lambda df: 1,
        'S': lambda df: df['S_t'],
        'E': lambda df: df['E_t'],
        'N': lambda df: df['N_t']
    }

    # Create a new DataFrame for predictor variables X
    X = pd.DataFrame({key: operations[key](df) for key in lib if key in operations})

    # Define response variables: take the exact subset from df
    response_vars = ['dn', 'de', 'dS'] if 'e' in df.columns else ['dn', 'dS']
    y = df[response_vars]  # Extract only the needed response variables

    census = df['census']

    return X, y, census

def do_regression(X, y, census, transition_function):

    model = LinearRegression()

    # Remove outliers from the data set
    threshold = 3
    z_scores = np.abs(stats.zscore(X))
    outliers = (z_scores > threshold)
    outlier_indices = np.where(outliers.any(axis=1))[0]
    X, y = X.drop(outlier_indices, axis=0), y.drop(outlier_indices, axis=0)
    print(f"Number of outliers removed: {len(outlier_indices)}")

    try:
        # A vif_score of > 10 could indicate colinearity
        X_vif = calculate_vif(X, threshold=10)
        model.fit(X_vif, y)
        X = X_vif
    except:
        print("Undoing VIF step")
        model.fit(X, y)

    # Error on training set
    y_pred = model.predict(X)

    rmse = (mean_squared_error(y, y_pred)) ** (1 / 2)
    r2 = r2_score(y, y_pred)
    print("f(n, e): RMSE = %.3f, \n R^2 = %.3f \n" % (rmse, r2))

    # Add year column back to X for coloring the scatterplot
    census = census.drop(outlier_indices, axis=0)

    # Scatterplot with color by YEAR
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(y, y_pred, c=census, cmap='viridis', alpha=0.7, zorder=1)
    plt.colorbar(scatter, label='Census')
    plt.xlabel(f'Observed {transition_function}')
    plt.ylabel(f'Predicted {transition_function}')
    plt.axis('square')

    plt.axline(xy1=(0, 0), slope=1, color='black', zorder=0, label=("r^2 = %f" % r2))  # Add 1-1 line
    plt.text(0.87, 0.92, "$r^2$ = %.3f" % r2, fontsize=12,
             horizontalalignment='right',
             transform=plt.gca().transAxes)

    # Calculate the combined range
    max_abs = max(abs(min(y)), abs(max(y)), abs(min(y_pred)), abs(max(y_pred)))
    min_val, max_val = -max_abs, max_abs

    # Set the same limits for both axes
    plt.xlim(min_val - 0.1 * max_val, max_val + 0.1 * max_val)
    plt.ylim(min_val - 0.1 * max_val, max_val + 0.1 * max_val)

    plt.gca().set_aspect('equal', adjustable='box')

    # Display the regression equation
    equation = ""
    for feature, coef in zip(model.feature_names_in_, model.coef_):
        if coef != 0:
            equation += f"{coef:.4f} {feature} + "
    equation = equation.rstrip(" + ")
    plt.text(0.5, 0.05, equation, fontsize=9, ha='center', transform=plt.gca().transAxes)

    plt.title(transition_function)
    plt.show()

    coefficients = {key: value for key, value in zip(model.feature_names_in_, model.coef_) if value != 0}
    print("Coefficients:")
    for key, value in coefficients.items():
        print(f"{key} = {value}")

    return coefficients


def METimE(X, y, census):

    all_coeff = []

    for transition_function in ['dn', 'de', 'dS']:
        if transition_function in y:
            coeff = do_regression(X, y[transition_function], census, transition_function)
            all_coeff.append({"transition_function": transition_function, "coeff": coeff})

    return all_coeff


def calculate_vif(X, threshold=10):
    """
    Iteratively calculates VIF scores and drops the feature with the highest VIF
    until all VIF scores are below the threshold.

    Parameters:
    X (pd.DataFrame): Feature DataFrame
    threshold (float): The VIF threshold above which columns are dropped

    Returns:
    pd.DataFrame: DataFrame with remaining features
    """
    while True:
        # Calculate VIF for each column
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        # Check if all VIFs are below the threshold
        max_vif = vif_data["VIF"].max()
        if max_vif <= threshold:
            break

        # Identify the column with the highest VIF
        max_vif_column = vif_data.loc[vif_data["VIF"].idxmax(), "Variable"]
        print(f"Dropping '{max_vif_column}' with VIF: {max_vif}")

        # Drop the column with the highest VIF
        X = X.drop(columns=[max_vif_column])

    return X

# def create_transition_dict(X, y, transition_functions):
#
#     # Initialize the dictionary
#     transition_dict = {}
#
#     # First function: 'n'
#     transition_dict['n'] = {
#         'name': 'n',
#         'function': lambda n, e: n,
#         'value': X['N']/X['S']
#     }
#
#     # Second function: 'ne' if 'e' is present in X
#     if 'e' in X:
#         transition_dict['ne'] = {
#             'name': 'ne',
#             'function': lambda n, e: n * e,
#             'value': X['E'] * X['S']
#         }
#
#     # Process each transition function
#     for tf in transition_functions:
#         tf_name = tf['transition_function']
#
#         # Summing all terms in the dictionary (assuming linear regression-like summation)
#         function = lambda n, e: sum(tf.get(term, 0) * X.get(term, 0) for term in tf)
#
#         if tf_name == 'dn':
#             value = X['dN'] / X['S']
#         elif tf_name == 'de':
#             value = X['dE'] / X['S']
#         elif tf_name == 'dS':
#             value = X['dS']
#
#         transition_dict[tf_name] = {
#             'name': tf_name,
#             'function': tf_name,
#             'value': value
#         }
#
#     return transition_dict

# def plot_histograms(df, data):
#
#     column_list = [i for i in df.columns if i in ['n', 'e', 'ne', 'nN', 'n^2', 'N_t']]
#
#     num_cols = len(column_list)
#     fig, axes = plt.subplots(nrows=1, ncols=num_cols, figsize=(5 * num_cols, 5))
#
#     for ax, col in zip(axes, column_list):
#         min_val, max_val = df[col].min(), df[col].max()
#         if col in df.columns:
#             ax.hist(df[col].dropna(), bins=20, edgecolor='black', alpha=0.7)
#             ax.set_title(f'Histogram of {col}')
#             ax.set_xlabel(col)
#             ax.set_ylabel('Frequency')
#
#             # Add min/max text inside the histogram
#             ax.text(max_val / 2, ax.get_ylim()[1] * 0.9, f'Min: {min_val:.2f}', color='red', fontsize=12, ha='right')
#             ax.text(max_val / 2, ax.get_ylim()[1] * 0.7, f'Max: {max_val:.2f}', color='blue', fontsize=12, ha='left')
#
#         else:
#             ax.set_visible(False)  # Hide plot if column not in DataFrame
#
#     plt.tight_layout()
#     plt.show()
#
#     return df


if __name__ == "__main__":

    for data_set in ['fish']:
        print("-----" + data_set + "-----")

        df = load_data(data_set)

        # For each census, only take the top 5 abundant species
        df_species = df.drop_duplicates(subset=[col for col in df.columns if col != 'TreeID'])

        df_dominant = df_species.groupby('census', group_keys=False).apply(
            lambda x: x[x['n'] >= x['n'].quantile(0.75)]
        )
        dominant_species = df_dominant.groupby('census')['species'].unique().reset_index()
        dominant_species = dominant_species.explode('species')

        df_rare = df_species.groupby('census', group_keys=False).apply(
            lambda x: x[x['n'] <= x['n'].quantile(0.25)]
        )
        rare_species = df_rare.groupby('census')['species'].unique().reset_index()
        rare_species = rare_species.explode('species')

        # Normal species are those not in dominant or rare
        df_dominant = df.merge(
            dominant_species,
            on=['census', 'species'],
            how='inner'
        ).reset_index(drop=True)

        df_rare = df.merge(
            rare_species,
            on=['census', 'species'],
            how='inner'
        ).reset_index(drop=True)

        df_normal = df[~df['species'].isin(dominant_species) & ~df['species'].isin(rare_species)].reset_index(drop=True)

        for df_subset in [df_dominant, df_normal, df_rare]:
            X, y, census = set_up_library(df_subset.copy(), data_set)
            transition_functions = METimE(X, y, census)

        #print(transition_functions)

        # Create a dictionary with columns: name, function, value
        # first function is always 'n' with name 'n' and value the column 'n' in X
        # second function is sometimes 'ne' with name 'ne', value 'n' * 'e' in X (but only if 'e' is present)
        # then, for each transition function in transition_functions (list of dicts)
        # the function name is the key 'transition_function' with function the regression model that you get byvalue
        # summing up al the terms in the dictionary, and value is:
        # - dN / S if transition_function == dn
        # - dE / S if transition_function == de
        # - dS if transition_function == dS



    # TODO:
        # For the BCI data set, we don't have e_t per species, but per individual. So the data that goes into the
        # regression model is different from the bird data, where e_t was the same for individuals within a
        # single species. For the fish data, there are no values for e_t whatsoever.

        # For the BCI data set, there are multiple Stem IDs per Tree ID, which causes duplicates. For now, I just took
        # the mean over the Stem IDs.

        # scaling e with 1/3 had a positive effect. Maybe other feature scaling can have a positive impact as well.


