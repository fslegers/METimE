import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pysindy as ps
from pysindy.feature_library import CustomLibrary, GeneralizedLibrary, PolynomialLibrary, FourierLibrary
from scipy import stats
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def load_data(data_set):

    if data_set == "fish":
        df = pd.read_csv('../data/fish_regression_library.csv')
        df = df[df['census'] >= 2000]
        df.reset_index(drop=True, inplace=True)

    if data_set == "birds":
        df = pd.read_csv('../data/birds_regression_library.csv')
        df = df[df['census'] >= 2007]
        df.reset_index(drop=True, inplace=True)

    if data_set == "BCI":
        df = pd.read_csv('../data/BCI_regression_library.csv')
        df = df[df['census'] >= 5]
        df.reset_index(drop=True, inplace=True)

    return df


def set_up_library(df, data_set, elaborate=False):
    """
    Sets up the library of predictor variables for the bilinear regression.
    :param df: data frame with columns that are used to form the library
    ('species', 'census', 'n_t', 'e_t', 'S_t', 'N_t', 'E_t', and 'n_t+1')
    :param lib: character strings that represent which predictor variables should be included in the regression library.
    Possible terms to include are: 'C', 'n', 'ne', 'e', 'n/N', 'e/E', 'S', 'N', and 'E'
    :return: a dataframe with columns of predictor variables and n_{t + 1}
    """

    if elaborate:
        if 'm' in df.columns:
            df = df.rename(columns={'m': 'e', 'next_m': 'next_e', 'dm': 'de'})

        if data_set == "fish":
            lib = [
                'n',
                'N',
                'S',
                'n2',
                'nN',
                'nS',
                'n/N',
                'n/S',
                'sin(n)',
                'cos(n)',
                'sin(2n)',
                'cos(2n)',
                'sin(3n)',
                'cos(3n)',
                'sigmoid(0.5n)',
                'sigmoid(n)',
                'sigmoid(2n)',
                'C']
        else:
            lib = ['n',
            'e',
            'N',
            'S',
            'E',
            'n2',
            'ne',
            'nN',
            'nS',
            'nE',
            'eN',
            'eS',
            'eE',
            'n/e',
            'n/N',
            'n/S',
            'n/E',
            'e/N',
            'e/S',
            'e/E',
            'sin(n)',
            'cos(n)',
            'sin(2n)',
            'cos(2n)',
            'sin(3n)',
            'cos(3n)',
            'sigmoid(0.5n)',
            'sigmoid(n)',
            'sigmoid(2n)',
            'C']


        # Add polynomial terms up to order 2 (except for e)
        operations = {
            'n': lambda df: df['n'],
            'e': lambda df: df['e'],
            'N': lambda df: df['N_t'],
            'S': lambda df: df['S_t'],
            'E': lambda df: df['E_t'],
            'n2': lambda df: df['n'] ** 2,
            'ne': lambda df: df['n'] * df['e'],
            'nN': lambda df: df['n'] * df['N_t'],
            'nS': lambda df: df['n'] * df['S_t'],
            'nE': lambda df: df['n'] * df['E_t'],
            'eN': lambda df: df['e'] * df['N_t'],
            'eS': lambda df: df['e'] * df['S_t'],
            'eE': lambda df: df['e'] * df['E_t'],
            'n/e': lambda df: df['n'] / df['e'],
            'n/N': lambda df: df['n'] / df['N_t'],
            'n/S': lambda df: df['n'] / df['S_t'],
            'n/E': lambda df: df['n'] / df['E_t'],
            'e/N': lambda df: df['e'] / df['N_t'],
            'e/S': lambda df: df['e'] / df['S_t'],
            'e/E': lambda df: df['e'] / df['E_t'],
            'sin(n)': lambda df: np.sin(df['n']),
            'cos(n)': lambda df: np.cos(df['n']),
            'sin(2n)': lambda df: np.sin(2*df['n']),
            'cos(2n)': lambda df: np.cos(2*df['n']),
            'sin(3n)': lambda df: np.sin(3 * df['n']),
            'cos(3n)': lambda df: np.cos(3 * df['n']),
            'sigmoid(0.5n)': lambda df: 1 / (1 + np.exp(-0.5 * df['n'])),
            'sigmoid(n)': lambda df: 1 / (1 + np.exp(-df['n'])),
            'sigmoid(2n)': lambda df: 1 / (1 + np.exp(-2 * df['n'])),
            'C':  lambda df: 1
        }

        # Create a new DataFrame for predictor variables X
        X = pd.DataFrame({key: operations[key](df) for key in lib if key in operations})

        # Define response variables: take the exact subset from df
        response_vars = ['dn', 'de', 'dS'] if 'e' in df.columns else ['dn', 'dS']
        y = df[response_vars]  # Extract only the needed response variables

        census = df['census']

    else:
        if data_set == "fish":
            lib = ['C', 'n', 'n^2', 'n^3', 'N', 'S', 'nN', 'nS', 'n/N']
        else:
            lib = ['C', 'n', 'n^2', 'n^3', 'ne', 'e/n', 'N', 'S', 'E', 'nN', 'nS', 'nE', 'n/N', 'n/S', 'e/E', 'neE']

        # Rename 'm' to 'e' and 'next_m' to 'next_e' if 'm' is present in columns
        if 'm' in df.columns:
            df = df.rename(columns={'m': 'e', 'next_m': 'next_e', 'dm': 'de'})

        df['weight'] = 1 #TODO: what to do with "weight"

        # Mapping of operations for predictor variables
        operations = {
            'n': lambda df: df['n'],
            'e/E': lambda df: df['e'] / df['E_t'],
            'n/N': lambda df: df['n'] / df['N_t'],
            'n/S': lambda df: df['n'] / df['S_t'],
            'n/E': lambda df: df['n'] / df['E_t'],
            'eE': lambda df: df['e'] * df['E_t'],
            'eN': lambda df: df['e'] * df['N_t'],
            'nN': lambda df: df['n'] * df['N_t'],
            'nS': lambda df: df['n'] * df['S_t'],
            'nE': lambda df: df['n'] * df['E_t'],
            'ne': lambda df: df['n'] * df['e'],
            'e/n': lambda df: df['e'] / df['n'],
            'n^2': lambda df: df['n'] ** 2,
            'n^3': lambda df: df['n'] ** 3,
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


def do_regression(X, y, census, transition_function, fig_title):

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
    #plt.savefig('C://Users/5605407/Documents/PhD/Chapter_2/Figures/Regression/{title}.png'.format(title=fig_title))
    plt.show()

    coefficients = {key: value for key, value in zip(model.feature_names_in_, model.coef_) if value != 0}
    print("Coefficients:")
    for key, value in coefficients.items():
        print(f"{key} = {value}")

    return coefficients


def do_sparse_regression(X, y, census, transition_function, fig_title):

    model = ElasticNet()

    try:
        X_vif = calculate_vif(X, threshold=10)
        model.fit(X_vif, y)
        X = X_vif
    except:
        print("Undoing VIF step")


    # Remove outliers from the data set
    threshold = 3
    z_scores = np.abs(stats.zscore(X))
    outliers = (z_scores > threshold)
    outlier_indices = np.where(outliers.any(axis=1))[0]
    X, y = X.drop(outlier_indices, axis=0), y.drop(outlier_indices, axis=0)
    print(f"Number of outliers removed: {len(outlier_indices)}")

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
    # plt.savefig('C://Users/5605407/Documents/PhD/Chapter_2/Figures/Regression/{title}.png'.format(title=fig_title))
    plt.show()

    coefficients = {key: value for key, value in zip(model.feature_names_in_, model.coef_) if value != 0}
    print("Coefficients:")
    for key, value in coefficients.items():
        print(f"{key} = {value}")



def METimE(X, y, census, fig_title):

    all_coeff = []

    for transition_function in ['dn', 'de', 'dS']:
        if transition_function in y:
            coeff = do_regression(X, y[transition_function], census, transition_function, fig_title="{fig_title}_{transition_function}".format(fig_title=fig_title, transition_function=transition_function))
            all_coeff.append({"transition_function": transition_function, "coeff": coeff})

    return all_coeff


def SINDy(X, y, census, fig_title):

    all_coeff = []

    for transition_function in ['dn', 'de', 'dS']:
        if transition_function in y:
            coeff = do_sparse_regression(X, y[transition_function], census, transition_function, fig_title="{fig_title}_{transition_function}".format(fig_title=fig_title, transition_function=transition_function))
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


def save_transition_functions(trans_dict, data_set):
    for entry in trans_dict:
        transition_name = entry['transition_function']  # File name
        coeff_dict = entry['coeff']  # Extract coefficients

        # Convert dictionary to DataFrame
        df = pd.DataFrame(list(coeff_dict.items()), columns=['Predictor Variable', 'Value'])

        # Save to CSV with the transition function name
        df.to_csv(f'../data/METimE_{transition_name}_{data_set}.csv', index=False)


def get_dominant_species(df, n):
    try:
        df.drop(columns=['e'], inplace=True)
        df = df.drop_duplicates()
    except:
        pass

    df = df[['species', 'n']]
    counts = df.groupby('species').sum()
    dominant_species = counts.sort_values(by='n', ascending=False).head(n)
    return list(dominant_species.index)


def plot_dynamics(df, data_set):
    fig, axes = plt.subplots(1, 3 if data_set == "BCI" else 2, figsize=(15, 5), sharex=True)

    # Plot n_t trajectories for each species
    ax1 = axes[0]
    sns.lineplot(data=df, x="census", y="n", hue="species", ax=ax1, legend=False)
    ax1.set_title("n_t trajectories per species")
    ax1.set_xlabel("Census")
    ax1.set_ylabel("n_t")

    # Plot e_t per individual if dataset is BCI
    if data_set == "BCI":
        ax2 = axes[1]
        df["e_t_per_individual"] = df["e"]
        sns.lineplot(data=df, x="census", y="e_t_per_individual", hue="species", ax=ax2, legend=False)
        ax2.set_title("e_t per individual")
        ax2.set_xlabel("Census")
        ax2.set_ylabel("e_t")

    # Plot N_t, E_t, and S_t per census
        ax3 = axes[2]
    else:
        ax3 = axes[1]

    for var in ["N_t", "S_t"]:
        if var in df.columns:
            sns.lineplot(data=df, x="census", y=var, ax=ax3, label=var)

    if "E_t" in df.columns:
        ax4 = ax3.twinx()
        sns.lineplot(data=df, x="census", y="E_t", ax=ax4, color="r", label="E_t")
        ax4.set_ylabel("E_t", color="r")
        ax4.tick_params(axis="y", colors="r")

    ax3.set_title("N_t, E_t, S_t per Census")
    ax3.set_xlabel("Census")
    ax3.set_ylabel("Value")
    ax3.legend()

    plt.suptitle(f"{data_set}", fontsize=20)
    plt.tight_layout()
    plt.show()

    # Try removing most dominant species from N_t and see what's left
    n = 10
    dominant_species = get_dominant_species(df, n)
    print("Dominant species of {data_set} data: {dominant_species}".format(data_set=data_set, dominant_species=dominant_species))

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and axes object

    # Define colormap for the gradient legend
    cmap = cm.viridis  # You can change this to 'plasma', 'coolwarm', etc.
    norm = mcolors.Normalize(vmin=0, vmax=len(dominant_species))

    # Plot initial full data
    sns.lineplot(data=df, x="census", y="dN", legend=False, hue=None, alpha=0.7, label="0", color=cmap(norm(0)),
                 ax=ax)

    # Iterate through species and progressively remove them
    for i, species in enumerate(dominant_species, start=1):
        sub_df = df[df['species'] == species]
        to_subtract = sub_df.groupby('census')['dn'].sum()

        # Subtract only where 'census' matches
        df['dN'] = df['dN'] - df['census'].map(to_subtract).fillna(0)

        # Plot with color from colormap
        sns.lineplot(data=df, x="census", y="dN", legend=False,
                     alpha=0.7, label=f"{i}", color=cmap(norm(i)), ax=ax)

    # Create a colorbar and associate it with the Axes
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Needed for the colorbar to work
    cbar = plt.colorbar(sm, ax=ax)  # Pass ax here to associate with the correct axes
    cbar.set_label("Species removed", fontsize=12)
    cbar.set_ticks(range(len(dominant_species) + 1))
    cbar.set_ticklabels([str(i) for i in range(len(dominant_species) + 1)])

    # Add title
    ax.set_title(data_set, fontsize=20)
    plt.show()

    if data_set == "BCI":
        # Plot initial full data
        sns.lineplot(data=df, x="census", y="dE", hue=None, legend=False, alpha=0.7, label="0", color=cmap(norm(0)),
                     ax=ax)

        # Iterate through species and progressively remove them
        for i, species in enumerate(dominant_species, start=1):
            sub_df = df[df['species'] == species]
            to_subtract = sub_df.groupby('census')['de'].sum()

            # Subtract only where 'census' matches
            df['dE'] = df['dE'] - df['census'].map(to_subtract).fillna(0)

            # Plot with color from colormap
            sns.lineplot(data=df, x="census", y="dE", legend=False,
                         alpha=1 - (i / (len(dominant_species) + 1)), label=f"{i}", color=cmap(norm(i)), ax=ax)

        # Create a colorbar and associate it with the Axes
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Species removed", fontsize=12)
        cbar.set_ticks(range(len(dominant_species) + 1))
        cbar.set_ticklabels([str(i) for i in range(len(dominant_species) + 1)])

        # Add title
        ax.set_title(data_set, fontsize=20)
        plt.show()


if __name__ == "__main__":

    for data_set in ['fish', 'birds', 'BCI']:
        print("-----" + data_set + "-----")

        df = load_data(data_set)
        #plot_dynamics(df, data_set)

        # # Averaging over individual trees:
        # if data_set == "BCI":
        #     df = df.drop(columns=['TreeID'], errors='ignore')  # Drop 'TreeID'
        #     df = df.groupby(['census', 'species'], as_index=False).mean()  # Group and take the mean

        X, y, census = set_up_library(df.copy(), data_set, elaborate=True)
        transition_functions = METimE(X, y, census, fig_title="METimE_{data_set}".format(data_set=data_set))
        save_transition_functions(transition_functions, data_set)





