import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sdeint
from copy import deepcopy
import sys
import os
from sklearn.linear_model import ElasticNet
from scipy import stats
from matplotlib.patches import Patch
from sklearn.metrics import r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.regression_on_simulation_dynamics import compute_deltas, polynomial_regression

def f(n, t, growth_rates, alpha):
    return n * growth_rates * (1 - alpha @ n)


def G(n, t, noise_term):
    noise_strength = noise_term
    return noise_strength * np.eye(len(n))


def plot_solutions(sol, tspan, model=""):
    group_size = sol.shape[1] // 3
    group_colors = sns.color_palette("Set1", 3)  # Colors for X, Y, Z

    # Map species to group colors
    species_colors = (
            [group_colors[0]] * group_size +  # X
            [group_colors[1]] * group_size +  # Y
            [group_colors[2]] * group_size  # Z
    )

    sns.set(style="whitegrid", context="talk")
    plt.figure(figsize=(14, 7))

    for i in range(sol.shape[1]):
        plt.plot(tspan, sol[:, i], lw=1.5, alpha=0.8, color=species_colors[i])

    # Add legend
    legend_handles = [
        Patch(color=group_colors[0], label="X"),
        Patch(color=group_colors[1], label="Y"),
        Patch(color=group_colors[2], label="Z")
    ]
    plt.legend(handles=legend_handles, title="Groups")

    plt.title("Lotka-Volterra Dynamics", fontsize=20)
    plt.xlabel("Time", fontsize=18)
    plt.ylabel("Population Size", fontsize=18)
    plt.grid(True, linestyle='--', linewidth=0.5)
    sns.despine()
    plt.tight_layout()
    #plt.show()
    path = "C://Users/5605407/Documents/PhD/Chapter_2/Results/LV/dynamics_" + model + ".png"
    plt.savefig(path)



def create_df(solutions):
    T, S = solutions.shape

    # Create base DataFrame
    df = pd.DataFrame({
        "census": np.repeat(np.arange(T), S),
        "species": np.tile(np.arange(S), T),
        "n": solutions.flatten()
    })

    # Pivot back into (T x S) matrix for easier computation of totals
    n_matrix = solutions

    # Compute N (total individuals) and S (species richness) over time
    total_N = n_matrix.sum(axis=1)  # Total individuals at each time
    richness_S = (n_matrix > 0).sum(axis=1)  # Species with n > 0

    # Broadcast totals into long format
    df["N_t"] = np.repeat(total_N, S)
    df["S_t"] = np.repeat(richness_S, S)

    # Compute dn per species (difference in n from previous step)
    dn_matrix = np.diff(n_matrix, axis=0, prepend=np.zeros((1, S)))
    df["dn"] = dn_matrix.flatten()

    # Compute dN and dS per time step
    dN = np.diff(total_N, prepend=0)
    dS = np.diff(richness_S, prepend=0)
    df["dN"] = np.repeat(dN, S)
    df["dS"] = np.repeat(dS, S)

    df = df[df['census'] > 0]

    return df


def three_groups_LV(model_func="food_web", T=50, var=0.0):
    S = 30
    N = 500

    noise_term = 0.0

    # Initialize matrix
    A = np.zeros((S, S))

    # Group boundaries
    group_size = S // 3
    group_indices = {
        'X': slice(0, group_size),
        'Y': slice(group_size, 2 * group_size),
        'Z': slice(2 * group_size, S)
    }

    def constant():
        # Growth rates: individual variability around mean 1.0
        growth_rates = np.random.normal(1.0, var, S)

        # Interaction matrix A: full pairwise variability
        A[:, :] = np.random.normal(0.5, var, (S, S))

        # Diagonal (self-competition) gets slightly higher values
        np.fill_diagonal(A, 1.0 + np.random.normal(0, var, S))

        # Initial populations: grouped by different ranges, still per individual
        initial_conditions = np.random.uniform(0, 1, S)
        initial_conditions /= np.sum(initial_conditions)

        return growth_rates, A, initial_conditions


    def food_web():
        # Growth rates with per-individual noise
        growth_rates = np.concatenate([
            np.repeat(0.3, group_size),
            np.repeat(0.7, group_size),
            np.repeat(0.9, group_size)
        ])
        growth_rates += np.random.normal(0, var, len(growth_rates))

        # Inter-group competition with per-individual variability
        A[group_indices['X'], group_indices['Y']] = 0.0  # Y on X
        A[group_indices['X'], group_indices['Z']] = 0.0  # Z on X

        A[group_indices['Y'], group_indices['X']] = 0.8 + np.random.normal(0, var,
                                                                                   (group_size, group_size))  # X on Y
        A[group_indices['Y'], group_indices['Z']] = 0.0  # Z on Y

        A[group_indices['Z'], group_indices['X']] = 0.0  # X on Z
        A[group_indices['Z'], group_indices['Y']] = 0.7 + np.random.normal(0, var,
                                                                                   (group_size, group_size))  # Y on Z

        # Intra-group competition: still per-individual (diagonal within group block)
        for group in group_indices.values():
            A[group, group] = 1.0 + np.random.normal(0, var, (group_size, group_size))

        # Initial populations with variability per individual
        initial_conditions = np.concatenate([
            np.random.uniform(0, 0.2, group_size),
            np.random.uniform(0, 0.4, group_size),
            np.random.uniform(0, 1.0, group_size)
        ])
        initial_conditions /= np.sum(initial_conditions)

        return growth_rates, A, initial_conditions


    def cascading_food_web():
        # Growth rates with per-individual noise
        growth_rates = np.concatenate([
            np.repeat(0.3, group_size),
            np.repeat(0.7, group_size),
            np.repeat(0.9, group_size)
        ])
        growth_rates += np.random.normal(0, var, len(growth_rates))

        # Inter-group competition with per-individual variability
        A[group_indices['X'], group_indices['Y']] = 0.0  # Y on X
        A[group_indices['X'], group_indices['Z']] = 0.0  # Z on X

        A[group_indices['Y'], group_indices['X']] = 0.8 + np.random.normal(0, var,
                                                                                   (group_size, group_size))  # X on Y
        A[group_indices['Y'], group_indices['Z']] = 0.0  # Z on Y

        A[group_indices['Z'], group_indices['X']] = 0.8 + np.random.normal(0, var,
                                                                                   (group_size, group_size))  # X on Z
        A[group_indices['Z'], group_indices['Y']] = 0.6 + np.random.normal(0, var,
                                                                                   (group_size, group_size))  # Y on Z

        # Intra-group competition: still per-individual (diagonal within group block)
        for group in group_indices.values():
            A[group, group] = 1.0 + np.random.normal(0, var, (group_size, group_size))

        # Initial populations with variability per individual
        initial_conditions = np.concatenate([
            np.random.uniform(0, 0.2, group_size),
            np.random.uniform(0, 0.4, group_size),
            np.random.uniform(0, 1.0, group_size)
        ])
        initial_conditions /= np.sum(initial_conditions)

        return growth_rates, A, initial_conditions


    def cyclic():
        # Growth rates with per-individual noise
        growth_rates = np.concatenate([
            np.repeat(0.3, group_size),
            np.repeat(0.6, group_size),
            np.repeat(0.9, group_size)
        ])
        growth_rates += np.random.normal(0, var, len(growth_rates))

        # Inter-group competition with per-individual variability
        A[group_indices['X'], group_indices['Y']] = 0.0  # Y on X
        A[group_indices['X'], group_indices['Z']] = 0.4 + np.random.normal(0, var,
                                                                                   (group_size, group_size)) # Z on X

        A[group_indices['Y'], group_indices['X']] = 0.8 + np.random.normal(0, var,
                                                                                   (group_size, group_size))  # X on Y
        A[group_indices['Y'], group_indices['Z']] = 0.0  # Z on Y

        A[group_indices['Z'], group_indices['X']] = 0.0  # X on Z
        A[group_indices['Z'], group_indices['Y']] = 0.6 + np.random.normal(0, var,
                                                                                   (group_size, group_size))  # Y on Z

        # Intra-group competition: still per-individual (diagonal within group block)
        for group in group_indices.values():
            A[group, group] = 1.0 + np.random.normal(0, var, (group_size, group_size))

        # Initial populations with variability per individual
        initial_conditions = np.concatenate([
            np.random.uniform(0, 0.2, group_size),
            np.random.uniform(0, 0.5, group_size),
            np.random.uniform(0, 1.0, group_size)
        ])
        initial_conditions /= np.sum(initial_conditions)

        return growth_rates, A, initial_conditions


    def cleaner_fish():
        # Growth rates with per-individual noise
        growth_rates = np.concatenate([
            np.repeat(0.3, group_size),
            np.repeat(0.6, group_size),
            np.repeat(1.2, group_size)
        ])
        growth_rates += np.random.normal(0, var, len(growth_rates))

        # Inter-group competition with per-individual variability
        A[group_indices['X'], group_indices['Y']] = -0.2  # Y on X
        A[group_indices['X'], group_indices['Z']] = 0.4 + np.random.normal(0, var,
                                                                                   (group_size, group_size)) # Z on X

        A[group_indices['Y'], group_indices['X']] = -0.2 + np.random.normal(0, var,
                                                                                   (group_size, group_size))  # X on Y
        A[group_indices['Y'], group_indices['Z']] = 0.0  # Z on Y

        A[group_indices['Z'], group_indices['X']] = 0.0  # X on Z
        A[group_indices['Z'], group_indices['Y']] = 0.5 + np.random.normal(0, var,
                                                                                   (group_size, group_size))  # Y on Z

        # Intra-group competition: still per-individual (diagonal within group block)
        for group in group_indices.values():
            A[group, group] = 1.0 + np.random.normal(0, var, (group_size, group_size))

        # Initial populations with variability per individual
        initial_conditions = np.concatenate([
            np.random.uniform(0, 0.2, group_size),
            np.random.uniform(0, 0.5, group_size),
            np.random.uniform(0, 1.0, group_size)
        ])
        initial_conditions /= np.sum(initial_conditions)

        return growth_rates, A, initial_conditions


    def resource_competition():
        # Growth rates with per-individual noise
        growth_rates = np.concatenate([
            np.repeat(0.3, group_size),
            np.repeat(0.4, group_size),
            np.repeat(0.8, group_size)
        ])
        growth_rates += np.random.normal(0, var, len(growth_rates))

        # Inter-group competition with per-individual variability
        A[group_indices['X'], group_indices['Y']] = 0.4 + np.random.normal(0, var,
                                                                                   (group_size, group_size)) # Y on X
        A[group_indices['X'], group_indices['Z']] = 0.0  # Z on X

        A[group_indices['Y'], group_indices['X']] = 0.2 + np.random.normal(0, var,
                                                                                   (group_size, group_size))  # X on Y
        A[group_indices['Y'], group_indices['Z']] = 0.0  # Z on Y

        A[group_indices['Z'], group_indices['X']] = 0.4 + np.random.normal(0, var,
                                                                                   (group_size, group_size)) # X on Z
        A[group_indices['Z'], group_indices['Y']] = 0.8 + np.random.normal(0, var,
                                                                                   (group_size, group_size))  # Y on Z

        # Intra-group competition: still per-individual (diagonal within group block)
        for group in group_indices.values():
            A[group, group] = 1.0 + np.random.normal(0, var, (group_size, group_size))

        # Initial populations with variability per individual
        initial_conditions = np.concatenate([
            np.random.uniform(0, 0.3, group_size),
            np.random.uniform(0, 0.4, group_size),
            np.random.uniform(0, 1.0, group_size)
        ])
        initial_conditions /= np.sum(initial_conditions)

        return growth_rates, A, initial_conditions


    def f_wrapped(n, t):
        return f(n, t, growth_rates, A)

    def G_wrapped(n, t):
        return G(n, t, noise_term)


    # TODO: Repeat 100 times and take average?
    if model_func == "constant":
        growth_rates, A, initial_conditions = constant()
    elif model_func == "food_web":
        growth_rates, A, initial_conditions = food_web()
    elif model_func == "cascading_food_web":
        growth_rates, A, initial_conditions = cascading_food_web()
    elif model_func == "cyclic":
        growth_rates, A, initial_conditions = cyclic()
    elif model_func == "cleaner_fish":
        growth_rates, A, initial_conditions = cleaner_fish()
    elif model_func == "resource_competition":
        growth_rates, A, initial_conditions = resource_competition()
    else:
        print("Invalid model")


    # Solve stochastic ode but ensure that populations don't become non-negative
    tspan = np.linspace(0, T, int(T * 100))
    sol = initial_conditions.copy()  # shape: (30,)
    solutions = [sol.copy()]  # list of 1D arrays

    for i in range(1, len(tspan)):
        t0, t1 = tspan[i - 1], tspan[i]

        sol = sdeint.itoint(f_wrapped, G_wrapped, sol, np.array([t0, t1]))[-1]

        sol = np.where(sol < 0.001, 0, sol)  # enforce lower bound
        sol = sol.flatten()  # ensure 1D
        solutions.append(sol.copy())

    solutions = np.array(solutions)  # shape: (timesteps, 30)
    solutions = solutions * N  # scale up populations

    plot_solutions(solutions, tspan, model_func)
    df = create_df(solutions)

    return df


def do_polynomial_regression(X, y, census, species, transition_function):
    # TODO: this should be a summation over the individuals of a species,
    #  so a summation of some of the rows in X

    #model = LinearRegression()
    model = ElasticNet()

    # Remove outliers from the data set
    threshold = 3
    z_scores = np.abs(stats.zscore(X))
    outliers = (z_scores > threshold)
    outlier_indices = np.where(outliers.any(axis=1))[0]
    #X, y = X.drop(outlier_indices, axis=0), y.drop(outlier_indices, axis=0)
    keep_indices = np.setdiff1d(np.arange(len(X)), outlier_indices)
    X = X.iloc[keep_indices]
    y = y.iloc[keep_indices]
    species = species.iloc[keep_indices]
    print(f"Number of outliers removed: {len(outlier_indices)}")

    # Error on training set
    model.fit(X, y)
    y_pred = model.predict(X)

    return y, y_pred, species


def set_up_regression(df, N_clusters=None, clustered=False):
    # Single observation interval
    all_census = sorted(df['census'].unique())
    reduced_census = deepcopy(all_census)[::150]

    # Filter to current censuses
    df_filtered = df[df['census'].isin(reduced_census)].copy().reset_index(drop=True)

    # Recompute dn, dN, dS
    df_deltas = compute_deltas(df_filtered, 'LV').reset_index(drop=True)

    if clustered:
        df_deltas = df_deltas.merge(N_clusters, on='census', how='left')

    # METimE regression
    cols_to_exclude = ['dN', 'n_next', 'N_next', 'S_next']
    df_for_setup = df_deltas.drop(columns=cols_to_exclude)
    X, y, census, species = polynomial_regression.set_up_library(df_for_setup, 3, False, False, False)

    return X, y, census, species


def plot_observed_vs_predicted(obs, pred, title, species=None):
    """
    Plots observed vs. predicted values, optionally colored by species.

    Parameters:
    - obs: list or array of observed values
    - pred: list or array of predicted values
    - title: str, the title of the plot
    - species: list or array of species IDs (same length as obs/pred), optional
    """
    r2 = r2_score(obs, pred)

    plt.figure()

    if species is not None:
        species = np.array(species)
        unique_species = np.unique(species)
        cmap = plt.get_cmap("tab20", len(unique_species))
        for idx, sp in enumerate(unique_species):
            mask = species == sp
            plt.scatter(np.array(obs)[mask], np.array(pred)[mask],
                        label=f"Species {sp}", color=cmap(idx), alpha=0.6, edgecolors='k')
        #plt.legend(loc='best', bbox_to_anchor=(1.05, 1.0), title='Species', fontsize='small')
    else:
        plt.scatter(obs, pred, alpha=0.6, edgecolors='k')

    plt.text(0.05, 0.95, f'R² = {r2:.2f}', ha='left', va='top', transform=plt.gca().transAxes,
             fontsize=14, bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray', boxstyle='round,pad=0.5'))

    plt.plot([min(obs), max(obs)], [min(obs), max(obs)], 'r--', linewidth=1)  # 1:1 line
    plt.title(title)
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.grid(True)
    plt.tight_layout()
    #plt.show()

    path = "C://Users/5605407/Documents/PhD/Chapter_2/Results/LV/" + title + ".png"
    plt.savefig(path)


def adjusted_r2_score(r2, n, k):
    """Compute adjusted R^2."""
    return 1 - (1 - r2) * (n - 1) / (n - k - 1) if n != k + 1 else np.nan


def get_cluster_state_variables(df):
    """
    For each time t, calculates the total number of individuals in each cluster
    and adds these as new columns to the dataframe.

    Assumes df contains at least: 't', 'cluster', and 'individuals' columns.
    """
    # Group by time and cluster, summing individuals
    cluster_totals = df.groupby(['census', 'cluster'])['n'].sum().unstack(fill_value=0)

    # Rename columns to reflect cluster totals
    cluster_totals.columns = [f'N_{col}' for col in cluster_totals.columns]
    cluster_totals = cluster_totals.reset_index()

    return cluster_totals


def get_metrics(model, var, T=30, repetitions=25):
    from collections import defaultdict

    results_r2 = defaultdict(list)
    results_adj_r2 = defaultdict(list)

    for _ in range(repetitions):
        df = three_groups_LV(model, T=T, var=var)

        # -------- GLOBAL REGRESSION --------
        X, y, census, species = set_up_regression(df)
        obs, pred, _ = do_polynomial_regression(X, y, census, species, 'dn')
        r2 = r2_score(obs['dn'], pred[:,0])
        adj_r2 = adjusted_r2_score(r2, len(obs['dn']), X.shape[1])
        results_r2['global'].append(r2)
        results_adj_r2['global'].append(adj_r2)

        # -------- SPECIES-SPECIFIC REGRESSION --------
        species_r2, species_adj_r2 = [], []
        for sp in df['species'].unique():
            df_sp = df[df['species'] == sp]
            X, y, census, species = set_up_regression(df_sp)
            y_true, y_pred, _ = do_polynomial_regression(X, y, census, species, 'dn')
            r2 = r2_score(y_true['dn'], y_pred[:,0])
            adj_r2 = adjusted_r2_score(r2, len(y_true), X.shape[1])
            species_r2.append(r2)
            species_adj_r2.append(adj_r2)
        results_r2['species_specific'].append(np.nanmean(species_r2))
        results_adj_r2['species_specific'].append(np.nanmean(species_adj_r2))

        # -------- CLUSTERED REGRESSION --------
        df['cluster'] = df['species'].apply(lambda x: 'x' if x < 10 else 'y' if x < 20 else 'z')
        X_clustered = get_cluster_state_variables(df)
        cluster_r2, cluster_adj_r2 = [], []
        for cluster in df['cluster'].unique():
            df_cluster = df[df['cluster'] == cluster]
            X, y, census, species = set_up_regression(df_cluster.drop(columns=['cluster']), X_clustered, clustered=True)
            y_true, y_pred, _ = do_polynomial_regression(X, y, census, species, 'dn')
            r2 = r2_score(y_true['dn'], y_pred[:,0])
            adj_r2 = adjusted_r2_score(r2, len(y_true), X.shape[1])
            cluster_r2.append(r2)
            cluster_adj_r2.append(adj_r2)
        results_r2['clustered'].append(round(np.nanmean(cluster_r2), 3))
        results_adj_r2['clustered'].append(round(np.nanmean(cluster_adj_r2), 3))

    # Convert to DataFrames: one row per treatment, one column per model-variance
    index = ['global', 'species_specific', 'clustered']
    avg_r2 = pd.DataFrame({f"{model},{var}": [np.nanmean(results_r2[k]) for k in index]}, index=index)
    avg_adj_r2 = pd.DataFrame({f"{model},{var}": [np.nanmean(results_adj_r2[k]) for k in index]}, index=index)

    return avg_r2, avg_adj_r2


if __name__ == "__main__":
    np.random.seed(42)

    all_r2 = []
    all_adj_r2 = []

    for model in ['constant',
                  'food_web',
                  'cascading_food_web',
                  'cyclic',
                  'cleaner_fish',
                  'resource_competition']:
        #for var in [0.0, 0.05, 0.1, 0.5]:
        for var in [0.1]:
            df = three_groups_LV(model, T=30, var=var)

#             # do non-species specific regression
#             X, y, census, species = set_up_regression(df)
#             obs, pred, species = do_polynomial_regression(X, y, census, species,'dn')
#             plot_observed_vs_predicted(obs['dn'], pred[:,0], species=species, title=f"{model}_{var}_global")
#
#             # do regression per species
#             obs, pred, species_list = [], [], []
#             for species in df['species'].unique():
#                 X, y, census, species = set_up_regression(df[df['species'] == species])
#                 y, y_pred, species = do_polynomial_regression(X, y, census, species,'dn')
#                 obs += (y['dn']).tolist()
#                 pred += (y_pred[:,0]).tolist()
#                 species_list += list(species * len(y))
#             plot_observed_vs_predicted(obs, pred, species= species_list, title=f"{model}_{var}_species_specific")
#
#             # do regression per cluster
#             df['cluster'] = df['species'].apply(lambda x: 'x' if x < 10 else 'y' if x < 20 else 'z')
#             X_clustered = get_cluster_state_variables(df)
#             obs, pred, cluster_list = [], [], []
#             for cluster in df['cluster'].unique():
#                 df_cluster = df[df['cluster'] == cluster]
#                 X, y, census, species = set_up_regression(df_cluster.copy().drop(columns=['cluster']), X_clustered, clustered=True)
#                 y, y_pred, species = do_polynomial_regression(X, y, census, species, 'dn')
#                 obs += (y['dn']).tolist()
#                 pred += (y_pred[:, 0]).tolist()
#                 cluster_list += list(cluster * len(y))
#             plot_observed_vs_predicted(obs, pred, species=cluster_list, title=f"{model}_{var}_clustered")
#
#             # Create table with R^2 and adjusted R^2
#             avg_r2, avg_adj_r2 = get_metrics(model, var)
#             all_r2.append(round(avg_r2, 3))
#             all_adj_r2.append(round(avg_adj_r2, 3))
#
# final_r2_df = pd.concat(all_r2, axis=1).transpose()
# final_adj_r2_df = pd.concat(all_adj_r2, axis=1).transpose()
#
# final_r2_df = final_r2_df.reset_index().rename(columns={'index': 'model'})
# final_adj_r2_df = final_adj_r2_df.reset_index().rename(columns={'index': 'model'})
#
# final_r2_df.to_csv('C:/Users/5605407/Documents/PhD/Chapter_2/Results/LV/r2.csv', index=False)
# final_adj_r2_df.to_csv('C:/Users/5605407/Documents/PhD/Chapter_2/Results/LV/adjusted_r2.csv', index=False)









