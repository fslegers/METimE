import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import seaborn as sns
import pandas as pd
from copy import deepcopy
import sys
import os

from scipy.integrate import solve_ivp

from sklearn.linear_model import LinearRegression
from matplotlib.patches import Patch
from sklearn.metrics import r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.regression_on_simulation_dynamics import compute_deltas


def f(n, t, growth_rates, alpha):
    n = np.clip(n, 0, None)
    return n * growth_rates * (1 - alpha @ n)


def plot_solutions(sol, tspan, model=""):
    """
    Plots time-series solutions of a dynamical system, grouping species by variable type (X, Y, Z).

    Parameters:
    - sol: np.ndarray, shape (time, species), the solution matrix
    - tspan: array-like, the time vector
    - model: str, optional title suffix and filename tag
    """
    group_size = sol.shape[1] // 3
    group_colors = sns.color_palette("Set1", 3)  # Colors for X, Y, Z

    # Assign colors based on species group
    species_colors = (
        [group_colors[0]] * group_size +  # X
        [group_colors[1]] * group_size +  # Y
        [group_colors[2]] * group_size    # Z
    )

    # Aesthetic settings
    sns.set(style="whitegrid", context="notebook", font_scale=2.0)
    plt.figure(figsize=(10, 6))

    for i in range(sol.shape[1]):
        plt.plot(tspan, sol[:, i], lw=1.8, alpha=0.9, color=species_colors[i])

    # Group legend
    legend_handles = [
        Patch(color=group_colors[2], label=r"$Z$"),
        Patch(color=group_colors[1], label=r"$Y$"),
        Patch(color=group_colors[0], label=r"$X$")
    ]
    plt.legend(handles=legend_handles, title="Genus", frameon=False, fontsize=16)

    # Titles and labels with LaTeX formatting
    #plt.title(r"\textbf{Lotka–Volterra System Dynamics}" + f" ({model})", fontsize=20)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Population Size (n)", fontsize=20)

    # Grid and layout
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    sns.despine()
    plt.tight_layout()

    # Save figure
    path = f"C://Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/LV/dynamics_{model}.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    #plt.show()


def create_df(solutions):
    T, S = solutions.shape

    # Increase the observation interval by only keeping every 10th time step                                            # TODO: REMOVE AGAIN BECAUSE THIS WILL HAPPEN LATER ON
    # so that dn is not so small
    sampled_censuses = np.arange(T)[::50]
    subsamples = solutions[sampled_censuses]
    T = subsamples.shape[0]

    # Create base DataFrame
    df = pd.DataFrame({
        "census": np.repeat(np.arange(1, T + 1), S),
        "species": np.tile(np.arange(S), T),
        "n": subsamples.flatten()
    })

    # Compute N (total individuals) and S (species richness) over time
    total_N = subsamples.sum(axis=1)  # Total individuals at each time
    richness_S = (subsamples > 0).sum(axis=1)  # Species with n > 0

    # Broadcast totals into long format
    df["N_t"] = np.repeat(total_N, S)
    df["S_t"] = np.repeat(richness_S, S)

    dn_matrix = np.diff(subsamples, axis=0, append=np.zeros((1, S)))

    # Replace with -n_t if species goes extinct at t+1
    extinct_mask = subsamples[1:] == 0  # shape (T-1, S)
    dn_matrix[:-1][extinct_mask] = -subsamples[:-1][extinct_mask]

    # Flatten into dataframe
    df["dn"] = dn_matrix.flatten()

    # Compute forward differences for N and S: value(t+1) - value(t)
    dN = np.diff(total_N, append=0)
    dS = np.diff(richness_S, append=0)

    df["dN/S"] = np.repeat(dN/S, S)
    df["dS"] = np.repeat(dS, S)

    df = df[df['census'] < 20]

    return df


def three_groups_LV(model_func="food_web", T=50, var=0.0):
    S = 12
    N = 100

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

    def a_LV():
        """
        Constant interaction network
        """
        growth_rates = np.ones(S)

        # Equal competition for all species
        A = 0.125 * np.ones((S, S)) + np.random.normal(0, var, (S, S))

        # Intra-group competition: still per-individual (diagonal within group block)
        group_means = [0.25, 0.25, 0.25]
        for i, group_slice in enumerate(group_indices.values()):
            group = np.arange(*group_slice.indices(S))  # Convert slice to array of ints
            mean = group_means[i]
            block = mean + np.random.normal(0, var, (len(group), len(group)))
            A[np.ix_(group, group)] = block

        # Initial populations with variability per individual
        initial_conditions = (np.random.uniform
                (0,100,12))
        initial_conditions /= np.sum(initial_conditions)
        return growth_rates, A, initial_conditions

    def e_LV():
        """
        Food chain
        """
        growth_rates = np.array([1]*4 + [0.2]*4 + [-0.001]*4)

        # Inter-group competition with per-individual variability
        A[group_indices['Y'], group_indices['Z']] = 0.125 + np.random.normal(0, var,(group_size, group_size)) # Z on Y
        A[group_indices['X'], group_indices['Y']] = 0.125 + np.random.normal(0, var,(group_size, group_size)) # Y on X

        A[group_indices['Z'], group_indices['Y']] = -0.025 + np.random.normal(0, var, (group_size, group_size))  # Y on Z
        A[group_indices['Y'], group_indices['X']] = -0.025 + np.random.normal(0, var, (group_size, group_size))  # X on Y

        group_means = [0.25, 0.025, 0.025]
        for i, group_slice in enumerate(group_indices.values()):
            group = np.arange(*group_slice.indices(S))  # Convert slice to array of ints
            mean = group_means[i]
            block = mean + np.random.normal(0, var, (len(group), len(group)))
            A[np.ix_(group, group)] = block

        # Initial populations with variability per individual
        initial_conditions = np.concatenate([
            np.random.uniform(1,55,4),
            np.random.uniform(1,30, 4),
            np.random.uniform(1, 15, 4)
        ])
        initial_conditions /= np.sum(initial_conditions)

        return growth_rates, A, initial_conditions

    def b_LV():
        """
        Two predators one prey
        """
        growth_rates = np.array([1.0]*4 + [-0.001]*8)

        # Inter-group competition with per-individual variability
        A[group_indices['X'], group_indices['Z']] = 0.125 + np.random.normal(0, var,(group_size, group_size)) # Z on X
        A[group_indices['X'], group_indices['Y']] = 0.125 + np.random.normal(0, var,(group_size, group_size)) # Y on X

        A[group_indices['X'], group_indices['Z']] = -0.025 + np.random.normal(0, var,(group_size, group_size)) # Z on X
        A[group_indices['X'], group_indices['Y']] = -0.025 + np.random.normal(0, var,(group_size, group_size)) # Y on X

        group_means = [0.25, 0.025, 0.025]
        for i, group_slice in enumerate(group_indices.values()):
            group = np.arange(*group_slice.indices(S))  # Convert slice to array of ints
            mean = group_means[i]
            block = mean + np.random.normal(0, var, (len(group), len(group)))
            A[np.ix_(group, group)] = block

        # Initial populations with variability per individual
        initial_conditions = np.concatenate([
            np.random.uniform(1,65, 4),
            np.random.normal(1, 35, 8)
        ])
        initial_conditions /= np.sum(initial_conditions)

        return growth_rates, A, initial_conditions

    def c_LV():
        """
        One predator two prey
        """
        growth_rates = np.array([1.0] * 8 + [-0.001] * 4)

        # Inter-group competition with per-individual variability
        A[group_indices['X'], group_indices['Z']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # Z on X
        A[group_indices['Y'], group_indices['Z']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # Z on Y

        A[group_indices['Z'], group_indices['X']] = -0.025 + np.random.normal(0, var, (group_size, group_size))  # Z on X
        A[group_indices['Z'], group_indices['Y']] = -0.025 + np.random.normal(0, var, (group_size, group_size))  # Z on Y

        group_means = [0.25, 0.25, 0.025]
        for i, group_slice in enumerate(group_indices.values()):
            group = np.arange(*group_slice.indices(S))  # Convert slice to array of ints
            mean = group_means[i]
            block = mean + np.random.normal(0, var, (len(group), len(group)))
            A[np.ix_(group, group)] = block

        # Initial populations with variability per individual
        initial_conditions = np.concatenate([
            np.random.uniform(1,65, 8),
            np.random.uniform(1, 35, 4)
        ])
        initial_conditions = np.clip(initial_conditions, 0.01, None)
        initial_conditions /= np.sum(initial_conditions)

        return growth_rates, A, initial_conditions

    def f_LV():
        """
        Food chain with omnivory
        """
        growth_rates = np.array([1.0] * 4 + [0.2] * 4 + [-0.001] * 4)

        # Inter-group competition with per-individual variability
        A[group_indices['X'], group_indices['Z']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # Z on X
        A[group_indices['Y'], group_indices['Z']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # Z on Y
        A[group_indices['X'], group_indices['Y']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # Y on X

        A[group_indices['Z'], group_indices['X']] = -0.025 + np.random.normal(0, var, (group_size, group_size))  # Z on X
        A[group_indices['Z'], group_indices['Y']] = -0.025 + np.random.normal(0, var, (group_size, group_size))  # Z on Y
        A[group_indices['Y'], group_indices['X']] = -0.025 + np.random.normal(0, var, (group_size, group_size))  # Y on X

        group_means = [0.25, 0.025, 0.025]
        for i, group_slice in enumerate(group_indices.values()):
            group = np.arange(*group_slice.indices(S))  # Convert slice to array of ints
            mean = group_means[i]
            block = mean + np.random.normal(0, var, (len(group), len(group)))
            A[np.ix_(group, group)] = block

        # Initial populations with variability per individual
        initial_conditions = np.concatenate([
            np.random.uniform(1, 55, 4),
            np.random.uniform(1, 30, 4),
            np.random.uniform(1, 15, 4)
        ])
        initial_conditions /= np.sum(initial_conditions)

        return growth_rates, A, initial_conditions

    def d_LV():
        """food chain with cycle"""
        growth_rates = np.array([1.0] * 8 + [-0.001] * 4)

        # Inter-group competition with per-individual variability
        A[group_indices['X'], group_indices['Z']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # Z on Y
        A[group_indices['X'], group_indices['Y']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # Y on X
        A[group_indices['Z'], group_indices['X']] = 0.125 + np.random.normal(0, var, (group_size, group_size))  # X on Z

        A[group_indices['Z'], group_indices['X']] += -0.025 + np.random.normal(0, var, (group_size, group_size))  # Z on Y
        A[group_indices['Y'], group_indices['X']] += -0.025 + np.random.normal(0, var, (group_size, group_size))  # Y on X
        A[group_indices['X'], group_indices['Z']] += -0.025 + np.random.normal(0, var, (group_size, group_size))  # X on Z

        group_means = [0.25, 0.25, 0.025]
        for i, group_slice in enumerate(group_indices.values()):
            group = np.arange(*group_slice.indices(S))  # Convert slice to array of ints
            mean = group_means[i]
            block = mean + np.random.normal(0, var, (len(group), len(group)))
            A[np.ix_(group, group)] = block

        # Initial populations with variability per individual
        initial_conditions = np.concatenate([
            np.random.uniform(1, 55, 4),
            np.random.uniform(1, 30, 4),
            np.random.uniform(1, 15, 4)
        ])
        initial_conditions /= np.sum(initial_conditions)

        return growth_rates, A, initial_conditions

    def f_wrapped(t, n):
        return f(n, t, growth_rates, A)

    if model_func == "a":
        growth_rates, A, initial_conditions = a_LV()
    elif model_func == "b":
        growth_rates, A, initial_conditions = b_LV()
    elif model_func == "c":
        growth_rates, A, initial_conditions = c_LV()
    elif model_func == "d":
        growth_rates, A, initial_conditions = d_LV()
    elif model_func == "e":
        growth_rates, A, initial_conditions = e_LV()
    elif model_func == "f":
        growth_rates, A, initial_conditions = f_LV()
    else:
        print("Invalid model")

    # Solve stochastic ode but ensure that populations don't become non-negative
    tspan = np.linspace(0, T, int(T * 100))

    solution = solve_ivp(
        f_wrapped,
        [tspan[0], tspan[-1]],
        initial_conditions,
        t_eval=tspan,
        method='RK45'
    )

    solutions = solution.y.T

    # enforce lower bound
    solutions = np.where(solutions < 0.001, 0, solutions)

    # scale up populations
    solutions = solutions * N

    #plot_solutions(solutions, tspan, model_func)
    df = create_df(solutions)

    if var == 0.05:
        plot_solutions(solutions, solution.t, model_func)
        df.to_csv(f'../../data/LV_{model_func}_regression_library.csv', index=False)

    return df


def do_polynomial_regression(df, LV_model, var, regression_type='global', cluster=""):
    model = LinearRegression()

    # Separate target and features
    y = df['dn']
    X = df.drop(columns='dn')

    # Compute polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)

    try:
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(X.columns)
    except ValueError:
        print("Polynomial features failed")
        return y, None

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    # Fit model
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    # De-standardize coefficients
    beta_std = model.coef_
    mu = scaler.mean_
    sigma = scaler.scale_

    beta_orig = beta_std / sigma
    intercept_orig = model.intercept_ - np.sum((beta_std * mu) / sigma)

    # Combine into DataFrame
    coeff_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': beta_orig
    })

    # Add intercept as a separate row (optional but useful)
    coeff_df.loc[len(coeff_df)] = ['Intercept', intercept_orig]

    # Save
    if var == 0.05:
        coeff_df.to_csv(
            f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Data sets/LV/METimE_{LV_model}_dn_{regression_type}{cluster}.csv',
            index=False
        )

    return y, y_pred, coeff_df


def set_up_regression(df, var, N_clusters=None, LV_model='constant', regression_type="global", cluster=""):
    # Single observation interval
    all_census = sorted(df['census'].unique())
    reduced_census = deepcopy(all_census)

    # Filter to current censuses
    df_filtered = df[df['census'].isin(reduced_census)].copy().reset_index(drop=True)

    # Recompute dn, dN, dS
    df_deltas = compute_deltas(df_filtered, 'LV').reset_index(drop=True)
    df_deltas = compute_deltas(df_filtered, 'LV').reset_index(drop=True)

    if regression_type=="clustered":
        df_deltas = df_deltas.merge(N_clusters, on='census', how='left')

    # METimE regression
    cols_to_exclude = ['dN', 'n_next', 'N_next', 'S_next', 'dS', 'census', 'species', 'S_t']
    df_for_setup = df_deltas.drop(columns=cols_to_exclude)
    # X, y, census, species = polynomial_regression.set_up_library(df_for_setup, 3, False, False, False)
    y, y_pred, coeffs = do_polynomial_regression(df_for_setup, LV_model, var, regression_type, cluster)

    return y, y_pred, coeffs


def plot_observed_vs_predicted(obs, pred, title, species=None, save=False, dpi=300):
    """
    Plots observed vs. predicted values with optional coloring by species,
    styled for academic publications.

    Parameters:
    - obs: list or array of observed values
    - pred: list or array of predicted values
    - title: str, the title of the plot
    - species: list or array of species IDs (same length as obs/pred), optional
    - save_path: str, full path to save the figure (with .png or .pdf extension)
    - dpi: int, resolution of the saved figure
    - return_fig: bool, whether to return the figure object
    """
    sns.set(style='whitegrid', font_scale=1.3, rc={
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'legend.fontsize': 10,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16
    })

    obs = np.array(obs)
    pred = np.array(pred)
    r2 = r2_score(obs, pred)

    fig, ax = plt.subplots(figsize=(6, 6))

    if species is not None:
        species = np.array(species)
        unique_species = np.unique(species)
        palette = sns.color_palette("tab20", len(unique_species))
        for idx, sp in enumerate(unique_species):
            mask = species == sp
            ax.scatter(obs[mask], pred[mask],
                       label=f"{sp}", color=palette[idx],
                       alpha=0.7, edgecolors='k', linewidths=0.5, s=50)
        ax.legend(title='Species', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    else:
        ax.scatter(obs, pred, alpha=0.7, edgecolors='k', linewidths=0.5, s=50, color='steelblue')

    # 1:1 line
    min_val = min(np.min(obs), np.min(pred))
    max_val = max(np.max(obs), np.max(pred))
    padding = 0.05 * (max_val - min_val)  # 5% padding
    lims = [min_val - padding, max_val + padding]

    ax.plot(lims, lims, 'r--', linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    #ax.set_title(title)
    ax.set_xlabel(r"$\Delta n_{t+1} \ {\mathrm{(observed)}}$")
    ax.set_ylabel(r"$\Delta n_{t+1} \ {\mathrm{(predicted)}}$")

    # R² annotation
    ax.text(0.05, 0.95, f"$R^2$ = {r2:.2f}", transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", edgecolor='gray', alpha=0.8))

    plt.tight_layout()

    if save:
        save_path = "C://Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/LV/" + title + ".png"
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        plt.close(fig)
    else:
        plt.show()


# def get_cluster_state_variables(df):
#     """
#     For each time t, calculates the total number of individuals in each cluster
#     and adds these as new columns to the dataframe.
#
#     Assumes df contains at least: 't', 'cluster', and 'individuals' columns.
#     """
#     # Group by time and cluster, summing individuals
#     cluster_totals = df.groupby(['census', 'cluster'])['n'].sum().unstack(fill_value=0)
#
#     # Rename columns to reflect cluster totals
#     cluster_totals.columns = [f'N_{col}' for col in cluster_totals.columns]
#     cluster_totals = cluster_totals.reset_index()
#
#     return cluster_totals


def get_metrics(model, var, T=20, repetitions=100):
    from collections import defaultdict

    results_r2 = defaultdict(list)

    for _ in range(repetitions):
        df = three_groups_LV(model, T=T, var=var)

        # -------- GLOBAL REGRESSION --------
        obs, pred, _ = set_up_regression(df, var, LV_model=model, regression_type="global")
        obs = obs.tolist()
        pred = pred.tolist()
        r2 = r2_score(obs, pred)
        results_r2['global'].append(r2)

        # -------- SPECIES-SPECIFIC REGRESSION --------
        species_r2, species_adj_r2 = [], []
        for sp in df['species'].unique():
            df_sp = df[df['species'] == sp]
            obs, pred, _ = set_up_regression(df_sp, var, LV_model=model, regression_type="species-specific")
            obs = obs.tolist()
            pred = pred.tolist()
            r2 = r2_score(obs, pred)
            species_r2.append(r2)
        results_r2['species_specific'].append(np.nanmean(species_r2))

        # # -------- CLUSTERED REGRESSION --------
        # df['cluster'] = df['species'].apply(lambda x: 'x' if x < 10 else 'y' if x < 20 else 'z')
        # X_clustered = get_cluster_state_variables(df)
        # cluster_r2 = []
        # for cluster in df['cluster'].unique():
        #     df_cluster = df[df['cluster'] == cluster]
        #     obs, pred = set_up_regression(df_cluster.drop(columns=['cluster']), var, X_clustered, regression_type="clustered", cluster=cluster, LV_model=model)
        #     obs = obs.tolist()
        #     pred = pred.tolist()
        #     r2 = r2_score(obs, pred)
        #     cluster_r2.append(r2)
        # results_r2['clustered'].append(round(np.nanmean(cluster_r2), 3))

    # Convert to DataFrames: one row per treatment, one column per model-variance
    index = ['global', 'species_specific']
    avg_r2 = pd.DataFrame({f"{model},{var}": [np.nanmean(results_r2[k]) for k in index]}, index=index)

    return avg_r2