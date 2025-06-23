import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from scipy.optimize import least_squares
from scipy.optimize import nnls
from matplotlib.patches import Patch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.METE_no_integrals import make_initial_guess, perform_optimization

def f(n, e, X, p):
    return (p['b'] - p['d'] * X['E']/p['Ec']) * n / e**(1/3) + p['m'] * n / X['N']

def h(n, e, X, p):
    return (p['w'] - p['d'] * X['E'] / p['Ec']) * n * e**(2/3) - p['w1'] * n * e / np.log(1 / X['beta'])**(2/3) + p['m'] * n / X['N']

def add_state_var(df):
    # Group by 'time' to calculate state variables
    state_vars = df.groupby('t').agg(
        N=('Tree_ID', pd.Series.nunique),
        S=('Species_ID', pd.Series.nunique),
        E=('e', 'sum')
    ).reset_index()

    # Merge state variables back into original DataFrame
    df = df.merge(state_vars, on='t', how='left')

    return df

def add_beta(df):
    beta_cache = {}

    # Iterate over unique (S_t, N_t, E_t) combinations
    for (S, N, E) in df[['S', 'N', 'E']].drop_duplicates().itertuples(index=False):

        # Compute theoretical guess
        theoretical_guess = make_initial_guess([S, N, E])

        # Perform optimization to find l1 and l2
        l1, l2 = perform_optimization([theoretical_guess], [S, N, E])
        beta = min(l1 + l2, 0.99)

        # Store results in the cache
        beta_cache[(S, N, E)] = (l1, l2, beta)

    # Map computed values back to the original DataFrame
    df[['l1', 'l2', 'beta']] = df.apply(
        lambda row: beta_cache[(row['S'], row['N'], row['E'])], axis=1, result_type="expand"
    )

    df = df.drop(columns=['l1', 'l2'], axis=1)
    return df

def scaled_residuals(scaled_params, df, df_grouped, param_scale):
    params = scaled_params * param_scale
    return residuals(params, df, df_grouped)

def residuals(params, df, df_grouped):
    # Step 1: Compute f and h at individual level
    df = df.copy()

    p = {'b': params[0],
         'd': params[1],
         'Ec': params[2],
         'm': params[3],
         'w': params[4],
         'w1': params[5]}

    X = {
        'S': df['S'],
        'N': df['N'],
        'E': df['E'],
        'beta': df['beta']
    }

    df['f_pred'] = f(df['n'], df['e'], X, p)
    df['h_pred'] = h(df['n'], df['e'], X, p)

    # Step 2: Aggregate to species level
    pred_grouped = df.groupby(['Species_ID', 't'])[['f_pred', 'h_pred']].mean().reset_index()
    merged = df_grouped.merge(pred_grouped, on=['Species_ID', 't'], how='left')

    # # Step 4: Compute residuals
    # res_f = np.abs(merged['dn'] - merged['f_pred'])
    # res_h = np.abs(merged['dm'] - merged['h_pred'])
    #
    # return np.concatenate([res_f.values, res_h.values])

    res_f = (merged['dn'] - merged['f_pred'])**2
    res_h = (merged['de'] - merged['h_pred'])**2
    #res_h = (merged['dn'] - merged['f_pred'])*0

    # Scale residuals by standard deviation to equalize their influence
    res_f_std = np.std(merged['dn']) if np.std(merged['dn']) != 0 else 1
    res_h_std = np.std(merged['de']) if np.std(merged['de']) != 0 else 1

    res_f_scaled = res_f / res_f_std
    res_h_scaled = res_h / res_h_std

    return np.concatenate([res_f_scaled.values, res_h_scaled.values])

def do_dynaMETE_regression(df, initial_guess, cluster):
    # Step 2: Group everything to species level
    df_grouped = df.groupby(['Species_ID', 't'], as_index=False).mean().drop(['Tree_ID'], axis=1)

    # Step 3: Calculate dn and dm
    df_grouped = df_grouped.sort_values(['Species_ID', 't'])
    df_grouped['dn'] = df_grouped.groupby('Species_ID')['n'].shift(-1) - df_grouped['n']
    df_grouped['de'] = df_grouped.groupby('Species_ID')['e'].shift(-1) - df_grouped['e']
    df_grouped = df_grouped.dropna()

    # # Step 4: Fit model
    # result = least_squares(fun=residuals, x0=initial_guess, verbose=2, xtol=1e-16, ftol=1e-16, loss='soft_l1', args=(df, df_grouped))
    # params = result.x

    # PARAMETER SCALING
    param_scale = np.array([0.1, 0.1, 1e7, 100, 1, 0.1])
    initial_guess_scaled = np.array(initial_guess) / param_scale

    # Optimizer
    result = least_squares(
        fun=scaled_residuals,
        x0=initial_guess_scaled,
        args=(df, df_grouped, param_scale),
        verbose=2,
        loss='soft_l1',
        max_nfev=20000,
        x_scale='jac'  # optional but helps
    )

    # Scale back to natural parameter values
    params = result.x * param_scale

    optimized_params = {'b': params[0],
         'd': params[1],
         'Ec': params[2],
         'm': params[3],
         'w': params[4],
         'w1': params[5]}
    print(optimized_params)

    # Save results
    optimized_params_df = pd.DataFrame([optimized_params])
    optimized_params_df.to_csv(f'C:/Users/5605407/Documents/PhD/Chapter_2/Data sets/BCI/dynaMETE_{cluster}.csv', index=False)

    # Step 5: Make predictions using optimized parameters
    df['f_pred'] = f(df['n'], df['e'], {'S': df['S'], 'N': df['N'], 'E': df['E'], 'beta': df['beta']}, optimized_params)
    df['h_pred'] = h(df['n'], df['e'], {'S': df['S'], 'N': df['N'], 'E': df['E'], 'beta': df['beta']}, optimized_params)

    # Group predictions to species-time level and reset index
    pred_df = df.groupby(['Species_ID', 't'])[['f_pred', 'h_pred']].mean().reset_index()

    # Now align with df_grouped by merging (guaranteed order match)
    df_grouped = df_grouped.merge(pred_df, on=['Species_ID', 't'], how='left')

    # Rename columns for clarity
    df_grouped = df_grouped.rename(columns={
        'f_pred': 'pred_dn',
        'h_pred': 'pred_dm'
    })

    return df_grouped['dn'], df_grouped['de'], df_grouped['pred_dn'], df_grouped['pred_dm'], df_grouped['Species_ID'], df_grouped['t'], list(params)


def plot_observed_vs_predicted(obs, pred, title, species=None):
    """
    Plots observed vs. predicted values, optionally colored by species.
    Displays R² on the plot and ensures a rectangular plot layout.

    Parameters:
    - obs: list or array of observed values
    - pred: list or array of predicted values
    - title: str, the title of the plot
    - species: list or array of species IDs (same length as obs/pred), optional
    """
    obs = np.array(obs)
    pred = np.array(pred)

    plt.figure(figsize=(8, 6))  # Set a consistent rectangular size (width > height)

    if species is not None:
        species = np.array(species)
        unique_species = np.unique(species)
        cmap = plt.get_cmap("tab20", len(unique_species))
        for idx, sp in enumerate(unique_species):
            mask = species == sp
            plt.scatter(obs[mask], pred[mask],
                        label=f"{sp}", color=cmap(idx), alpha=0.6, edgecolors='k')
        #plt.legend(title="Species", fontsize="small", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(obs, pred, alpha=0.6, edgecolors='k')

    # Plot 1:1 line
    lims = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
    plt.plot(lims, lims, 'r--', linewidth=1)

    # R² calculation
    r2 = r2_score(obs, pred)
    plt.text(0.05, 0.95, f"$R^2$ = {r2:.2f}", transform=plt.gca().transAxes,
             ha='left', va='top', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))

    # Labels and title
    plt.title(title)
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def add_clusters(df):
    cluster_info = pd.read_csv("C:/Users/5605407/Documents/PhD/Chapter_2/Data sets/BCI/SpeciesID_to_Cluster.csv").drop_duplicates()
    cluster_info.rename({'SpeciesID': 'Species_ID'}, axis=1, inplace=True)
    cluster_info = cluster_info[['Species_ID', 'Cluster']]
    df = df.merge(cluster_info, on='Species_ID', how='left')
    df['Cluster'] = df['Cluster'].fillna(4)

    # Add cluster's state variables
    # For each cluster and census, calculate S, N and E and add these cluster values to all rows with the same census
    cluster_stats = df.groupby(['t', 'Cluster']).agg(
        S=('Species_ID', 'nunique'),
        N=('Tree_ID', 'nunique'),
        E=('e', 'sum')
    ).reset_index()

    # 2. Pivot to wide format so each cluster's state variables are columns
    wide_stats = cluster_stats.pivot(index='t', columns='Cluster')

    # Flatten MultiIndex columns
    wide_stats.columns = [f"{stat}_{cluster}" for stat, cluster in wide_stats.columns]

    # 3. Merge wide census-level stats back into the main DataFrame on 'census'
    df = df.merge(wide_stats, on='t', how='left')

    return df


def plot_solutions(df, save_dir="C://Users/5605407/Documents/PhD/Chapter_2/Results/BCI"):
    # Setup
    sns.set(style="whitegrid", context="talk")
    group_colors = sns.color_palette("Set1", 4)
    cluster_labels = sorted(df['Cluster'].unique())

    # --------- Plot all species together ----------
    plt.figure(figsize=(14, 7))
    for cluster in cluster_labels:
        cluster_df = df[df['Cluster'] == cluster]
        for species in cluster_df['Species_ID'].unique():
            species_df = cluster_df[cluster_df['Species_ID'] == species]
            mean_df = species_df.groupby('t')['e'].mean().reset_index()
            # plt.plot(species_df['t'], species_df['e'], lw=1.5, alpha=0.8, color=group_colors[cluster - 1])
            plt.plot(mean_df['t'], mean_df['e'], lw=1.5, alpha=0.8, color=group_colors[cluster - 1])

    # Legend
    legend_handles = [Patch(color=group_colors[i - 1], label=f"Cluster {i}") for i in cluster_labels]
    plt.legend(handles=legend_handles, title="Cluster")
    plt.xlabel("Census", fontsize=18)
    #plt.ylabel("Population Size", fontsize=18)
    plt.ylabel("Average metabolic rate", fontsize=18)
    #plt.title("Population Dynamics of All Species")
    plt.grid(True, linestyle='--', linewidth=0.5)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "e_dynamics_all_species.png"))
    plt.close()

    # --------- Plot one figure per cluster ----------
    for cluster in cluster_labels:
        plt.figure(figsize=(14, 7))
        cluster_df = df[df['Cluster'] == cluster]
        for species in cluster_df['Species_ID'].unique():
            species_df = cluster_df[cluster_df['Species_ID'] == species]
            mean_df = species_df.groupby('t')['e'].mean().reset_index()
            #plt.plot(species_df['t'], species_df['e'], lw=1.5, alpha=0.8, color=group_colors[cluster - 1])
            plt.plot(mean_df['t'], mean_df['e'], lw=1.5, alpha=0.8, color=group_colors[cluster - 1])

        plt.xlabel("Census", fontsize=18)
        #plt.ylabel("Population Size", fontsize=18)
        plt.ylabel("Average metabolic rate", fontsize=18)
        #plt.title(f"Population Dynamics - Cluster {cluster}")
        plt.grid(True, linestyle='--', linewidth=0.5)
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"e_dynamics_cluster_{cluster}.png"))
        plt.close()

# def k_means_clustering(df, ncluster):
#     # Calculate dn
#     n_df = df[['Species_ID', 't', 'n']].drop_duplicates()
#     n_df = n_df.sort_values(['Species_ID', 't'])
#     n_df['dn'] = n_df.groupby('Species_ID')['n'].shift(-1) - n_df['n']
#
#     # Compute summary stats per species
#     summary = n_df.groupby('Species_ID')['dn'].agg(['mean', 'std', 'median']).reset_index()
#
#     # Flatten MultiIndex columns
#     summary.columns = ['Species_ID'] + [f"{'dn'}_{stat}" for stat in ['mean', 'std', 'median']]
#     summary.fillna(0, inplace=True)
#
#     # Step 3: Standardize the features
#     X = summary.drop(columns='Species_ID')
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # Step 4: Run K-Means Clustering (choose 2 or 3 clusters)
#     kmeans = KMeans(n_clusters=ncluster, random_state=42)
#     summary['cluster'] = kmeans.fit_predict(X_scaled)
#
#     # Visualize clusters
#     sns.scatterplot(data=summary, x='dn_mean', y='dn_std', hue='cluster', palette='viridis')
#     plt.title('Species Clustering Based on Summary Stats')
#     plt.xlabel('Mean dn')
#     plt.ylabel('Std Dev dn')
#     plt.show()
#
#     # Step 6: Merge cluster labels back to the original DataFrame
#     df_with_clusters = df.merge(summary[['Species_ID', 'cluster']], on='Species_ID', how='left')
#
#     return df_with_clusters

# def make_initial_guess(df):
#     # Compute b, d, m
#     df['b'] = df['n'] * df['e'] ** (-1 / 3)
#     df['d'] = df['E'] / (2 * 10 ** 7) * df['n'] * df['e'] ** (-1 / 3)
#     df['m'] = df['n'] / df['N']
#
#     # Drop duplicates for (species_ID, t, n, N)
#     df_n_m = df[['Species_ID', 't', 'n', 'N']].drop_duplicates()
#
#     # Add average of e per species_ID
#     avg_e = df.groupby('Species_ID')['e'].mean().reset_index().rename(columns={'e': 'avg_e'})
#     df_n_m = df_n_m.merge(avg_e, on='Species_ID', how='left')
#
#     # Group by species_ID and t and sum b, d, m
#     df_grouped = df[['Species_ID', 't', 'b', 'd', 'm']].groupby(['Species_ID', 't']).sum().reset_index()
#
#     # Prepare data for linear regression
#     X = df_grouped[['b', 'd', 'm']]
#     y = df_n_m['n']
#
#     coef, rnorm = nnls(X, y)
#
#     # You can optionally return the model coefficients
#     return coef


if __name__ == '__main__':

    data = 'BCI'

    # Simulated BCI
    # df = pd.read_csv("C:/Users/5605407/Documents/PhD/Chapter_2/Results/BCI/simulated_dynaMETE_snapshots.csv")

    # Empirical BCI data:
    df = pd.read_csv("../../data/BCI_regression_library.csv")
    df = df.rename(columns={'species': 'Species_ID',
                            'TreeID': 'Tree_ID',
                            'S_t': 'S',
                            'N_t': 'N',
                            'E_t': 'E',
                            'census': 't'})

    initial_guess = [0.2, 0.2, 2 * 10 ** 7, 437.3, 1.0, 0.42]

    # Add beta
    df = add_beta(df)

    # All species together
    obs_dn, obs_dm, pred_dn, pred_dm, species_ID, census, _ = do_dynaMETE_regression(df, initial_guess, 'global')
    plot_observed_vs_predicted(obs_dn, pred_dn / df['N'].mean(), "dn", species_ID)
    plot_observed_vs_predicted(obs_dm, pred_dm / df['N'].mean(), "dm", species_ID)

    # All species separately
    all_obs_dn, all_pred_dn, all_obs_dm, all_pred_dm, all_species = [], [], [], [], []
    for species in df['Species_ID'].unique():
        df_species = df[df['Species_ID'] == species]
        obs_dn, obs_dm, pred_dn, pred_dm, _, _, _ = do_dynaMETE_regression(df_species, initial_guess, f'species_specific_{species}')
        all_obs_dn.extend(obs_dn)
        all_pred_dn.extend(pred_dn)
        all_obs_dm.extend(obs_dm)
        all_pred_dm.extend(pred_dm)
        all_species.extend([species] * len(obs_dn))
    plot_observed_vs_predicted(all_obs_dn, all_pred_dn, "Species-specific (dn)", species=all_species)
    plot_observed_vs_predicted(all_obs_dm, all_pred_dm, "Species-specific (dm)", species=all_species)

    # Clusters
    df = add_clusters(df)
    all_obs_dn, all_pred_dn, all_obs_dm, all_pred_dm, all_clusters = [], [], [], [], []
    for cluster in df['Cluster'].unique():
        df_cluster = df[df['Cluster'] == cluster].drop(columns='Cluster')
        obs_dn, obs_dm, pred_dn, pred_dm, _, _, _ = do_dynaMETE_regression(df_cluster, initial_guess, f'clustered_{cluster}')
        all_obs_dn.extend(obs_dn)
        all_pred_dn.extend(pred_dn)
        all_obs_dm.extend(obs_dm)
        all_pred_dm.extend(pred_dm)
        all_clusters.extend([cluster] * len(obs_dn))
        plot_observed_vs_predicted(obs_dn, pred_dn, f"Clustered (dn) cluster {cluster}", species=[cluster] * len(obs_dn))
        plot_observed_vs_predicted(obs_dm, pred_dm, f"Clustered (dm) cluster {cluster}", species=[cluster] * len(obs_dn))
    plot_observed_vs_predicted(all_obs_dn, all_pred_dn, "Clustered (dn)", species=all_clusters)
    plot_observed_vs_predicted(all_obs_dm, all_pred_dm, "Clustered (dm)", species=all_clusters)