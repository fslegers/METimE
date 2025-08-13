import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import ElasticNet, LinearRegression, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import re

def calculate_vif(X):
    """Calculate VIF for each feature in a DataFrame."""
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def remove_high_vif_features(X, threshold=25.0):
    while True:
        vif = calculate_vif(X)
        max_vif = vif['VIF'].max()
        if max_vif > threshold:
            feature_to_drop = vif.sort_values('VIF', ascending=False).iloc[0]['feature']
            print(f"Dropping '{feature_to_drop}' with VIF = {max_vif:.2f}")
            X = X.drop(columns=[feature_to_drop])
        else:
            break
    return X

def do_polynomial_regression(df, target='dn', level='individuals', cluster='global'):
    if 'dn' not in df.columns and target == 'dn':
        # Step 1: Compute dn = n(t+1) - n(t) per species
        n_df = df[['Species_ID', 'census', 'n']].drop_duplicates()
        n_df = n_df.sort_values(['Species_ID', 'census'])
        n_df['dn'] = n_df.groupby('Species_ID')['n'].shift(-1) - n_df['n']

        # Step 2: Filter original df to only rows where dn is defined
        df = df.merge(n_df[['Species_ID', 'census', 'dn']], on=['Species_ID', 'census'], how='left')

    if 'de' not in df.columns and target == 'de':
        # Step 1: Compute dn = n(t+1) - n(t) per species
        e_df = df[['Tree_ID', 'census', 'e']].drop_duplicates()
        e_df = e_df.sort_values(['Tree_ID', 'census'])
        e_df['de'] = e_df.groupby('Tree_ID')['e'].shift(-1) - e_df['e']

        # Step 2: Filter original df to only rows where dn is defined
        df = df.merge(e_df[['Tree_ID', 'census', 'de']], on=['Tree_ID', 'census'], how='left')

    if target == 'dn':
        df = df.dropna(subset=['dn'])
        y = df['dn']
    else:
        df = df.dropna(subset=['de'])
        y = df['de']

    # Step 3: Prepare input features (drop ID and outcome columns)
    feature_cols = df.drop(columns=['Species_ID', 'census', 'dn'], errors='ignore')

    if level == 'individuals' and 'Tree_ID' in feature_cols.columns:
        feature_cols = feature_cols.drop(columns=['Tree_ID'])

    if 'de' in feature_cols.columns:
        feature_cols = feature_cols.drop(columns=['de'])

    for col in ['dN', 'dE', 'dS', 'de']:
        if col in feature_cols.columns:
            feature_cols = feature_cols.drop(columns=[col])

    if level != 'individuals' and cluster == 'global':
        # order columns e, n, S, N, E

    # Step 4: Compute polynomial features
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(feature_cols)
    feature_names = poly.get_feature_names_out(feature_cols.columns)
    X = pd.DataFrame(X_poly, columns=feature_names, index=df.index)
    del(X_poly)

    # Step 5: take mean over individuals per species
    if level == 'individuals':
        X['Species_ID'] = df['Species_ID']
        X['census'] = df['census']
        y.index = df.index  # ensure alignment
        X = X.groupby(['Species_ID', 'census']).mean() # TODO: check if this should be mean instead of sum
        y = y.groupby([df['Species_ID'], df['census']]).mean()
    else:
        X.index = df.index
        y.index = df.index

    cols_to_drop = [
        col for col in X.columns
        if pattern.search(col) or ('n' not in col and 'e' not in col)
    ]

    X = X.drop(columns=cols_to_drop)

    # # Step 5a: remove colinear features
    # X = remove_high_vif_features(X)

    # Step 6: Fit model and predict
    model = Lasso()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Step 7: Return predictions with IDs
    species_ID = X.reset_index()['Species_ID'] if level == 'individuals' else df['Species_ID']
    census = X.reset_index()['census'] if level == 'individuals' else df['census']

    # Save transition functions
    coeff_df = pd.DataFrame({'Feature': model.feature_names_in_,
                            'Coefficient': model.coef_})

    # coeff_df.to_csv(
    #     f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Data sets/BCI/METimE_{target}_{cluster}.csv',
    #     index=False)
    # coeff_df.to_csv(
    #     f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Data sets/BCI_quadrat_2/METimE_{target}_{cluster}.csv',
    #     index=False)
    coeff_df.to_csv(
        f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Data sets/simulated_BCI/METimE_{target}_{cluster}.csv',
        index=False)
    print(coeff_df)
    return y, y_pred, species_ID, census


# def do_dynaMETE_regression(df, target='dn', level='individuals', cluster='global'):
#     if 'dn' not in df.columns and target == 'dn':
#         # Step 1: Compute dn = n(t+1) - n(t) per species
#         n_df = df[['Species_ID', 'census', 'n']].drop_duplicates()
#         n_df = n_df.sort_values(['Species_ID', 'census'])
#         n_df['dn'] = n_df.groupby('Species_ID')['n'].shift(-1) - n_df['n']
#
#         # Step 2: Filter original df to only rows where dn is defined
#         df = df.merge(n_df[['Species_ID', 'census', 'dn']], on=['Species_ID', 'census'], how='left')
#
#     if 'de' not in df.columns and target == 'de':
#         # Step 1: Compute dn = n(t+1) - n(t) per species
#         e_df = df[['Tree_ID', 'census', 'e']].drop_duplicates()
#         e_df = e_df.sort_values(['Tree_ID', 'census'])
#         e_df['de'] = e_df.groupby('Tree_ID')['e'].shift(-1) - e_df['e']
#
#         # Step 2: Filter original df to only rows where dn is defined
#         df = df.merge(e_df[['Tree_ID', 'census', 'de']], on=['Tree_ID', 'census'], how='left')
#
#     if target == 'dn':
#         df = df.dropna(subset=['dn'])
#         y = df['dn']
#     else:
#         df = df.dropna(subset=['de'])
#         y = df['de']
#
#     # Step 3: Prepare input features (drop ID and outcome columns)
#     feature_cols = df.drop(columns=['Species_ID', 'census', 'dn'], errors='ignore')
#
#     if level == 'individuals' and 'Tree_ID' in feature_cols.columns:
#         feature_cols = feature_cols.drop(columns=['Tree_ID'])
#
#     if 'de' in feature_cols.columns:
#         feature_cols = feature_cols.drop(columns=['de', 'dS', 'dN', 'dE'])
#
#     # Step 4: Compute features
#     X = pd.DataFrame({
#         'n_e_neg_1_3': df['n'] * df['e'] ** (-1 / 3),
#         'n_e_neg_1_3_E': df['E'] * df['n'] * df['e'] ** (-1 / 3),
#         'n_div_N': df['n'] / df['N'],
#         'n_e_2_3': df['n'] * df['e'] ** (2 / 3),
#         'n_e': df['n'] * df['e']
#     }, index=df.index)
#
#     # Step 5: take mean over individuals per species
#     if level == 'individuals':
#         X['Species_ID'] = df['Species_ID']
#         X['census'] = df['census']
#         y.index = df.index  # ensure alignment
#         X = X.groupby(['Species_ID', 'census']).mean() # TODO: check if this should be mean instead of sum
#         y = y.groupby([df['Species_ID'], df['census']]).mean()
#     else:
#         X.index = df.index
#         y.index = df.index
#
#
#     # Step 5a: remove colinear features
#     X = remove_high_vif_features(X)
#
#     # Step 6: Fit model and predict
#     model = LinearRegression()
#     model.fit(X, y)
#     y_pred = model.predict(X)
#
#     # Step 7: Return predictions with IDs
#     species_ID = X.reset_index()['Species_ID'] if level == 'individuals' else df['Species_ID']
#     census = X.reset_index()['census'] if level == 'individuals' else df['census']
#
#     # Save transition functions
#     coeff_df = pd.DataFrame({'Feature': model.feature_names_in_,
#                             'Coefficient': model.coef_})
#
#     coeff_df.to_csv(f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Data sets/BCI/METimE_{target}_{cluster}.csv', index=False)
#
#     return y, y_pred, species_ID, census\


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
             ha='left', va='top', fontsize=16, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))

    # Labels and title
    plt.title(title)
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    #plt.savefig(f"C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/BCI/Empirical_BCI_Poly_Regression_{title}.png")


def add_clusters(df):
    cluster_info = pd.read_csv("C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Data sets/BCI/SpeciesID_to_Cluster.csv").drop_duplicates()
    cluster_info.rename({'SpeciesID': 'Species_ID'}, axis=1, inplace=True)
    cluster_info = cluster_info[['Species_ID', 'Cluster']]
    df = df.merge(cluster_info, on='Species_ID', how='left')
    df['Cluster'] = df['Cluster'].fillna(4)

    # Add cluster's state variables
    # For each cluster and census, calculate S, N and E and add these cluster values to all rows with the same census
    cluster_stats = df.groupby(['census', 'Cluster']).agg(
        S=('Species_ID', 'nunique'),
        N=('Tree_ID', 'nunique'),
        E=('e', 'sum')
    ).reset_index()

    # 2. Pivot to wide format so each cluster's state variables are columns
    wide_stats = cluster_stats.pivot(index='census', columns='Cluster')

    # Flatten MultiIndex columns
    wide_stats.columns = [f"{stat}_{cluster}" for stat, cluster in wide_stats.columns]

    # 3. Merge wide census-level stats back into the main DataFrame on 'census'
    df = df.merge(wide_stats, on='census', how='left')

    return df


if __name__ == '__main__':
    for target in ['dn', 'de']:
        data = 'BCI'

        # Simulated BCI data:
        df = pd.read_csv("../../data/simulated_BCI_regress_lib.csv")
        df = df.rename(columns={'t':'census'})

        # # Empirical BCI data (all):
        # df = pd.read_csv("../../data/BCI_regression_library.csv")
        # df = df.rename(columns={'species': 'Species_ID',
        #                         'TreeID': 'Tree_ID'})

        # IMPORTANT! Change the location where the coefficients are saved when a different data set is used!

        # # Empirical BCI data (per plot):
        # df = pd.read_csv("../../data/BCI_regression_library_quadrat_2.csv")
        # df = df.rename(columns={'species': 'Species_ID',
        #                         'TreeID': 'Tree_ID'})

        # All species together
        y_obs, y_pred, species_ID, census = do_polynomial_regression(df, target=target)
        plot_observed_vs_predicted(y_obs, y_pred, f"Global ({target})", species_ID)

        # # All species separately
        # all_obs, all_pred, all_species = [], [], []
        # for species in df['Species_ID'].unique():
        #     df_species = df[df['Species_ID'] == species]
        #     y_obs, y_pred, _, _ = do_polynomial_regression(df_species, target=target)
        #     all_obs.extend(y_obs)
        #     all_pred.extend(y_pred)
        #     all_species.extend([species] * len(y_obs))
        # plot_observed_vs_predicted(all_obs, all_pred, f"Species-specific ({target})", species=all_species)

        # Clusters
        df = add_clusters(df)
        all_obs, all_pred, all_clusters = [], [], []
        for cluster in df['Cluster'].unique():
            df_cluster = df[df['Cluster'] == cluster].drop(columns='Cluster')
            y_obs, y_pred, _, _ = do_polynomial_regression(df_cluster, target=target, cluster=cluster)
            all_obs.extend(y_obs)
            all_pred.extend(y_pred)
            all_clusters.extend([cluster] * len(y_obs))
            plot_observed_vs_predicted(y_obs, y_pred, f"Clustered_{cluster} ({target})", species=[cluster] * len(y_obs))
        plot_observed_vs_predicted(all_obs, all_pred, f"Clustered ({target})", species=all_clusters)

    # only having the first order of e doesn't give good results