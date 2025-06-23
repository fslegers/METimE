import pandas as pd
import numpy as np
from copy import deepcopy
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.parametrize_transition_functions import polynomial_regression


def compute_deltas(df, data_set): # TODO: are we losing too many rows here?
    if data_set == "LV":
        df = df.sort_values(['species', 'census'])
        df['n_next'] = df.groupby('species')['n'].shift(-1)
        df['N_next'] = df.groupby('species')['N_t'].shift(-1)
        df['S_next'] = df.groupby('species')['S_t'].shift(-1)

        df['dn'] = df['n_next'] - df['n']
        df['dN'] = df['N_next'] - df['N_t']
        df['dS'] = df['S_next'] - df['S_t']
    else:
        df = df.sort_values(['species', 'treeID', 'census'])
        df['n_next'] = df.groupby('species')['n'].shift(-1)
        df['e_next'] = df.groupby('species')['e'].shift(-1)
        df['N_next'] = df.groupby('species')['N_t'].shift(-1)
        df['S_next'] = df.groupby('species')['S_t'].shift(-1)
        df['E_next'] = df.groupby('species')['E_t'].shift(-1)

        df['dn'] = df['n_next'] - df['n']
        df['de'] = df['e_next'] - df['e']
        df['dN'] = df['N_next'] - df['N_t']
        df['dS'] = df['S_next'] - df['S_t']
        df['dE'] = df['E_next'] - df['E_t']

    df.dropna(inplace=True)

    return df


def load_data(data_set):
    if data_set == "LV":
        df = pd.read_csv('../data/LV.csv')
    else:
        df = pd.read_csv(r"C:/Users/5605407/Documents/PhD/Chapter_2/Results/simulated_dynaMETE.csv")

        # Create the vector of observed years starting from min(df['t'])
        t_observed = np.arange(df['t'].min(), df['t'].max(), (df['t'].max() - df['t'].min())/100)

        # Find the closest available `t` in df for each `t_observed`
        df_sorted = np.sort(df['t'])
        selected_t = np.array([df_sorted[df_sorted <= t].max() if np.any(df_sorted <= t) else None for t in t_observed])

        # use this t-vector to select rows from df
        df = df[df['t'].isin(selected_t)]
        df['census'] = df['t'].rank(method="dense").astype(int)
        df = df.drop(columns=['t'])

    return df


def do_repeated_regression(df):
    iteration = 1

    while len(reduced_census) >= 2:
        print(f"\n--- Iteration {iteration} ---")
        print(f"Censuses used: {len(reduced_census)}")

        # Filter to current censuses
        df_filtered = df[df['census'].isin(reduced_census)].copy().reset_index(drop=True)

        # Recompute dn, dN, dS
        df_deltas = compute_deltas(df_filtered, data_set).reset_index(drop=True)

        # METimE regression
        if data_set == "BCI":
            cols_to_exclude = ['species', 'treeID', 'dN', 'dE', 'n_next', 'e_next', 'N_next', 'S_next', 'E_next']
        else:
            cols_to_exclude = ['species', 'dN', 'n_next', 'N_next', 'S_next']
        df_for_setup = df_deltas.drop(columns=cols_to_exclude)
        X, y, census = polynomial_regression.set_up_library(df_for_setup, 3, False, False, False)
        _ = polynomial_regression.METimE(X, y, census,
                                         fig_title=f"simulated_METimE_interval_{iteration}")

        # Update census list by removing every second census
        reduced_census = reduced_census[::2]
        iteration += 1


if __name__ == '__main__':

    # Load data
    data_set = "BCI"
    df = load_data(data_set)
    #df = load_data("LV")
    all_census = sorted(df['census'].unique())

    reduced_census = deepcopy(all_census)

    # # For BCI data, already remove some censuses
    # df_filtered = df[df['census'].isin(reduced_census)].copy().reset_index(drop=True)
    # reduced_census = reduced_census[::4]
    iteration = 1

    while len(reduced_census) >= 2:
        print(f"\n--- Iteration {iteration} ---")
        print(f"Censuses used: {len(reduced_census)}")

        # Filter to current censuses
        df_filtered = df[df['census'].isin(reduced_census)].copy().reset_index(drop=True)

        # Recompute dn, dN, dS
        df_deltas = compute_deltas(df_filtered, data_set).reset_index(drop=True)

        # METimE regression
        if data_set == "BCI":
            cols_to_exclude = ['species', 'treeID', 'dN', 'dE', 'n_next', 'e_next', 'N_next', 'S_next', 'E_next']
        else:
            cols_to_exclude = ['species', 'dN', 'n_next', 'N_next', 'S_next']
        df_for_setup = df_deltas.drop(columns=cols_to_exclude)
        X, y, census = polynomial_regression.set_up_library(df_for_setup, 3, False, False, False)
        transition_functions = polynomial_regression.METimE(X, y, census, fig_title=f"simulated_METimE_interval_{iteration}")

        # Update census list by removing every second census
        reduced_census = reduced_census[::2]
        iteration += 1