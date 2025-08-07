import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.MaxEnt_inference import zero_point_METE
import sys
import os
import random

from src.MaxEnt_inference import empirical_METimE_quadrat as METimE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def partition_function_given_n(X, n, lambdas):
    return (np.exp(-lambdas[0] * n) - np.exp(-lambdas[0] * n - X['E'] * lambdas[1] * n)) / (lambdas[1] * n)


def get_SAD(lambdas, X):
    """ Calculates species abundance distribution from Lagrange multipliers """
    l1, l2 = lambdas
    S, N, E = int(X['S']), int(X['N']), X['E']
    n = np.arange(1, N + 1)
    beta = l1 + E * l2
    Z = -1/l2 * sum(1/n * (np.exp(-n * beta) - np.exp(-l1 * n)))
    denom = Z * l2 * n
    num = np.exp(-l1 * n) * (1 - np.exp(- E * l2 * n))
    sad = num / denom
    sad = sad / sum(sad)
    return sad


def cum_SAD(lambdas, X):
    sad = get_SAD(lambdas, X)
    cum_sad = []

    p = 0
    for n in range(int(X['N'])):
        p += sad[n]
        cum_sad.append(p)

    return cum_sad


def sample_community(X):
    """ Samples community that follows abundance and metabolic rate distributions as predicted by METE """
    # Add beta value
    beta_constraint = lambda b, X: b * np.log(1 / (1 - np.exp(-b))) - X['S'] / X['N']
    beta = fsolve(beta_constraint, 0.0001, args=X)[0]
    X['beta'] = beta

    # Find Lagrange multipliers
    lambdas = zero_point_METE.make_initial_guess(X)
    lambdas, _ = zero_point_METE.find_roots(lambdas,
                                            state_variables=X,
                                            method="METE")

    # Sample species
    p = np.random.uniform(0, 1, X['S'])
    cum_sad = cum_SAD(lambdas, X)

    populations = []
    for prob in p:
        n = int(np.searchsorted(cum_sad, prob) + 1)
        populations.append(n)
    species_indices = np.cumsum(populations)
    species_indices = np.concatenate(([0], species_indices))

    # Sample individual metabolic rates
    individuals = []
    for pop in populations:
        Z_n = partition_function_given_n(X, pop, lambdas)
        CDF_inverse = lambda u: -(np.log(1 - Z_n * lambdas[1] * n * u * np.exp(lambdas[0] * n)))/(lambdas[1] * n)
        u_samples = np.random.uniform(0, 1, pop)
        samples = [float(CDF_inverse(u).real) for u in u_samples] # checked: there is no imaginary part
        while np.isnan(samples).any():
            u_samples = np.random.uniform(0, min(1, 1/(Z_n * lambdas[1] * n * np.exp(lambdas[0] * n))), pop)
            samples = [float(CDF_inverse(u).real) for u in u_samples]
        individuals += samples
    tree_id_list = list(range(len(individuals)))

    return individuals, species_indices, tree_id_list


def update_metabolic_rates(e, X, dt, param):
    """ Euler method to update metabolic rates """
    de_dt = np.maximum(0, param['w'] * e ** (2 / 3) - param['w1'] * e)
    #de_dt = param['w'] * e ** (2 / 3) - param['w1'] * e
    new_e = e + dt * de_dt
    X['E'] = np.sum(new_e)
    return new_e, X


def update_event_rates(species_ids, abundances, metabolic_rates, X, p):
    """ Compute birth and death rates """
    n = np.array([abundances[sp] for sp in species_ids])
    birth_rates = p['b'] * n * (metabolic_rates ** (-1/3))
    death_rates = (p['d0'] + p['d1'] * n + p['d'] * (X['E'] / p['Ec'])) * n * (metabolic_rates ** (-1/3))
    migration_rates = p['m'] * np.array(list(abundances.values())) / X['N']
    R = birth_rates.sum() + death_rates.sum() + migration_rates.sum()
    return birth_rates, death_rates, migration_rates, R


def what_event_happened(birth_rates, death_rates, migration_rates, R, q):
    event_rates = np.concatenate([birth_rates, death_rates, migration_rates])
    cumulative_rates = np.cumsum(event_rates)

    index = np.searchsorted(cumulative_rates, q * R)

    if index < len(birth_rates):
        return ('birth', index)
    elif index < len(birth_rates) + len(death_rates):
        return ('death', index - len(birth_rates))
    else:
        return ('migration', index - len(birth_rates) - len(death_rates))


def perform_event(tree_ids, species_ids, metabolic_rates, abundances, next_tree_id, X, event_info, params):
    event_type, idx = event_info

    if (event_type == 'birth'):
        tree_ids = np.append(tree_ids, next_tree_id)
        species_ids = np.append(species_ids, species_ids[idx])
        metabolic_rates = np.append(metabolic_rates, 1.0)
        abundances[species_ids[idx]] += 1
        next_tree_id += 1
        X['N'] += 1
        X['E'] += 1

    elif event_type == 'death':
        tree_ids = np.delete(tree_ids, idx)
        abundances[species_ids[idx]] -= 1
        if abundances[species_ids[idx]] == 0:
            del abundances[species_ids[idx]]
            X['S'] -= 1
        species_ids = np.delete(species_ids, idx)
        X['E'] -= metabolic_rates[idx]
        X['N'] -= 1
        metabolic_rates = np.delete(metabolic_rates, idx)

    else: # Migration
        prob_new_species = np.exp(-params['mu_meta'] * X['S'] - np.euler_gamma)
        if np.random.rand() < prob_new_species:
            # Create a new species
            new_species_id = max(species_ids) + 1 if len(species_ids) > 0 else 0
            tree_ids = np.append(tree_ids, next_tree_id)
            species_ids = np.append(species_ids, new_species_id)
            metabolic_rates = np.append(metabolic_rates, 1.0)
            abundances[new_species_id] = 1
            X['N'] += 1
            X['S'] += 1
            X['E'] += 1.0
            next_tree_id += 1
        else:
            # Add to existing species (species_ids[idx])
            tree_ids = np.append(tree_ids, next_tree_id)
            species_ids = np.append(species_ids, species_ids[idx])
            metabolic_rates = np.append(metabolic_rates, 1.0)
            abundances[species_ids[idx]] += 1
            X['N'] += 1
            X['E'] += 1.0
            next_tree_id += 1

    return tree_ids, species_ids, metabolic_rates, abundances, next_tree_id, X


# def plot_community_histograms(community, X, t):
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#
#     # Histogram for 'e'
#     axes[0].hist(community['e'], bins=20, color='skyblue', edgecolor='black')
#     axes[0].set_title('Histogram of e')
#     axes[0].set_xlabel('e')
#     axes[0].set_ylabel('Frequency')
#
#     # Histogram for 'n'
#     axes[1].hist(community['n'], bins=20, color='salmon', edgecolor='black')
#     axes[1].set_title('Histogram of n')
#     axes[1].set_xlabel('n')
#     axes[1].set_ylabel('Frequency')
#
#     # Main title with observation time
#     fig.suptitle(f'Community Metabolic State at Time {t}', fontsize=14)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()


def gillespie(metabolic_rates, species_indices, tree_id_list, X, p, t_max, max_iter, obs_interval, save_final_state=False):
    # some preparations
    species_ids = np.zeros(int(species_indices[-1]), dtype=int)
    for i in range(1, len(species_indices)):
        start = species_indices[i - 1]
        end = species_indices[i]
        species_ids[start:end] = i

    observation_times = np.arange(0, t_max, obs_interval)
    obs_pointer = 0

    # # create df of individual trees
    # community = pd.DataFrame({
    #     'Tree_ID': tree_id_list,
    #     'Species_ID': species_ids,
    #     'e': metabolic_rates
    # })
    # community['n'] = community['Species_ID'].map(community['Species_ID'].value_counts())

    # Initialization
    tree_ids = np.array(tree_id_list)
    species_ids = np.array(species_ids)
    metabolic_rates = np.array(metabolic_rates)
    abundances = dict(Counter(species_ids))

    next_tree_id = tree_ids.max() + 1

    # Compute state variables and event rates
    X = {'S':len(np.unique(species_ids)), 'N': len(np.unique(tree_ids)), 'E': sum(metabolic_rates)}
    birth_rates, death_rates, migration_rates, R = update_event_rates(species_ids, abundances, metabolic_rates, X, p)

    # Store snapshots here
    snapshots = []

    # Start simulation
    t = 0
    n_iter = 0
    while t < t_max and n_iter < max_iter:
        # Sample event time
        u = np.random.uniform(0, 1)
        time_until_event = -np.log(u) / R
        t += time_until_event
        n_iter += 1
        #print(n_iter, t)

        # In case the event happens *after* the current observation time, save a snapshot of the community
        while obs_pointer < len(observation_times) and t > observation_times[obs_pointer]:
            # Save snapshot
            snapshots.append({
                't': observation_times[obs_pointer],
                'species_ids': species_ids,
                'tree_ids': tree_ids,
                'S': X['S'],
                'N': X['N'],
                'E': X['E'],
                'n': np.array([abundances[sp] for sp in species_ids]),
                'e': metabolic_rates.copy()
            })

            # Plot histogram of n and e
            #plot_community_histograms(community, X, observation_times[obs_pointer])

            # Progress observation time
            obs_pointer += 1

        # if obs_pointer >= len(observation_times):
        #     break

        # Update individual metabolic rates
        metabolic_rates, X = update_metabolic_rates(metabolic_rates, X, time_until_event, p)

        # Update event rates based on new metabolic rates e
        birth_rates, death_rates, migration_rates, R = update_event_rates(species_ids, abundances, metabolic_rates, X, p)

        # Determine what event (birth or death) happened
        q = np.random.uniform(0, 1)
        event = what_event_happened(birth_rates, death_rates, migration_rates, R, q)
        tree_ids, species_ids, metabolic_rates, abundances, next_tree_id, X = perform_event(tree_ids, species_ids, metabolic_rates, abundances, next_tree_id, X, event, p)

    # Flatten list of snapshowts with one row per individual per time step              # TODO: should also save species_id and tree_id!
    rows = []
    for snap in snapshots:
        n_individuals = len(snap['e'])
        for i in range(n_individuals):
            rows.append({
                't': snap['t'],
                'species_id': snap['species_ids'][i],
                'tree_id': snap['tree_ids'][i],
                'S': snap['S'],
                'N': snap['N'],
                'E': snap['E'],
                'n': snap['n'][i],
                'e': snap['e'][i]
            })
    df = pd.DataFrame(rows)

    if save_final_state:
        output_file = "C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/BCI/simulated_dynaMETE_snapshots.csv"
        df.to_csv(output_file, index=False)

        # Save final community state to resume later
        final_state_file = "C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/BCI/final_community_state.npz"

        # Reconstruct species_indices from species_ids
        unique_species, counts = np.unique(species_ids, return_counts=True)
        species_indices_final = np.cumsum(np.insert(counts, 0, 0))  # like original format

        np.savez(
            final_state_file,
            metabolic_rates=metabolic_rates,
            species_indices=species_indices_final,
            tree_id_list=tree_ids,
            X=np.array([X['S'], X['N'], X['E']])
        )

        # Save param dictionary as JSON
        param_file = final_state_file.replace(".npz", "_param.json")
        with open(param_file, 'w') as f:
            json.dump(p, f)

    return df


def plot_state_var(df):
    # Create three horizontally aligned subplots
    fig, axes = plt.subplots(ncols=3, figsize=(18, 5), sharex=True)

    # Plot S vs time
    axes[0].plot(df['t'], df['S'], color='tab:blue', marker='o')
    axes[0].set_title('S (Entropy) vs Time')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('S (Entropy)')
    axes[0].grid(True)

    # Plot N vs time
    axes[1].plot(df['t'], df['N'], color='tab:green', marker='s')
    axes[1].set_title('N (Abundance) vs Time')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('N (Abundance)')
    axes[1].grid(True)

    # Plot E vs time
    axes[2].plot(df['t'], df['E'], color='tab:red', marker='^')
    axes[2].set_title('E (Energy) vs Time')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('E (Energy)')
    axes[2].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def do_polynomial_regression(df):
    # Select the columns to apply polynomial features
    poly_cols = ['e', 'n', 'S', 'N', 'E']

    # Generate polynomial features
    poly = PolynomialFeatures(degree=3, include_bias=False)
    poly_features = poly.fit_transform(df[poly_cols])

    # Create a new DataFrame with polynomial features
    poly_feature_names = poly.get_feature_names_out(poly_cols)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)

    # Concatenate polynomial features back to the original DataFrame
    df = pd.concat([df.drop(columns=poly_cols), poly_df], axis=1)

    # Drop 'tree_id'
    if 'tree_id' in df.columns:
        df = df.drop(columns='tree_id')

    # Group by (t, species_id) and sum all features
    df_grouped = df.groupby(['t', 'species_id']).sum().reset_index()

    # Now fit the linear regression model
    dn_obs = df_grouped['dn']
    de_obs = df_grouped['de']
    X = df_grouped.drop(columns=['t', 'species_id', 'dn', 'de'])

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    output = []
    for y in[dn_obs, de_obs]:
        model = LinearRegression()
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
            'Feature': poly_feature_names,
            'Coefficient': beta_orig
        })

        # Add intercept as a separate row (optional but useful)
        coeff_df.loc[len(coeff_df)] = ['Intercept', intercept_orig]

        # Calculate r2
        r2 = r2_score(y, y_pred)

        output.append(coeff_df)
        output.append(r2)

    return output

def do_dynaMETE_regression(df):
    """
    TODO:
        Implement dynaMETE regression
    """
    coeffs, r2_dn, r2_de = 0, 0, 0
    return coeffs, r2_de, r2_de

def fit_transition_functions(df):
    # METimE
    df_METimE = df.copy()
    coeffs_dn, r2_dn, coeffs_de, r2_de = do_polynomial_regression(df_METimE)

    # DynaMETE
    df_dynaMETE = df.copy()
    coeffs_dynaMETE, r2_dynaMETE_dn, r2_dynaMETE_de = do_dynaMETE_regression(df_dynaMETE)

    return [coeffs_dn, r2_dn, coeffs_de, r2_de], [coeffs_dynaMETE, r2_dynaMETE_dn, r2_dynaMETE_de]

def do_maxent_inference(df, alphas, betas):
    functions = METimE.get_functions()

    # Create rank abundance distribution
    df_rad = df[['species_id', 'n']].drop_duplicates()
    df_rad = df_rad.sort_values(by='n', ascending=False).reset_index(drop=True)
    df_rad['rank'] = df_rad.index + 1
    empirical_rad = df_rad[['rank', 'n']].rename(columns={'n': 'abundance'})

    df = df.rename(columns={'t': 'census', 'S': 'S_t', 'N': 'N_t', 'E': 'E_t'})
    df = df[['census', 'e', 'n', 'S_t', 'N_t', 'E_t', 'dN/S', 'dE/S']]

    mete_aics, mete_maes, metime_aics, metime_maes = [], [], [], []

    for census in df['census'].unique():
        print(f"\n Census: {census} \n")
        input_census = df[df['census'] == census]

        X = input_census[[
            'S_t', 'N_t', 'E_t',
        ]].drop_duplicates().iloc[0]

        macro_var = {
            'N/S': float(X['N_t'] / X['S_t']),
            'E/S': float(X['E_t'] / X['S_t']),
            'dN/S': input_census['dN/S'].unique()[0],
            'dE/S': input_census['dE/S'].unique()[0]
        }

        # Make initial guess
        initial_lambdas = METimE.make_initial_guess(X)
        print(f"Initial guess (theoretical): {initial_lambdas}")

        print("Starting METE for initial guess")
        METE_lambdas = METimE.run_optimization(initial_lambdas[:2],
                                                  functions[:2],
                                                  {'N/S': float(X['N_t'] / X['S_t']),'E/S': float(X['E_t'] / X['S_t'])},
                                                  X, alphas, betas)
        print("Optimized lambdas (METE): {}".format(METE_lambdas))
        METE_lambdas = np.append(METE_lambdas, [0, 0])
        constraint_errors = METimE.check_constraints(METE_lambdas, input_census, functions, alphas, betas)
        aic, mae = METimE.evaluate_model(METE_lambdas, functions, X, alphas, betas, empirical_rad['abundance'].values, constraint_errors, 'METE', census)
        mete_aics.append(aic)
        mete_maes.append(mae)

        # Perform optimization
        METimE_lambdas = METimE.run_optimization(METE_lambdas, functions, macro_var, X, alphas, betas)
        print("Optimized lambdas (METimE): {}".format(METimE_lambdas))
        constraint_errors = METimE.check_constraints(METimE_lambdas, input_census, functions, alphas, betas)
        aic, mae = METimE.evaluate_model(METimE_lambdas, functions, X, alphas, betas, empirical_rad['abundance'].values, constraint_errors, 'METimE',census)
        metime_aics.append(aic)
        metime_maes.append(mae)

    pass

def run_simulation(X, p, t_max=30, max_iter = 1e08, obs_interval=0.25, start_from_prev=False, save_final_state=False):
    if start_from_prev:
        final_state_file = "C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/BCI/final_community_state.npz"
        data = np.load(final_state_file)

        metabolic_rates = data['metabolic_rates']
        species_indices = data['species_indices']
        tree_id_list = data['tree_id_list']
        X_vals = data['X']
        X = {'S': int(X_vals[0]), 'N': int(X_vals[1]), 'E': float(X_vals[2])}

    else:
        metabolic_rates, species_indices, tree_id_list = sample_community(X)
        X['S'], X['N'], X['E'] = len(species_indices) - 1, len(metabolic_rates), sum(metabolic_rates)

    # Generate trajectories of its state variables
    df = gillespie(metabolic_rates, species_indices, tree_id_list, X, p, t_max=t_max, max_iter = max_iter, obs_interval=obs_interval, save_final_state=save_final_state)
    plot_state_var(df)

    # Rescale t
    df['t'] = (df['t'] * (1 / obs_interval)).astype(int)

    # Add dn and de
    df = df.sort_values(by=['tree_id', 't'])
    df['dn'] = df.groupby('tree_id')['n'].shift(-1) - df['n']
    df['de'] = df.groupby('tree_id')['e'].shift(-1) - df['e']

    # Add dN/S and dE/S
    df_maxent = df.drop(columns=['dn', 'de'])
    df_maxent['dN/S'] = (df_maxent.groupby('tree_id')['N'].shift(-1) - df_maxent['N']) / df_maxent['S']
    df_maxent['dE/S'] = (df_maxent.groupby('tree_id')['E'].shift(-1) - df_maxent['E']) / df_maxent['S']
    df_maxent = df_maxent.dropna()

    METimE_output, _ = fit_transition_functions(df.dropna())
    aic, mae = do_maxent_inference(df_maxent, list(METimE_output[0]['Coefficient'].values), list(METimE_output[2]['Coefficient'].values))

    return METimE_output[1], aic, mae

def load_simple_dynaMETE():
    param = {
        'b': 0.2, 'd': 0.2, 'Ec': 450, 'm': 437.3,
        'w': 1.0, 'w1': 0.42, 'mu_meta': 0.0215,
        'd0': 0,
        'd1': 0
    }

    X = {
        'E': 460,
        'N': 160,
        'S': 55,
        'beta': 0.0001
    }
    return param, X

def load_new_dynaMETE():
    param = {
            'b': 0.5, 'd': 0.15504447, 'Ec': 1e07, 'm': 1000,                                                               # param['d'] is d2
            'w': 1.0, 'w1': 0.083626, 'mu_meta': 0.0269564,
            'd0': 0.1,
            'd1': 0.0000324
    }

    X = {
            'E': 2e07,
            'N': 230000,
            'S': 320,
            'beta': 0.0001
    }

    return param, X

if __name__ == '__main__':
    random.seed(123)

    param, X = load_simple_dynaMETE()
    #_, _, _ = run_simulation(X, param, max_iter = 1e06, save_final_state=True) # Only needed once

    # Then, repeatedly run simulation from equilibrium
    all_r2, all_aic, all_mae = [], [], []
    for iter in range(2):
        r2, aic, mae = run_simulation(X, param, t_max=5.1, obs_interval=0.5, start_from_prev=True)
        all_r2.append(r2)
        all_aic.append(aic)
        all_mae.append(mae)

    # Let's see what happens when we change the death rate
    param['d'] *= 2
    run_simulation(X, param, start_from_prev=True)
    r2, aic, mae = run_simulation(X, param, t_max=5.1, obs_interval=0.5, start_from_prev=True)



