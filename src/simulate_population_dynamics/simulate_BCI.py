import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from src.MaxEnt_inference.empirical_METimE_riemann import run_optimization as riemann_optimization
from src.MaxEnt_inference.empirical_METimE_riemann import get_empirical_RAD, evaluate_model, get_function_values, make_initial_guess, plot_RADs, do_polynomial_regression, get_functions, check_constraints

from src.MaxEnt_inference import zero_point_METE
import sys
import os
import random

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
        if np.random.rand() < prob_new_species or len(tree_ids) == 0:
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


def gillespie(metabolic_rates, species_indices, tree_id_list, X, p, t_max, max_iter, obs_interval, save_final_state=False):
    # some preparations
    species_ids = np.zeros(int(species_indices[-1]), dtype=int)
    for i in range(1, len(species_indices)):
        start = species_indices[i - 1]
        end = species_indices[i]
        species_ids[start:end] = i

    observation_times = np.arange(0, t_max, obs_interval)
    obs_pointer = 0

    # Initialization
    tree_ids = np.array(tree_id_list)
    species_ids = np.array(species_ids)
    metabolic_rates = np.array(metabolic_rates)
    abundances = dict(Counter(species_ids))

    try:
        next_tree_id = tree_ids.max() + 1
    except:
        next_tree_id = 0

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
        if len(tree_id_list) == 0:
            time_until_event = 0
        else:
            u = np.random.uniform(0, 1)
            time_until_event = -np.log(u) / R
        t += time_until_event
        n_iter += 1

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

            # Progress observation time
            obs_pointer += 1

        # Update individual metabolic rates
        metabolic_rates, X = update_metabolic_rates(metabolic_rates, X, time_until_event, p)

        # Update event rates based on new metabolic rates e
        birth_rates, death_rates, migration_rates, R = update_event_rates(species_ids, abundances, metabolic_rates, X, p)

        # Determine what event (birth or death) happened
        q = np.random.uniform(0, 1)
        event = what_event_happened(birth_rates, death_rates, migration_rates, R, q)
        tree_ids, species_ids, metabolic_rates, abundances, next_tree_id, X = perform_event(tree_ids, species_ids, metabolic_rates, abundances, next_tree_id, X, event, p)

    # Flatten list of snapshowts with one row per individual per time step
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


def plot_state_var(df, frac):
    # Create three horizontally aligned subplots
    fig, axes = plt.subplots(ncols=3, figsize=(18, 5), sharex=True)

    # Plot S vs time
    axes[0].plot(df['t'], df['S'], color='tab:blue', marker='o')
    #axes[0].set_title('S (Species count) vs Time')
    axes[0].set_xlabel('time (t)')
    axes[0].set_ylabel('S_t')
    axes[0].grid(True)

    # Plot N vs time
    axes[1].plot(df['t'], df['N'], color='tab:green', marker='s')
    #axes[1].set_title('N (Abundance) vs Time')
    axes[1].set_xlabel('time (t)')
    axes[1].set_ylabel('N_t')
    axes[1].grid(True)

    # Plot E vs time
    axes[2].plot(df['t'], df['E'], color='tab:red', marker='^')
    #axes[2].set_title('E (Energy) vs Time')
    axes[2].set_xlabel('time (t)')
    axes[2].set_ylabel('E_t')
    axes[2].grid(True)

    # Adjust layout
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/BCI/simulated_BCI_state_variables_{frac}.png')
    plt.close()

def do_dynaMETE_regression(df):
    """
    TODO:
        Implement dynaMETE regression
    """
    coeffs, r2_dn, r2_de = 0, 0, 0
    return coeffs, r2_de, r2_de

def get_empirical_RAD(df, census):
    df = df[df['census'] == census]
    df = df[['species', 'n']].drop_duplicates()

    # Create rank abundance distribution
    df = df.sort_values(by='n', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    rad = df[['rank', 'n']].rename(columns={'n': 'abundance'})

    return rad


def remove_fraction(frac, metabolic_rates, species_indices, tree_id_list):
    # Remove fraction of community
    # indices to remove
    indices_to_remove = np.random.choice(len(tree_id_list), int(frac * len(tree_id_list)), replace=False)

    # remove from tree_id_list and metabolic_rates
    # create species_id from species_indices
    species_id = np.empty(len(tree_id_list), dtype=int)
    for i, start_idx in enumerate(species_indices):
        end_idx = species_indices[i + 1] if i + 1 < len(species_indices) else len(tree_id_list)
        species_id[start_idx:end_idx] = i

    tree_id_list = np.array([tree for i, tree in enumerate(tree_id_list) if i not in indices_to_remove])
    metabolic_rates = np.array([rate for i, rate in enumerate(metabolic_rates) if i not in indices_to_remove])
    species_id = np.array([species for i, species in enumerate(species_id) if i not in indices_to_remove])

    unique_species, counts = np.unique(species_id, return_counts=True)
    species_indices = np.concatenate(([0], np.cumsum(counts)))

    # recalculate X
    S = len(species_indices) - 1
    N = len(tree_id_list)
    E = metabolic_rates.sum()
    X = {'S': S, 'N': N, 'E': E}

    return metabolic_rates, species_indices, tree_id_list, X


def run_simulation(X, p, frac, n_iter=1, t_max=30, max_iter = 1e08, obs_interval=0.25, start_from_prev=False, save_final_state=False):
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

    metabolic_rates, species_indices, tree_id_list, X = remove_fraction(frac, metabolic_rates, species_indices, tree_id_list)

    results_list = []
    for iter in range(n_iter):
        # Generate trajectories of its state variables
        df = gillespie(metabolic_rates, species_indices, tree_id_list, X, p, t_max=t_max, max_iter = max_iter, obs_interval=obs_interval, save_final_state=save_final_state)

        if iter == 0:
            plot_state_var(df, frac)

        # Rescale t
        df['t'] = (df['t'] * (1 / obs_interval)).astype(int)

        # Add dn and de
        df = df.sort_values(by=['tree_id', 't'])
        df['dn'] = df.groupby('tree_id')['n'].shift(-1) - df['n']
        df['de'] = df.groupby('tree_id')['e'].shift(-1) - df['e']

        df['dN/S'] = (df.groupby('tree_id')['N'].shift(-1) - df['N']) / df['S']
        df['dE/S'] = (df.groupby('tree_id')['E'].shift(-1) - df['E']) / df['S']
        df = df.dropna()

        # Rename columns
        df = df.rename(columns={
            'S': 'S_t',
            'N': 'N_t',
            'E': 'E_t',
            't': 'census',
            'tree_id': 'TreeID',
            'species_id': 'species'
        })

        # Compute polynomial coefficients
        alphas, r2_dn, betas, r2_de = do_polynomial_regression(df, show_plot=True)
        r2s = pd.DataFrame({'r2_dn': [r2_dn], 'r2_de': [r2_de]})
        print(r2s)
        alphas = alphas['Coefficient'].values
        betas = betas['Coefficient'].values
        functions = get_functions()

        # Create list to store results
        results_list = []

        # For 5 cencuses, perform MaxEnt inference
        for census in np.linspace(0, len(np.unique(df['census'])) - 1, 4, dtype=int):
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

            print("macro_var")
            print(macro_var)

            # Get empirical rank abundance distribution
            empirical_rad = get_empirical_RAD(input_census, census)['abundance']

            # Precompute functions(n, e)
            de = 1.0
            func_vals, _ = get_function_values(functions, X, alphas, betas, de)

            # Make initial guess
            initial_lambdas = make_initial_guess(X)
            print(f"Initial guess (theoretical): {initial_lambdas}")

            #######################################
            #####            METE             #####
            #######################################

            # Optimizer SLSQP gives errors because inequality constraints are not satisfiable

            print(" ")
            print("----------METE----------")
            METE_lambdas = riemann_optimization(
                initial_lambdas[:2],
                {
                    'N/S': float(X['N_t'] / X['S_t']),
                    'E/S': float(X['E_t'] / X['S_t'])
                },
                X,
                func_vals[:2],
                de,
                optimizer='trust-constr'
            )
            print("Optimized lambdas (METE): {}".format(METE_lambdas))
            METE_lambdas = np.append(METE_lambdas, [0, 0])
            constraint_errors = check_constraints(METE_lambdas, input_census, func_vals)
            METE_results, METE_rad = evaluate_model(METE_lambdas, X, func_vals, empirical_rad, de, constraint_errors)
            print(f"AIC: {METE_results['AIC'].values[0]}, MAE: {METE_results['MAE'].values[0]}")

            #######################################
            #####           METimE            #####
            #######################################
            print(" ")
            print("----------METimE----------")
            METimE_lambdas = riemann_optimization(METE_lambdas, macro_var, X, func_vals, de, 'trust-constr')
            print("Optimized lambdas: {}".format(METimE_lambdas))
            constraint_errors = check_constraints(METimE_lambdas, input_census, func_vals)
            METimE_results, METimE_rad = evaluate_model(METimE_lambdas, X, func_vals, empirical_rad, de, constraint_errors)
            print(f"AIC: {METimE_results['AIC'].values[0]}, MAE: {METimE_results['MAE'].values[0]}")

            ##########################################
            #####           Save results         #####
            ##########################################
            results_list.append({
                'census': census,
                'iter': iter,
                'METE_AIC': METE_results['AIC'].values[0],
                'METE_MAE': METE_results['MAE'].values[0],
                'METE_RMSE': METE_results['RMSE'].values[0],
                'METimE_AIC': METimE_results['AIC'].values[0],
                'METimE_MAE': METimE_results['MAE'].values[0],
                'METimE_RMSE': METimE_results['RMSE'].values[0],
                'r2_dn': r2s['r2_dn'].values[0],
                'r2_de': r2s['r2_de'].values[0]
            })

            if iter == 0:
                ext = f"simulated_census_{census}_frac_{frac}.png"
                plot_RADs(empirical_rad, METE_rad, METimE_rad, ext, use_log=True)

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/BCI/simulated_BCI_{frac}.csv', index=False)

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
    #_, _, _ = run_simulation(X, param, max_iter = 1e06, save_final_state=True) # Only needed once to get to an equilibrium

    # Then, repeatedly run simulation from equilibrium, where different fractions of the initial population are removed
    # thereby disturbing the equilibrium
    # for each "level of disturbance" (fraction of initial population removed) repeat 5 times
    for frac in [0.8, 0.6, 0.4, 0.2, 0.0, 1.0]:
        run_simulation(X, param, frac, n_iter=1, t_max=2.1, obs_interval=0.1, start_from_prev=True)



