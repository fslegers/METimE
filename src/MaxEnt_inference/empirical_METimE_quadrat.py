from token import N_TOKENS

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar, minimize
from scipy.integrate import quad
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import time
import METE_no_integrals
from sklearn.metrics import mean_squared_error
from scipy.stats import rv_discrete
import mpmath as mp

import warnings

warnings.filterwarnings("ignore")

"""
Entropy Maximization Script
---------------------------
This script maximizes the Shannon entropy subject to constraints on macroscopic ecosystem variables.
It supports different methods (METE, METimE, DynaMETE) that determine the functions that are used in
 the constraints and appear in the ecosystem structure function, and also the macro-variables and the 
 number of lambda parameters to optimize.

Components:
    1. Methods: Choose how to setup the functions and constraints.
    2. Functions: Define the constraints and the form of the ecosystem structure function.
    3. Macro-variables: Right-hand side values for the constraints (At least N/S and E/S).
    4. State variables: Values used to calculate functions and perform averaging (S, N, and E).
    5. Census: each gets its own optimization.

Usage:
    Provide a data set (CSV file) and select a method. The data set should at least contain the 
    following columns: 'census', 'n', 'e', 'S_t', 'N_t', 'E_t'. For methods METimE and DynaMETE, the
    following colums should also be available: 'dN', 'dE'.

    The script performs optimization and outputs the optimized lambda values along with constraint errors 
    and resulting entropy.

Assumptions:
    R(n, e) = exp( - lambda_1 f_1 - lambda_2 f_2 ...) / Z
    Z = sum integral exp( - lambda_1 f_1 - lambda_2 f_2 ...) de

TODO:
    - Expand available methods (e.g., DynaMETE).
    - Implement further scaling and additional transition functions if needed.
    - Improve logging / error reporting.

"""

###############################################
### Ecosystem Structure Function Components ###
###############################################

def exp_in_R(n, e, X, functions, lambdas, alphas, betas, scaling_factors=[1, 1, 1, 1]):
    """
    Compute the exponent term: -lambda1*f1 - ... - lambdak*fk
    """
    exponent = sum(-lambdas[i] * functions[i](n, e, X, alphas, betas, scaling_factors) for i in range(len(functions)))
    return exponent


def partition_function(lambdas, functions, X, alphas, betas, scaling_factors=[1, 1, 1, 1]):
    def integrand(e, n):
        exponent = sum(-lambdas[i] * functions[i](n, e, X, alphas, betas, scaling_factors) for i in range(len(functions)))
        return np.exp(exponent)

    Z = 0
    for n in range(1, int(X['N_t']) + 1):
        Z += quad(lambda e: integrand(e, n),  0, X['E_t'])[0]

    return Z


def entropy(lambdas, functions, X, alphas, betas, scaling_factors=[1, 1, 1, 1]):
    """
    Compute Shannon entropy for the given lambdas and functions.
    """
    def integrand(e, n, Z):
        exponent = sum(-lambdas[i] * functions[i](n, e, X, alphas, betas, scaling_factors) for i in range(len(functions)))
        return np.exp(exponent) / Z * (np.log(1/Z) + exp_in_R(n, e, X, functions, lambdas, alphas, betas, scaling_factors))

    Z = partition_function(lambdas, functions, X, alphas, betas, scaling_factors)

    H = 0
    for n in range(1, int(X['N_t']) + 1):
        H += quad(lambda e: integrand(e, n, Z), 0, X['E_t'])[0]

    if np.any(np.isnan(H)) or np.any(np.isinf(H)):
        print("Invalid values detected in entropy")
        H = 1e10

    return H


def beta_function(beta, S, N):
    """
    Beta function used to generate the initial guess for Lagrange multipliers.
    """
    return (1 - np.exp(-beta)) / (np.exp(-beta) - np.exp(-beta * (N + 1))) * np.log(1.0 / beta) - S / N


def make_initial_guess(X):
    """
    A function that makes an initial guess for the Lagrange multipliers lambda1 and lambda2.
    Based on Eq 7.29 from Harte 2011 and meteR'diag function meteESF.mete.lambda

    :param state_variables: state variables S, S and E
    :return: initial guess for the Lagrange multipliers lambda1 and lambda2
    """
    S, N, E = int(X['S_t']), int(X['N_t']), float(X['E_t'])
    interval = [1.0 / N, S / N]

    beta = root_scalar(beta_function, x0=0.001, args=(S, N), method='brentq', bracket=interval)

    l2 = S / (E - N)
    l1 = beta.root - l2

    return [l1, l2, 0, 0]


###############################################
###               Optimization              ###
###############################################

def single_constraint(X, f_k, F_k, all_f, lambdas, alphas, betas, scaling_factors=[1, 1, 1, 1]):
    def integrand(e, n):
        exponent = sum(-lambdas[i] * functions[i](n, e, X, alphas, betas, scaling_factors) for i in range(len(functions)))
        return np.exp(exponent) * f_k(n, e, X, alphas, betas)

    Z = partition_function(lambdas, all_f, X, alphas, betas, scaling_factors)

    expected_value = 0
    for n in range(1, int(X['N_t']) + 1):
        expected_value += quad(lambda e: integrand(e, n), 0, X['E_t'])[0]
    expected_value /= Z

    if np.any(np.isnan(expected_value)) or np.any(np.isinf(expected_value)):
        print("Invalid values detected in single constraint")
        expected_value = 1e10

    return (expected_value - F_k) / np.abs(F_k) # TODO: scaled down so that all constraints weigh equally


# def constraint(f_k, lambdas, functions, F_k, X, alphas, betas):
#     """
#     Calculates the expected value of a single constraint function f_k over the ecosystem structure function:
#     Σ ʃ f_k R(n, e) / Z de - F_k
#     Parallelized version.
#     """
#     with ProcessPoolExecutor(max_workers=2) as executor:
#         futures = [
#             executor.submit(
#                 single_constraint,
#                 n, X['E_t'], X, f_k, functions, lambdas, alphas, betas
#             )
#             for n in range(1, int(X['N_t']) + 1)
#         ]
#         contributions = [f.result() for f in futures]
#
#     I = sum(contributions)
#     Z = integrate_with_cutoff(X, functions, lambdas)
#     return I / Z - F_k


def compute_bounds(X, alphas, betas):
    N_max = X['N_t']
    E_max = X['E_t']

    # Define the boundary values to check
    n_values = [1, N_max]
    e_values = [0, E_max]

    max_abs_val_dn = 0
    max_abs_val_de = 0

    for n in n_values:
        for e in e_values:
            val_dn = np.abs(f_dn(n, e, X, alphas, betas))
            val_de = np.abs(f_de(n, e, X, alphas, betas))

            if val_dn > max_abs_val_dn:
                max_abs_val_dn = val_dn

            if val_de > max_abs_val_de:
                max_abs_val_de = val_de

    if max_abs_val_dn == 0:
        bounds_dn = (-1, 1)  # TODO: is dit te klein?
    else:
        bounds_dn = (-400/max_abs_val_dn, 400/max_abs_val_dn)

    if max_abs_val_de == 0:
        bounds_de = (-1, 1)   # TODO: is dit te klein?
    else:
        bounds_de = (-400/max_abs_val_de, 400/max_abs_val_de)

    print(f"Bounds for dn: {bounds_dn}")
    print(f"Bounds for de: {bounds_de}")

    return bounds_dn, bounds_de


def perform_optimization(lambdas, functions, macro_var, X, alphas, betas):
    # Set bounds and scale all lambas to be of order of magnitude ~10
    bounds_dn, bounds_de = compute_bounds(X, alphas, betas)

    scaling_factors = [10 / lambdas[0],
                       10 / lambdas[1],
                       10 / bounds_dn[1],
                       10 / bounds_de[1]]

    bounds = [(None, None),
              (None, None),
              (bounds_dn[0] * scaling_factors[2], bounds_dn[1] * scaling_factors[2]),
              (bounds_de[0] * scaling_factors[3], bounds_de[1] * scaling_factors[3])]

    # Scale up the initial guess
    lambdas[0] *= scaling_factors[0]
    lambdas[1] *= scaling_factors[1]

    # Collect all constraints
    constraints = [{
        'type': 'eq',
        'fun': lambda lambdas, functions=functions, f_k=f, F_k=macro_var[name], X=X, scaling=scaling_factors:
        single_constraint(X, f_k, F_k, functions, lambdas, alphas, betas, scaling_factors)
    } for f, name in zip(functions, macro_var)]

    # Perform optimization
    print("Starting Optimizing with constraints...")
    result = minimize(entropy,
                      lambdas,
                      args=(functions, X, alphas, betas, scaling_factors),
                      constraints=constraints,
                      bounds=bounds,
                      method="trust-constr",
                      options={'initial_tr_radius': 0.1,
                               'disp': True,
                               'verbose': 3
                               })

    optimized_lambdas = result.x

    # Revert scaling
    optimized_lambdas[0] /= scaling_factors[0]
    optimized_lambdas[1] /= scaling_factors[1]
    optimized_lambdas[2] /= scaling_factors[2]
    optimized_lambdas[3] /= scaling_factors[3]

    return optimized_lambdas

########################
### Set-up and check ###
########################

def f_n(n, e, X, alphas, betas, scaling_factors=[1, 1, 1, 1]):
    return n / scaling_factors[0]

def f_ne(n, e, X, alphas, betas, scaling_factors=[1, 1, 1, 1]):
    return n * e / scaling_factors[1]

def f_dn(n, e, X, alphas, betas, scaling_factors=[1, 1, 1, 1]):
    return (
    alphas[0] * e +
    alphas[1] * n +
    alphas[2] * e ** 2 +
    alphas[3] * e * X['S_t'] +
    alphas[4] * e * n +
    alphas[5] * e * X['N_t'] +
    alphas[6] * e * X['E_t'] +
    alphas[7] * X['S_t'] * n +
    alphas[8] * n ** 2 +
    alphas[9] * n * X['N_t'] +
    alphas[10] * n * X['E_t'] +
    alphas[11] * e ** 3 +
    alphas[12] * e ** 2 * X['S_t'] +
    alphas[13] * e ** 2 * n +
    alphas[14] * e ** 2 * X['N_t'] +
    alphas[15] * e ** 2 * X['E_t'] +
    alphas[16] * e * X['S_t'] ** 2 +
    alphas[17] * e * X['S_t'] * n +
    alphas[18] * e * X['S_t'] * X['N_t'] +
    alphas[19] * e * X['S_t'] * X['E_t'] +
    alphas[20] * e * n ** 2 +
    alphas[21] * e * n * X['N_t'] +
    alphas[22] * e * n * X['E_t'] +
    alphas[23] * e * X['N_t'] ** 2 +
    alphas[24] * e * X['N_t'] * X['E_t'] +
    alphas[25] * e * X['E_t'] ** 2 +
    alphas[26] * X['S_t'] ** 2 * n +
    alphas[27] * X['S_t'] * n ** 2 +
    alphas[28] * X['S_t'] * n * X['N_t'] +
    alphas[29] * X['S_t'] * n * X['E_t'] +
    alphas[30] * n ** 3 +
    alphas[31] * n ** 2 * X['N_t'] +
    alphas[32] * n ** 2 * X['E_t'] +
    alphas[33] * n * X['N_t'] ** 2 +
    alphas[34] * n * X['N_t'] * X['E_t'] +
    alphas[35] * n * X['E_t'] ** 2) / scaling_factors[2]

def f_de(n, e, X, alphas, betas, scaling_factors=[1, 1, 1, 1]):
    return (
    betas[0] * e +
    betas[1] * n +
    betas[2] * e ** 2 +
    betas[3] * e * X['S_t'] +
    betas[4] * e * n +
    betas[5] * e * X['N_t'] +
    betas[6] * e * X['E_t'] +
    betas[7] * X['S_t'] * n +
    betas[8] * n ** 2 +
    betas[9] * n * X['N_t'] +
    betas[10] * n * X['E_t'] +
    betas[11] * e ** 3 +
    betas[12] * e ** 2 * X['S_t'] +
    betas[13] * e ** 2 * n +
    betas[14] * e ** 2 * X['N_t'] +
    betas[15] * e ** 2 * X['E_t'] +
    betas[16] * e * X['S_t'] ** 2 +
    betas[17] * e * X['S_t'] * n +
    betas[18] * e * X['S_t'] * X['N_t'] +
    betas[19] * e * X['S_t'] * X['E_t'] +
    betas[20] * e * n ** 2 +
    betas[21] * e * n * X['N_t'] +
    betas[22] * e * n * X['E_t'] +
    betas[23] * e * X['N_t'] ** 2 +
    betas[24] * e * X['N_t'] * X['E_t'] +
    betas[25] * e * X['E_t'] ** 2 +
    betas[26] * X['S_t'] ** 2 * n +
    betas[27] * X['S_t'] * n ** 2 +
    betas[28] * X['S_t'] * n * X['N_t'] +
    betas[29] * X['S_t'] * n * X['E_t'] +
    betas[30] * n ** 3 +
    betas[31] * n ** 2 * X['N_t'] +
    betas[32] * n ** 2 * X['E_t'] +
    betas[33] * n * X['N_t'] ** 2 +
    betas[34] * n * X['N_t'] * X['E_t'] +
    betas[35] * n * X['E_t'] ** 2) / scaling_factors[3]

def get_functions():
    return [f_n, f_ne, f_dn, f_de]

def check_constraints(lambdas, input, functions, alphas, betas):
    """
    Returns the error on constraints given some lambda values
    Given in percentage of the observed value
    """
    S, N, E = (int(input['S_t'].drop_duplicates().iloc[0]),
               int(input['N_t'].drop_duplicates().iloc[0]),
               input['E_t'].drop_duplicates().iloc[0])

    X = {
        'S_t': S,
        'N_t': N,
        'E_t': E
    }

    macro_var = {
        'N/S': X['N_t'] / X['S_t'],
        'E/S': X['E_t'] / X['S_t'],
        'dN': input['dN'].unique()[0],
        'dE': input['dE'].unique()[0]
    }

    Z = partition_function(lambdas, functions, X, alphas, betas)
    print(f"Z = {Z}")

    absolute_errors = []
    percentage_errors = []

    for f, (key, v) in zip(functions, macro_var.items()):
        # Compute integral with upper bound
        integral_value = 0
        for n in range(1, int(X['N_t'] + 1)):
            edges = np.linspace(0, 1, 11 + 1) ** 2 * X['E_t']
            integral_value += sum(
                quad(
                    lambda e: f(n, e, X, alphas, betas) * np.exp(exp_in_R(n, e, X, functions, lambdas, alphas, betas)),
                    a, b
                )[0]
                for a, b in zip(edges[:-1], edges[1:])
            )

        integral_value /= Z  # TODO: check

        # Compute constraint error
        abs_error = np.abs(integral_value - v)
        pct_error = abs_error / np.abs(v) * 100

        absolute_errors.append(abs_error)
        percentage_errors.append(pct_error)

    print("\n Errors on constraints:")
    print(f"{'Constraint':<10} {'Abs Error':>15} {'% Error':>15}")
    print("-" * 42)
    for key, abs_err, pct_err in zip(macro_var.keys(), absolute_errors, percentage_errors):
        print(f"{key:<10} {abs_err:15.6f} {pct_err:15.2f}")

    return absolute_errors

def get_empirical_RAD(file, census):
    # Load relevant data
    df = pd.read_csv(file)

    if 'census' not in df.columns:
        df = df.rename(columns={'t': 'census', 'Species_ID': 'species'})

    df = df[df['census'] == census]
    df = df[['species', 'n']].drop_duplicates()

    # Create rank abundance distribution
    df = df.sort_values(by='n', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    rad = df[['rank', 'n']].rename(columns={'n': 'abundance'})

    return rad


# Discretize the domain
# def plot_dn(X):
#     # Create a grid over the domain
#     n_vals = np.linspace(1, X['N_t'] + 1, 100)
#     e_vals = np.linspace(0, X['E_t'], 100)
#     N, E = np.meshgrid(n_vals, e_vals)
#
#     # Evaluate f_dn over the grid
#     Z = f_dn(N, E, X)
#
#     fig = go.Figure(data=[go.Surface(z=Z, x=N, y=E, colorscale='Viridis')])
#     fig.update_layout(
#         title='Interactive 3D Plot of f_dn(n, e)',
#         scene=dict(
#             xaxis_title='n',
#             yaxis_title='e',
#             zaxis_title='f_dn(n, e)'
#         )
#     )
#     fig.show()
#
#     return np.min(Z), np.max(Z)
#
# def plot_de(X):
#     # Create a grid over the domain
#     n_vals = np.linspace(1, X['N_t'] + 1, 100)
#     e_vals = np.linspace(0, X['E_t'], 100)
#     N, E = np.meshgrid(n_vals, e_vals)
#
#     # Evaluate f_dn over the grid
#     Z = f_de(N, E, X)
#
#     fig = go.Figure(data=[go.Surface(z=Z, x=N, y=E, colorscale='Viridis')])
#     fig.update_layout(
#         title='Interactive 3D Plot of f_de(n, e)',
#         scene=dict(
#             xaxis_title='n',
#             yaxis_title='e',
#             zaxis_title='f_de(n, e)'
#         )
#     )
#     fig.show()
#
#     return np.min(Z), np.max(Z)

def get_rank_abundance(sad, X):
    """
    Generate a predicted rank-abundance distribution using the quantile method.
    Ensures exactly S_t values by clipping quantiles and handling edge cases.
    """
    S = int(X['S_t']) + 1

    # Create the discrete distribution
    n_vals = np.arange(1, len(sad) + 1)
    dist = rv_discrete(name='sad_dist', values=(n_vals, sad))

    # Safer quantiles: strictly within (0, 1)
    epsilon = 1e-6
    quantiles = (np.arange(1, S + 1) - 0.5) / S
    quantiles = np.clip(quantiles, epsilon, 1 - epsilon)

    # Evaluate quantiles
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_abundances = dist.ppf(quantiles).astype(int)

    # Fix any zeros or nans (can happen if ppf fails)
    pred_abundances = np.where(pred_abundances < 1, 1, pred_abundances)
    pred_abundances = np.nan_to_num(pred_abundances, nan=1).astype(int)

    # Ensure output length = S_t
    if len(pred_abundances) != S:
        raise ValueError(f"Expected {S} predicted abundances, got {len(pred_abundances)}.")

    return np.sort(pred_abundances)[::-1]  # descending order


def evaluate_model(lambdas, functions, X, alphas, betas, empirical_rad, constraint_errors, model, census, ext=""):
    # Compute SAD
    Z = partition_function(lambdas, functions, X, alphas, betas)
    sad = np.zeros(int(X['N_t']))

    if abs(Z) > 1e-300:
        for n in range(1, int(X['N_t']) + 1):
            sad[n - 1] = 1 / Z * quad(
                lambda e: np.exp(sum([-lambdas[i] * functions[i](n, e, X, alphas, betas) for i in range(4)])),
                0, X['E_t']
            )[0]
    else:
        for n in range(1, int(X['N_t']) + 1):
            sad[n - 1] = mp.quad(
                lambda e: mp.exp(sum([-lambdas[i] * functions[i](n, e, X, alphas, betas)
                                      for i in range(4)])),
                [0, X['E_t']]
            )
        # Convert back to
        Z = mp.fsum(sad)
        sad = np.array([float(x/Z) for x in sad], dtype=float)

    # Resize to match empirical_rad length
    rad = get_rank_abundance(sad, X)
    rad = rad[:len(empirical_rad)]
    empirical_rad = empirical_rad[:len(rad)]

    # RMSE
    rmse = np.sqrt(mean_squared_error(empirical_rad, rad))

    # AIC
    eps = 1e-10
    log_probs = np.log(sad[rad - 1] + eps)  # -1 for indexing
    log_likelihood = np.sum(log_probs)
    k = len(lambdas)
    aic = -2 * log_likelihood + 2 * k

    plt.figure(figsize=(8, 5))
    ranks = np.arange(1, len(empirical_rad) + 1)
    plt.plot(ranks, empirical_rad, 'o-', label='Empirical RAD', color='blue')
    plt.plot(ranks, rad, 's--', label='Predicted RAD', color='red')
    plt.xlabel('Rank')
    plt.ylabel('Abundance')
    plt.legend()

    # Annotate RMSE and AIC
    textstr = f'RMSE: {rmse:.3f}\nAIC: {aic:.2f}'
    plt.text(0.95, 0.95, textstr,
                     transform=plt.gca().transAxes,
                     fontsize=16,
                     verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    plt.tight_layout()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    # plt.show()
    # plt.savefig(f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/BCI/{ext}/{model}_{census}.png')
    plt.savefig(
        f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/BCI/simulated_{model}_{census}.png')

    results_data = {
        'RMSE': [rmse],
        'AIC': [aic],
    }

    # Add lambdas to dictionary
    for i, lam in enumerate(lambdas):
        results_data[f'lambda_{i}'] = [lam]


    for constr, error in zip(['N/S', 'E/S', 'dN', 'dE'], constraint_errors):
        results_data[f'{constr}'] = error

    # Create DataFrame
    results_df = pd.DataFrame(results_data)

    # Save to CSV (same name as PNG but .csv extension)
    # results_df.to_csv(f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/BCI/{ext}{model}_{census}.csv', index=False)
    results_df.to_csv(f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/BCI/simulated_{model}_{census}.png')

if __name__ == "__main__":
    ext = ''
    #input = pd.read_csv(f'../../data/BCI_regression_library{ext}.csv')
    input = pd.read_csv(f'../../data/simulated_BCI_regress_lib.csv')
    functions = get_functions()
    
    if 'census' not in input.columns:
        input = input.rename(columns={'t': 'census', 'S': 'S_t', 'N': 'N_t', 'E': 'E_t'})

        # Get only one row per census (e.g., the first one)
        census_df = input.drop_duplicates(subset='census', keep='first').sort_values('census')

        # Compute dN and dE
        census_df['dN'] = census_df['N_t'].diff().shift(-1)  # N(t+1) - N(t)
        census_df['dE'] = census_df['E_t'].diff().shift(-1)  # E(t+1) - E(t)

        # If you want to merge back to original input
        input = input.merge(census_df[['census', 'dN', 'dE']], on='census', how='left')
        input = input.dropna(subset=['dN', 'dE'])

    for census in input['census'].unique():
        print(f"\n Census: {census} \n")
        input_census = input[input['census'] == census]

        X = input_census[[
            'S_t', 'N_t', 'E_t',
        ]].drop_duplicates().iloc[0]

        macro_var = {
            'N/S': float(X['N_t'] / X['S_t']),
            'E/S': float(X['E_t'] / X['S_t']),
            'dN': input_census['dN'].unique()[0],
            'dE': input_census['dE'].unique()[0]
        }

        # alphas = pd.read_csv(
        #     f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Data sets/BCI{ext}/METimE_dn_global.csv')[
        #     'Coefficient'].values

        alphas = pd.read_csv(
            f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Data sets/simulated_BCI/METimE_dn_global.csv')[
            'Coefficient'].values

        # betas = pd.read_csv(
        #     f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Data sets/BCI{ext}/METimE_de_global.csv')[
        #     'Coefficient'].values

        betas = pd.read_csv(
            f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Data sets/simulated_BCI/METimE_de_global.csv')[
            'Coefficient'].values
        
        # Get empirical rank abundance distribution
        # empirical_rad = get_empirical_RAD(f'../../data/BCI_regression_library{ext}.csv', census)['abundance']
        empirical_rad = get_empirical_RAD(f'../../data/simulated_BCI_regress_lib.csv', census)['abundance']

        # Make initial guess
        initial_lambdas = make_initial_guess(X)
        print(f"Initial guess (theoretical): {initial_lambdas}")

        print("Starting METE for initial guess")
        initial_lambdas = METE_no_integrals.perform_optimization([initial_lambdas[:2]], X).tolist() + [0,0]               # TODO: doesn't change the initial guess?
        constraint_errors = check_constraints(initial_lambdas, input_census, functions, alphas, betas)
        evaluate_model(initial_lambdas, functions, X, alphas, betas, empirical_rad, constraint_errors,'METE', census, ext)
        
        # Perform optimization
        optimized_lambdas = perform_optimization(initial_lambdas, functions, macro_var, X, alphas, betas)
        print("Optimized lambdas: {}".format(optimized_lambdas))
        constraint_errors = check_constraints(optimized_lambdas, input_census, functions, alphas, betas)
        evaluate_model(optimized_lambdas, functions, X, alphas, betas, empirical_rad, constraint_errors,'METimE', census, ext)

        # Are the initial_lambdas and optimized_lambdas the correct scale?