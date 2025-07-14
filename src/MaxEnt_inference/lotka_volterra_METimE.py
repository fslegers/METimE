import numpy as np
import pandas as pd
from scipy.optimize import root_scalar, minimize
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
from scipy.stats import rv_discrete
from sklearn.metrics import mean_squared_error

import warnings

#warnings.filterwarnings("ignore")

"""
Entropy Maximization Script
---------------------------
This script maximizes the Shannon entropy subject to constraints on macroscopic ecosystem variables.
It supports different methods (METE, METimE) that determine the functions that are used in the constraints 
and appear in the ecosystem structure function, and also the macro-variables and the  number of lambda 
parameters to optimize.

Components:
    1. Methods: Choose how to setup the functions and constraints.
    2. Functions: Define the constraints and the form of the ecosystem structure function.
    3. Macro-variables: Right-hand side values for the constraints (N/S).
    4. State variables: Values used to calculate functions and perform averaging (S and N).
    5. Census: each gets its own optimization.

Usage:
    Provide a data set (CSV file) and select a method. The data set should at least contain the 
    following columns: 'census', 'n', 'S_t', and 'N_t'. For method METimE, the 'dN' should also
    be available.

    The script performs optimization and outputs the optimized lambda values along with constraint errors 
    and resulting entropy.

Assumptions:
    R(n) = exp(- lambda_1 n - lambda_2 f(n, X)) / Z
    Z = sum exp( - lambda_1 f_1 - lambda_2 f_2 f(n, X))

"""

def entropy(lambdas, functions, X, coeffs):
    """
    Compute Shannon entropy for the given lambdas and functions.
    Change sign because we will be minimizing instead of maximizing.
    """
    n = np.arange(1, int(X['N_t']) + 1)

    exponent_arg = np.zeros_like(n, dtype=float)
    for i in range(len(functions)):
        exponent_arg += lambdas[i] * functions[i](n, X, coeffs)
    exponent = np.exp(-exponent_arg)

    Z = np.sum(exponent) # partition function

    p = exponent / Z # probabilities

    neg_H = np.sum(p * exponent_arg)

    return neg_H


def beta_function(beta, S, N):
    """
    Beta function used to generate the initial guess for Lagrange multipliers.
    """
    return (1 - np.exp(-beta)) / (np.exp(-beta) - np.exp(-beta * (N + 1))) * np.log(1.0 / beta) - S / N


def make_initial_guess(X, method):
    """
    A function that makes an initial guess for the Lagrange multipliers lambda1 and lambda2.
    Based on Eq 7.29 from Harte 2011 and meteR'diag function meteESF.mete.lambda

    :param state_variables: state variables S, N and E
    :return: initial guess for the Lagrange multipliers lambda1 and lambda2
    """
    S, N = int(X['S_t']), int(X['N_t'])
    interval = [1.0 / N, S / N]

    beta = root_scalar(beta_function, x0=0.001, args=(S, N), method='brentq', bracket=interval)

    if method == "METE":
        return [beta.root]
    else:
        return [beta.root, 0]

###############################################
###               Optimization              ###
###############################################

def constraint(f_k, lambdas, functions, F_k, X, coeffs):
    """
    Calculates the expected value of a single constraint function f_k over the ecosystem structure function:
    Σ f_k p_n / Z - F_k
    """
    n = np.arange(1, int(X['N_t']) + 1)

    exponent_arg = np.zeros_like(n, dtype=float)
    for i in range(len(functions)):
        exponent_arg += lambdas[i] * functions[i](n, X, coeffs)
    exponent = np.exp(-exponent_arg)

    Z = np.sum(exponent)  # partition function

    p = exponent / Z  # probabilities

    lhs = np.sum(p * f_k(n, X, coeffs))

    return lhs - F_k


def perform_optimization(lambdas, functions, macro_var, X, coeffs):
    # Collect all constraints
    constraints = [{
        'type': 'eq',
        'fun': lambda lambdas, functions=functions, f_k=f, F_k=macro_var[name], X=X, coeffs=coeffs:
        constraint(f_k, lambdas, functions, F_k, X, coeffs)
    } for f, name in zip(functions, macro_var)]

    # Set bounds
    if len(functions) == 1:
        bounds = [(0, None)]
    else:
        min_l2, max_l2 = find_extremes(X, coeffs)
        bounds = [(0, None),(min_l2, max_l2)]

    # Perform optimization
    print("Starting Optimizing with constraints...")
    result = minimize(entropy,
                      lambdas,
                      args=(functions, X, coeffs),
                      constraints=constraints,
                      bounds=bounds,
                      method="trust-constr",
                      options={'maxiter': 200,
                               'xtol': 1e-6,
                               'gtol': 1e-12,
                               'barrier_tol': 1e-12,
                               'disp': True,
                               'verbose': 3
                               })

    optimized_lambdas = result.x

    return optimized_lambdas


def check_constraints(lambdas, functions, X, macro_var, coeffs):
    """
    Calculates the expected value of a single constraint function f_k over the ecosystem structure function:
    Σ f_k p_n / Z - F_k
    """
    absolute_errors = []
    percentage_errors = []

    for f_k, (key, F_k) in zip(functions, macro_var.items()):
        error = np.abs(constraint(f_k, lambdas, functions, F_k, X, coeffs))
        pct_error = error / np.abs(F_k) * 100

        absolute_errors.append(error)
        percentage_errors.append(pct_error)

    print("\n Errors on constraints:")
    print(f"{'Constraint':<10} {'Abs Error':>15} {'% Error':>15}")
    print("-" * 42)
    for key, abs_err, pct_err in zip(macro_var.keys(), absolute_errors, percentage_errors):
        print(f"{key:<10} {abs_err:15.6f} {pct_err:15.2f}")


def f_1(n, X, coeffs):
    return n

def dn(n, X, coeffs):
    c0, c1, c2, c3, c4, c5, c6 = coeffs
    return c0 + c1 * n + c2 * X['N_t'] + c3 * n**2 + c4 * n * X['N_t'] + c5 * X['N_t']**2 + c6

def find_extremes(X, coeffs):
    min_f = np.inf
    max_f = -np.inf

    extrema = [1, X['N_t']]
    if coeffs[2] != 0:
        extremum = -(coeffs[0] + coeffs[3] * X['N_t']) / (2 * coeffs[2]) # this is the zero point of the derivative of dn with respect to n
        if extremum > 0:
            extrema.append(extremum)

    # Find extrema of f_n
    for n in extrema:
        function_value = dn(n, X, coeffs)
        if function_value > max_f:
            max_f = function_value
        if function_value < min_f:
            min_f = function_value

    max_abs_f = max(abs(min_f), abs(max_f))

    return -500 / max_abs_f, 500 / max_abs_f    # calculate bounds given extrema of f_n

def compute_SAD_probabilities(lambdas, functions, X, coeffs):
    """
    Compute p_n = exp(-sum(lambda_i * f_i(n))) / Z for n in 1..N_max.
    Returns the SAD as a normalized numpy array.
    """
    n = np.arange(1, int(X['N_t']) + 1)
    exponent = np.zeros_like(n, dtype=float)

    for lam, f in zip(lambdas, functions):
        exponent += lam * f(n, X, coeffs)

    unnorm_p = np.exp(-exponent)
    Z = unnorm_p.sum()
    return unnorm_p / Z


def get_rank_abundance(p_n, X):
    """
    Generate a predicted rank-abundance distribution using the quantile method.
    Ensures exactly S_t values by clipping quantiles and handling edge cases.
    """
    S = int(X['S_t']) + 1

    # Create the discrete distribution
    n_vals = np.arange(1, len(p_n) + 1)
    dist = rv_discrete(name='sad_dist', values=(n_vals, p_n))

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


def compare_SADs(lambdas, functions, X, coeffs, empirical_rap, method, model, census, plot=True):
    """
    Compare predicted vs empirical rank abundance distributions.

    Parameters:
        lambdas: list of lambda values
        functions: list of constraint functions f_i(n, X, coeffs)
        X: dictionary of state variables (e.g., S_t, N_t, etc.)
        coeffs: parameters used by f_i
        empirical_rap: observed rank-abundance list or array
        plot: whether to display the plot (default: True)

    Returns:
        predicted_rap: predicted abundances
        rmse: root mean squared error
        aic: Akaike Information Criterion
    """

    # SAD from lambda parameters
    p_n = compute_SAD_probabilities(lambdas, functions, X, coeffs)

    # Predicted rank-abundance
    predicted_rap = get_rank_abundance(p_n, X)
    predicted_rap = predicted_rap[:len(empirical_rap)]
    empirical_rap = empirical_rap[:len(predicted_rap)]

    # RMSE
    rmse = np.sqrt(mean_squared_error(empirical_rap, predicted_rap))

    # AIC
    eps = 1e-10
    log_probs = np.log(p_n[predicted_rap - 1] + eps)  # -1 for indexing
    log_likelihood = np.sum(log_probs)
    k = len(initial_lambdas)
    aic = -2 * log_likelihood + 2 * k

    # Plot
    if plot:
        plt.figure(figsize=(8, 5))
        ranks = np.arange(1, len(empirical_rap) + 1)
        plt.plot(ranks, empirical_rap, 'o-', label='Empirical RAP', color='blue')
        plt.plot(ranks, predicted_rap, 's--', label='Predicted RAP', color='red')
        plt.xlabel('Rank')
        plt.ylabel('Abundance')
        #plt.yscale('log')
        #plt.title('Rank-Abundance Plot: Predicted vs. Empirical')
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
        #plt.show()
        plt.savefig(f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/LV/{method}/{model}_{census}.png')

    return predicted_rap, rmse, aic


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 16})

    # # METE
    # functions = [f_1]
    # for model in ['constant', 'food_web', 'cascading_food_web', 'cyclic', 'cleaner_fish', 'resource_competition']:
    # #for model in ['constant']:
    #     input = pd.read_csv(f'../../data/LV_{model}_regression_library.csv')
    #
    #     for census in input['census'].unique()[::500]:
    #         input_census = input[input['census'] == census]
    #
    #         X = input_census[[
    #             'S_t', 'N_t'
    #         ]].drop_duplicates().iloc[0]
    #
    #         macro_var = {
    #             'N/S': float(X['N_t'] / X['S_t'])
    #         }
    #
    #         grouped = input_census.groupby('species')['n'].sum()
    #         empirical_RAP = grouped.sort_values(ascending=False).values
    #
    #         # Make initial guess
    #         initial_lambdas = make_initial_guess(X, 'METE')
    #         initial_errors = check_constraints(initial_lambdas, functions, X, macro_var, [])
    #         print(f"Initial lambdas: {initial_lambdas}")
    #         #error = compare_SADs(initial_lambdas, functions, X, [], empirical_RAP)
    #
    #         # Perform optimization
    #         optimized_lambdas = perform_optimization(initial_lambdas, functions, macro_var, X, [])
    #         constraint_errors = check_constraints(optimized_lambdas, functions, X, macro_var, [])
    #         print(f"Optimized lambdas: {optimized_lambdas}")
    #         error = compare_SADs(optimized_lambdas, functions, X, [], empirical_RAP, 'METE', model, census)

            # METimE
    for model in ['constant', 'food_web', 'cascading_food_web', 'cyclic', 'cleaner_fish', 'resource_competition']:
        input = pd.read_csv(f'../../data/LV_{model}_regression_library.csv')
        coeffs = pd.read_csv(f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Data sets/LV/METimE_{model}_dn_global.csv')
        coeffs = coeffs['Coefficient'].tolist()
        functions = [f_1, dn]

        for census in input['census'].unique()[::500]:
            input_census = input[input['census'] == census]

            X = input_census[[
                'S_t', 'N_t'
            ]].drop_duplicates().iloc[0]

            macro_var = {
                'N/S': float(X['N_t'] / X['S_t']),
                'dN': input_census['dN'].unique()[0]
            }

            grouped = input_census.groupby('species')['n'].sum()
            empirical_RAP = grouped.sort_values(ascending=False).values

            # Make initial guess
            initial_lambdas = make_initial_guess(X, 'METimE')
            print(f"Initial lambdas: {initial_lambdas}")
            initial_errors = check_constraints(initial_lambdas, functions, X, macro_var, coeffs)
            #error = compare_SADs(initial_lambdas, functions, X, coeffs, empirical_RAP)

            # Perform optimization
            optimized_lambdas = perform_optimization(initial_lambdas, functions, macro_var, X, coeffs)
            print(f"Optimized lambdas: {optimized_lambdas}")
            constraint_errors = check_constraints(optimized_lambdas, functions, X, macro_var, coeffs)
            error = compare_SADs(optimized_lambdas, functions, X, coeffs, empirical_RAP, 'METimE', model, census)
