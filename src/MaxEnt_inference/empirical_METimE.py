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

def exp_in_R(n, e, X, functions, lambdas):
    """
    Compute the exponent term: -lambda1*f1 - ... - lambdak*fk
    """
    exponent = sum(-lambdas[i] * functions[i](n, e, X) for i in range(len(functions)))
    return exponent

def integrate_bin(n, a, b, lambdas, alphas, betas):
    #return quad(lambda e: np.exp(exp_in_R(n, e, X, functions, lambdas)), a, b)[0]
    I = quad(lambda e: np.exp(- lambdas[1] * n * e
                                  - lambdas[2] * (alphas[3] * e + alphas[4] * e * n + alphas[5] * e ** 2)
                                  - lambdas[3] * (betas[3] * e + betas[4] * e * n + betas[5] * e ** 2)
                               ), a, b)[0]

    if np.isinf(I):
        print("Warning: I may be invalid")

        # TODO: I is soms oneindig, dus de bounds voor lambdas[2] en lambas[3] zijn niet goed genoeg

    return I

def integrate_for_n(args):
    n, e_max, X, functions, lambdas, alphas, betas = args
    edges = np.linspace(0, 1, 11 + 1) ** 2 * e_max
    return (np.exp(- lambdas[2] * alphas[0] * X['E_t']
                  - lambdas[3] * betas[0] * X['E_t'])
            * sum(
                np.exp(- lambdas[0] * n
                       - lambdas[2] * (alphas[1] * n + alphas[2] * n ** 2)
                       - lambdas[3] * (betas[1] * n + betas[2] * n ** 2)) *
                integrate_bin(n, a, b, lambdas, alphas, betas)
        for a, b in zip(edges[:-1], edges[1:])
    ))

def integrate_with_cutoff(X, functions, lambdas):
    """
    scipy.integrate.quad() gave warnings and suggested doing the computation on multiple subranges.
    So we split it into 5 regions, and parallelize over `n` to speed up computations.
    """
    alphas = [- 4.2577836468050255e-07,
              0.004115429750029052,
              - 1.2340258201126042e-06,
              0.0007050089504230784,
              4.240319168064214e-06,
              - 8.857872661055533e-09]

    betas = [- 5.0790489992858706e-05,
             - 0.04641707200559591,
             1.196180145098891e-06,
             0.25103426364018466,
             0.0005510138124147476,
             7.71564817664272e-06]

    # TODO: moeten we steeds X, functions en lambdas meegeven?
    task_args = [
        (n, X['E_t'], X, functions, lambdas, alphas, betas)
        for n in range(1, int(X['N_t']))
    ]

    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(integrate_for_n, task_args))

    Z = sum(results)

    if np.isnan(Z) or np.isinf(Z) or Z == 0:
        print("Warning: Partition function Z may be invalid")

    return Z


###############################################
###              Initial Guess              ###
###############################################

# def compute_entropy_contribution(n, e_max, Z, X, functions, lambdas):
#     log_Z = np.log(Z)
#     edges = np.linspace(0, e_max, 6)
#
#     def integrand(e):
#         rexp = exp_in_R(n, e, X, functions, lambdas)
#         return np.exp(rexp) * (rexp - log_Z)
#
#     results = [quad(integrand, a, b)[0] for a, b in zip(edges[:-1], edges[1:])]
#     return np.sum(results)

def compute_entropy_contribution(n, e_max, Z, X, functions, lambdas, alphas, betas):
    log_Z = np.log(Z)
    edges = np.linspace(0, 1, 11 + 1) ** 2 * e_max

    def integrand(e):
        rexp = exp_in_R(n, e, X, functions, lambdas)

        return np.exp((-lambdas[1] * n * e
                      - lambdas[2] * (alphas[3] * e + alphas[4] * e * n + alphas[5] * e ** 2)
                      - lambdas[3] * (betas[3] * e + betas[4] * e * n + betas[5] * e ** 2)
                       )) * (rexp - log_Z)

    results = [np.exp(-(lambdas[0] * n
                        + lambdas[2] * (alphas[1] * n + alphas[2] * n ** 2)
                        + lambdas[3] * (betas[1] * n + betas[2] * n ** 2)))
               * quad(integrand, a, b)[0] for a, b in zip(edges[:-1], edges[1:])]
    return np.exp(-(lambdas[2] * alphas[0] + lambdas[3] * betas[0]) * X['E_t']) * np.sum(results)

def entropy(lambdas, functions, X, alphas, betas):
    """
    Compute Shannon entropy for the given lambdas and functions.
    Parallelized version.
    """
    Z = integrate_with_cutoff(X, functions, lambdas)

    # Find the maximum of n for which we are doing calculations
    n_max = int(max(4, -np.log(0.001) / lambdas[0]) + 1)

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                compute_entropy_contribution,
                n, (np.log(1000) - lambdas[0] * n) / (lambdas[1] * n), Z, X, functions, lambdas, alphas, betas
            )
            for n in range(1, n_max)
        ]
        contributions = np.fromiter((f.result() for f in futures), dtype=np.float64)

    I = contributions.sum() / Z

    if np.any(np.isnan(I)) or np.any(np.isinf(I)):
        print("Invalid values detected in entropy")

    return I


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

    # TODO: method can be either 'brentq' or 'bisect', but 'bisect' hasn't been tested yet

    # TODO: root_scalar can take multiple guesses x0, so maybe we can also give it the optimized values from the
    #  previous year

    return [l1, l2, 0, 0]


###############################################
###               Optimization              ###
###############################################

def compute_constraint_contribution(n, e_max, X, f_k, all_f, lambdas, alphas, betas):
    # def integrand(e):
    #     return f_k(n, e, X) * np.exp(exp_in_R(n, e, X, all_f, lambdas))

    def integrand(e):
        return f_k(n, e, X) * np.exp(-lambdas[1] * n * e
                                     - lambdas[2] * (alphas[3] * e + alphas[4] * e * n + alphas[5] * e ** 2)
                                     - lambdas[3] * (betas[3] * e + betas[4] * e * n + betas[5] * e ** 2)
                                     )

    edges = np.linspace(0, 1, 11 + 1) ** 2 * e_max
    return (np.exp(- lambdas[2] * alphas[0] * X['E_t']
                  - lambdas[3] * betas[0] * X['E_t']) *
            sum(
                np.exp(- lambdas[0] * n
                       - lambdas[2] * (alphas[1] * n + alphas[2] * n ** 2)
                       - lambdas[3] * (betas[1] * n + betas[2] * n ** 2)
                       ) *
            quad(integrand, a, b)[0]
                for a, b in zip(edges[:-1], edges[1:])
            )
            )


def constraint(f_k, lambdas, functions, F_k, X, alphas, betas):
    """
    Calculates the expected value of a single constraint function f_k over the ecosystem structure function:
    Σ ʃ f_k R(n, e) / Z de - F_k
    Parallelized version.
    """
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                compute_constraint_contribution,
                n, X['E_t'], X, f_k, functions, lambdas, alphas, betas
            )
            for n in range(1, int(X['N_t']) + 1)
        ]
        contributions = [f.result() for f in futures]

    I = sum(contributions)
    Z = integrate_with_cutoff(X, functions, lambdas)
    return I / Z - F_k


def perform_optimization(lambdas, functions, macro_var, X, alphas, betas):
    # Collect all constraints
    constraints = [{
        'type': 'eq',
        'fun': lambda lambdas, functions=functions, f_k=f, F_k=macro_var[name], X=X:
        constraint(f_k, lambdas, functions, F_k, X, alphas, betas)
    } for f, name in zip(functions, macro_var)]

    # Set bounds                                                                                                        # TODO: why these bounds?
    bounds = [(0, None), (0, None), (-8.18e-05, 2.49e-05), (-9.38e-08, 9.14e-07)] # these last two bounds should prevent overflow in exp()

    # -l3 f(n, e) < 500
    # l3 > 500 / f_dn(n, e, X) for all n, e
    # - l4 h(n, e) < 500

    def my_callback(xk, state):
        print("Current grad norm:", np.linalg.norm(state.grad))

    # Perform optimization
    print("Starting Optimizing with constraints...")
    result = minimize(entropy,
                      lambdas,
                      args=(functions, X, alphas, betas),
                      constraints=constraints,
                      bounds=bounds,
                      method="trust-constr",
                      callback=my_callback,
                      options={'maxiter': 100,
                               'xtol': 1e-6,
                               'disp': True,
                               'verbose': 3
                               })

    optimized_lambdas = result.x

    return optimized_lambdas

########################
### Set-up and check ###
########################

def f_n(n, e, X):
    return n

def f_ne(n, e, X):
    return n * e

def f_dn(n, e, X):
    return (0.0007050089504230784 * e
            + 0.004115429750029052 * n
            - 4.2577836468050255e-07 * X['E_t']
            - 8.857872661055533e-09 * e ** 2
            + 4.240319168064214e-06 * e * n
            - 1.2340258201126042e-06 * n ** 2)

def f_de(n, e, X):
    return (0.25103426364018466 * e
            - 0.04641707200559591 * n
            - 5.0790489992858706e-05 * X['E_t']
            - 7.71564817664272e-06 * e ** 2
            + 0.0005510138124147476 * e * n
            + 1.196180145098891e-06 * n ** 2)

def get_functions():
    return [f_n, f_ne, f_dn, f_de]                                                                                      # TODO: Maybe some scaling of the transition functions is beneficial

def check_constraints(lambdas, input, functions):
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

    Z = integrate_with_cutoff(X, functions, lambdas)
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
                    lambda e: f(n, e, X) * np.exp(exp_in_R(n, e, X, functions, lambdas)),
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


# Discretize the domain
def plot_dn(X):
    # Create a grid over the domain
    n_vals = np.linspace(1, X['N_t'] + 1, 100)
    e_vals = np.linspace(0, X['E_t'], 100)
    N, E = np.meshgrid(n_vals, e_vals)

    # Evaluate f_dn over the grid
    Z = f_dn(N, E, X)

    fig = go.Figure(data=[go.Surface(z=Z, x=N, y=E, colorscale='Viridis')])
    fig.update_layout(
        title='Interactive 3D Plot of f_dn(n, e)',
        scene=dict(
            xaxis_title='n',
            yaxis_title='e',
            zaxis_title='f_dn(n, e)'
        )
    )
    fig.show()

    return np.min(Z), np.max(Z)

def plot_de(X):
    # Create a grid over the domain
    n_vals = np.linspace(1, X['N_t'] + 1, 100)
    e_vals = np.linspace(0, X['E_t'], 100)
    N, E = np.meshgrid(n_vals, e_vals)

    # Evaluate f_dn over the grid
    Z = f_de(N, E, X)

    fig = go.Figure(data=[go.Surface(z=Z, x=N, y=E, colorscale='Viridis')])
    fig.update_layout(
        title='Interactive 3D Plot of f_de(n, e)',
        scene=dict(
            xaxis_title='n',
            yaxis_title='e',
            zaxis_title='f_de(n, e)'
        )
    )
    fig.show()

    return np.min(Z), np.max(Z)


if __name__ == "__main__":
    input = pd.read_csv('../../data/BCI_regression_library.csv')
    functions = get_functions()

    for census in input['census'].unique():
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

        alphas = [- 4.2577836468050255e-07,
                  0.004115429750029052,
                  - 1.2340258201126042e-06,
                  0.0007050089504230784,
                  4.240319168064214e-06,
                  - 8.857872661055533e-09]

        betas = [- 5.0790489992858706e-05,
                 - 0.04641707200559591,
                 1.196180145098891e-06,
                 0.25103426364018466,
                 0.0005510138124147476,
                 7.71564817664272e-06]

        #min_f, max_f = plot_dn(X)
        #min_h, max_h = plot_de(X)

        # (-8.187843758111391e-05, 2.491595315291577e-05), (-9.382999736191917e-08, 9.141265779971718e-07)

        # Make initial guess
        initial_lambdas = make_initial_guess(X)
        initial_errors = check_constraints(initial_lambdas, input_census, functions)
        print(f"Initial lambdas: {initial_lambdas}")

        # Perform optimization
        optimized_lambdas = perform_optimization(initial_lambdas, functions, macro_var, X, alphas, betas)
        constraint_errors = check_constraints(optimized_lambdas, input_census, functions)
        print(f"Optimized lambdas: {optimized_lambdas}")

# TODO: replace initial_guess with solution for previous census?
# TODO: the integral cutoffs for certain n and e maybe don't make sense with the new lambdas involved

"""
Uit tests blijkt dat het wel zin heeft (qua rekentijd) om de integraalterm zo simpel mogelijk te maken,
oftewel het zo ver mogelijk uitwerken
"""

# l1, l2, l3, l4 = [0.0001, 0.00002, 0.0003, 0.0005]
# N, E = int(2.353380e+05), 2.629646e+07
#
# start1 = time.time()
# sum1 = 0
# for n in range(1, N):
#     sum1 += quad(lambda e: np.exp(-l1 * n - l2 * n * e - l3 * (
#         7.05 * e + 4.11 * n - 4.25 * E - 8.86 * e ** 2 + 4.24 * e * n - 1.23 * n ** 2
#     ) - l4 * (
#         7.25 * e + 4.61 * n - 4.05 * E - 9.86 * e ** 2 + 4.14 * e * n - 1.73 * n ** 2
#     )), 0, E)[0]
# end1 = time.time()
#
# start2 = time.time()
# sum2 = 0
# for n in range(1, N):
#     sum2 += np.exp(-l1 * n - l3 * (4.11 * n - 1.23 * n ** 2) - l4 * (4.61 * n - 1.73 * n ** 2)) * quad(lambda e: np.exp(- l2 * n * e - l3 * (
#         7.05 * e  - 8.86 * e ** 2 + 4.24 * e * n ) - l4 * (
#         7.25 * e - 9.86 * e ** 2 + 4.14 * e * n )), 0, E)[0]
# sum2 *= np.exp(-(l3 * 4.25 + l4 * 4.05))
# end2 = time.time()
#
# print(f"Time for version 1: {end1 - start1:.6f} seconds")
# print(f"Time for version 2: {end2 - start2:.6f} seconds")
# print(f"Result version 1: {sum1}")
# print(f"Result version 2: {sum2}")
