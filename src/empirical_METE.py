import numpy as np
import pandas as pd
from scipy.optimize import root_scalar, minimize
from scipy.integrate import quad
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor

import warnings
#warnings.filterwarnings("ignore")

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

def integrate_bin(n, a, b, X, functions, lambdas):
    return quad(lambda e: np.exp(exp_in_R(n, e, X, functions, lambdas)), a, b)[0]

def integrate_for_n(args):
    n, e_max, X, functions, lambdas = args
    edges = np.linspace(0, e_max, 6)  # 5 bins
    return sum(
        integrate_bin(n, a, b, X, functions, lambdas)
        for a, b in zip(edges[:-1], edges[1:])
    )

def integrate_with_cutoff(X, functions, lambdas):
    """
    scipy.integrate.quad() gave warnings and suggested doing the computation on multiple subranges.
    So we split it into 5 regions, and parallelize over `n` to speed up computations.
    """
    # Find the maximum of n for which we are doing calculations
    n_max = int(max(4, -np.log(0.001)/lambdas[0]) + 1)

    task_args = [
        (n, (np.log(1000) - lambdas[0] * n)/(lambdas[1] * n), X, functions, lambdas)
        for n in range(1, n_max)
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

def compute_entropy_contribution(n, e_max, Z, X, functions, lambdas, scaling):
    def integrand(e):
        rexp = exp_in_R(n, e, X, functions, lambdas)
        return np.exp(rexp) * (rexp - np.log(Z))

    edges = np.linspace(0, e_max, 6)
    return sum(
        quad(integrand, a, b)[0]
        for a, b in zip(edges[:-1], edges[1:])
    )

def entropy(lambdas, functions, X, scaling=False):
    """
    Compute Shannon entropy for the given lambdas and functions.
    Parallelized version.
    """
    Z = integrate_with_cutoff(X, functions, lambdas)

    # Find the maximum of n for which we are doing calculations
    n_max = int(max(4, -np.log(0.001)/lambdas[0]) + 1)

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                compute_entropy_contribution,
                n, (np.log(1000) - lambdas[0] * n) / (lambdas[1] * n), Z, X, functions, lambdas, scaling
            )
            for n in range(1, n_max)
        ]
        contributions = [f.result() for f in futures]

    I = sum(contributions) / Z

    if np.any(np.isnan(I)) or np.any(np.isinf(I)):
        print("Invalid values detected in entropy")

    return I

def beta_function(beta, S, N):
    """
    Beta function used to generate the initial guess for Lagrange multipliers.
    """
    return (1 - np.exp(-beta)) / (np.exp(-beta) - np.exp(-beta*(N + 1))) * np.log(1.0/beta) - S/N

def make_initial_guess(X, scaling=False):
    """
    A function that makes an initial guess for the Lagrange multipliers lambda1 and lambda2.
    Based on Eq 7.29 from Harte 2011 and meteR'diag function meteESF.mete.lambda

    :param state_variables: state variables S, S and E
    :return: initial guess for the Lagrange multipliers lambda1 and lambda2
    """
    S, N, E = int(X['S_t']), int(X['N_t']), float(X['E_t'])
    interval = [1.0/N, S/N]

    beta = root_scalar(beta_function, x0=0.001, args=(S, N), method='brentq', bracket=interval)

    l2 = S / (E - N)
    l1 = beta.root - l2

    # TODO: method can be either 'brentq' or 'bisect', but 'bisect' hasn't been tested yet

    # TODO: root_scalar can take multiple guesses x0, so maybe we can also give it the optimized values from the
    #  previous year

    if scaling:
        l1 = l1 * N
        l2 = l2 * (N * E)

    return [l1, l2]

###############################################
###               Optimization              ###
###############################################

def compute_constraint_contribution(n, e_max, X, f_k, all_f, lambdas):
    def integrand(e):
        return f_k(n, e, X) * np.exp(exp_in_R(n, e, X, all_f, lambdas))

    edges = np.linspace(0, e_max, 6)  # 5 bins
    return sum(
        quad(integrand, a, b)[0]
        for a, b in zip(edges[:-1], edges[1:])
    )

def constraint(f_k, lambdas, functions, F_k, X):
    """
    Calculates the expected value of a single constraint function f_k over the ecosystem structure function:
    Σ ʃ f_k R(n, e) / Z de - F_k
    Parallelized version.
    """
    # Find the maximum of n for which we are doing calculations
    # If an integral returns a value smaller than 0.001, neglect it
    n_max = int(max(4, -np.log(0.001)/lambdas[0]) + 1)

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                compute_constraint_contribution,
                n, (np.log(1000) - lambdas[0] * n) / (lambdas[1] * n), X, f_k, functions, lambdas
            )
            for n in range(1, n_max)
        ]
        contributions = [f.result() for f in futures]

    I = sum(contributions)
    Z = integrate_with_cutoff(X, functions, lambdas)
    return I / Z - F_k

def perform_optimization(lambdas, functions, macro_var, X):
    # Collect all constraints
    constraints = [{
        'type': 'eq',
        'fun': lambda lambdas, functions=functions, f_k=f, F_k=macro_var[name], X=X:
        constraint(f_k, lambdas, functions, F_k, X)
    } for f, name in zip(functions, macro_var)]

    # Set bounds for realistic lambda values                                                                            # TODO: why these bounds?
    bounds = [(0, None), (0, None)]

    def my_callback(xk, state):
        print("Current grad norm:", np.linalg.norm(state.grad))

    # Perform optimization
    print("Starting Optimizing with constraints...")
    result = minimize(entropy,
                      lambdas,
                      args=(functions, X),
                      constraints=constraints,
                      bounds=bounds,
                      method="trust-constr",
                      callback=my_callback,
                      options={'maxiter': 100,
                               'gtol': 1e-5,
                               'disp': True,
                               'verbose': 3
                               })

    optimized_lambdas = result.x

    return optimized_lambdas

def transform_lambdas(lambdas, N, E):
    """
    Transform the lambdas according to the given scaling rules.
    Input are optimized lagrange multipliers, which need to be scaled so that they are smaller again
    """
    lambda_1, lambda_2 = lambdas[0], lambdas[1]

    # Apply transformations:
    lambda_1_transformed = lambda_1 / N
    lambda_2_transformed = lambda_2 / (N * E)

    return [lambda_1_transformed, lambda_2_transformed]

########################
### Set-up and check ###
########################

def f_n(n, e, X):
    return n

def f_ne(n, e, X):
    return n * e

def get_functions():
    return [f_n, f_ne]                                                                                                  # TODO: Maybe some scaling of the transition functions is beneficial

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
        'N/S': X['N_t']/X['S_t'],
        'E/S': X['E_t']/X['S_t']
    }

    Z = integrate_with_cutoff(X, functions, lambdas)
    # We checked that Z is off only 0.13% from the analytically calculated Z

    absolute_errors = []
    percentage_errors = []

    # Faster computations: use n_max
    n_max = int(max(4, -np.log(0.001) / lambdas[0]) + 1)

    for f, (key, v) in zip(functions, macro_var.items()):
        # Compute integral with upper bound
        integral_value = 0
        for n in range(1, n_max):
            e_max = (np.log(1000) - lambdas[0] * n) / (lambdas[1] * n)
            edges = np.linspace(0, e_max, 6)  # 5 bins
            integral_value += sum(
                quad(
                lambda e: f(n, e, X) * np.exp(exp_in_R(n, e, X, functions, lambdas)),
                a, b
            )[0]
                for a,b in zip(edges[:-1], edges[1:])
            )

        integral_value /= Z                                                                                             # TODO: check

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

# def check_constraints(lambdas, input, functions):
#     """
#     Returns the error on constraints given some lambda values
#     Given in percentage of the observed value
#     """
#     S, N, E = (int(input['S_t'].drop_duplicates().iloc[0]),
#                int(input['N_t'].drop_duplicates().iloc[0]),
#                input['E_t'].drop_duplicates().iloc[0])
#
#     X = {
#         'S_t': S,
#         'N_t': N,
#         'E_t': E
#     }
#
#     macro_var = {
#         'N/S': X['N_t'] / X['S_t'],
#         'E/S': X['E_t'] / X['S_t']
#     }
#
#     Z = integrate_with_cutoff(X, functions, lambdas)
#
#     def compute_integral_for_function(args):
#         f, n, e_max = args
#         edges = np.linspace(0, e_max, 6)
#         return sum(
#             quad(
#                 lambda e: f(n, e, X) * np.exp(exp_in_R(n, e, X, functions, lambdas)),
#                 a, b
#             )[0]
#             for a, b in zip(edges[:-1], edges[1:])
#         )
#
#     n_max = int(max(4, -np.log(0.001) / lambdas[0]) + 1)
#     absolute_errors = []
#     percentage_errors = []
#
#     with ProcessPoolExecutor(max_workers=2) as executor:
#         for f, (key, v) in zip(functions, macro_var.items()):
#             # Prepare all tasks for this function
#             tasks = [(f, n, (np.log(1000) - lambdas[0] * n) / (lambdas[1] * n))
#                      for n in range(1, n_max)]
#
#             # Compute all integrals in parallel
#             results = list(executor.map(compute_integral_for_function, tasks))
#             integral_value = sum(results) / Z
#
#             # Compute errors
#             abs_error = np.abs(integral_value - v)
#             pct_error = abs_error / np.abs(v) * 100
#             absolute_errors.append(abs_error)
#             percentage_errors.append(pct_error)
#
#     print("\n Errors on constraints:")
#     print(f"{'Constraint':<10} {'Abs Error':>15} {'% Error':>15}")
#     print("-" * 42)
#     for key, abs_err, pct_err in zip(macro_var.keys(), absolute_errors, percentage_errors):
#         print(f"{key:<10} {abs_err:15.6f} {pct_err:15.2f}")
#
#     return absolute_errors


if __name__ == "__main__":
    input = pd.read_csv('../data/BCI_regression_library.csv')
    functions = get_functions()

    for census in input['census'].unique():
        input_census = input[input['census'] == census]

        X = input_census[[
            'S_t', 'N_t', 'E_t',
        ]].drop_duplicates().iloc[0]

        macro_var = {
                'N/S': float(X['N_t'] / X['S_t']),
                'E/S': float(X['E_t'] / X['S_t'])
            }

        # Make initial guess                                                                                            # TODO: what does scaling do?
        initial_lambdas = make_initial_guess(X, scaling=False)                                                          # TODO: start from previous guess?
        #initial_errors = check_constraints(initial_lambdas, input_census, functions)

        # Perform optimization
        optimized_lambdas = perform_optimization(initial_lambdas, functions, macro_var, X)
        constraint_errors = check_constraints(optimized_lambdas, input_census, functions)
