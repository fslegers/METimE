import numpy as np
import pandas as pd
from scipy.optimize import root_scalar, minimize
from scipy.integrate import quad
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor

import warnings
# warnings.filterwarnings("ignore")

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

def R_exponent(n, e, X, functions, lambdas, scaling=False):
    """
    Compute the exponent term: -lambda1*f1 - ... - lambdak*fk.
    If scaling is True, each term is multiplied by an appropriate scalar.
    """
    if scaling:                                                                                                         # TODO: what is this based on?
        scalars = [1/X['S_t'], 1/(X['S_t']*X['E_t']),
                   1/max(X['S_t']**2, X['E_t']),
                   1/max(X['S_t']**2, X['E_t']),
                   1/max(X['S_t']**2, X['E_t'])]
        exponent = sum(-lambdas[i] * scalars[i] * functions[i](n, e, X) for i in range(len(functions)))
    else:
        exponent = sum(-lambdas[i] * functions[i](n, e, X) for i in range(len(functions)))

    return exponent

def find_support_cutoff(n, X, functions, lambdas, e_vals, threshold=5):
    """Find e where the integrand becomes negligible for a specific n."""
    log_vals = np.array([
        R_exponent(n, e, X, functions, lambdas) for e in e_vals
    ])
    valid = np.where(log_vals > threshold)[0]

    if len(valid) == 0:
        return e_vals[-1]

    return e_vals[valid[-1]]

def estimate_cutoff_function(X, functions, lambdas, num_samples=100, threshold=5):
    """Sample e_cutoff(n) and fit a smooth interpolated function."""
    sample_ns = np.linspace(1, int(X['N_t']), num_samples, dtype=int)
    e_vals = np.linspace(0, X['E_t'], 1000)

    cutoff_points = []
    for n in sample_ns:
        e_cut = find_support_cutoff(n, X, functions, lambdas, e_vals, threshold)
        cutoff_points.append(e_cut)

    # Fit smooth function (interpolation or parametric model)
    cutoff_func = interp1d(sample_ns, cutoff_points, kind='cubic', fill_value="extrapolate")

    return cutoff_func

# def integrate_with_cutoff(X, functions, lambdas, threshold=5):
#     """
#     scipy.integrate.quad() gave warnings and suggested doing the computation on multiple subranges.
#     So we split it into 5 regions.
#     """
#     cutoff_func = estimate_cutoff_function(X, functions, lambdas, threshold=threshold)
#
#     Z = 0.0
#     for n in range(1, int(X['N_t']) + 1):
#         e_max = float(cutoff_func(n))
#         edges = np.linspace(0, e_max, 6) # 5 bins
#         Z += sum(
#             quad(lambda e: np.exp(R_exponent(n, e, X, functions, lambdas)), a, b)[0]
#             for a, b in zip(edges[:-1], edges[1:])
#         )
#
#     if np.isnan(Z) or np.isinf(Z) or Z == 0:
#         print("Warning: Partition function Z may be invalid")
#
#     return Z

def integrate_bin(n, a, b, X, functions, lambdas):
    return quad(lambda e: np.exp(R_exponent(n, e, X, functions, lambdas)), a, b)[0]

def integrate_for_n(n, cutoff_func, X, functions, lambdas):
    e_max = float(cutoff_func(n))
    edges = np.linspace(0, e_max, 6)  # 5 bins
    return sum(
        integrate_bin(n, a, b, X, functions, lambdas)
        for a, b in zip(edges[:-1], edges[1:])
    )

def integrate_with_cutoff(X, functions, lambdas, threshold=5):
    """
    scipy.integrate.quad() gave warnings and suggested doing the computation on multiple subranges.
    So we split it into 5 regions, and parallelize over `n` to speed up computations.
    """
    cutoff_func = estimate_cutoff_function(X, functions, lambdas, threshold=threshold)
    N_t = int(X['N_t'])

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(integrate_for_n, n, cutoff_func, X, functions, lambdas)
            for n in range(1, N_t + 1)
        ]
        results = [f.result() for f in futures]

    Z = sum(results)

    if np.isnan(Z) or np.isinf(Z) or Z == 0:
        print("Warning: Partition function Z may be invalid")

    return Z

def partition_function(lambdas, functions, X, scaling=False):
    """
    Compute the partition function Z.
    Z = sum ∫ de exp( R_exponent(n,e,X) )
    """
    # def integrand(e, m):
    #     return np.exp(R_exponent(m, e, X, functions, lambdas, scaling=scaling))
    #
    # N, E = int(X['N_t']), X['E_t']

    # quad is not stable when evaluated on a region where a lot of zero-values are sampled
    # so we need to split the area of interest (large negative exponents) from the area with very low values
    # (small negative and positive exponents), which can happen in dynaMETE and METE

    # TODO:
    # Idea is to define a function (of n) that determines a cutoff for the region of e for integration,
    # by sampling e and n values and checking when the value is smaller than some threshold. Use this
    # region in quad for more stable results.

    # TODO:
    # Do we run into problems with too high values of quad? Do we need to bound the lambda values to prevent this
    # from happening, based on the functions and state variables?

    Z = integrate_with_cutoff(X, functions, lambdas)

    # # Create bin edges for left and right of the cut
    # num_bins = 1
    # left_edges = np.linspace(0, first_cut, num_bins + 1)
    # right_edges = np.linspace(first_cut, E, num_bins + 1)
    #
    # Z = 0
    #
    # for m in range(1, N + 1):
    #     for i in range(num_bins):
    #         # Left side bin
    #         a, b = left_edges[i], left_edges[i + 1]
    #         result = quad(lambda e: integrand(e, m), a, b)[0]
    #         Z += result
    #
    #         # Right side bin
    #         a, b = right_edges[i], right_edges[i + 1]
    #         result = quad(lambda e: integrand(e, m), a, b)[0]
    #         Z += result

    # Z = sum(
    #     fixed_quad(
    #         lambda e: np.exp(R_exponent(m, e, X, functions, lambdas, scaling=scaling)),
    #         0, split_point
    #     )[0] +
    #     fixed_quad(
    #         lambda e: np.exp(R_exponent(m, e, X, functions, lambdas, scaling=scaling)),
    #         split_point, E, n=5
    #     )[0]
    #     for m in range(1, N + 1)
    # )

    # if np.isnan(Z) or np.isinf(Z) or Z == 0:
    #     print("Invalid values encountered in partition function Z")

    return Z

# def entropy(lambdas, functions, X, scaling=False):
#     """
#     Compute Shannon entropy for the given lambdas and functions
#     """
#     N, E = int(X['N_t']), X['E_t']
#
#     Z = partition_function(lambdas, functions, X, scaling)
#
#     def integrand(e, m, state_variables, functions, lambdas, scaling):
#         rexp = R_exponent(m, e, state_variables, functions, lambdas, scaling)
#         return np.exp(rexp) * (rexp - np.log(Z))
#
#     I = 0
#     cutoff_func = estimate_cutoff_function(X, functions, lambdas, scaling, threshold=5)
#     for n in range(1, int(X['N_t']) + 1):
#         e_max = float(cutoff_func(n))
#         edges = np.linspace(0, e_max, 6)
#         I += sum(
#             quad(lambda e: integrand(e, n, X, functions, lambdas, scaling), a, b)[0]
#             for a, b in zip(edges[:-1], edges[1:])
#         )
#     I /= Z
#
#     # I = sum(                                                                                                            # TODO: what does split_point do?
#     #     quad(
#     #         lambda e: integrand(e, i, X, functions, lambdas, scaling),
#     #         0, E
#     #     )[0]
#     #     for i in range(1, N + 1)
#     # ) / Z
#
#     if np.any(np.isnan(I)) or np.any(np.isinf(I)):
#         print("Invalid values detected in entropy")
#
#     return I


###############################################
###              Initial Guess              ###
###############################################

def compute_entropy_contribution(n, e_max, Z, X, functions, lambdas, scaling):
    def integrand(e):
        rexp = R_exponent(n, e, X, functions, lambdas, scaling)
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
    N = int(X['N_t'])

    Z = partition_function(lambdas, functions, X, scaling)
    cutoff_func = estimate_cutoff_function(X, functions, lambdas, scaling, threshold=5)
    e_max_list = [float(cutoff_func(n)) for n in range(1, N + 1)]

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                compute_entropy_contribution,
                n, e_max_list[n - 1], Z, X, functions, lambdas, scaling
            )
            for n in range(1, N + 1)
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

def make_initial_guess(X, method, scaling=False):
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

    if method == "METE":
        return [l1, l2]

    return [l1, l2, 0, 0]


###############################################
###               Optimization              ###
###############################################

# def constr_integrand(e, n, X, f_k, all_f, lambdas):
#     """
#     the integrand used to calculate the expected value of a constraint function over the ecosystem structure function:
#     f_k R(n, e) / Z
#     """
#     return f_k(n, e, X) * np.exp(R_exponent(n, e, X, all_f, lambdas))                                                   # TODO: order of parameters is off

# def constraint(f_k, lambdas, functions, F_k, X):
#     """
#     Calculates the expected value of a single constraint function f_k over the ecosystem structure function:
#     Σ ʃ f_k R(n, e) / Z de - F_k
#     """
#     cutoff_func = estimate_cutoff_function(X, functions, lambdas)
#
#     I = 0
#     for n in range(1, int(X['N_t']) + 1):
#         e_max = float(cutoff_func(n))
#         edges = np.linspace(0, e_max, 6)  # 5 bins
#         I += sum(
#             quad(constr_integrand, a, b, args=(n, X, f_k, functions, lambdas))[0]
#             for a, b in zip(edges[:-1], edges[1:])
#         )
#     # integral_value = sum(                                                                                               # TODO: change this quad to work with bounds
#     #     quad(constr_integrand, 0, X['E_t'],
#     #                args=(n, X, f_k, functions, lambdas))[0]
#     #     for n in range(1, int(X['N_t']))
#     # )
#     constr = I /  partition_function(lambdas, functions, X) - F_k
#     return constr

def compute_constraint_contribution(n, e_max, X, f_k, all_f, lambdas):
    def integrand(e):
        return f_k(n, e, X) * np.exp(R_exponent(n, e, X, all_f, lambdas))

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
    cutoff_func = estimate_cutoff_function(X, functions, lambdas)
    N = int(X['N_t'])
    e_max_list = [float(cutoff_func(n)) for n in range(1, N + 1)]

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                compute_constraint_contribution,
                n, e_max_list[n - 1], X, f_k, functions, lambdas
            )
            for n in range(1, N + 1)
        ]
        contributions = [f.result() for f in futures]

    I = sum(contributions)
    Z = partition_function(lambdas, functions, X)
    return I / Z - F_k

def perform_optimization(lambdas, functions, macro_var, X):
    # Collect all constraints
    constraints = [{
        'type': 'eq',
        'fun': lambda lambdas, functions=functions, f_k=f, F_k=macro_var[name], X=X:
        constraint(f_k, lambdas, functions, F_k, X)
    } for f, name in zip(functions, macro_var)]

    # Set bounds for realistic lambda values
    if len(lambdas) == 2:                                                                                               # TODO: why these bounds?
        bounds = [(0, None), (0, None)]
    else:
        bounds = [(-141 / X['N_t'], None), (-141 / (X['N_t'] * X['E_t']), None), (-141 / X['N_t'], 141 / X['N_t']),
                  (-141 / X['E_t'], 141 / X['E_t']), (-141 / X['S_t'], 141 / X['S_t'])]

    # Perform optimization
    result = minimize(entropy, lambdas, args=(functions, X), constraints=constraints, bounds=bounds,
                      method="trust-constr", options={'maxiter': 3, 'disp': True})

    optimized_lambdas = result.x

    return optimized_lambdas

def transform_lambdas(lambdas, N, E):
    """
    Transform the lambdas according to the given scaling rules.
    Input are optimized lagrange multipliers, which need to be scaled so that they are smaller again
    """
    lambda_1, lambda_2, lambda_3, lambda_4, lambda_5 = lambdas[0], lambdas[1], lambdas[2], lambdas[3], lambdas[4]

    # Apply transformations:
    lambda_1_transformed = lambda_1 / N
    lambda_2_transformed = lambda_2 / (N * E)
    lambda_3_transformed = lambda_3 / max(N**2, E)
    lambda_4_transformed = lambda_4 / max(N ** 2, E)
    lambda_5_transformed = lambda_5 / max(N ** 2, E)

    return [lambda_1_transformed, lambda_2_transformed, lambda_3_transformed, lambda_4_transformed, lambda_5_transformed]

########################
### Set-up and check ###
########################

def get_functions(method):
    functions = [lambda n, e, X: n,
                 lambda n, e, X: n * e]

    if method == 'METimE':
        functions.append(lambda n, e, X: 0.0007050089504230784 * e +
                                         0.004115429750029052 * n -
                                         4.2577836468050255e-07 * X['E'] -
                                         8.857872661055533e-09 * e ** 2 +
                                         4.240319168064214e-06 * n * e -
                                         1.2340258201126042e-06 * n ** 2
                         )

        functions.append(lambda n, e, X: 0.25103426364018466 * e +
                                         -0.04641707200559591 * n +
                                         -5.0790489992858706e-05 * X['E'] +
                                         -7.71564817664272e-06 * e ** 2 +
                                         0.0005510138124147476 * n * e +
                                         1.196180145098891e-06 * n ** 2
                         )

    # elif method == 'dynaMETE':
    #     p = pd.read_csv('../../data/dynaMETE_parameters_BCI.csv')
    #
    #     functions.append(lambda n, e, X: (p['b'] - p['d'] * X[2] / p['Ec']) * n / e ** (1 / 3) + p['m'] * n / X[1])
    #     functions.append(
    #         lambda n, e, X: (p['w'] - p['d'] * X[2] / p['Ec']) * n * e ** (2 / 3) - p['w1'] * n * e / np.log(1 / X[3]) ** (
    #                 2 / 3) + p['m'] * n / X[1])
    #     functions.append(lambda n, e, X: p['m'] * np.exp(-p['mu_meta'] * X[0] - np.euler_gamma) + (
    #             p['sigma_1'] * p['K'] / (p['K'] + X[0]) + p['sigma_2'] * p['b'] * n / e ** (1 / 3) - (
    #             int(np.rint(n)) == 1) * p['d'] / e ** (1 / 3) * X[2] / p['Ec']) * X[0])

    # TODO: add sim_BCI

    # TODO: Maybe some scaling of the transition functions is beneficial

    return functions

def check_constraints(lambdas, input, functions):
    """
    Returns the error on constraints given some lambda values
    Given in percentage of the observed value
    """
    S, N, E = int(input['S_t'].drop_duplicates().iloc[0]), int(input['N_t'].drop_duplicates().iloc[0]), input['E_t'].drop_duplicates().iloc[0]

    X = {
        'S_t': S,
        'N_t': N,
        'E_t': E
    }

    macro_var = {
        'N/S': X['N_t']/X['S_t'],
        'E/S': X['E_t']/X['S_t']
    }

    if method != "METE":
        macro_var['dN'] = input['dN'].drop_duplicates().iloc[0]
        macro_var['dE'] = input['dE'].drop_duplicates().iloc[0]

    Z = partition_function(lambdas, functions, X)

    absolute_errors = []
    percentage_errors = []

    for f, (key, v) in zip(functions, macro_var.items()):
        # Compute integral with upper bound
        integral_value = 0
        for n in range(1, N + 1):
            integral_value += quad(
                lambda e: f(n, e, X) * np.exp(R_exponent(n, e, X, functions, lambdas)),
                0, E
            )[0]

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


# def get_input():
#     input = pd.read_csv('../../data/BCI_regression_library.csv')
#
#     # Split into micro and macro
#     micro = input[['species', 'TreeID', 'census', 'e', 'n']]
#     macro = input[['census', 'S_t', 'N_t', 'E_t', 'dS', 'dN', 'dE']]
#     macro = macro.drop_duplicates()
#
#     return micro, macro


if __name__ == "__main__":

    input = pd.read_csv('../../data/BCI_regression_library.csv')

    for method in ['METE', 'METimE']:

        functions = get_functions(method)

        for census in input['census'].unique():
            input_census = input[input['census'] == census]

            X = input_census[[
                'S_t', 'N_t', 'E_t',
            ]].drop_duplicates().iloc[0]

            macro_var = {
                    'N/S': float(X['N_t'] / X['S_t']),
                    'E/S': float(X['E_t'] / X['S_t'])
                }

            if method != 'METE':
                macro_var['dN'] = float(input_census['dN'].drop_duplicates().iloc[0])
                macro_var['dE'] = float(input_census['dE'].drop_duplicates().iloc[0])

            # Make initial guess                                                                                        # TODO: what does scaling do?
            initial_lambdas = make_initial_guess(X, method, scaling=False)                                              # TODO: start from previous guess?
            initial_errors = check_constraints(initial_lambdas, input_census, functions)

            # Perform optimization
            optimized_lambdas = perform_optimization(initial_lambdas, functions, macro_var, X)
            constraint_errors = check_constraints(optimized_lambdas, input_census, functions)
