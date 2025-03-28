import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
from scipy.optimize import minimize
from scipy.integrate import quad, fixed_quad
from itertools import islice

import warnings
warnings.filterwarnings("ignore")

"""
Requires:
    f_k = list of functions (at least n, ne, fut maybe also transition functions)
    F_k = list of values
    N, E to determine the summation and integral ranges

Assumption:
    R(n, e) = exp( - lambda_1 f_1 - lambda_2 f_2 ...) / Z

Will maximize entropy over R(n, e) while satisfying constraints sum int f_i R(n, e) = F_i for all i
"""

def R_exponent(n, e, X, functions, lambdas, scaling=True):
    if scaling:
        scalars = [1/X['N'], 1/(X['N']*X['E']), 1/max(X['N']**2, X['E']), 1/max(X['N']**2, X['E']), 1/max(X['N']**2, X['E'])]
        exponent = sum(-lambdas[i] * scalars[i] * functions[i](n, e, X) for i in range(len(functions)))
    else:
        exponent = sum(-lambdas[i] * functions[i](n, e, X) for i in range(len(functions)))

    if np.any(np.isnan(exponent)) or np.any(np.isinf(exponent)):
        print("Invalid values detected in exponent")

    return exponent


def integrand_Z(e, m, X, functions, lambdas, scaling):
    return np.exp(R_exponent(m, e, X, functions, lambdas, scaling=scaling))



def partition_function(lambdas, functions, state_variables, scaling=True):
    _, N, E, _ = state_variables
    split_point = min((-np.log(0.1) + lambdas[0] / lambdas[1], E))
    N = int(N)

    Z = sum(
        fixed_quad(
            lambda e: np.exp(R_exponent(m, e, state_variables, functions, lambdas, scaling=scaling)),
            0, split_point
        )[0] +
        fixed_quad(
            lambda e: np.exp(R_exponent(m, e, state_variables, functions, lambdas, scaling=scaling)),
            split_point, E, n=5
        )[0]
        for m in range(1, N + 1)
    )

    if np.isnan(Z) or np.isinf(Z) or Z == 0:
        print("Invalid values")

    return Z


def entropy(lambdas, functions, state_variables, scaling=True):
    _, N, E, _ = state_variables
    N = int(N)
    split_point = min((-np.log(0.1) + lambdas[0] / lambdas[1], E))

    Z = partition_function(lambdas, functions, state_variables, scaling)

    def integrand(e, m, state_variables, functions, lambdas, scaling):
        rexp = R_exponent(m, e, state_variables, functions, lambdas, scaling)
        return np.exp(rexp) * (rexp - np.log(Z))

    I = sum(
        fixed_quad(
            lambda e: integrand(e, i, state_variables, functions, lambdas, scaling),
            0, split_point, n=5
        )[0] +
        fixed_quad(
            lambda e: integrand(e, i, state_variables, functions, lambdas, scaling),
            split_point, E, n=5
        )[0]
        for i in range(1, N + 1)
    ) / Z

    if np.any(np.isnan(I)) or np.any(np.isinf(I)):
        print("Invalid values detected in exponent")

    return I


def beta_function(beta, S, N):
    return (1 - np.exp(-beta)) / (np.exp(-beta) - np.exp(-beta*(N + 1))) * np.log(1.0/beta) - S/N


def make_initial_guess(state_variables, scaling=True):
    """
    A function that makes an initial guess for the Lagrange multipliers lambda1 and lambda2.
    Based on Eq 7.29 from Harte 2011 and meteR's function meteESF.mete.lambda

    :param state_variables: state variables S, N and E
    :return: initial guess for the Lagrange multipliers lambda1 and lambda2
    """
    S, N, E, beta = state_variables
    S, N = int(S), int(N)
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

    return [l1, l2, 0, 0, 0]


def check_constraints(lambdas, state_variables, functions, values):
    """
    Requires lambda values (so optional scaling has to be reversed beforehand), which is done automatically in
    perform_optimization.
    :param lambdas:
    :param state_variables:
    :param functions:
    :param values:
    :return:
    """
    S, N, E, beta = state_variables
    S, N = int(S), int(N)
    Z = partition_function(lambdas, functions, state_variables, scaling=False)

    constraint_errors = []

    for f, v in zip(functions, values):
        # Compute integral with upper bound
        integral_value = sum(
            quad(
                lambda e: f(n, e, state_variables) * np.exp(R_exponent(n, e, state_variables, functions, lambdas, scaling=False)),
                0,
                min(E, np.ceil((-np.log(0.1) + lambdas[0]) / max(lambdas[1], 1e-8)))  # Avoid division by zero
            )[0]
            for n in range(1, N + 1)
        )
        integral_value = integral_value / Z # TODO: check

        # Compute constraint error
        error = integral_value - v
        constraint_errors.append(error)

    # # Print constraint errors
    # for i, error in enumerate(constraint_errors):
    #     print("Error on constraint {i} ({val}): {error}".format(i=i, error=error, val=values[i]))
    return constraint_errors


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


def perform_optimization(lambdas, functions, values, state_variables, scaling=True):
    S, N, E, beta = state_variables
    split_point = min((-np.log(0.1) + lambdas[0] / lambdas[1], E))
    N = int(N)

    def constraint_func(state_variables, functions, lambdas, func, val):
        def integrand(e, m, state_variables, functions, lambdas, func):
            return func(m, e, state_variables) * np.exp(R_exponent(m, e, state_variables, functions, lambdas))

        integral_value = sum(
            fixed_quad(integrand, 0, split_point, n=10, args=(i, state_variables, functions, lambdas, func))[0] +
            fixed_quad(integrand, split_point, E, n=5, args=(i, state_variables, functions, lambdas, func))[0]
            for i in range(1, int(state_variables['N'])+1)
        )
        return integral_value / partition_function(lambdas, functions, state_variables) - val

    constraints = [{'type': 'eq', 'fun': lambda lambdas, f=f, v=v: constraint_func(state_variables, functions, lambdas, f, v)} for f, v in
                   zip(functions, values)]

    bounds = [(-141 / N, None), (-141 / (N * E), None), (-141 / N, 141 / N), (-141 / E, 141 / E), (-141 / S, 141 / S)]

    result = minimize(entropy, lambdas, args=(functions, state_variables), constraints=constraints, bounds=bounds,
                      method="trust-constr", options={'maxiter': 1000, 'disp': True})

    optimized_lambdas = result.x

    if scaling:
        transform_lambdas(optimized_lambdas, N, E)

    return optimized_lambdas


def get_functions(data_set, method):
    functions = [lambda n, e, X: n / X['N'],                #TODO: check if this scaling helps
                 lambda n, e, X: n * e / (X['N'] * X['E'])] #TODO: check if this scaling helps

    if data_set == 'fish' and method == 'METimE':
        # f(n, e)
        functions.append(lambda n, e, X: (-0.0004042815829500911*n**2 + 600.0316142392553*n/X['N'] + 0.3664776728697916*X['S'])/max(X['N']**2, X['E']))

        # q(n, e)
        functions.append(lambda n, e, X: (1.7336323484424592*10**(-5) * n**2 - 28.51933175306022*n/X['N'] - 0.5985205959246045*X['S'])/max(X['N']**2, X['E']))

    elif data_set == 'birds':
        
        if method == 'METimE':
            functions.append(lambda n, e, X: 0)
            functions.append(lambda n, e, X: 0)
            functions.append(lambda n, e, X: 0)
            
        elif method == 'dynaMETE':
            p = pd.read_csv('../../data/dynaMETE_parameters_birds.csv')
            p = dict(zip(p['Unnamed: 0'], p['Value']))
            
            functions.append(lambda n, e, X: (p['b'] - p['d'] * X[2]/p['Ec']) * n / e**(1/3) + p['m'] * n / X[1])
            functions.append(lambda n, e, X: (p['w'] - p['d'] * X[2]/p['Ec']) * n * e**(2/3) - p['w1'] * n * e / np.log(1 / X[3])**(2/3) + p['m'] * n / X[1])
            functions.append(lambda n, e, X: p['m'] * np.exp(-p['mu_meta'] * X[0] - np.euler_gamma) + (p['sigma_1'] * p['K'] / (p['K'] + X[0]) + p['sigma_2'] * p['b'] * n / e**(1/3) - (int(np.rint(n)) == 1) * p['d'] / e**(1/3) * X[2] / p['Ec']) * X[0])

    elif data_set == 'BCI':
        functions.append(lambda n, e, X: n * e)
        
        if method == 'METimE':
            functions.append(lambda n, e, X: 0)
            functions.append(lambda n, e, X: 0)
            functions.append(lambda n, e, X: 0)
            
        elif method == 'dynaMETE':
            p = pd.read_csv('../../data/dynaMETE_parameters_BCI.csv')
    
            functions.append(lambda n, e, X: (p['b'] - p['d'] * X[2] / p['Ec']) * n / e ** (1 / 3) + p['m'] * n / X[1])
            functions.append(
                lambda n, e, X: (p['w'] - p['d'] * X[2] / p['Ec']) * n * e ** (2 / 3) - p['w1'] * n * e / np.log(1 / X[3]) ** (
                            2 / 3) + p['m'] * n / X[1])
            functions.append(lambda n, e, X: p['m'] * np.exp(-p['mu_meta'] * X[0] - np.euler_gamma) + (
                        p['sigma_1'] * p['K'] / (p['K'] + X[0]) + p['sigma_2'] * p['b'] * n / e ** (1 / 3) - (
                            int(np.rint(n)) == 1) * p['d'] / e ** (1 / 3) * X[2] / p['Ec']) * X[0])
    return functions


def get_input(data_set):
    values = pd.read_csv('../../data/{data_set}_METimE_values.csv'.format(data_set=data_set))
    state_var = pd.read_csv('../../data/{data_set}_regression_library.csv'.format(data_set=data_set))

    # If E is missing, add to column
    if 'E_t' not in state_var.columns:
        state_var['E_t'] = state_var['N_t'] * 1e6 # E_t defaults to N0 * 1e6 if not specified (from meteR package)
        values['E/S'] = state_var['E_t'] / state_var['S_t']

    if 'beta' not in state_var.columns:
        state_var['beta'] = np.inf

    # Select columns that exist in the DataFrame
    state_var = state_var[[col for col in ['S_t', 'N_t', 'E_t', 'beta'] if col in state_var.columns]]
    state_var = state_var.rename(columns={'S_t': 'S', 'N_t': 'N', 'E_t': 'E'})

    # Filter values based on the method
    if method == 'METE':
        values = values[[col for col in ['N/S', 'E/S'] if col in values.columns]]

    return values, state_var


if __name__ == "__main__":

    for data_set in ['fish', 'birds', 'BCI']:

        print("------------{data_set}-------------".format(data_set=data_set))

        method_errors = {}

        if data_set == 'fish':
            methods = ['METE', 'METimE']
        else:
            methods = ['METE', 'METimE', 'dynaMETE']

        for method in methods:
            print("Performing {method}...".format(method=method))

            functions = get_functions(data_set, method)
            values_df, state_var_df = get_input(data_set)

            # change beta=inf to beta=-1
            state_var_df['beta'] = [
                -1 if i == np.inf else i for i in state_var_df['beta']
            ]

            # Initialize error collection
            all_errors = []

            #for (idx1, values), (idx2, state_var) in zip(values_df.iterrows(), state_var_df.iterrows()):
            for (idx1, values), (idx2, state_var) in islice(zip(values_df.iterrows(), state_var_df.iterrows()), 2):

                if idx1 == 0:
                    initial_lambdas = make_initial_guess(state_var, scaling=True)
                else:
                    initial_lambdas = prev_lambdas # use previous solution as initial guess

                optimized_lambdas = perform_optimization(initial_lambdas,
                                                         state_variables=state_var,
                                                         functions=functions,
                                                         values=values)

                prev_lambdas = optimized_lambdas

                constraint_errors = check_constraints(optimized_lambdas, state_var, functions, values)
                all_errors.append(constraint_errors)

            # Aggregate errors (mean absolute error per constraint)
            aggregated_errors = np.mean(np.abs(all_errors), axis=0)
            method_errors[method] = aggregated_errors

        # Print aggregated errors for each method
        for method, errors in method_errors.items():
            print(f"Aggregated errors for {method}: {errors}")