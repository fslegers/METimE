import numpy as np
import pandas as pd
from IPython.display import clear_output
from matplotlib import pyplot as plt
from scipy.optimize import root_scalar
from scipy.optimize import minimize
from scipy.integrate import quad
import ast

"""
Requires:
    f_k = list of functions (at least n, ne, fut maybe also transition functions)
    F_k = list of values
    N, E to determine the summation and integral ranges

Assumption:
    R(n, e) = exp( - lambda_1 f_1 - lambda_2 f_2 ...) / Z

Will maximize entropy over R(n, e) while satisfying constraints sum int f_i R(n, e) = F_i for all i
"""

def R_exponent(n, e, functions, lambdas):
    exponent = sum(-l * func(n, e) for func, l in zip(functions, lambdas))
    return exponent


def partition_function(lambdas, functions, state_variables):
    _, N, E = state_variables
    upper_bound = min(E, np.ceil((-np.log(0.1) + lambdas[0]) / lambdas[1]))

    Z = 0
    for n in range(1, N+1):
        Z += quad(lambda e: np.exp(R_exponent(n, e, functions, lambdas)), 0, E, points = [1, upper_bound])[0]

    return Z


def entropy(lambdas, functions, state_variables):
    _, N, E = state_variables
    Z = partition_function(lambdas, functions, state_variables)

    I = 0
    upper_bound = min(E, np.ceil((-np.log(0.1) + lambdas[0]) / lambdas[1]))
    for n in range(1, N+1):
        integral = quad(lambda e: np.exp(R_exponent(n, e, functions, lambdas)) * (R_exponent(n, e, functions, lambdas) - np.log(Z)),1, E, points = [1, upper_bound])[0]
        I += integral
    I = I / Z

    return I


def beta_function(beta, S, N):
    return (1 - np.exp(-beta)) / (np.exp(-beta) - np.exp(-beta*(N + 1))) * np.log(1.0/beta) - S/N


def make_initial_guess(state_variables):
    """
    A function that makes an initial guess for the Lagrange multipliers lambda1 and lambda2.
    Based on Eq 7.29 from Harte 2011 and meteR's function meteESF.mete.lambda

    :param state_variables: state variables S, N and E
    :return: initial guess for the Lagrange multipliers lambda1 and lambda2
    """
    S, N, E = state_variables
    interval = [1.0/N, S/N]

    beta = root_scalar(beta_function, x0=0.001, args=(S, N), method='brentq', bracket=interval)

    l2 = S / (E - N)
    l1 = beta.root - l2

    # TODO: method can be either 'brentq' or 'bisect', but 'bisect' hasn't been tested yet

    # TODO: root_scalar can take multiple guesses x0, so maybe we can also give it the optimized values from the
    #  previous year

    return [l1, l2]


def check_constraints(lambdas, state_variables, functions, values):
    S, N, E = state_variables
    Z = partition_function(lambdas, functions, state_variables)

    constraint_errors = []

    for f, v in zip(functions, values):
        # Compute integral with upper bound
        integral_value = sum(
            quad(
                lambda e: f(n, e) * np.exp(R_exponent(n, e, functions, lambdas)),
                0,
                min(E, np.ceil((-np.log(0.1) + lambdas[0]) / max(lambdas[1], 1e-8)))  # Avoid division by zero
            )[0]
            for n in range(1, N + 1)
        )

        # Compute constraint error
        error = integral_value - v
        constraint_errors.append(error)

    # Print constraint errors
    for i, error in enumerate(constraint_errors):
        print("Error on constraint {i} ({val}): {error}".format(i=i, error=error, val=values[i]))

    pass


def perform_optimization(initial_lambdas, functions, values, state_variables):
    """
    Performs optimization (scipy.minimize) to find Lagrange multipliers,
    given an initial guess for lambda_1 and lambda_2 and with METEs ratio constraints on the state variables.
    :param initial_lambdas: Initial guess for Lagrange multipliers
    :param state_variables: S, N, E
    :return: optimized Lagrange multipliers
    """
    print("----- performing optimization -----")

    S, N, E = state_variables

    constraints = [
        {
            'type': 'eq',
            'fun': lambda lambdas, func=f, val=v: sum(
                quad(
                    lambda e: func(n, e) * np.exp(R_exponent(n, e, functions, lambdas)),
                    0,
                    min(E, np.ceil((-np.log(0.1) + lambdas[0]) / lambdas[1]))
                )[0]
                for n in range(1, N + 1)
            ) - v
        }
        for f, v in zip(functions, values)
    ]

    boundaries = ((- 709/(N*E), None), (- 709/(N*E), None)) # TODO: can/should we incorporate bounds?

    result = minimize(entropy, initial_lambdas, args=(functions, state_variables), constraints=constraints, bounds=boundaries)
    optimized_lambdas = result.x

    clear_output(wait=False)
    print("Optimized Lagrange multipliers:", optimized_lambdas)
    print("Maximum value of I:", result.fun)

    return optimized_lambdas


def get_functions(data_set, method):

    if data_set == 'fish':
        functions = [lambda n, e: n]

        if method != 'METE':
            functions.append(lambda n, e: )

    elif data_set == 'birds':
        functions = []
        values = []
        state_var = []

    elif data_set == 'BCI':
        functions = []
        values = []
        state_var = []


    return functions, values, state_var


if __name__ == "__main__":

    for data_set in ['fish', 'birds', 'BCI']:
        for method in ['METE', 'METimE', 'dynaMETE']:
            
            functions, values, state_var = get_input(data_set, method)
        
            initial_lambdas = make_initial_guess(state_var)
        
            optimized_lambdas = perform_optimization(initial_lambdas, state_variables=state_var, functions=functions, values=values)
        
            check_constraints(optimized_lambdas, state_var, functions, values)