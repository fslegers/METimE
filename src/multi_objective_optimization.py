import numpy as np
import pandas as pd
from IPython.display import clear_output
from matplotlib import pyplot as plt
from scipy.optimize import root_scalar, minimize
from scipy.integrate import quad
from mpmath import exp, log


def partition_function(lambdas, X):
    l1, l2 = lambdas
    n = np.arange(1, X['S'] + 1)
    return - 1/l2 * np.sum(1/n * ([exp(-l1 * i - X['E'] * l2 * i) for i in n] - np.exp(-l1 * n)))


def R(n, e, lambdas, X):
    l1, l2 = lambdas
    Z = partition_function(lambdas, X)
    return exp(-l1 * n - l2 * n * e) / Z


def entropy(lambdas, X):
    l1, l2 = lambdas
    n = np.arange(1, X['S'] + 1)
    return - 1/l2 * np.sum(1/n * ([exp(-l1 * i - X['E'] * l2 * i) for i in n] * (l1 * n - [exp(X['E'] * l2 * i) for i in n] - l1 * n * [exp(X['E'] * l2 * i) for i in n] + X['E'] * l2 * n + 1)))


def kullback_leibler_divergence(lambdas, X, prev_lambdas):
    KL = 0
    for n in range(1, int(X['S']) + 1):
        KL += -(quad(lambda e: R(n, e, lambdas, X) * log(R(n, e, lambdas, X) / R(n, e, prev_lambdas, X)), 0, X['E'])[0])
    print("Kullback-Leibler divergence:", KL)
    return KL
    # TODO: log(R/Q) can be simplified to (-l1 * n - l2 * n * e + a1 * n + a2 * n * e)


def objective_function(all_lambdas, all_X):
    lambdas = [[all_lambdas[i], all_lambdas[i + 1]] for i in range(0, len(all_lambdas), 2)]
    f = 0
    for i in range(1, len(lambdas)):
        #f += -1 * entropy(lambdas[i], all_X[i])
        f += -1 * kullback_leibler_divergence(lambdas[i], all_X[i], lambdas[i - 1])
    return f


def constr_NS(lambdas, X):
    lambdas_list = [[lambdas[i], lambdas[i + 1]] for i in range(0, len(lambdas), 2)]
    error = 0

    for ls, Xs in zip(lambdas_list, X):
        l1, l2 = ls
        S, N, E = int(Xs['S']), int(Xs['S']), Xs['E']

        # Common terms
        E_l2_N1 = exp(E * l2 * (N + 1))
        E_l2_l1 = exp(E * l2 + l1)

        lhs = (exp(-E * l2 * (N + 1))
               * exp(-l1 * (N + 1))
               * (exp(2 * l1 + E * l2)
                  + E_l2_N1 * exp(l1)
                  - E_l2_l1
                  - E_l2_N1 * exp(2 * l1 + E * l2)
                  - E_l2_N1 * exp(l1 * (N + 2))
                  + E_l2_N1 * E_l2_l1 * exp(l1 * (N + 1))))

        rhs = N / S
        error += (np.abs(lhs - rhs))

    print('Error in S/S:', error)
    return error


def constr_ES(lambdas, X):
    lambdas_list = [[lambdas[i], lambdas[i + 1]] for i in range(0, len(lambdas), 2)]

    error = 0

    for ls, Xs in zip(lambdas_list, X):
        l1, l2 = ls
        S, N, E = int(Xs['S']), int(Xs['S']), Xs['E']

        n = np.arange(1, N + 1)
        lhs = - 1/(l2 ** 2) * np.sum(1/n * ([exp(-l1 * i - E * l2 * i) for i in n] * (E * l2 * n + 1) - np.exp(-l1 * n)))
        rhs = E/S

        error += np.abs(lhs - rhs)

    print('Error in E/S:', error)
    return error


def constr_dNS(lambdas, X):
    lambdas_list = [[lambdas[i], lambdas[i + 1]] for i in range(0, len(lambdas), 2)]

    expected_NS = []
    observed_NS = []

    for ls, Xs in zip(lambdas_list, X):
        l1, l2 = ls
        S, N, E = int(Xs['S']), int(Xs['S']), Xs['E']

        # Common terms
        E_l2_N1 = exp(E * l2 * (N + 1))
        E_l2_l1 = exp(E * l2 + l1)

        lhs = (exp(-E * l2 * (N + 1))
               * exp(-l1 * (N + 1))
               * (exp(2 * l1 + E * l2)
                  + E_l2_N1 * exp(l1)
                  - E_l2_l1
                  - E_l2_N1 * exp(2 * l1 + E * l2)
                  - E_l2_N1 * exp(l1 * (N + 2))
                  + E_l2_N1 * E_l2_l1 * exp(l1 * (N + 1))))

        expected_NS.append(float(lhs))
        observed_NS.append(N/S)

    # Calculate differences with the next value
    diff_expected_NS = np.array(expected_NS[1:]) - np.array(expected_NS[:-1])
    diff_observed_NS = np.array(observed_NS[1:]) - np.array(observed_NS[:-1])

    print("Error in dN:", np.sum(np.abs(diff_expected_NS - diff_observed_NS)))

    return np.sum(np.abs(diff_expected_NS - diff_observed_NS))


def constr_dES(lambdas, X):
    lambdas_list = [[lambdas[i], lambdas[i + 1]] for i in range(0, len(lambdas), 2)]

    expected_ES = []
    observed_ES = []

    for ls, Xs in zip(lambdas_list, X):
        l1, l2 = ls
        S, N, E = int(Xs['S']), int(Xs['S']), Xs['E']

        n = np.arange(1, N + 1)
        lhs = - 1 / (l2 ** 2) * np.sum(1 / n * ([exp(-l1 * i - E * l2 * i) for i in n] * (E * l2 * n + 1) - np.exp(-l1 * n)))

        expected_ES.append(float(lhs))
        observed_ES.append(E / S)

    # Calculate differences with the next value
    diff_expected_ES = np.array(expected_ES[1:]) - np.array(expected_ES[:-1])
    diff_observed_ES = np.array(observed_ES[1:]) - np.array(observed_ES[:-1])

    print("Error in dE:", np.sum(np.abs(diff_expected_ES - diff_observed_ES)))
    return np.sum(np.abs(diff_expected_ES - diff_observed_ES))


def initial_guess(all_X):

    def single_initial_guess(X):
        """
        A function that makes an initial guess for the Lagrange multipliers lambda1 and lambda2.
        Based on Eq 7.29 from Harte 2011 and meteR'diag function meteESF.mete.lambda

        :param state_variables: state variables S, S and E
        :return: initial guess for the Lagrange multipliers lambda1 and lambda2
        """

        def beta_function(beta, X):
            return (1 - np.exp(-beta)) / (np.exp(-beta) - np.exp(-beta * (X['S'] + 1))) * np.log(1.0 / beta) - X['S'] / \
                X['S']

        interval = [1.0 / X['S'], X['S'] / X['S']]

        beta = root_scalar(beta_function, x0=0.001, args=(X,), method='brentq', bracket=interval)

        l2 = X['S'] / (X['E'] - X['S'])
        l1 = beta.root - l2

        return [l1, l2]


    initial_guess = []

    for X in all_X:
        single_guess = single_initial_guess(X)
        initial_guess = initial_guess + single_guess

    return initial_guess


def load_data(data_set):
    if data_set == "BCI":
        #filename = 'C:/Users/5605407/Documents/PhD/Chapter_2/Data sets/BCI/METE_Input_BCI.csv'
        filename = '../data/BCI_METimE_values.csv'

    else:
        print("Did not recognize data set")

    df = pd.read_csv(filename)
    return df


def perform_optimization(lambdas, X):
    """
    Performs optimization (scipy.minimize) to find Lagrange multipliers lambda_1 and lambda_2,
    given an initial guess for lambda_1 and lambda_2 and with METEs ratio constraints on the state variables,
    assuring positive values are returned for lambda_1 and lambda_2.
    :param initial_lambdas: Initial guess for Lagrange multipliers
    :param state_variables: S, S, E
    :return: optimized Lagrange multipliers
    """
    print("----- performing optimization -----")

    constraints = [
        {'type': 'eq', 'fun': constr_NS, 'args': (X,)},
        {'type': 'eq', 'fun': constr_ES, 'args': (X,)},
        {'type': 'eq', 'fun': constr_dNS, 'args': (X,)},
        {'type': 'eq', 'fun': constr_dES, 'args': (X,)}]

    boundaries = tuple((0, None) for _ in lambdas)

    result = minimize(objective_function, lambdas, args=(X,), constraints=constraints, bounds=boundaries, options={'disp': True})
    optimized_lambdas = result.x

    clear_output(wait=False)
    print("Optimized Lagrange multipliers:", optimized_lambdas)
    print("Maximum value of I:", result.fun)

    return optimized_lambdas


if __name__ == '__main__':

    data_set = "BCI"
    df = load_data(data_set)

    all_X = [{'S': row['S'], 'S': row['S'], 'E': row['E']} for _, row in df.iterrows()]

    initial_lambdas = initial_guess(all_X)

    optimized_lambdas = perform_optimization(initial_lambdas, all_X)

    print(optimized_lambdas)