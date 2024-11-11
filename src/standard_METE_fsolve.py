import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import fsolve, root_scalar
from scipy.integrate import quad
import ast


def partition_function(x, state_variables):
    S, N, E = state_variables
    l1, l2 = x[0]**2, x[1]**2

    n_vec = np.arange(1, N+1)
    a = np.exp(-(l1 + l2)*n_vec)
    b = np.exp(-(l1+l2*E)*n_vec)
    Z = sum(1/l2 * (a - b)/n_vec)

    return Z


def all_constraints(x, state_variables):
    S, N, E = state_variables
    l1, l2 = x[0]**2, x[1]**2


    print("Lambda_1 = %.9f, Lambda_2 = %.9f" % (l1, l2))


    Z = partition_function(x, state_variables)
    if Z == 0:
        print("Warning. Partition function is zero.")


    # Calculate 1/Z * partial derivative of Z with respect to lambda_1
    a = np.exp(-(l1 + l2)*N) - 1
    b = np.exp(-(l1 + l2)) - 1
    c = 1 - np.exp(-(l1 + l2*E)*N)
    d = 1 - np.exp(-(l1 + l2*E))

    partial_l1 = 1.0 / l2 * 1.0 / Z * (a/b - c/d)


    # Calculate 1/Z * partial derivative of Z with respect to lambda_2
    n_vec = np.arange(1, N+1)
    f = l2 * n_vec * E - 1
    g = np.exp(-(l1 + l2*E)*n_vec)
    h = np.exp(-(l1 + l2)*n_vec)

    partial_l2 = sum(1.0 / Z * 1.0/(l2**2) * (f * g - h)/n_vec)


    return [partial_l1 - N/S,
            partial_l2 - E/S]


def constraint1(lambdas, state_variables):
    S, N, E = state_variables
    l1, l2 = lambdas

    Z = partition_function(lambdas, state_variables)

    a = np.exp(-(l1 + l2)*N) * (np.exp(l1 + l2)*N - 1)
    b = np.exp(l1 + l2) - 1
    c = 1 - np.exp(-(l1 + l2*E)*N)
    d = 1 - np.exp(-(l1 + l2*E))
    lhs = 1.0/Z * 1.0/l2 * (a/b + c/d)

    return lhs - N/S


def constraint2(lambdas, state_variables, tol=0.1):
    S, N, E = state_variables
    l1, l2 = lambdas

    Z = partition_function(state_variables, lambdas)

    upper_bound = np.ceil((-np.log(tol) + l1) / l2)

    lhs = Z * E / S

    rhs = 0
    for n in range(1, N-S+1):
        rhs += n * quad(lambda e: e * np.exp(R_exponent(n, e, lambdas)), 1, E, points = [1, upper_bound])[0]

    return lhs - rhs


def load_data(data_set):
    if data_set == "BCI":
        filename = 'C:/Users/5605407/Documents/PhD/Chapter_2/Data sets/BCI/METE_Input_BCI.csv'

    elif data_set == "birds":
        filename = 'C:/Users/5605407/Documents/PhD/Chapter_2/Data sets/BioTIME/METE_Input_39.csv'

    df = pd.read_csv(filename)
    return df


def beta_function(beta, S, N):
    return (1 - np.exp(-beta)) / (np.exp(-beta) - np.exp(-beta*(N + 1))) * np.log(1.0/beta) - S/N


def beta_derivative(beta, S, N):
    term1 = -(1 - np.exp(-beta)) / (beta*(np.exp(-beta) - np.exp(-beta*(N + 1))))
    term2 = -(1 - np.exp(-beta))*((-N-1)*(-np.exp(-beta*(N+1)))-np.exp(-beta))*np.log(1/beta) / (np.exp(-beta) - np.exp(-beta*(N+1)))**2
    term3 = (np.exp(-beta) * np.log(1/beta)) / (np.exp(-beta) - np.exp(-beta*(N+1)))
    return term1 + term2 + term3


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



if __name__ == '__main__':

    data_set = "birds"
    df = load_data(data_set)

    #for row in range(0, df.size):
    for row in [0]:
        S = int(df['S'][row])
        N = int(df['N'][row])
        E = df['E'][row]

        empirical_sad = df['SAD'][row]
        empirical_sad = ast.literal_eval(empirical_sad)

        if data_set == "birds":
            year = df['YEAR'][row]
        elif data_set == "BCI":
            year = df['PlotCensusNumber'][row]

        print("State variables: S0 = %f, N0 = %f, E0 = %f" % (S, N, E))
        print("Constraints: N0/S0 = %f, E0/S0 = %f" % (N/S, E/S))

        state_variables = [S, N, E]
        initial_lambdas = [1.29*10**(-4), 1.25*10**(-7)]

        print("Errors on constraint (before fsolve): \n %f (N/S)" % (constraint1(initial_lambdas, state_variables)))

        optimized_lambdas = fsolve(all_constraints, args=state_variables, x0=initial_lambdas)
        optimized_lambdas = [optimized_lambdas[0]**2, optimized_lambdas[1]**2]

        print("Optimized Lagrange multipliers: l1 = %.9f, l2 = %.9f" % (optimized_lambdas[0], optimized_lambdas[1]))
        print("Partition function Z:", partition_function(optimized_lambdas, state_variables))

        # TODO (write in report):
        #  fsolve can find negative values for lambda_1 and lambda_2. In this case, exponents can become too large
        #  positive values, and np.exp() returns infinity.
        #  we cannot change fsolves settings to make it find positive values.
        #  however, we can define y = lambda^2 and use y in calculations.


        # Check constraints
        constr_1 = constraint1(optimized_lambdas, state_variables)
        #constr_2 = constraint2(optimized_lambdas, state_variables)
        constr_2 = 0
        print("Errors on constraints: \n %f (N/S), \n %f (E/S)" % (constr_1, constr_2))


        # # Turn results into SAD
        # meteSAD = []
        #
        # upper_bound = np.ceil((-np.log(0.1) + optimized_lambdas[0]) / optimized_lambdas[1])
        #
        # total=0
        # Z = partition_function(state_variables, optimized_lambdas)
        # Z_test = 0
        # for n in range(1, N + 1):
        #     p_n = quad(lambda e: np.exp(R_exponent(n, e, optimized_lambdas)), 1, E, points = [1, upper_bound])[0]
        #     Z_test += p_n
        #     p_n = p_n/Z
        #     meteSAD.append(p_n)
        #
        # print("Z = %f, sum of p_n = %f, sum of meteSAD = %f" % (Z, Z_test, sum(meteSAD)))
        #
        # plot_rank_SAD(meteSAD, empirical_sad, 'BCI', year)