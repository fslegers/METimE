import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize, root_scalar
import ast


def R_exponent(n, e, lambdas):
    l1, l2 = lambdas
    return -l1 * n - l2 * n * e


def partition_function(x, state_variables):
    S, N, E = state_variables
    l1, l2 = x[0], x[1]

    n_vec = np.arange(1, N+1)
    a = np.exp(-(l1 + l2)*n_vec)
    b = np.exp(-(l1+l2*E)*n_vec)
    Z = sum((a - b)/n_vec) / l2

    return Z


def all_constraints(x, state_variables):
    S, N, E = state_variables
    l1, l2 = x[0], x[1]

    Z = partition_function(x, state_variables)
    if Z == 0:
        print("Warning. Partition function is zero.")


    # Calculate partial derivative of ln(Z) with respect to lambda_1
    a = np.exp(-(l1 + l2*E)) * (np.exp(-(l1 + l2*E)*N) - 1)
    b = np.exp(-(l1 + l2*E)) - 1
    c = np.exp(-(l1 + l2)) * (np.exp(-(l1 + l2)*N) - 1)
    d = np.exp(-(l1 + l2)) - 1
    partial_l1 = 1/Z * 1/l2 * (a/b - c/d)

    # Calculate partial derivative of ln(Z) with respect to lambda_2
    a = np.exp(-(l1 + l2)) * (np.exp(-(l1 + l2)*N) - 1)
    b = np.exp(-(l1 + l2)) - 1
    c = np.exp(-(l1 + l2*E)) * (np.exp(-(l1 + l2*E)*N) - 1)
    d = np.exp(-(l1 + l2*E)) - 1
    partial_l2 = 1/Z * 1/l2 * (-a/b - E*c/d) - 1/l2

    #print("Error: %f" % ((partial_l1 - N/S)**2 + (partial_l2 - E/S)**2))
    #return (partial_l1 - N/S)**2 + (partial_l2 - E/S)**2

    print("Error: %f" % ((partial_l1 - N / S) ** 2))
    return (partial_l1 - N/S)**2

    # return [partial_l1 - N/S,
    #         partial_l2 - E/S]


def load_data(data_set):
    if data_set == "BCI":
        filename = 'C:/Users/5605407/Documents/PhD/Chapter_2/Data sets/BCI/METE_Input_BCI.csv'

    elif data_set == "birds":
        filename = 'C:/Users/5605407/Documents/PhD/Chapter_2/Data sets/BioTIME/METE_Input_39.csv'

    df = pd.read_csv(filename)
    return df


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


def beta_function(beta, S, N):
    return (1 - np.exp(-beta)) / (np.exp(-beta) - np.exp(-beta*(N + 1))) * np.log(1.0/beta) - S/N


def beta_derivative(beta, S, N):
    term1 = -(1 - np.exp(-beta)) / (beta*(np.exp(-beta) - np.exp(-beta*(N + 1))))
    term2 = -(1 - np.exp(-beta))*((-N-1)*(-np.exp(-beta*(N+1)))-np.exp(-beta))*np.log(1/beta) / (np.exp(-beta) - np.exp(-beta*(N+1)))**2
    term3 = (np.exp(-beta) * np.log(1/beta)) / (np.exp(-beta) - np.exp(-beta*(N+1)))
    return term1 + term2 + term3


def check_constraints(initial_lambdas, state_variables):
    constr_1 = constraint_1(initial_lambdas, state_variables)
    constr_2 = constraint_2(initial_lambdas, state_variables)
    print("Errors on constraints: \n %f (N/S), \n %f (E/S)" % (constr_1, constr_2))
    pass


def constraint_1(lambdas, state_variables):
    """Calculates the difference between the observed average number of individuals per species (N/S), and the expected
    number of individuals per species, calculated from the ecosystem structure function (determined by the optimized
    Lagrange multipliers lambda_1 and lambda_2)"""
    S, N, E = state_variables
    l1, l2 = lambdas

    Z = partition_function(lambdas, state_variables)

    a = np.exp(-(l1 + l2)) * (np.exp(-(l1 + l2)*N) - 1)
    b = np.exp(-(l1 + l2)) - 1
    c = np.exp(-(l1 + l2*E)) * (np.exp(-(l1 + l2*E)*N) - 1)
    d = np.exp(-(l1 + l2*E)) - 1

    rhs = 1/Z * 1/l2 * (a/b - c/d)

    return rhs - N/S


def constraint_2(lambdas, state_variables):
    """Calculates the difference between the observed average metabolic rate per species (E/S), and the expected
    metabolic rate per species, calculated from the ecosystem structure function (determined by the optimized
    Lagrange multipliers lambda_1 and lambda_2)"""
    S, N, E = state_variables
    l1, l2 = lambdas

    Z = partition_function(lambdas, state_variables)

    rhs = 0
    for n in range(1, N-S+1):
        a = -l2*E*n - 1
        b = l2 * n + 1
        rhs += 1/n * np.exp(-l1*n) * (a * np.exp(-l2*E*n) + b * np.exp(-l2*n))

    return 1/Z * 1/(l2**2) * rhs - E/S


def fetch_census_data(df, row):
    """
    A function that fetches the census data from a given data set and census/row.
    :param df: data set (bird or BCI data)
    :param row: which row to fetch census data from
    :return: state variables, census number and empirical species abundance distribution
    """
    S = int(df['S'][row])
    N = int(df['N'][row])
    E = df['E'][row]

    empirical_sad = df['SAD'][row]
    empirical_sad = ast.literal_eval(empirical_sad)

    if data_set == "birds":
        census = df['YEAR'][row]
    elif data_set == "BCI":
        census = df['PlotCensusNumber'][row]

    print("State variables: S0 = %f, N0 = %f, E0 = %f" % (S, N, E))
    print("Constraints: N0/S0 = %f, E0/S0 = %f" % (N / S, E / S))

    return [S, N, E], census, empirical_sad


def perform_optimization(lambdas, state_variables):
    #objective_function = lambda x, state_variables: all_constraints(x, state_variables)
    objective_function = lambda x, state_variables: constraint_1(x, state_variables)**2 + constraint_2(x, state_variables)**2

    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0]},
        {'type': 'ineq', 'fun': lambda x: x[1]}]

    lambdas = minimize(objective_function, lambdas,
                       args=(state_variables,),
                       method='SLSQP',
                       options={'eps': 1e-10, 'disp':True},
                       constraints=constraints,
                       tol=1e-10)

    return lambdas.x


def plot_rank_SAD(S, N, lambdas, empirical_sad, data_set, census):
    l1, l2 = lambdas

    meteSAD = []
    Z = partition_function(lambdas, state_variables)

    for n in range(1, N-S+1):
        p_n = np.exp(-l1*n) * (np.exp(-l2*n) - np.exp(-l2*n*E))
        p_n = p_n / (Z * l2 * n)
        meteSAD.append(p_n)

    print("CHECK: sum of meteSAD = %f \n \n" % sum(meteSAD))

    # Expected number of species with abundance n
    # for a community with n_species
    exp_n_species = [S * i for i in meteSAD]

    # Reverse so abundance goes from high to low
    exp_n_species.reverse()

    # Add abundances to df
    exp_n_species = pd.DataFrame(exp_n_species, columns=['exp_n_species'])
    abundances = pd.DataFrame(range(N-S, 0, -1), columns=['abundance'])
    df = pd.concat([exp_n_species, abundances], axis = 1)

    # Add a column with cummulative n species
    df['cummulative'] = df['exp_n_species'].cumsum()

    # Create plot
    x = list(df['cummulative'][:-1])
    x.insert(0, 0)

    # Plot predicted rank sad
    plt.bar(x = x,
            height = df['abundance'],
            width = df['exp_n_species'],
            align = 'edge',
            label = 'theoretical')

    # Add empirical sad
    plt.plot([i for i in range(1, S + 1)], empirical_sad, color='orange', label = 'empirical')
    plt.scatter([i for i in range(1, S + 1)], empirical_sad, color='orange', linestyle='dashed')

    if data_set == "birds":
        y_lim = ((-0.1, 125))

    elif data_set == "BCI":
        y_lim = ((0, 55000))

    plt.title(data_set + "\n %d" % int(census))

    plt.xlabel("Rank")
    plt.ylabel("Abundance (n)")
    plt.ylim(y_lim)
    plt.legend(loc=1)
    plt.show()

    #path = 'C:/Users/5605407/Documents/PhD/Chapter_2'
    # plt.savefig(path + "/standard_METE_minimize_" + str(data_set) + "_" + str(int(year)) + ".png")
    #plt.close()


if __name__ == '__main__':

    #data_set = "BCI"
    data_set = "birds"
    df = load_data(data_set)

    for row in range(0, len(df)):
        state_variables, census, empirical_sad = fetch_census_data(df, row)
        S, N, E = state_variables


        # Determine starting lambdas
        initial_lambdas = make_initial_guess(state_variables)
        check_constraints(initial_lambdas, state_variables)


        # Run optimization
        optimized_lambdas = perform_optimization(initial_lambdas, state_variables)
        check_constraints(optimized_lambdas, state_variables)


        # Plot rank SADs
        plot_rank_SAD(S, N, optimized_lambdas, empirical_sad, data_set, census)






########################################################################
###                         Notes on meteR                           ###
########################################################################


# Equations that are solved in meteR:

#     b < - La[1] + La[2]
#     s < - La[1] + E0 * La[2]
#
#     n < - 1: N0
#
#     g.bn < - exp(-b * n)
#     g.sn < - exp(-s * n)
#
#     univ.denom < - sum((g.bn - g.sn) / n)
#     rhs.7.19.num < - sum(g.bn - g.sn)
#     rhs.7.20.num < - sum(g.bn - E0 * g.sn)
#
#     ##  the two functions to solve
#     f < - rep(NA, 2)
#     f[1] < - rhs.7.19.num / univ.denom - N0 / S0
#     f[2] < - (1 / La[2]) + rhs.7.20.num / univ.denom - E0 / S0
#
#     return (f)