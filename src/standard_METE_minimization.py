import numpy as np
import pandas as pd
from IPython.display import clear_output
from matplotlib import pyplot as plt
from scipy.optimize import root_scalar
from scipy.optimize import minimize
from scipy.integrate import quad
import ast


def R_exponent(n, e, lambdas):
    l1, l2 = lambdas
    return -l1 * n - l2 * n * e


def partition_function(state_variables, lambdas):
    S, N, E = state_variables
    l1, l2 = lambdas

    upper_bound = (-np.log(0.1) + l1) / l2

    Z = 0
    for n in range(1, N-S+1):
        Z += quad(lambda e: np.exp(R_exponent(n, e, lambdas)), 1, E, points = [1, upper_bound])[0]

    return Z


def entropy(lambdas, state_variables):
    S, N, E = state_variables
    l1, l2 = lambdas

    Z = partition_function(state_variables, lambdas)

    I = 0
    upper_bound = np.ceil((-np.log(0.1) + l1) / l2)
    for n in range(1, N-S+1):
        I += quad(lambda e: np.exp(R_exponent(n, e, lambdas)) * (R_exponent(n, e, lambdas) - np.log(Z)),1, E, points = [1, upper_bound])[0]
    I = I / Z

    #print("l1 = %f, l2 = %f, I = %f, Z = %f" % (l1, l2, I, Z))

    return I


def constraint_1(lambdas, state_variables):
    """Calculates the difference between the observed average number of individuals per species (N/S), and the expected
    number of individuals per species, calculated from the ecosystem structure function (determined by the optimized
    Lagrange multipliers lambda_1 and lambda_2)"""
    S, N, E = state_variables
    l1, l2 = lambdas

    Z = partition_function(state_variables, lambdas)

    rhs = 0
    upper_bound = np.ceil((-np.log(0.1) + l1) / l2)
    for n in range(1, N-S+1):
        rhs += n * quad(lambda e: np.exp(R_exponent(n, e, lambdas)), 1, E, points = [1, upper_bound])[0]

    return 1/Z * rhs - N/S


def constraint_2(lambdas, state_variables):
    """Calculates the difference between the observed average metabolic rate per species (E/S), and the expected
    metabolic rate per species, calculated from the ecosystem structure function (determined by the optimized
    Lagrange multipliers lambda_1 and lambda_2)"""
    S, N, E = state_variables
    l1, l2 = lambdas

    Z = partition_function(state_variables, lambdas)

    rhs = 0
    upper_bound = np.ceil((-np.log(0.1) + l1) / l2)
    for n in range(1, N-S+1):
        rhs += n * quad(lambda e: e * np.exp(R_exponent(n, e, lambdas)), 1, E, points = [1, upper_bound])[0]

    return 1/Z * rhs - E/S


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


def load_data(data_set):
    if data_set == "BCI":
        filename = 'C:/Users/5605407/Documents/PhD/Chapter_2/Data sets/BCI/METE_Input_BCI.csv'

    elif data_set == "birds":
        filename = 'C:/Users/5605407/Documents/PhD/Chapter_2/Data sets/BioTIME/METE_Input_39.csv'

    df = pd.read_csv(filename)
    return df


def plot_rank_SAD(S, N, lambdas, empirical_sad, data_set, census):
    l1, l2 = lambdas[0], lambdas[1]

    meteSAD = []
    Z = partition_function(state_variables, lambdas)
    upper_bound = np.ceil((-np.log(0.1) + l1) / l2)
    for n in range(1, N-S+1):
        p_n = quad(lambda e: np.exp(R_exponent(n, e, optimized_lambdas)), 1, E, points=[1, upper_bound])[0]
        p_n = p_n / Z
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
    else:
        y_lim = ((-0.1, 125))

    plt.title(data_set + "\n %d" % int(census))
    plt.ylim(y_lim)
    plt.xlabel("Rank")
    plt.ylabel("Abundance (n)")
    plt.legend()
    plt.show()

    #path = 'C:/Users/5605407/Documents/PhD/Chapter_2'
    # plt.savefig(path + "/standard_METE_minimize_" + str(data_set) + "_" + str(int(year)) + ".png")
    #plt.close()


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


def check_constraints(initial_lambdas, state_variables):
    constr_1 = constraint_1(initial_lambdas, state_variables)
    constr_2 = constraint_2(initial_lambdas, state_variables)
    print("Errors on constraints: \n %f (N/S), \n %f (E/S)" % (constr_1, constr_2))
    pass


def perform_optimization(initial_lambdas, state_variables):
    """
    Performs optimization (scipy.minimize) to find Lagrange multipliers lambda_1 and lambda_2,
    given an initial guess for lambda_1 and lambda_2 and with METEs ratio constraints on the state variables,
    assuring positive values are returned for lambda_1 and lambda_2.
    :param initial_lambdas: Initial guess for Lagrange multipliers
    :param state_variables: S, N, E
    :return: optimized Lagrange multipliers
    """
    print("----- performing optimization -----")

    constraints = [
        {'type': 'eq', 'fun': constraint_1, 'args': (state_variables,)},
        {'type': 'eq', 'fun': constraint_2, 'args': (state_variables,)}]
    boundaries = ((0, None), (0, None))

    result = minimize(entropy, initial_lambdas, args=(state_variables,), constraints=constraints, bounds=boundaries)
    optimized_lambdas = result.x

    clear_output(wait=False)
    print("Optimized Lagrange multipliers:", optimized_lambdas)
    print("Maximum value of I:", result.fun)

    return optimized_lambdas


if __name__ == '__main__':

    #data_set = "BCI"
    data_set = "birds"
    df = load_data(data_set)

    for row in range(0, df.size):
        state_variables, census, empirical_sad = fetch_census_data(df, row)
        S, N, E = state_variables


        # Determine starting lambdas
        initial_lambdas = make_initial_guess(state_variables)
        check_constraints(initial_lambdas, state_variables)


        # Run the optimization
        optimized_lambdas = perform_optimization(initial_lambdas, state_variables)
        check_constraints(optimized_lambdas, state_variables)


        # Plot rank SADs
        plot_rank_SAD(S, N, optimized_lambdas, empirical_sad, data_set, census)





