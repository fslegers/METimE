import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import root_scalar
from scipy.optimize import minimize
from itertools import islice
from scipy.stats import chi2
from mpmath import mp, mpf, exp
from scipy import integrate
from functools import partial
import time
import scipy
from scipy.optimize import basinhopping


import warnings
warnings.filterwarnings("ignore")

"""
Assumption:
    R(n, e) = exp( - lambda_1 f_1 - lambda_2 f_2 ...) / Z
    Z = sum from n = 1 to N of int e^(- lambda_1 n - lambda_2 n e) from e = 0 to E
    
Requires:
    f_k = list of partial derivatives (wrt lambda_1 and lambda_2) of Z
    F_k = list of values
    
Will find zero points of f_k(lambdas) - F_k
"""


def make_initial_guess(state_variables):
    """
    A function that makes an initial guess for the Lagrange multipliers lambda1 and lambda2.
    Based on Eq 7.29 from Harte 2011 and meteR's function meteESF.mete.lambda

    :param state_variables: state variables S, N and E
    :return: initial guess for the Lagrange multipliers lambda1 and lambda2
    """
    def beta_function(beta, S, N):
        return (1 - np.exp(-beta)) / (np.exp(-beta) - np.exp(-beta * (N + 1))) * np.log(1.0 / beta) - S / N


    S, N, E = int(state_variables['S']), int(state_variables['N']), state_variables['E']
    interval = [1.0/N, S/N]

    beta = root_scalar(beta_function, x0=0.001, args=(S, N), method='brentq', bracket=interval)

    l2 = S / (E - N)
    l1 = beta.root - l2

    return [l1, l2]


def find_roots(initial_lambdas, state_variables, values={'dN/S': 0, 'dS': 0}, method="METimE", data_set="dummy", scaling=False):
    """
    Solves for the optimal lambda values by finding the roots of the system of equations.

    :param initial_lambdas: List of initial guesses for lambda values.
    :param state_variables: Dictionary of state variables (e.g., S, N, E).
    :param values: Dictionary of additional parameters for METimE and dynaMETE.
    :param method: The model to use ("METimE", "METE", "dynaMETE", "meteR").
    :param data_set: The dataset used (e.g., "fish", "birds", "BCI", "dummy").
    :param scaling: Boolean to indicate whether to scale the function.
    :return: Optimized lambda values if successful, otherwise None.
    """

    # Store function and lambda values in a single dictionary
    method_info = {
        "METimE": {
            "fish": [METimE_fish_functions, initial_lambdas + [0, 0]],
            "birds": [METimE_birds_functions, initial_lambdas + [0, 0]],
            "BCI": [METimE_BCI_functions, initial_lambdas + [0, 0, 0]]
        },
        "METE": [METE_functions, initial_lambdas],
        "dynaMETE": [dynaMETE_functions, initial_lambdas + [0, 0, 0]],
        "meteR": [meteR_functions, initial_lambdas]
    }

    # Retrieve function and lambda values, defaulting to meteR_functions if method is unknown
    method_data = method_info.get(method, [meteR_functions, initial_lambdas])

    # If method is METimE, use dataset-specific functions
    if isinstance(method_data, dict):
        method_data = method_data.get(data_set, [meteR_functions, initial_lambdas])

    func, lambdas = method_data

    # Define arguments for fsolve
    def beta_constraint(b, s):
        return b * np.log(1 / (1 - np.exp(-b))) - s['S'] / s['N']

    beta = fsolve(beta_constraint, 0.0001, args=state_variables)[0]
    args = (beta, state_variables, values, data_set, scaling) if method == "dynaMETE" else (state_variables, values, scaling) if method == "METimE" else (state_variables, scaling)

    # Solve the system using fsolve
    max_iterations = 25

    lambda_history = []
    func_history = []

    start_time = time.time()

    for _ in range(max_iterations):
        func_values = func(lambdas, *args)
        lambda_history.append(lambdas.copy())
        func_history.append(func_values.copy())

        if np.linalg.norm(func_values) < 1e-15:
            break

        # Update lambda values using fsolve for one iteration
        lambdas = fsolve(func, lambdas, args=args, maxfev=1, xtol=1e-15)

    elapsed_time = time.time() - start_time

    # Convert history to numpy arrays
    lambda_history = np.array(lambda_history)
    func_history = np.array(func_history)

    # Plot results
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    for i in range(lambda_history.shape[1]):
        axs[0].plot(lambda_history[:, i], label=f'Lambda {i + 1}')
    axs[0].set_title('Lambda Values per Iteration')
    axs[0].legend()

    for i in range(func_history.shape[1]):
        axs[1].plot(func_history[:, i], label=f'Func {i + 1}')
    axs[1].set_title('Function Values per Iteration')
    axs[1].legend()

    # Set the main figure title
    if scaling == "True":
        plot_title = f'{method}, {data_set}, weighted constraints'
    else:
        plot_title = f'{method}, {data_set}, unweighted constraints'
    fig.suptitle(plot_title, fontsize=14)

    plt.tight_layout()

    # Display final function values as text next to the plot
    final_lambda_values = "\n".join([f'L{i + 1}: {val:.2e}' for i, val in enumerate(lambda_history[-1])])
    axs[0].text(1.02, 0.5, final_lambda_values, fontsize=12, verticalalignment='center', transform=axs[0].transAxes,
                bbox=dict(facecolor='white', alpha=0.7))

    final_func_values = "\n".join([f'Func {i + 1}: {val:.2e}' for i, val in enumerate(func_history[-1])])
    axs[1].text(1.02, 0.5, final_func_values, fontsize=12, verticalalignment='center', transform=axs[1].transAxes,
                bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.show()

    return lambdas, elapsed_time


def minimize(initial_lambdas, state_variables, values={'dN/S': 0, 'dS': 0}, method="METimE", data_set="dummy", scaling=False):
    """
    Solves for the optimal lambda values by finding the roots of the system of equations.

    :param initial_lambdas: List of initial guesses for lambda values.
    :param state_variables: Dictionary of state variables (e.g., S, N, E).
    :param values: Dictionary of additional parameters for METimE and dynaMETE.
    :param method: The model to use ("METimE", "METE", "dynaMETE", "meteR").
    :param data_set: The dataset used (e.g., "fish", "birds", "BCI", "dummy").
    :param scaling: Boolean to indicate whether to scale the function.
    :return: Optimized lambda values if successful, otherwise None.
    """
    def beta_constraint(b, s):
        return b * np.log(1 / (1 - np.exp(-b))) - s['S'] / s['N']

    # Solve the system using fsolve
    max_iterations = 1000

    lambda_history = []
    func_history = []

    start_time = time.time()

    if method in ["METE", "meteR"]:
        lambdas = initial_lambdas
    elif method == "METimE" and data_set != "BCI":
        lambdas = initial_lambdas + [0, 0]
    else:
        lambdas = initial_lambdas + [0, 0, 0]

    for _ in range(max_iterations):
        # Update lambda values using fsolve for one iteration
        if method == "METE":
            objective_function = lambda x, state_variables, scaling: np.sum(
                METE_functions(x, X=state_variables, scaling=scaling) ** 2)
            bounds = [(0, None), (0, None)]
            lambdas = scipy.optimize.minimize(objective_function,
                                              lambdas,
                                              args=(state_variables, scaling),
                                              method="L-BFGS-B",
                                              bounds=bounds,
                                              options={
                                                  'maxiter': 1000,
                                                  'gtol': 1e-14,
                                                  'ftol': 1e-14,
                                                  'eps': 1e-8
                                              }
                                              )['x']
            lambdas = basinhopping(objective_function,
                                   lambdas,
                                   minimizer_kwargs={"method": "L-BFGS-B",
                                                     "args":(state_variables, scaling)})['x']
            func_values = METE_functions(lambdas, X=state_variables, scaling=scaling)
        elif method == "meteR":
            objective_function = lambda x, state_variables, scaling: np.sum(
                meteR_functions(x, X=state_variables, scaling=scaling) ** 2
            )
            bounds = [(0, None), (0, None)]
            lambdas = scipy.optimize.minimize(objective_function,
                                              lambdas,
                                              args=(state_variables, scaling),
                                              method="L-BFGS-B",
                                              bounds=bounds,
                                              options={
                                                  'maxiter': 10,
                                                  'gtol': 1e-14,
                                                  'ftol': 1e-14,
                                                  'eps': 1e-8
                                              }
                                              )['x']
            lambdas = basinhopping(objective_function,
                                   lambdas,
                                   minimizer_kwargs={"method": "L-BFGS-B",
                                                     "args":(state_variables, scaling)})['x']
            func_values = meteR_functions(lambdas, X=state_variables, scaling=scaling)
        elif method == "METimE":
            if data_set == "fish":
                objective_function = lambda x, state_variables, values, scaling: np.sum(
                    METimE_fish_functions(x, X=state_variables, values=values, scaling=scaling) ** 2
                )
                bounds = [(0, None), (0, None), (None, None), (None, None)]
                lambdas = scipy.optimize.minimize(objective_function,
                                                  lambdas,
                                                  args=(state_variables, values, scaling),
                                                  method="trust-constr",
                                                  bounds=bounds,
                                                  options={
                                                      'maxiter': 10,
                                                      'initial_tr_radius': 1e-14,
                                                      'gtol': 1e-15,
                                                      'finite_diff_rel_step': 1e-14
                                                  })['x']
                func_values = METimE_fish_functions(lambdas, X=state_variables, values=values, scaling=scaling)
            elif data_set == "birds":
                objective_function = lambda x, state_variables, values, scaling: np.sum(
                    METimE_birds_functions(x, X=state_variables, values=values, scaling=scaling) ** 2
                )
                bounds = [(0, None), (0, None), (None, None), (None, None)]
                lambdas = scipy.optimize.minimize(objective_function,
                                                  lambdas,
                                                  args=(state_variables, values, scaling),
                                                  method="trust-constr",
                                                  bounds=bounds,
                                                  options={
                                                      'maxiter': 10,
                                                      'initial_tr_radius': 1e-14,
                                                      'gtol': 1e-15,
                                                      'finite_diff_rel_step': 1e-14
                                                  })['x']
                func_values = METimE_birds_functions(lambdas, X=state_variables, values=values, scaling=scaling)
            else:
                objective_function = lambda x, state_variables, values, scaling: np.sum(
                    METimE_BCI_functions(x, X=state_variables, values=values, scaling=scaling) ** 2
                )
                bounds = [(0, None), (0, None), (None, None), (None, None), (None, None)]
                lambdas = scipy.optimize.minimize(objective_function,
                                                  lambdas,
                                                  args=(state_variables, values, scaling),
                                                  method="trust-constr",
                                                  bounds=bounds,
                                                  options={
                                                      'maxiter': 10,
                                                      'initial_tr_radius': 1e-14,
                                                      'gtol': 1e-15,
                                                      'finite_diff_rel_step': 1e-14
                                                  })['x']
                func_values = METimE_BCI_functions(lambdas, X=state_variables, values=values, scaling=scaling)
        else:
            beta = fsolve(beta_constraint, 0.0001, args=state_variables)[0]
            objective_function = lambda x, beta, state_variables, values, data_set, scaling: np.sum(
                dynaMETE_functions(x, beta=beta, X=state_variables, values=values, data_set=data_set,
                                   scaling=scaling) ** 2
            )
            bounds = [(0, None), (0, None), (None, None), (None, None), (None, None)]
            lambdas = scipy.optimize.minimize(objective_function,
                                              lambdas,
                                              bounds=bounds,
                                              args=(beta, state_variables, values, data_set, scaling),
                                              method="trust-constr",
                                              options={
                                                  'maxiter': 10,
                                                  'initial_tr_radius': 1e-14,
                                                  'gtol': 1e-15,
                                                  'finite_diff_rel_step': 1e-14
                                              })['x']
            func_values = dynaMETE_functions(lambdas, beta=beta, X=state_variables, values=values, data_set=data_set,
                                   scaling=scaling)

        lambda_history.append(lambdas.copy())
        func_history.append(func_values.copy())

    elapsed_time = time.time() - start_time

    # Convert history to numpy arrays
    lambda_history = np.array(lambda_history)
    func_history = np.array(func_history)

    # Plot results
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    for i in range(lambda_history.shape[1]):
        axs[0].plot(lambda_history[:, i], label=f'Lambda {i + 1}')
    axs[0].set_title('Lambda Values per Iteration')
    axs[0].legend()

    for i in range(func_history.shape[1]):
        axs[1].plot(func_history[:, i], label=f'Func {i + 1}')
    axs[1].set_title('Function Values per Iteration')
    axs[1].legend()

    # Set the main figure title
    if scaling == "True":
        plot_title = f'Minimize, {method}, {data_set}, weighted constraints'
    else:
        plot_title = f'Minimize, {method}, {data_set}, unweighted constraints'
    fig.suptitle(plot_title, fontsize=14)

    plt.tight_layout()

    # Display final function values as text next to the plot
    final_lambda_values = "\n".join([f'L{i + 1}: {val:.2e}' for i, val in enumerate(lambda_history[-1])])
    axs[0].text(1.02, 0.5, final_lambda_values, fontsize=12, verticalalignment='center', transform=axs[0].transAxes,
                    bbox=dict(facecolor='white', alpha=0.7))

    final_func_values = "\n".join([f'Func {i + 1}: {val:.2e}' for i, val in enumerate(func_history[-1])])
    axs[1].text(1.02, 0.5, final_func_values, fontsize=12, verticalalignment='center', transform=axs[1].transAxes,
                    bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.show()

    return lambdas

def METE_functions(lambdas, X, scaling):
    l1, l2 = lambdas[0], lambdas[1]
    S, N, E = int(X['S']), int(X['N']), X['E']

    n = np.arange(1, N + 1)

    f = np.empty(2)

    beta = l1 + E * l2
    a = [np.exp(x) for x in -beta * n]
    b = [np.exp(x) for x in -l1 * n]
    sum_term = [(i - j) / k for i, j, k in zip(a, b, n)]
    sum_ = sum(sum_term)

    f[0] =  (((1 - np.exp(-N * beta))/(np.exp(beta) - 1) + (np.exp(-N * l1) - 1)/(np.exp(l1) - 1)) / sum_ - N/S)
    f[1] =  ((sum_/l2 + (E - E * np.exp(-N * beta)) / (np.exp(beta) - 1))/ sum_ - E/S)

    if scaling == "True":
        f[0] *= S/N
        f[1] *= S/E

    return f


def meteR_functions(lambdas, X, scaling):
    l1, l2 = lambdas
    S, N, E = int(X['S']), int(X['N']), X['E']

    b = l1 + l2
    s = l1 + E * l2

    n = np.arange(1, N+1)

    g_bn = np.exp(-b*n)
    g_sn = np.exp(-s*n)

    univ_denom = np.sum((g_bn - g_sn) / n)
    rhs_7_19_num = np.sum(g_bn - g_sn)
    rhs_7_20_num = np.sum(g_bn - E * g_sn)

    f = np.empty(2)
    f[0] = (rhs_7_19_num / univ_denom - N / S)
    f[1] = ((1 / l2) + rhs_7_20_num / univ_denom - E / S)

    if scaling == "True":
        f[0] *= S/N
        f[1] *= S/E

    return f


def METimE_fish_functions(lambdas, X, values, scaling):
    l1, l2, l3, l4 = lambdas

    a1 = -0.0002966770904530696
    a2 = -0.01150577791675056
    a3 = 398.29616068871843
    b1 = -2.0626245432725737e-06
    b2 = 0.001510662695163621
    b3 = 0.5211389172589189

    S, N, E = int(X['S']), int(X['N']), X['E']
    dN_S, dS = values['dN/S'], values['dS']

    n = np.arange(1, N + 1)

    # Calculate all exponentials separately
    a1l3n2 = [exp(x) for x in -a1 * l3 * n**2]
    b1l4n2 = [exp(x) for x in -b1 * l4 * n**2]
    l1n = [exp(x) for x in -l1*n]
    a3l3nN = [exp(x) for x in -a3*l3*n/N]
    b3l4nN = [exp(x) for x in -b3 * l4 * n / N]
    El2n = [exp(x) for x in -E*l2*n]
    a2b2N = exp(N * a2 * l3) * exp(N * b2 * l4)

    # Calculate the common term A (Lambda in appendix)
    A = [
        a * b * c * d * e * (1 - f)
        for a, b, c, d, e, f in zip(a1l3n2, b1l4n2, l1n, a3l3nN, b3l4nN, El2n)
    ]
    sum_A = sum(A)
    sum_A_over_n = sum(a / b for a, b in zip(A, n))
    sum_A_times_n = sum(a * b for a, b in zip(A, n))
    sum_not_A = sum(a * b * c * d * e for a, b, c, d, e in zip(a1l3n2, b1l4n2, l1n, a3l3nN, b3l4nN))

    # Compute the difference with the partial derivatives and state variables
    f = [mpf(0)] * 4
    f[0] = ((sum_A / sum_A_over_n) + N/S)
    f[1] = ((1 / sum_A_over_n * (E * sum_not_A - E * a2b2N * sum_A - 1 / l2 * sum_A_over_n)) + E / S)
    f[2] = ((1 / sum_A_over_n * (a3 / N * sum_A + a1 * a2b2N * sum_A_times_n + N * a2 * sum_A_over_n) + dN_S))
    f[3] = ((1 / sum_A_over_n * (b3 / N * sum_A + b1 * a2b2N * sum_A_times_n + N * b2 * sum_A_over_n) + dS))

    f = np.array([float(val) for val in f], dtype=np.float64)

    if scaling == "True":
        f[0] *= S/N
        f[1] *= S/E
        f[2] /= dN_S
        f[3] /= dS

    return f


def METimE_birds_functions(lambdas, X, values, scaling):
    l1, l2, l3, l4 = lambdas

    a1 = 0.002680866657705244
    a2 = -0.13872925115082604
    a3 = 0.7132348935661652
    a4 = -2.3095257445787665e-05
    c1 = -0.002006965502034695
    c2 = -1.0026770297907317
    c3 = -8.962314123706848
    c4 = 1.5038243092790893e-05

    S, N, E = int(X['S']), int(X['N']), X['E']
    dN_S, dE_S, dS = values['dN/S'], values['dE/S'], values['dS']

    n = np.arange(1, N + 1)

    a1_l3_n2 = np.exp(-a1 * l3 * n ** 2)
    c1_l4_n2 = np.exp(-c1 * l4 * n ** 2)
    l1_n = np.exp(-l1 * n)
    E2_a4_l3_n = np.exp(-E ** 2 * a4 * l3 * n)
    E2_c4_l4_n = np.exp(-E ** 2 * c4 * l4 * n)
    E_l2_n = np.exp(-E * l2 * n)
    a3_l3 = np.exp(a3 * l3)
    c3_l4 = np.exp(c3 * l4)

    LAMBDA = a1_l3_n2 * c1_l4_n2 * l1_n * (a3_l3 * c3_l4 - E2_a4_l3_n * E2_c4_l4_n * E_l2_n)
    PHI = a3 * l3 + c3 * l4 + E * l2 * n + E ** 2 * a4 * l3 * n + E ** 2 * c4 * l4 * n
    PSI = a1_l3_n2 * c1_l4_n2 * l1_n * a3_l3 * c3_l4 * (a3 + E**2 * a4 * n)
    OMEGA = a1_l3_n2 * c1_l4_n2 * l1_n * a3_l3 * c3_l4 * (c3 + E**2 * c4 * n)

    sum_Lambda_over_Phi = sum(LAMBDA/PHI)
    sum_n_times_Lambda_over_Phi = sum(n * LAMBDA/PHI)
    sum_n_times_Lambda_over_Phi_squared = sum(n * LAMBDA / PHI**2)
    sum_lambda_2 = sum(a3_l3 * c3_l4 * n * a1_l3_n2 * c1_l4_n2 * l1_n / PHI)
    sum_lambda_3 = sum(PSI / PHI - LAMBDA * (a4 * n * E**2 + a3)/ PHI**2 - a1*n**2*LAMBDA / PHI - E**2 * a4 * n * LAMBDA / PHI)
    sum_lambda_4 = sum(OMEGA / PHI - LAMBDA * (c4 * n * E**2 + c3)/ PHI**2 - c1*n**2*LAMBDA / PHI - E**2 * c4 * n * LAMBDA / PHI)

    f = [mpf(0)] * 4
    f[0] = (-sum_n_times_Lambda_over_Phi / sum_Lambda_over_Phi + N/S)
    f[1] = (-(E * sum_n_times_Lambda_over_Phi + E * sum_n_times_Lambda_over_Phi_squared - E * sum_lambda_2) / sum_Lambda_over_Phi + E/S)
    f[2] = (-(a3 * sum_Lambda_over_Phi - sum_lambda_3 + S * a2 * sum_Lambda_over_Phi)/sum_Lambda_over_Phi + dN_S)
    f[3] = (-(c3 * sum_Lambda_over_Phi - sum_lambda_4 + S * c2 * sum_Lambda_over_Phi)/sum_Lambda_over_Phi + dS)

    f = np.array([float(val) for val in f], dtype=np.float64)

    if scaling == "True":
        f[0] *= S/N
        f[1] *= S/E
        f[2] /= dN_S
        f[3] /= dS

    return f


def METimE_BCI_functions(lambdas, X, values, scaling):
    mp.dps = 15
    l1, l2, l3, l4, l5 = lambdas

    a1 = -1.7695099217664192e-06
    a2 = 5.738028749752286e-05
    a3 = -0.00033872905701187863
    a4 = -1976982.6907172424
    b1 = -2.7150893249221894e-09
    b2 = -2.728124059500504e-06
    b3 = 2.8467469505310685e-07
    b4 = 2515657.311392835
    c1 = 2.9856597972591785e-12
    c2 = -2.73442670272408e-09
    c3 = -3.740825794959918e-06
    c4 = -58.63031895973122

    S, N, E = int(X['S']), int(X['N']), X['E']
    dN_S, dE_S, dS = values['dN/S'], values['dE/S'], values['dS']

    f = np.empty(5)
    n = np.arange(1, N + 1)

    # Precompute common factors
    abc4 = a4 * l3 + b4 * l4 + c4 * l5
    labc2 = l2 + a2 * l3 + b2 * l4 + c2 * l5
    F = np.exp(np.clip(a4 * l3 + b4 * l4 + c4 * l5, -700, 700))

    exp_a1_l3_n2 = np.exp(np.clip(-a1 * l3 * n ** 2, -700, 700))
    exp_b1_l4_n2 = np.exp(np.clip(-b1 * l4 * n ** 2, -700, 700))
    exp_c1_l5_n2 = np.exp(np.clip(-c1 * l5 * n ** 2, -700, 700))
    exp_l1_n = np.exp(np.clip(-l1 * n, -700, 700))

    A = abc4 + labc2 * E * n

    B = np.exp(np.clip(-E * a2 * l3 * n, -700, 700)) * \
        np.exp(np.clip(-E * b2 * l4 * n, -700, 700)) * \
        np.exp(np.clip(-E * c2 * l5 * n, -700, 700)) * \
        exp_a1_l3_n2 * exp_b1_l4_n2 * exp_c1_l5_n2 * exp_l1_n * \
        (E * F - np.exp(np.clip(-E * l2 * n, -700, 700)))

    D = exp_a1_l3_n2 * exp_b1_l4_n2 * exp_c1_l5_n2 * exp_l1_n

    sum_ = np.sum(B / A)
    sum_times_n = np.sum(n * B / A)
    sum_times_n_over_A = np.sum(n * B / (A**2))
    sum_D = np.sum(n * D / A)
    sum_3 = np.sum(D * F * (a4 +E * a2 * n)/A - B * (a4 * E * a2 * n) / A**2 - a1 * n**2 * B / A - E * a2 * n * B / A)
    sum_4 = np.sum(D * F * (b4 +E * b2 * n)/A - B * (b4 * E * b2 * n) / A**2 - b1 * n**2 * B / A - E * b2 * n * B / A)
    sum_5 = np.sum(D * F * (c4 +E * c2 * n)/A - B * (c4 * E * c2 * n) / A**2 - c1 * n**2 * B / A - E * c2 * n * B / A)

    f[0] = (sum_times_n / sum_ - N/S)
    f[1] = (E * (sum_times_n + sum_times_n_over_A - F * sum_D) / sum_ - E/S)
    f[2] = (E * a3 + a4 - sum_3/sum_ - dN_S)
    f[3] = (E * b3 + b4 - sum_4/sum_ - dE_S)
    f[4] = (E * c3 + c4 - sum_5/sum_ - dS)

    if scaling == "True":
        f[0] *= S/N
        f[1] *= S/E
        f[2] /= dN_S
        f[3] /= dE_S
        f[4] /= dS

    return f


def dynaMETE_functions(lambdas, beta, X, values, data_set, scaling):

    def f(n, e, s, p):
        return (p['b0'] - p['d0'] * s['E'] / p['Ec']) * n / e ** (1 / 3) + p['m0'] / s['N'] * n

    def h(n, e, s, p, beta):
        return (p['w0'] - p['d0'] * s['E'] / p['Ec']) * n * e ** (2 / 3) - p['w10'] / np.log(1 / beta) ** (
                    2 / 3) * n * e + p['m0'] / s['N'] * n

    def q(n, e, s, p):
        kn1 = int(np.rint(n)) == 1
        return p['m0'] * np.exp(-p['mu'] * s['S'] - np.euler_gamma) - kn1 * p['d0'] / p['Ec'] * s['E'] * s['S'] / e ** (
                    1 / 3)

    def R(n, e, l, s, p, beta):
        return np.exp(-l[0] * n - l[1] * n * e - l[2] * f(n, e, s, p) - l[3] * h(n, e, s, p, beta) - l[4] * q(n, e, s, p))

    def Rsum(e, l, s, p, beta):
        l14 = l[0] + l[1] * e + l[2] * f(1, e, s, p) + l[3] * h(1, e, s, p, beta)
        l5_1 = l[4] * q(1, e, s, p)
        l5_0 = l[4] * q(0, e, s, p)
        return np.exp(-l14 - l5_1) + np.exp(-2 * l14 - l5_0) * (1 - np.exp(-l14 * (s['N'] - 1))) / (1 - np.exp(-l14))

    def z(l, s, p, beta):
        return integrate.quad(lambda loge: np.exp(loge) * Rsum(np.exp(loge), l, s, p, beta), 0, np.log(s['E']), points=[0, 300])[0]

    def nRsum(e, l, s, p, beta):
        l14 = l[0] + l[1] * e + l[2] * f(1, e, s, p) + l[3] * h(1, e, s, p, beta)
        l5_1 = l[4] * q(1, e, s, p)
        l5_0 = l[4] * q(0, e, s, p)
        t1 = np.exp(-l14 - l5_1)
        t2 = np.exp(-l14 - l5_0)
        t3fac = np.exp(-l14 - l5_0)
        t3num = 1 + s['N'] * np.exp(-l14 * (s['N'] + 2)) - (s['N'] + 1) * np.exp(-l14 * (s['N'] + 1))
        t3denom = (1 - np.exp(-l14)) ** 2
        return t1 - t2 + t3fac * t3num / t3denom

    def n2Rsum(e, l, s, p, beta):
        l14 = l[0] + l[1] * e + l[2] * f(1, e, s, p) + l[3] * h(1, e, s, p, beta)
        l5_1 = l[4] * q(1, e, s, p)
        l5_0 = l[4] * q(0, e, s, p)
        t1 = np.exp(-l14 - l5_1)
        t2 = np.exp(-l14 - l5_0)
        t3fac = np.exp(-l14 - l5_0)
        t3num = np.exp(-l14) + 1 - np.exp(-l14 * (s['N'])) * (s['N'] + 1) ** 2 + \
                np.exp(-l14 * (s['N'] + 1)) * (2 * (s['N'] + 1) * s['N'] - 1) - s['N'] ** 2 * np.exp(
            -l14 * (s['N'] + 2))
        t3denom = (1 - np.exp(-l14)) ** 3
        return t1 - t2 + t3fac * t3num / t3denom

    def fn_e13(s, p):
        return p['b0'] - p['d0'] * s['E'] / p['Ec']

    def fn(s, p):
        return p['m0'] / s['N']

    def hne23(s, p):
        return p['w0'] - p['d0'] * s['E'] / p['Ec']

    def hne(s, p, beta):
        return -p['w10'] / np.log(1 / beta) ** (2 / 3)

    def hn(s, p):
        return p['m0'] / s['N']

    def qc(s, p):
        return p['m0'] * np.exp(-p['mu'] * s['S'] - np.euler_gamma)

    def qdn_e13(s, p):
        return -s['S'] * p['d0'] * s['E'] / p['Ec']

    def fm(s, p, m):
        return fn_e13(s, p) * m['n_e13'] + fn(s, p) * m['n']

    def hm(s, p, m, beta):
        return hne23(s, p) * m['ne23'] + hne(s, p, beta) * m['ne'] + hn(s, p) * m['n']

    def qm(s, p, m):
        return qc(s, p) + qdn_e13(s, p) * m['dn_e13']

    def get_means(l, s, p, beta):
        epow_arr_n = np.array([0, -1 / 3, 2 / 3, 1])
        epow_arr_n2 = np.array([0, -1 / 3, -2 / 3, 1 / 3, 2 / 3, 1, 4 / 3, 5 / 3, 2])
        epow_arr_dn = np.array([-1 / 3, -2 / 3, 1 / 3, 2 / 3])

        logE = np.log(X['E'])

        nR_e_arr = \
        integrate.quad_vec(lambda loge: np.exp((epow_arr_n + 1) * loge) * nRsum(np.exp(loge), lambdas, X, p, beta), 0, logE,
                           points=[0, 300])[0]

        n2R_e_arr = \
        integrate.quad_vec(lambda loge: np.exp((epow_arr_n2 + 1) * loge) * n2Rsum(np.exp(loge), lambdas, X, p, beta), 0, logE,
                           points=[0, 300])[0]

        dnR_e_arr = \
        integrate.quad_vec(lambda loge: np.exp((epow_arr_dn + 1) * loge) * R(1, np.exp(loge), lambdas, X, p, beta), 0, logE,
                           points=[0, 300])[0]

        Z = z(lambdas, X, p, beta)
        nR_e_arr /= Z
        n2R_e_arr /= Z
        dnR_e_arr /= Z

        labels = ['n', 'n_e13', 'ne23', 'ne', \
                  'n2', 'n2_e13', 'n2_e23', 'n2e13', 'n2e23', 'n2e', 'n2e43', 'n2e53', 'n2e2', \
                  'dn_e13', 'dn_e23', 'dne13', 'dne23']

        means = pd.Series(np.concatenate([nR_e_arr, n2R_e_arr, dnR_e_arr]), index=labels)
        return means

    # Birds
    if data_set == "birds":
        p = {
            'b0': -0.2218923375417,
            'd0': -1421.493999137449,
            'Ec': 65626237.008849576,
            'm0': 10.612520502746746,
            'w0': -0.08297982541695764,
            'w10': -0.03821797759852467,
            'mu': 0.0219
        }
    else:
        p = {
            'b0': -0.056057535246620105,
            'd0': 0.20740623009583398,
            'Ec': 60001041.59510769,
            'm0': -1359.736649270577,
            'w0': 0.006064118341616576,
            'w10': 0.0002519484464155175,
            'mu': 0.03227502160956679
        }

    S, N, E = X['S'], X['N'], X['E']
    dN_S, dE_S, dS = values['dN/S'], values['dE/S'], values['dS']

    m = get_means(lambdas, X, p, beta)

    sums = np.array([m['n'], m['ne'], S * fm(X, p, m), S * hm(X, p, m, beta), qm(X, p, m)])

    f = np.empty(5)
    f[0] = (N/S - sums[0])
    f[1] = (E/S - sums[1])
    f[2] = (dN_S * S - sums[2])
    f[3] = (dE_S * S - sums[3])
    f[4] = (dS - sums[4])

    if scaling == "True":
        f[0] *= S/N
        f[1] *= S/E
        f[2] /= dN_S
        f[3] /= dE_S
        f[4] /= dS

    return f


def get_SAD_METimE(data_set, lambdas, X):

    if data_set == "fish":
        l1, l2, l3, l4 = lambdas

        a1 = -0.0002966770904530696
        a3 = 398.29616068871843
        b1 = -2.0626245432725737e-06
        b3 = 0.5211389172589189

        S, N, E = int(X['S']), int(X['N']), X['E']
        n = np.arange(1, N + 1)

        # Calculate all exponentials separately
        a1l3n2 = [exp(x) for x in -a1 * l3 * n ** 2]
        b1l4n2 = [exp(x) for x in -b1 * l4 * n ** 2]
        l1n = [exp(x) for x in -l1 * n]
        a3l3nN = [exp(x) for x in -a3 * l3 * n / N]
        b3l4nN = [exp(x) for x in -b3 * l4 * n / N]
        El2n = [exp(x) for x in -E * l2 * n]

        # Calculate the common term A (Lambda in appendix)
        A = [
            a * b * c * d * e * (1 - f)
            for a, b, c, d, e, f in zip(a1l3n2, b1l4n2, l1n, a3l3nN, b3l4nN, El2n)
        ]

        sum_A_over_n = sum(a / b for a, b in zip(A, n))

        sad = A / (n * sum_A_over_n)

    elif data_set == "birds": # TODO: check implementation
        l1, l2, l3, l4 = lambdas

        a1 = 0.002680866657705244
        a3 = 0.7132348935661652
        a4 = -2.3095257445787665e-05
        c1 = -0.002006965502034695
        c3 = -8.962314123706848
        c4 = 1.5038243092790893e-05

        S, N, E = int(X['S']), int(X['N']), X['E']

        n = np.arange(1, N + 1)

        a1_l3_n2 = np.exp(-a1 * l3 * n ** 2)
        c1_l4_n2 = np.exp(-c1 * l4 * n ** 2)
        l1_n = np.exp(-l1 * n)
        E2_a4_l3_n = np.exp(-E ** 2 * a4 * l3 * n)
        E2_c4_l4_n = np.exp(-E ** 2 * c4 * l4 * n)
        E_l2_n = np.exp(-E * l2 * n)
        a3_l3 = np.exp(a3 * l3)
        c3_l4 = np.exp(c3 * l4)

        LAMBDA = a1_l3_n2 * c1_l4_n2 * l1_n * (a3_l3 * c3_l4 - E2_a4_l3_n * E2_c4_l4_n * E_l2_n)

        PHI = a3 * l3 + c3 * l4 + E * l2 * n + E**2 * a4 * l3 * n + E**2 * c4 * l4 * n

        sum_Lambda_over_Phi = sum(LAMBDA / PHI)

        sad = LAMBDA / (PHI * sum_Lambda_over_Phi)

    elif data_set == "BCI":
        l1, l2, l3, l4, l5 = lambdas

        a1 = -1.7695099217664192e-06
        a2 = 5.738028749752286e-05
        a4 = -1976982.6907172424
        b1 = -2.7150893249221894e-09
        b2 = -2.728124059500504e-06
        b4 = 2515657.311392835
        c1 = 2.9856597972591785e-12
        c2 = -2.73442670272408e-09
        c4 = -58.63031895973122

        S, N, E = int(X['S']), int(X['N']), X['E']

        n = np.arange(1, N + 1)

        abc4 = a4 * l3 + b4 * l4 + c4 * l5
        labc2 = l2 + a2 * l3 + b2 * l4 + c2 * l5
        F = np.exp(np.clip(a4 * l3 + b4 * l4 + c4 * l5, -700, 700))

        A = abc4 + labc2 * E * n

        B = np.exp(np.clip(-E * a2 * l3 * n, -700, 700)) * \
            np.exp(np.clip(-E * b2 * l4 * n, -700, 700)) * \
            np.exp(np.clip(-E * c2 * l5 * n, -700, 700)) * \
            np.exp(np.clip(-a1 * l3 * n ** 2, -700, 700)) * \
            np.exp(np.clip(-b1 * l4 * n ** 2, -700, 700)) * \
            np.exp(np.clip(-c1 * l5 * n ** 2, -700, 700)) * \
            np.exp(np.clip(-l1 * n, -700, 700)) * \
            (E * F - np.exp(np.clip(-E * l2 * n, -700, 700)))

        sum_ = sum(B / A)

        sad = B / (A * sum_)

    sad = sad / sum(sad)

    return sad


def get_SAD_meteR(lambdas, X):
    l1, l2 = lambdas
    S, N, E = int(X['S']), int(X['N']), X['E']

    n = np.arange(1, N + 1)

    beta = l1 + l2
    sigma = l1 + E * l2

    t1 = S / (l1 * N)
    t2 = (np.exp(np.clip(-beta, -700, 700)) - np.exp(np.clip(-beta * (N + 1), -700, 700)))/(1 - np.exp(np.clip(-beta, -700, 700)))
    t3 = (np.exp(np.clip(-sigma, -700, 700)) - np.exp(np.clip(-sigma * (N + 1), -700, 700)))/(1 - np.exp(np.clip(-sigma, -700, 700)))
    Z = t1 * (t2 - t3)

    sad = (np.exp(np.clip(-beta * n, -700, 700)) - np.exp(np.clip(-sigma * n, -700, 700))) / (l2 * Z * n)

    sad = sad / sum(sad)

    return sad


def get_SAD_METE(lambdas, X):
    l1, l2 = lambdas
    S, N, E = int(X['S']), int(X['N']), X['E']

    n = np.arange(1, N + 1)

    beta = l1 + E * l2

    Z = -1/l2 * sum(1/n * (np.exp(-n * beta) - np.exp(-l1 * n)))

    denom = Z * l2 * n
    num = np.exp(-l1 * n) * (1 - np.exp(- E * l2 * n))

    sad = num / denom

    sad = sad / sum(sad)

    return sad


def get_SAD_dynaMETE(data_set, dynaMETE_lambdas, X):
    if data_set == "birds":
        p = {
            'b0': -0.2218923375417,
            'd0': -1421.493999137449,
            'Ec': 65626237.008849576,
            'm0': 10.612520502746746,
            'w0': -0.08297982541695764,
            'w10': -0.03821797759852467,
            'mu': 0.0219
        }
    elif data_set == "BCI":
        p = {
            'b0': -0.056057535246620105,
            'd0': 0.20740623009583398,
            'Ec': 60001041.59510769,
            'm0': -1359.736649270577,
            'w0': 0.006064118341616576,
            'w10': 0.0002519484464155175,
            'mu': 0.03227502160956679
        }
    def beta_constraint(b, s):
        return b * np.log(1 / (1 - np.exp(-b))) - s['S'] / s['N']

    def f(n, e, s, p):
        return (p['b0'] - p['d0'] * s['E'] / p['Ec']) * n / e ** (1 / 3) + p['m0'] / s['N'] * n

    def h(n, e, s, p, beta):
        return (p['w0'] - p['d0'] * s['E'] / p['Ec']) * n * e ** (2 / 3) - p['w10'] / np.log(1 / beta) ** (
                    2 / 3) * n * e + p['m0'] / s['N'] * n

    def q(n, e, s, p):
        kn1 = int(np.rint(n)) == 1
        return p['m0'] * np.exp(-p['mu'] * s['S'] - np.euler_gamma) - kn1 * p['d0'] / p['Ec'] * s['E'] * s['S'] / e ** (
                    1 / 3)

    def z(l, X, p, beta):
        Z = 0
        upper_bound = np.log(X['E'])
        if upper_bound > 700:
            print("upper_bound is larger than 700")
        for n in range(1, int(X['N']) + 1):
            integral = lambda loge: (np.exp(loge) *
                                     np.exp( -l[0] * n
                                            - l[1] * n * np.exp(loge)
                                            - l[2] * f(n, np.exp(loge), X, p)
                                            - l[3] * h(n, np.exp(loge), X, p, beta)
                                            - l[4] * q(n, np.exp(loge), X, p)))
            Z += integrate.quad(integral, 0, upper_bound, points=[0, 300])[0]

        return Z

    n_range = np.arange(1, X['N'] + 1)
    beta = fsolve(beta_constraint, 0.0001, args=X)[0]

    sad = np.empty(int(X['N']))
    l1, l2, l3, l4, l5 = dynaMETE_lambdas

    for n in n_range:
        esf_log = lambda loge: np.exp(loge) * np.exp(-l1 * n - l2 * n * np.exp(loge) - l3 * f(n, np.exp(loge), X, p) - l4 * h(n, np.exp(loge), X, p, beta) - l5 * q(n, np.exp(loge), X, p))
        sad[int(n - 1)] = integrate.quad(esf_log, 0, np.log(X['E']), points=[0, 300])[0]

    Z = z(dynaMETE_lambdas, X, p, beta)
    sad = sad / Z
    sad = sad / sum(sad)
    #sad = [1/X['N'] for i in n_range] # TODO: for BCI, do the SADs need to have some minimum threshold?

    return sad


def plot_SADs(theoretical_sads, empirical_sad, X, census):
    S, N, E = int(X['S']), int(X['N']), X['E']

    try:
        sad_meteR, sad_METE, sad_METimE, sad_dynaMETE = theoretical_sads
    except:
        sad_meteR, sad_METE, sad_METimE = theoretical_sads
        sad_dynaMETE = [0, 0, 0, 0, 0]

    plot_rank_abundance(N, S, sad_METimE, sad_METE, sad_meteR, sad_dynaMETE, empirical_sad, census)

    pass


def get_SAD(lambdas, X, method, data_set):
    if method == "METE":
        sad = get_SAD_METE(lambdas, X)
    elif method == "meteR":
        sad = get_SAD_meteR(lambdas, X)
    elif method == "METimE":
        sad = get_SAD_METimE(data_set, lambdas, X)
    else:
        sad = get_SAD_dynaMETE(data_set, lambdas, X)
    return sad


def plot_rank_abundance(N, S, sad_METimE, sad_METE, sad_meteR, sad_dynaMETE, empirical_rank_sad, census):
    """Plots rank abundance distributions with population size on y-axis and expected species count as bar width.
    Three panels: METE, meteR, METimE."""

    n = np.arange(1, N + 1)

    # Compute quantiles & expected number of species for each population size
    expected_species_METE = sad_METE * S
    expected_species_METimE = sad_METimE * S
    expected_species_meteR = sad_meteR * S

    quantile_METE = quantile_function(sad_METE, S, N)
    quantile_METimE = quantile_function(sad_METimE, S, N)
    quantile_meteR = quantile_function(sad_meteR, S, N)

    if data_set != "fish":
        expected_species_dynaMETE = sad_dynaMETE * S
        quantile_dynaMETE = quantile_function(sad_dynaMETE, S, N)

    # Sort n from largest to smallest
    sorted_indices = np.argsort(n)[::-1]
    sorted_n = np.array(n)[sorted_indices]
    sorted_expected_METE = expected_species_METE[sorted_indices]
    sorted_expected_METimE = expected_species_METimE[sorted_indices]
    sorted_expected_meteR = expected_species_meteR[sorted_indices]

    if data_set != "fish":
        sorted_expected_dynaMETE = expected_species_dynaMETE[sorted_indices]
        x_positions_dynaMETE = np.insert(np.cumsum(sorted_expected_dynaMETE[:-1]), 0, 0)

    # Compute bar positions (cumulative sum to place bars next to each other)
    x_positions_METE = np.insert(np.cumsum(sorted_expected_METE[:-1]), 0, 0)
    x_positions_METimE = np.insert(np.cumsum(sorted_expected_METimE[:-1]), 0, 0)
    x_positions_meteR = np.insert(np.cumsum(sorted_expected_meteR[:-1]), 0, 0)

    # For BCI, we need to zoom in
    sorted_n_METE, sorted_n_METimE, sorted_n_meteR, sorted_n_dynaMETE = sorted_n, sorted_n, sorted_n, sorted_n

    if data_set == "BCI":
        index_METE = np.searchsorted(np.cumsum(sorted_expected_METE), 20, side="right")
        x_positions_METE = x_positions_METE[:index_METE]
        sorted_expected_METE = sorted_expected_METE[:index_METE]
        sorted_n_METE = sorted_n[:index_METE]

        index_METimE = np.searchsorted(np.cumsum(sorted_expected_METimE), 20, side="right")
        x_positions_METimE = x_positions_METimE[:index_METimE]
        sorted_expected_METimE = sorted_expected_METimE[:index_METimE]
        sorted_n_METimE = sorted_n[:index_METimE]

        index_meteR = np.searchsorted(np.cumsum(sorted_expected_meteR), 20, side="right")
        x_positions_meteR = x_positions_meteR[:index_meteR]
        sorted_expected_meteR = sorted_expected_meteR[:index_meteR]
        sorted_n_meteR = sorted_n[:index_meteR]

        index_dynaMETE = np.searchsorted(np.cumsum(sorted_expected_dynaMETE), 20, side="right")
        x_positions_dynaMETE = x_positions_dynaMETE[:index_dynaMETE]
        sorted_expected_dynaMETE = sorted_expected_dynaMETE[:index_dynaMETE]
        sorted_n_dynaMETE = sorted_n[:index_dynaMETE]

    # Create the figure with 3 subplots vertically
    if data_set != "fish":
        fig, axes = plt.subplots(4, 1, figsize=(10, 20))
    else:
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # First panel: METE
    axes[0].bar(x_positions_METE, width=sorted_expected_METE, height=sorted_n_METE, alpha=0.9, label='METE',
                color='blue', align="edge")
    axes[0].scatter(x_positions_METE, sorted_n_METE, color='blue', s=0.1, zorder=3)
    axes[0].set_ylabel('n', fontsize=20)
    axes[0].set_title('METE', fontsize=22)
    #axes[0].legend(fontsize=15)

    # Second panel: meteR
    axes[1].bar(x_positions_meteR, width=sorted_expected_meteR, height=sorted_n_meteR, alpha=0.9,
                label='meteR', color='green', align="edge")
    axes[1].scatter(x_positions_meteR, sorted_n_meteR, color='green', s=0.1, zorder=3)
    axes[1].set_ylabel('n', fontsize=20)
    axes[1].set_title('meteR', fontsize=22)
    #axes[1].legend(fontsize=15)

    # Third panel: METimE
    axes[2].bar(x_positions_METimE, width=sorted_expected_METimE, height=sorted_n_METimE, alpha=0.9,
                label='METimE', color='purple', align="edge")
    axes[2].scatter(x_positions_METimE, sorted_n_METimE, color='purple', s=0.1, zorder=3)
    axes[2].set_xlabel('Rank', fontsize=14)
    axes[2].set_ylabel('n', fontsize=20)
    axes[2].set_title('METimE', fontsize=22)
    #axes[2].legend(fontsize=15)

    if data_set != "fish":
        # Fourth panel: dynaMETE
        axes[3].bar(x_positions_dynaMETE, width=sorted_expected_dynaMETE, height=sorted_n_dynaMETE, alpha=0.9,
                    label='DynaMETE', color='orange', align="edge")
        axes[3].scatter(x_positions_dynaMETE, sorted_n_dynaMETE, color='orange', s=0.1, zorder=3)
        axes[3].set_xlabel('Rank', fontsize=14)
        axes[3].set_ylabel('n', fontsize=20)
        axes[3].set_title('dynaMETE', fontsize=22)
        #axes[3].legend(fontsize=15)

    if data_set == "BCI":
        quantiles = [quantile_METE, quantile_meteR, quantile_METimE, quantile_dynaMETE]
    else:
        quantiles = [quantile_METE, quantile_meteR, quantile_METimE]

    if data_set == "BCI":
        ax0 = axes[0].twinx()
        ax1 = axes[1].twinx()
        ax2 = axes[2].twinx()
        ax3 = axes[3].twinx()
        axes = [ax0, ax1, ax2, ax3]
        for i in [0, 1, 2, 3]:
            ax = axes[i]
            ax.scatter(np.arange(1, len(empirical_rank_sad[:20]) + 1), empirical_rank_sad[:20], color='red',
                        label='Empirical Rank Abundance', marker='s', zorder=3)
            ax.tick_params(axis='y', labelcolor='red')
            ax.legend(loc='upper right', fontsize=12)
            ax.scatter(np.arange(1, len(quantiles[i][:20]) + 1), quantiles[i][:20], color='black',
                            label='Quantile Rank Abundance', marker='*', zorder=3)
    else:
        for i in [0, 1, 2]:
            axes[i].scatter(np.arange(1, len(empirical_rank_sad) + 1), empirical_rank_sad, color='red',
                            label='Empirical Rank Abundance', marker='s', zorder=3)
            axes[i].scatter(np.arange(1, len(quantiles[i]) + 1), quantiles[i], color='black',
                            label='Quantile Rank Abundance', marker='*', zorder=3)
        if data_set != "fish":
            axes[3].scatter(np.arange(1, len(empirical_rank_sad) + 1), empirical_rank_sad, color='red',
                            label='Empirical Rank Abundance', marker='s', zorder=3)
            axes[3].scatter(np.arange(1, len(quantiles[i]) + 1), quantiles[i], color='black',
                            label='Quantile Rank Abundance', marker='*', zorder=3)

    # Adjust the layout
    plt.tight_layout()

    fig.suptitle("Rank Abundance Distributions", fontsize = 30)
    fig.text(0.04, 0.5, 'Population size', ha='center', va='center', rotation='vertical', fontsize=22)
    fig.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, 0.96), ncol=2)

    file_name = f"../../results/{scaling}_{data_set}_{census}.png"
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    #plt.savefig(file_name, dpi=150)
    plt.show()
    plt.close()


def calculate_goodness_of_fit(theoretical, empirical, num_params):
    """Calculates RMSE, R², χ², and AICc."""

    # Transform theoretical to np.array of floats if neccesary
    theoretical = np.array([float(x) for x in theoretical], dtype=np.float64)[:len(empirical)]

    empirical = np.array(empirical)

    rmse = np.sqrt(np.mean((empirical - theoretical) ** 2))
    ss_total = np.sum((empirical - np.mean(empirical)) ** 2)
    ss_residual = np.sum((empirical - theoretical) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    # Chi-Square test
    chi_sq = np.sum((empirical - theoretical) ** 2 / (theoretical + 1e-6))
    p_value = 1 - chi2.cdf(chi_sq, df=len(empirical) - num_params)

    # AICc calculation
    n = len(empirical)
    k = num_params
    aic = n * np.log(ss_residual / n + 1e-6) + 2 * k
    aicc = aic + (2 * k * (k + 1)) / (n - k - 1)

    return {"RMSE": rmse, "R^2": r2, "Chi-Square": chi_sq, "p-value": p_value, "AICc": aicc}


def sample_goodness_of_fit(empirical, lambdas, X, data_set, method, n_iter=100):
    empirical = np.array(empirical)

    if method == "METE":
        sad = get_SAD_METE(lambdas, X)
        num_params = 2
    elif method == "meteR":
        sad = get_SAD_meteR(lambdas, X)
        num_params = 2
    elif method == "METimE":
        sad = get_SAD_METimE(data_set, lambdas, X)
        if data_set == "fish":
            num_params = 4
        else:
            num_params = 5
    else:
        sad = get_SAD_dynaMETE(data_set, lambdas, X)
        num_params = 5

    # Turn sad into cummulative probabilities
    cdf = np.cumsum(sad, axis=0)

    rmse_list, r2_list, chi_sq_list, p_list, AICc_list = [], [], [], [], []

    for i in range(n_iter):
        samples = np.random.uniform(low=0, high=1, size=len(empirical))
        theoretical = [np.searchsorted(cdf, s) + 1 for s in samples]
        theoretical.sort(reverse=True)

        rmse = np.sqrt(np.mean((empirical - theoretical) ** 2))
        ss_total = np.sum((empirical - np.mean(empirical)) ** 2)
        ss_residual = np.sum((empirical - theoretical) ** 2)
        r2 = 1 - (ss_residual / ss_total)

        # Chi-Square test
        chi_sq = np.sum((empirical - theoretical) ** 2 / (theoretical))
        p_value = 1 - chi2.cdf(chi_sq, df=len(empirical) - num_params)

        # AICc calculation
        n = len(empirical)
        k = num_params
        aic = n * np.log(ss_residual / n + 1e-6) + 2 * k
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)

        rmse_list.append(rmse)
        r2_list.append(r2)
        chi_sq_list.append(chi_sq)
        p_list.append(p_value)
        AICc_list.append(aicc)

    return {"RMSE": np.mean(rmse_list), "R^2": np.mean(r2_list), "Chi-Square": np.mean(chi_sq_list), "p-value": np.mean(p_list), "AICc": np.mean(AICc_list)}


def get_input(data_set):
    values = pd.read_csv('../../data/{data_set}_METimE_values.csv'.format(data_set=data_set))

    # If E is missing, add to column
    if 'E' not in values.columns:
        values['E'] = values['N'] * 1e6
        values['E/S'] = values['E'] / values['S']

    # Select columns that exist in the DataFrame
    state_var = values[[col for col in ['S', 'N', 'E'] if col in values.columns]]

    # Ensure SAD is stored as a list of lists
    sad = values['SAD'].apply(eval).tolist() if values['SAD'].dtype == 'object' else values['SAD'].tolist()
    rank_sad = values['rank_SAD'].apply(eval).tolist() if values['SAD'].dtype == 'object' else values['SAD'].tolist()

    values = values.drop(['census', 'SAD', 'rank_SAD', 'S', 'N', 'E'], axis=1)

    return values, state_var, sad, rank_sad


def quantile_function(sad, S, N):
    S = int(S)
    N = int(N)

    cumulative_sad = np.cumsum(sad)

    quantiles = [i/S + 1/(2*S) for i in range(S)]
    quantile_values = []

    n = 1
    for quantile in quantiles:
        while (quantile > cumulative_sad[n - 1]) and (n < N):
            n += 1
        quantile_values.append(n)

    return quantile_values[::-1]



if __name__ == "__main__":

    for scaling in ['False']:
        for data_set in ['birds']:

            print("------------{data_set}-------------".format(data_set=data_set))

            methods = ["meteR", "METE", "METimE"]
            if data_set != "fish":
                methods += ['dynaMETE']

            values_df, state_var_df, _, rank_df = get_input(data_set)

            for (idx1, values), (idx2, state_var), (rank_sad) in islice(zip(values_df.iterrows(), state_var_df.iterrows(), rank_df), 1):

                initial_lambdas = make_initial_guess(state_var)
                print("Initial guess: \n", initial_lambdas)

                theoretical_sads = []
                alternative_sads = []
                all_results = []
                alternative_results = []

                for method in methods:
                    print(method)

                    # Get correct number of initial lambdas
                    if (method == "METimE" and data_set == "BCI") or (method == "dynaMETE"):
                        n_param = 5
                    elif (method == "METimE" and data_set != "BCI"):
                        n_param = 4
                    else:
                        n_param = 2

                    # Find optimal lambdas
                    lambdas, elapsed_time = find_roots(initial_lambdas,
                                         state_variables=state_var,
                                         values = values,
                                         method=method,
                                         data_set=data_set,
                                         scaling=scaling)

                    alternative_lambdas = minimize(initial_lambdas,
                                         state_variables=state_var,
                                         values = values,
                                         method=method,
                                         data_set=data_set,
                                         scaling=scaling)

                    # Calculate and save species abundance distribution
                    theoretical_sad = get_SAD(lambdas, state_var, method, data_set)
                    theoretical_sads.append(theoretical_sad)

                    alternative_sad = get_SAD(alternative_lambdas, state_var, method, data_set)
                    alternative_sads.append(alternative_sad)

                    # Compute errors based on quantiles and sampling
                    sample_errors = sample_goodness_of_fit(rank_sad, lambdas, state_var, data_set, method, 250)
                    quantile_errors = calculate_goodness_of_fit(
                        quantile_function(theoretical_sad, state_var['S'], state_var['N']), rank_sad, n_param)

                    sample_errors_2 = sample_goodness_of_fit(rank_sad, alternative_lambdas, state_var, data_set, method, 250)
                    quantile_errors_2 = calculate_goodness_of_fit(
                        quantile_function(alternative_sad, state_var['S'], state_var['N']), rank_sad, n_param)

                    # Pad the values to ensure exactly 5 columns
                    lambdas = np.concatenate([lambdas, np.full(5 - len(lambdas), -np.inf)])
                    row = ([method] + [float(i) for i in lambdas] + [elapsed_time]
                           + list(sample_errors.values())
                           + list(quantile_errors.values()))
                    all_results.append(row)

                    alternative_lambdas = np.concatenate([alternative_lambdas, np.full(5 - len(alternative_lambdas), -np.inf)])
                    row = ([method] + [float(i) for i in alternative_lambdas] + [elapsed_time]
                           + list(sample_errors_2.values())
                           + list(quantile_errors_2.values()))
                    alternative_results.append(row)

                plot_SADs(theoretical_sads, rank_sad, state_var, idx1)
                plot_SADs(alternative_sads, rank_sad, state_var, idx1)

                # # Generate the sheet name based on the combination
                columns = ['Method', 'Lambda1', 'Lambda2', 'Lambda3', 'Lambda4', 'lambda5', 'Elapsed time', 'RMSE', 'R2', 'Chi2', 'P-Value', 'AICc', 'RMSE_int', 'R2_int', 'Chi2_int', 'P-Value_int', 'AICc_int']
                df = pd.DataFrame(all_results, columns=columns)
                sheet_name = f"{data_set}_{'Scaled' if scaling == 'True' else 'Unscaled'}_{idx1}"
                #df.to_csv("../../results/" + sheet_name + ".csv")


