import numpy as np
import pandas as pd
from scipy.optimize import root_scalar, minimize
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete
from sklearn.metrics import r2_score, mean_absolute_error
from matplotlib.lines import Line2D
import warnings
import os

from src.simulate_population_dynamics.simulate_LV import three_groups_LV, set_up_regression

"""
Script for analysis of Lotka Volterra model
---------------------------

For six intereaction networks, does the following 20 times:
    (a) Generate trajectories of species abundances using the LV model
    (b) Transform into time series with regular time intervals
    (c) Parametrize transition function f(n, N) 
    (d) Find optimal lambda values for METE
    (e) Find optimal lambda values for METimE (= METE with additional constraint E[f(n, N)] = N_{t + 1} - N_t)
    (f) Compare predicted vs empirical rank-abundance distributions
    
Reports the following metrics (averaged over the 20 repetitions):
    (a) R2 of observed and predicted (by parametrized f(n, N)) values of n_{t + 1} - n_t
    (b) AIC of empirical rank abundance distribution of METE and METimE 
    (C) MAE of empirical rank abundance distribution of METE and METimE 

Assumptions:
    f(n, N) = dn/dt = c0 + c1 * N + c2 * n^2 + c3 * n * N + c4 * N^2 + c5 * n^3
    lambda_1 must be strictly positive
"""

def safe_entropy(lambdas, functions, X, coeffs):
    """
    Compute Shannon entropy for the given lambdas and functions.
    Change sign because we will be minimizing instead of maximizing.
    """
    n = np.arange(1, int(X['N_t']) + 1)

    exponent_arg = np.zeros_like(n, dtype=float)
    for i in range(len(functions)):
        exponent_arg += lambdas[i] * functions[i](n, X, coeffs)
    exponent = np.exp(np.clip(-exponent_arg, 1e-12, 1e12))

    Z = np.max([1e-12, np.sum(exponent)])# partition function

    p = exponent / Z # probabilities

    neg_H = np.sum(p * exponent_arg)

    return neg_H

def entropy(lambdas, functions, X, coeffs):
    """
    Compute Shannon entropy for the given lambdas and functions.
    Change sign because we will be minimizing instead of maximizing.
    """
    n = np.arange(1, int(X['N_t']) + 1)

    exponent_arg = np.zeros_like(n, dtype=float)
    for i in range(len(functions)):
        exponent_arg += lambdas[i] * functions[i](n, X, coeffs)
    exponent = np.exp(-exponent_arg)

    Z = np.sum(exponent) # partition function

    p = exponent / Z # probabilities

    neg_H = np.sum(p * exponent_arg)

    return neg_H


def beta_function(beta, S, N):
    """
    Beta function used to generate the initial guess for Lagrange multipliers.
    """
    return (1 - np.exp(-beta)) / (np.exp(-beta) - np.exp(-beta * (N + 1))) * np.log(1.0 / beta) - S / N


def make_initial_guess(X, method):
    """
    A function that makes an initial guess for the Lagrange multipliers lambda1 and lambda2.
    Based on Eq 7.29 from Harte 2011 and meteR'diag function meteESF.mete.lambda

    :param state_variables: state variables S, N and E
    :return: initial guess for the Lagrange multipliers lambda1 and lambda2
    """
    S, N = int(X['S_t']), int(X['N_t'])
    interval = [1.0 / N, S / N]

    try:
        beta = root_scalar(beta_function, x0=0.001, args=(S, N), method='brentq', bracket=interval)
    except:
        if method == "METE":
            return [1/N]
        else:
            return [1/N, 0]

    if method == "METE":
        return [beta.root]
    else:
        return [beta.root, 0]

###############################################
###               Optimization              ###
###############################################

def constraint(f_k, lambdas, functions, F_k, X, coeffs):
    """
    Calculates the expected value of a single constraint function f_k over the ecosystem structure function:
    Σ f_k p_n / Z - F_k
    """
    n = np.arange(1, int(X['N_t']) + 1)

    exponent_arg = np.zeros_like(n, dtype=float)
    for i in range(len(functions)):
        exponent_arg += lambdas[i] * functions[i](n, X, coeffs)
    exponent = np.exp(-exponent_arg)

    Z = np.sum(exponent)  # partition function

    p = exponent / Z  # probabilities

    lhs = np.sum(p * f_k(n, X, coeffs))

    return (lhs - F_k)
    #return (lhs - F_k) / np.abs(F_k)


def perform_optimization(lambdas, functions, macro_var, X, coeffs):
    # Collect all constraints
    constraints = [{
        'type': 'eq',
        'fun': lambda lambdas, functions=functions, f_k=f, F_k=macro_var[name], X=X, coeffs=coeffs:
        constraint(f_k, lambdas, functions, F_k, X, coeffs)
    } for f, name in zip(functions, macro_var)]

    # Set bounds
    if len(functions) == 1:
        bounds = [(0, None)]
    else:
        min_l2, max_l2 = find_extremes(X, coeffs)
        bounds = [(0, None),(min_l2, max_l2)]

    # Perform optimization
    print("Starting Optimizing with constraints...")

    try:

        result = minimize(entropy,
                          lambdas,
                          args=(functions, X, coeffs),
                          constraints=constraints,
                          bounds=bounds,
                          method="trust-constr",
                          options={'initial_tr_radius': 0.001,
                                   'disp': True,
                                   'verbose': 3
                                   })

        optimized_lambdas = result.x

    except:

        result = minimize(safe_entropy,
                          lambdas,
                          args=(functions, X, coeffs),
                          constraints=constraints,
                          bounds=bounds,
                          method='L-BFGS-B',
                          options={'xtol': 1e-6,
                                   'gtol': 1e-12,
                                   'barrier_tol': 1e-12,
                                   'disp': True,
                                   'verbose': 3
                                   })

        optimized_lambdas = result.x

    return optimized_lambdas


def check_constraints(lambdas, functions, X, macro_var, coeffs):
    """
    Calculates the expected value of a single constraint function f_k over the ecosystem structure function:
    Σ f_k p_n / Z - F_k
    """
    absolute_errors = []
    percentage_errors = []

    for f_k, (key, F_k) in zip(functions, macro_var.items()):
        error = np.abs(constraint(f_k, lambdas, functions, F_k, X, coeffs))
        pct_error = error / np.abs(F_k) * 100

        absolute_errors.append(error)
        percentage_errors.append(pct_error)

    print("\n Errors on constraints:")
    print(f"{'Constraint':<10} {'Abs Error':>15} {'% Error':>15}")
    print("-" * 42)
    for key, abs_err, pct_err in zip(macro_var.keys(), absolute_errors, percentage_errors):
        print(f"{key:<10} {abs_err:15.6f} {pct_err:15.2f}")


def f_1(n, X, coeffs):
    return n

def dn(n, X, coeffs):
    c0, c1, c2, c3, c4, c5 = coeffs
    return c0 * n + c1 * X['N_t'] + c2 * n**2 + c3 * n * X['N_t'] + c4 * X['N_t']**2 + c5

def find_extremes(X, coeffs):
    min_f = np.inf
    max_f = -np.inf

    extrema = [1, X['N_t']]

    if coeffs[2] != 0:
        extremum = -(coeffs[0] + coeffs[3] * X['N_t']) / (2 * coeffs[2]) # this is the zero point of the derivative of dn with respect to n
        if extremum > 0:
            extrema.append(extremum)

    # Find extrema of f_n
    for n in extrema:
        function_value = dn(n, X, coeffs)
        if function_value > max_f:
            max_f = function_value
        if function_value < min_f:
            min_f = function_value

    max_abs_f = max(abs(min_f), abs(max_f))

    return -500 / max_abs_f, 500 / max_abs_f    # calculate bounds given extrema of f_n

def compute_SAD_probabilities(lambdas, functions, X, coeffs):
    """
    Compute p_n = exp(-sum(lambda_i * f_i(n))) / Z for n in 1..N_max.
    Returns the SAD as a normalized numpy array.
    """
    n = np.arange(1, int(X['N_t']) + 1)
    exponent = np.zeros_like(n, dtype=float)

    for lam, f in zip(lambdas, functions):
        exponent += lam * f(n, X, coeffs)

    unnorm_p = np.exp(-exponent)
    Z = unnorm_p.sum()

    return unnorm_p / Z


def compute_entropy(p):
    entropy = 0

    for p_n in p:
        if p_n != 0:
            entropy += p_n * np.log(p_n)

    return -entropy


def get_rank_abundance(p_n, X):
    """
    Generate a predicted rank-abundance distribution using the quantile method.
    Ensures exactly S_t values by clipping quantiles and handling edge cases.
    """
    S = int(X['S_t']) + 1

    # Create the discrete distribution
    n_vals = np.arange(1, len(p_n) + 1)
    dist = rv_discrete(name='sad_dist', values=(n_vals, p_n))

    # Safer quantiles: strictly within (0, 1)
    epsilon = 1e-6
    quantiles = (np.arange(1, S + 1) - 0.5) / S
    quantiles = np.clip(quantiles, epsilon, 1 - epsilon)

    # Evaluate quantiles
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_abundances = dist.ppf(quantiles).astype(int)

    # Fix any zeros or nans (can happen if ppf fails)
    pred_abundances = np.where(pred_abundances < 1, 1, pred_abundances)
    pred_abundances = np.nan_to_num(pred_abundances, nan=1).astype(int)

    # Ensure output length = S_t
    if len(pred_abundances) != S:
        raise ValueError(f"Expected {S} predicted abundances, got {len(pred_abundances)}.")

    return np.sort(pred_abundances)[::-1]  # descending order


def compare_SADs(lambdas, functions, X, coeffs, empirical_rad, method, model, census, plot=True):
    """
    Compare predicted vs empirical rank abundance distributions.

    Parameters:
        lambdas: list of lambda values
        functions: list of constraint functions f_i(n, X, coeffs)
        X: dictionary of state variables (e.g., S_t, N_t, etc.)
        coeffs: parameters used by f_i
        empirical_rad: observed rank-abundance list or array
        plot: whether to display the plot (default: True)

    Returns:
        predicted_rad: predicted abundances
        rmse: root mean squared error
        aic: Akaike Information Criterion
    """

    # SAD from lambda parameters
    p_n = compute_SAD_probabilities(lambdas, functions, X, coeffs)

    if np.abs(np.sum(p_n) - 1) > 0.0005:
        print(f"Warning: sum(p_n) = {sum(p_n):.4f} != 1.0")

    # Compute entropy
    entropy = compute_entropy(p_n)

    # Predicted rank-abundance
    predicted_rad = get_rank_abundance(p_n, X)
    predicted_rad = predicted_rad[:len(empirical_rad)]
    empirical_rad = empirical_rad[:len(predicted_rad)]

    # MAE
    mae = mean_absolute_error(empirical_rad, predicted_rad)

    # AIC
    if method == "METE":
        k = 2
    elif method == "METimE":
        k = 3

    log_likelihood = 0
    for i in range(len(empirical_rad)):
        n_i = int(empirical_rad[i])
        p_i = max(p_n[n_i-1], 1e-8)
        log_likelihood += np.log(p_i)
    aic = 2*k - 2*log_likelihood

    # Plot
    if plot:
        plt.rcParams.update({
            'font.size': 16,  # base font size
            'axes.labelsize': 18,  # x and y labels
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14
        })

        plt.figure(figsize=(5, 2))
        ranks = np.arange(1, len(empirical_rad) + 1)
        plt.plot(ranks, empirical_rad, 'o-', label='Empirical RAP', color='blue')
        plt.plot(ranks, predicted_rad, 's--', label='Predicted RAP', color='red')
        plt.xlabel('Rank')
        plt.ylabel('Abundance')
        #plt.yscale('log')
        #plt.title('Rank-Abundance Plot: Predicted vs. Empirical')
        plt.legend()

        # Annotate RMSE and AIC
        textstr = f'RMSE: {mae:.3f}\nAIC: {aic:.2f}'
        plt.text(0.95, 0.95, textstr,
                 transform=plt.gca().transAxes,
                 fontsize=16,
                 verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

        plt.tight_layout()
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        #plt.show()
        plt.savefig(f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/LV/{method}/{model}_{census}.png')
        plt.close()

    return predicted_rad, mae, aic, entropy


def plot_combined_SAD(empirical, mete, metime, model, var, census_id, macro_var):
    ranks = np.arange(1, len(empirical) + 1)

    # Define custom colors
    redish = "#ef8a62"
    blueish = "#67a9cf"
    greyish = "#4D4D4D"  # darker grey

    plt.rcParams.update({
        'font.size': 20,  # base font size
        'axes.labelsize': 22,  # x and y labels
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 22
    })

    plt.figure(figsize=(6, 4))

    # Plot with updated styles
    plt.plot(ranks, empirical, 'o-', color=greyish, markersize=8, linewidth=3, label='Empirical')
    plt.plot(ranks, mete, 's--', color=blueish, markersize=8, linewidth=3, label='METE')
    plt.plot(ranks, metime, '^--', color=redish, markersize=8, linewidth=3, label='METimE')

    plt.xlabel("Rank", fontsize=16)
    plt.ylabel("Abundance", fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_path = f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/LV/RAD_{model}_{census}.png'
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    #plt.show()


# def plot_combined_SAD(empirical, mete, metime, model, var, census_id, macro_var):
#     ranks = np.arange(1, len(empirical) + 1)
#
#     # Custom colors
#     redish = "#ef8a62"
#     blueish = "#67a9cf"
#     greyish = "#4D4D4D"
#
#     plt.rcParams.update({
#         'font.size': 20,
#         'axes.labelsize': 22,
#         'xtick.labelsize': 16,
#         'ytick.labelsize': 16,
#         'legend.fontsize': 14
#     })
#
#     plt.figure(figsize=(6, 4))
#
#     # Plot actual data
#     plt.plot(ranks, empirical, 'o-', color=greyish, markersize=8, linewidth=3, label='Empirical')
#     plt.plot(ranks, mete, 's--', color=blueish, markersize=8, linewidth=3, label='METE')
#     plt.plot(ranks, metime, '^--', color=redish, markersize=8, linewidth=3, label='METimE')
#
#     # Add fake entries for N/S and dN/S
#     ns_text = f"N/S = {macro_var['N/S']:.2f}"
#     dns_text = f"1/S ΔN = {macro_var['dN/S']:.2f}"
#
#     # Create invisible handles
#     empty_handle = Line2D([], [], color='none', label=ns_text)
#     empty_handle2 = Line2D([], [], color='none', label=dns_text)
#
#     # Legend with all elements
#     plt.legend(handles=[
#         plt.Line2D([], [], color=greyish, marker='o', linestyle='-', markersize=8, linewidth=3, label='Empirical'),
#         plt.Line2D([], [], color=blueish, marker='s', linestyle='--', markersize=8, linewidth=3, label='METE'),
#         plt.Line2D([], [], color=redish, marker='^', linestyle='--', markersize=8, linewidth=3, label='METimE'),
#         empty_handle,
#         empty_handle2
#     ], loc='best', frameon=True)
#
#     plt.xlabel("Rank", fontsize=16)
#     plt.ylabel("Abundance", fontsize=16)
#
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.tight_layout()
#
#     # Fix file save path
#     save_path = f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/LV/RAD_{model}_{census_id}.png'
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path, dpi=300)
#     plt.close()


def generate_aic_mae_entropy_table(df):
    metrics = [
        'AIC_mete', 'MAE_mete', 'entropy_mete',
        'AIC_metime', 'MAE_metime', 'entropy_metime'
    ]

    summary = df.groupby(['model', 'var'])[metrics].mean().reset_index()
    summary = summary.sort_values(['model', 'var'])
    summary = summary.round(3)  # More precision for entropy if needed

    # Optional: Rename columns for prettier LaTeX output
    summary.columns = [
        'Model', 'Variance',
        'METE AIC', 'METE MAE', 'METE Entropy',
        'METimE AIC', 'METimE MAE', 'METimE Entropy'
    ]

    # Format numeric columns to 3 decimal places
    for col in summary.columns[2:]:
        summary[col] = summary[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "--")

    return summary.to_latex(
        index=False,
        caption="Mean AIC, MAE, and entropy for METE and METimE by model and inter-genus variance",
        label="tab:metrics_entropy_by_model_variance",
        column_format='llrrr|rrr'
    )


def generate_r2_table(df):
    r2_table = df.groupby(['var', 'model'])['r^2_transition'].first().unstack()
    r2_table = r2_table.sort_index()

    # Format values as strings with 3 decimal places (and replace NaN with '--')
    formatted_table = r2_table.applymap(
        lambda x: f"{x:.3f}" if pd.notnull(x) else "--"
    )
    return formatted_table.to_latex(
        caption="Transition $R^2$ by model and inter-genus variance",
        label="tab:r2_by_model_variance",
        index_names=True,
        column_format='l' + 'c' * len(r2_table.columns),
        escape=False
    )

if __name__ == "__main__":
    np.random.seed(123)

    results = []

    for model in ['a', 'b', 'c', 'd', 'e', 'f']:
        for var in [0.0, 0.05, 0.1, 0.2]:
            for realization in range(20):
                df = three_groups_LV(model, T=10, var=var)

                # choose only a small number of censuses to do the analysis on
                censuses = df['census'].unique()[::4]

                print("Number of censuses: ", len(censuses), "\n")

                y, y_pred, coeffs = set_up_regression(df, var, LV_model=model, regression_type="global")
                r2_transition = r2_score(y, y_pred)
                coeffs = coeffs['Coefficient'].tolist()

                for census in censuses:
                    input_census = df[df['census'] == census]
                    X = input_census[['S_t', 'N_t']].drop_duplicates().iloc[0]

                    macro_mete = {'N/S': float(X['N_t'] / X['S_t'])}
                    macro_metime = {
                        'N/S': float(X['N_t'] / X['S_t']),
                        'dN/S': input_census['dN'].unique()[0] / X['S_t']
                    }

                    grouped = input_census.groupby('species')['n'].sum()
                    empirical_RAP = grouped.sort_values(ascending=False).values

                    # METE
                    functions_mete = [f_1]
                    initial_lambdas_mete = make_initial_guess(X, 'METE')
                    lambdas_mete = perform_optimization(initial_lambdas_mete, functions_mete, macro_mete, X, [])
                    predicted_mete, mae_mete, aic_mete, entropy_mete = compare_SADs(lambdas_mete, functions_mete, X, [], empirical_RAP, 'METE',
                                                               model, census, plot=False)

                    # METimE
                    functions_metime = [f_1, dn]
                    initial_lambdas_metime = make_initial_guess(X, 'METimE')
                    lambdas_metime = perform_optimization(initial_lambdas_metime, functions_metime, macro_metime, X, coeffs)
                    predicted_metime, mae_metime, aic_metime, entropy_metime = compare_SADs(lambdas_metime, functions_metime, X, coeffs,
                                                                   empirical_RAP, 'METimE', model, census, plot=False)

                    # Plot
                    length_RAD = min(len(empirical_RAP), len(predicted_mete), len(predicted_metime))
                    empirical_RAP = empirical_RAP[:length_RAD]

                    if var == 0 and realization == 0:
                        plot_combined_SAD(empirical_RAP, predicted_mete, predicted_metime, model, var, census, macro_metime)

                    results.append({
                        'model': model,
                        'var': var,
                        'AIC_mete': aic_mete,
                        'MAE_mete': mae_mete,
                        'entropy_mete': entropy_mete,
                        'AIC_metime': aic_metime,
                        'MAE_metime': mae_metime,
                        'entropy_metime': entropy_metime,
                        'N/S': macro_mete['N/S'],
                        'dN/S': macro_metime['dN/S'],
                        'r^2_transition': r2_transition
                    })

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/METimE_vs_METE_summary.csv", index=False)

    # Create and print latex tables
    latex_table1 = generate_r2_table(results_df)
    print(latex_table1)
    print("\n" + "-" * 80 + "\n")

    latex_table2 = generate_aic_mae_entropy_table(results_df)
    print(latex_table2)


