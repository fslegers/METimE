import numpy as np
import pandas as pd
from scipy.optimize import root_scalar, minimize
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete
import mpmath as mp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error

import warnings

warnings.filterwarnings("ignore")

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

def do_polynomial_regression(df):
    # Select the columns to apply polynomial features
    poly_cols = ['e', 'n', 'S_t', 'N_t', 'E_t']

    # Generate polynomial features
    poly = PolynomialFeatures(degree=3, include_bias=False)
    poly_features = poly.fit_transform(df[poly_cols])

    # Create a new DataFrame with polynomial features
    poly_feature_names = poly.get_feature_names_out(poly_cols)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)

    # Concatenate polynomial features back to the original DataFrame
    df = pd.concat([df.drop(columns=poly_cols), poly_df], axis=1)

    # Drop 'tree_id' and dN/S and dE/S columns
    df = df.drop(columns=['TreeID', 'dN/S', 'dE/S'])
    if 'dS' in df.columns:
        df = df.drop(columns=['dS'])

    # Group by (t, species_id) and sum all features
    df_grouped = df.groupby(['census', 'species']).sum().reset_index()

    # Now fit the linear regression model
    dn_obs = df_grouped['dn']
    de_obs = df_grouped['de']
    X = df_grouped.drop(columns=['census', 'species', 'dn', 'de'])

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    output = []
    for y in[dn_obs, de_obs]:
        model = LinearRegression()
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)

        # De-standardize coefficients
        beta_std = model.coef_
        mu = scaler.mean_
        sigma = scaler.scale_

        beta_orig = beta_std / sigma
        intercept_orig = model.intercept_ - np.sum((beta_std * mu) / sigma)

        # Combine into DataFrame
        coeff_df = pd.DataFrame({
            'Feature': poly_feature_names,
            'Coefficient': beta_orig
        })

        # Add intercept as a separate row (optional but useful)
        coeff_df.loc[len(coeff_df)] = ['Intercept', intercept_orig]

        # Calculate r2
        r2 = r2_score(y, y_pred)

        output.append(coeff_df)
        output.append(r2)

    return output

def plot_RADs(empirical_rad, METE_rad, METimE_rad, save_name):
    ranks = np.arange(1, len(empirical_rad) + 1)

    # Define custom colors
    redish = "#ef8a62"
    blueish = "#67a9cf"
    greyish = "#4D4D4D"

    plt.rcParams.update({
        'font.size': 20,  # base font size
        'axes.labelsize': 22,  # x and y labels
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 22
    })

    plt.figure(figsize=(6, 4))

    # Plot with updated styles
    plt.plot(ranks, empirical_rad, 'o-', color=greyish, markersize=6, linewidth=2, label='Empirical')
    plt.plot(ranks, METE_rad, 's--', color=blueish, markersize=6, linewidth=2, label='METE')
    plt.plot(ranks, METimE_rad, '^--', color=redish, markersize=6, linewidth=2, label='METimE')

    plt.xlabel("Rank", fontsize=16)
    plt.ylabel("Abundance", fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_path = f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/BCI/{save_name}.png'
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    pass

###############################################
### Ecosystem Structure Function Components ###
###############################################

def exp_in_R(n, e, X, functions, lambdas, alphas, betas):
    """
    Compute the exponent term: -lambda1*f1 - ... - lambdak*fk
    """
    exponent = sum(-lambdas[i] * functions[i](n, e, X, alphas, betas) for i in range(len(functions)))
    return exponent


def partition_function(lambdas, functions, X, alphas, betas):
    def integrand(e, n):
        exponent = sum(
            -lambdas[i] * functions[i](n, e, X, alphas, betas)
            for i in range(len(functions))
        )
        return np.exp(exponent)

    Z = 0
    num_bins = 11  # or make this a parameter if you want flexibility
    edges = (np.linspace(0, 1, num_bins + 1) ** 2) * X['E_t']

    for n in range(1, int(X['N_t']) + 1):
        Z += sum(
            quad(lambda e: integrand(e, n), a, b)[0]
            for a, b in zip(edges[:-1], edges[1:])
        )

    return Z


def entropy(lambdas, functions, X, alphas, betas, scales):
    """
    Compute Shannon entropy for the given lambdas and functions.
    """
    lambdas = lambdas * scales

    def integrand(e, n, Z):
        exponent = sum(-lambdas[i] * functions[i](n, e, X, alphas, betas) for i in range(len(functions)))
        return np.exp(exponent) / Z * (np.log(1/Z) + exp_in_R(n, e, X, functions, lambdas, alphas, betas))

    Z = partition_function(lambdas, functions, X, alphas, betas)

    H = 0
    for n in range(1, int(X['N_t']) + 1):
        H += quad(lambda e: integrand(e, n, Z), 0, X['E_t'])[0]

    if np.any(np.isnan(H)) or np.any(np.isinf(H)):
        print("Invalid values detected in entropy")
        H = 1e10

    return H


def beta_function(beta, S, N):
    """
    Beta function used to generate the initial guess for Lagrange multipliers.
    """
    return (1 - np.exp(-beta)) / (np.exp(-beta) - np.exp(-beta * (N + 1))) * np.log(1.0 / beta) - S / N


def make_initial_guess(X):
    """
    A function that makes an initial guess for the Lagrange multipliers lambda1 and lambda2.
    Based on Eq 7.29 from Harte 2011 and meteR'diag function meteESF.mete.lambda

    :param state_variables: state variables S, S and E
    :return: initial guess for the Lagrange multipliers lambda1 and lambda2
    """
    S, N, E = int(X['S_t']), int(X['N_t']), float(X['E_t'])
    interval = [1.0 / N, S / N]

    beta = root_scalar(beta_function, x0=0.001, args=(S, N), method='brentq', bracket=interval)

    l2 = S / (E - N)
    l1 = beta.root - l2

    if l1 < 0 or l2 < 0: # Assumption based on "Derivations of the Core Functions of METE": l1 and l2 are strictly positive
        l1 = 0.1         # this assumption comes from somewhere else but not sure where
        l2 = 0.01

    return [l1, l2, 0, 0]


###############################################
###               Optimization              ###
###############################################

def single_constraint(X, f_k, F_k, all_f, lambdas, alphas, betas, scales):
    def integrand(e, n):
        exponent = sum(
            -lambdas[i] * all_f[i](n, e, X, alphas, betas)
            for i in range(len(all_f))
        )
        return np.exp(exponent) * f_k(n, e, X, alphas, betas)

    lambdas = lambdas * scales

    Z = partition_function(lambdas, all_f, X, alphas, betas)

    expected_value = 0
    for n in range(1, int(X['N_t']) + 1):
        edges = np.linspace(0, 1, 11 + 1) ** 2 * X['E_t']  # quadratic binning
        expected_value += sum(
            quad(lambda e: integrand(e, n), a, b)[0]
            for a, b in zip(edges[:-1], edges[1:])
        )
    expected_value /= Z

    if np.any(np.isnan(expected_value)) or np.any(np.isinf(expected_value)):
        print("Invalid values detected in single constraint")
        expected_value = 1e10

    return expected_value - F_k
    # TODO: optional scaled down so that all constraints weigh equally

# def constraint(f_k, lambdas, functions, F_k, X, alphas, betas):
#     """
#     Calculates the expected value of a single constraint function f_k over the ecosystem structure function:
#     Σ ʃ f_k R(n, e) / Z de - F_k
#     Parallelized version.
#     """
#     with ProcessPoolExecutor(max_workers=2) as executor:
#         futures = [
#             executor.submit(
#                 single_constraint,
#                 n, X['E_t'], X, f_k, functions, lambdas, alphas, betas
#             )
#             for n in range(1, int(X['N_t']) + 1)
#         ]
#         contributions = [f.result() for f in futures]
#
#     I = sum(contributions)
#     Z = integrate_with_cutoff(X, functions, lambdas)
#     return I / Z - F_k


def compute_bounds(X, alphas, betas):
    N_max = X['N_t']
    E_max = X['E_t']

    # Define the boundary values to check
    n_values = [1, N_max]
    e_values = [0, E_max]

    max_abs_val_dn = 0
    max_abs_val_de = 0

    for n in n_values:
        for e in e_values:
            val_dn = np.abs(f_dn(n, e, X, alphas, betas))
            val_de = np.abs(f_de(n, e, X, alphas, betas))

            if val_dn > max_abs_val_dn:
                max_abs_val_dn = val_dn

            if val_de > max_abs_val_de:
                max_abs_val_de = val_de

    if max_abs_val_dn == 0:
        bounds_dn = (-1, 1)  # TODO: is dit te klein?
    else:
        bounds_dn = (-400/max_abs_val_dn, 400/max_abs_val_dn)

    if max_abs_val_de == 0:
        bounds_de = (-1, 1)   # TODO: is dit te klein?
    else:
        bounds_de = (-400/max_abs_val_de, 400/max_abs_val_de)

    # print(f"Bounds for dn: {bounds_dn}")
    # print(f"Bounds for de: {bounds_de}")

    return bounds_dn, bounds_de


def run_optimization(lambdas, functions, macro_var, X, alphas, betas):
    # Set bounds and scale all lambas to be of order of magnitude ~10
    if len(lambdas) == 4:
        bounds_dn, bounds_de = compute_bounds(X, alphas, betas)
        values = np.asarray([lambdas[0], lambdas[1], bounds_dn[1], bounds_de[1]], dtype=float)
    else:
        values = np.asarray([lambdas[0], lambdas[1]], dtype=float)

    scales = np.where(values != 0,10.0 ** np.floor(np.log10(np.abs(values))),1.0)
    lambdas = lambdas / scales

    bounds = ([(0, 18) / scales[0],
              (0, 18) / scales[1]])

    if len(lambdas) == 4:
        bounds.append([(bounds_dn[0], bounds_dn[1]) / scales[2],
                      (bounds_de[0], bounds_de[1]) / scales[3]])

    # Collect all constraints
    constraints = [{
        'type': 'eq',
        'fun': lambda lambdas, functions=functions, f_k=f, F_k=macro_var[name], X=X:
        single_constraint(X, f_k, F_k, functions, lambdas, alphas, betas, scales)
    } for f, name in zip(functions, macro_var)]

    # Perform optimization
    result = minimize(entropy,
                      lambdas,
                      args=(functions, X, alphas, betas, scales),
                      constraints=constraints,
                      bounds=bounds[:len(lambdas)],
                      method="trust-constr",
                      options={'initial_tr_radius': 0.1,
                               'gtol': 1e-12,
                               'disp': True,
                               'verbose': 3
                               })

    optimized_lambdas = result.x * scales

    return optimized_lambdas

########################
### Set-up and check ###
########################

def f_n(n, e, X, alphas, betas):
    return n

def f_ne(n, e, X, alphas, betas):
    return n * e

def f_dn(n, e, X, alphas, betas):
    """
    expects coefficients alphas are ordened
    first, order columns: e, n, S, N, E
    then, polynomial features, order 3, include_bias=False
    """
    return (alphas[0] * e +
    alphas[1] * n +
    alphas[2] * X['S_t'] +
    alphas[3] * X['N_t'] +
    alphas[4] * X['E_t'] +
    alphas[5] * e ** 2 +
    alphas[6] * e * n +
    alphas[7] * e * X['S_t'] +
    alphas[8] * e * X['N_t'] +
    alphas[9] * e * X['E_t'] +
    alphas[10] * n ** 2 +
    alphas[11] * n * X['S_t'] +
    alphas[12] * n * X['N_t'] +
    alphas[13] * n * X['E_t'] +
    alphas[14] * X['S_t'] ** 2 +
    alphas[15] * X['S_t'] * X['N_t'] +
    alphas[16] * X['S_t'] * X['E_t'] +
    alphas[17] * X['N_t'] ** 2 +
    alphas[18] * X['N_t'] * X['E_t'] +
    alphas[19] * X['E_t'] ** 2 +
    alphas[20] * e ** 3 +
    alphas[21] * e ** 2 * n +
    alphas[22] * e ** 2 * X['S_t'] +
    alphas[23] * e ** 2 * X['N_t'] +
    alphas[24] * e ** 2 * X['E_t'] +
    alphas[25] * e * n ** 2 +
    alphas[26] * e * n * X['S_t'] +
    alphas[27] * e * n * X['N_t'] +
    alphas[28] * e * n * X['E_t'] +
    alphas[29] * e * X['S_t'] ** 2 +
    alphas[30] * e * X['S_t'] * X['N_t'] +
    alphas[31] * e * X['S_t'] * X['E_t'] +
    alphas[32] * e * X['N_t'] ** 2 +
    alphas[33] * e * X['N_t'] * X['E_t'] +
    alphas[34] * e * X['E_t'] ** 2 +
    alphas[35] * n ** 3 +
    alphas[36] * n ** 2 * X['S_t'] +
    alphas[37] * n ** 2 * X['N_t'] +
    alphas[38] * n ** 2 * X['E_t'] +
    alphas[39] * n * X['S_t'] ** 2 +
    alphas[40] * n * X['S_t'] * X['N_t'] +
    alphas[41] * n * X['S_t'] * X['E_t'] +
    alphas[42] * n * X['N_t'] ** 2 +
    alphas[43] * n * X['N_t'] * X['E_t'] +
    alphas[44] * n * X['E_t'] ** 2 +
    alphas[45] * X['S_t'] ** 3 +
    alphas[46] * X['S_t'] ** 2 * X['N_t'] +
    alphas[47] * X['S_t'] ** 2 * X['E_t'] +
    alphas[48] * X['S_t'] * X['N_t'] ** 2 +
    alphas[49] * X['S_t'] * X['N_t'] * X['E_t'] +
    alphas[50] * X['S_t'] * X['E_t'] ** 2 +
    alphas[51] * X['N_t'] ** 3 +
    alphas[52] * X['N_t'] ** 2 * X['E_t'] +
    alphas[53] * X['N_t'] * X['E_t'] ** 2 +
    alphas[54] * X['E_t'] ** 3 +
    alphas[55])

def f_de(n, e, X, alphas, betas):
    return (betas[0] * e +
    betas[1] * n +
    betas[2] * X['S_t'] +
    betas[3] * X['N_t'] +
    betas[4] * X['E_t'] +
    betas[5] * e ** 2 +
    betas[6] * e * n +
    betas[7] * e * X['S_t'] +
    betas[8] * e * X['N_t'] +
    betas[9] * e * X['E_t'] +
    betas[10] * n ** 2 +
    betas[11] * n * X['S_t'] +
    betas[12] * n * X['N_t'] +
    betas[13] * n * X['E_t'] +
    betas[14] * X['S_t'] ** 2 +
    betas[15] * X['S_t'] * X['N_t'] +
    betas[16] * X['S_t'] * X['E_t'] +
    betas[17] * X['N_t'] ** 2 +
    betas[18] * X['N_t'] * X['E_t'] +
    betas[19] * X['E_t'] ** 2 +
    betas[20] * e ** 3 +
    betas[21] * e ** 2 * n +
    betas[22] * e ** 2 * X['S_t'] +
    betas[23] * e ** 2 * X['N_t'] +
    betas[24] * e ** 2 * X['E_t'] +
    betas[25] * e * n ** 2 +
    betas[26] * e * n * X['S_t'] +
    betas[27] * e * n * X['N_t'] +
    betas[28] * e * n * X['E_t'] +
    betas[29] * e * X['S_t'] ** 2 +
    betas[30] * e * X['S_t'] * X['N_t'] +
    betas[31] * e * X['S_t'] * X['E_t'] +
    betas[32] * e * X['N_t'] ** 2 +
    betas[33] * e * X['N_t'] * X['E_t'] +
    betas[34] * e * X['E_t'] ** 2 +
    betas[35] * n ** 3 +
    betas[36] * n ** 2 * X['S_t'] +
    betas[37] * n ** 2 * X['N_t'] +
    betas[38] * n ** 2 * X['E_t'] +
    betas[39] * n * X['S_t'] ** 2 +
    betas[40] * n * X['S_t'] * X['N_t'] +
    betas[41] * n * X['S_t'] * X['E_t'] +
    betas[42] * n * X['N_t'] ** 2 +
    betas[43] * n * X['N_t'] * X['E_t'] +
    betas[44] * n * X['E_t'] ** 2 +
    betas[45] * X['S_t'] ** 3 +
    betas[46] * X['S_t'] ** 2 * X['N_t'] +
    betas[47] * X['S_t'] ** 2 * X['E_t'] +
    betas[48] * X['S_t'] * X['N_t'] ** 2 +
    betas[49] * X['S_t'] * X['N_t'] * X['E_t'] +
    betas[50] * X['S_t'] * X['E_t'] ** 2 +
    betas[51] * X['N_t'] ** 3 +
    betas[52] * X['N_t'] ** 2 * X['E_t'] +
    betas[53] * X['N_t'] * X['E_t'] ** 2 +
    betas[54] * X['E_t'] ** 3 +
    betas[55])

def get_functions():
    return [f_n, f_ne, f_dn, f_de]

def check_constraints(lambdas, input, functions, alphas, betas):
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
        'N/S': X['N_t'] / X['S_t'],
        'E/S': X['E_t'] / X['S_t'],
        'dN/S': input['dN/S'].unique()[0],
        'dE/S': input['dE/S'].unique()[0]
    }

    Z = partition_function(lambdas, functions, X, alphas, betas)
    print("Z (quadrat): {}".format(Z))

    absolute_errors = []
    percentage_errors = []

    for f, (key, v) in zip(functions, macro_var.items()):
        # Compute integral with upper bound
        integral_value = 0
        for n in range(1, int(X['N_t'] + 1)):
            edges = np.linspace(0, 1, 11 + 1) ** 2 * X['E_t']
            integral_value += sum(
                quad(
                    lambda e: f(n, e, X, alphas, betas) * np.exp(exp_in_R(n, e, X, functions, lambdas, alphas, betas)),
                    a, b
                )[0]
                for a, b in zip(edges[:-1], edges[1:])
            )

        integral_value /= Z  # TODO: check

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

def get_empirical_RAD(file, census):
    # Load relevant data
    df = pd.read_csv(file)

    if 'census' not in df.columns:
        df = df.rename(columns={'t': 'census', 'Species_ID': 'species'})

    df = df[df['census'] == census]
    df = df[['species', 'n']].drop_duplicates()

    # Create rank abundance distribution
    df = df.sort_values(by='n', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    rad = df[['rank', 'n']].rename(columns={'n': 'abundance'})

    return rad

def get_rank_abundance(sad, X):
    """
    Generate a predicted rank-abundance distribution using the quantile method.
    Ensures exactly S_t values by clipping quantiles and handling edge cases.
    """
    S = int(X['S_t']) + 1

    if np.sum(sad) > 0:
        sad = sad / np.sum(sad)
    else:
        sad = np.ones_like(sad) / len(sad)

    # Create the discrete distribution
    n_vals = np.arange(1, len(sad) + 1)
    dist = rv_discrete(name='sad_dist', values=(n_vals, sad))

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


def get_sad(lambdas, functions, X, alphas, betas, Z):
    sad = np.zeros(int(X['N_t']))

    if abs(Z) > 1e-300:
        for n in range(1, int(X['N_t']) + 1):
            sad[n - 1] = 1 / Z * quad(
                lambda e: np.exp(sum([-lambdas[i] * functions[i](n, e, X, alphas, betas) for i in range(4)])),
                0, X['E_t']
            )[0]
            Z = sum(sad)
            sad = np.array([x/Z for x in sad], dtype=float)
    else:
        for n in range(1, int(X['N_t']) + 1):
            sad[n - 1] = mp.quad(
                lambda e: mp.exp(sum([-lambdas[i] * functions[i](n, e, X, alphas, betas)
                                      for i in range(4)])),
                [0, X['E_t']]
            )
        # Convert back to
        Z = mp.fsum(sad)
        sad = np.array([float(x/Z) for x in sad], dtype=float)
    return sad


def evaluate_model(lambdas, functions, X, alphas, betas, empirical_rad, constraint_errors):
    # Compute SAD
    Z = partition_function(lambdas, functions, X, alphas, betas)
    sad = get_sad(lambdas, functions, X, alphas, betas, Z)

    # Resize to match empirical_rad length
    rad = get_rank_abundance(sad, X)
    rad = rad[:len(empirical_rad)]
    empirical_rad = empirical_rad[:len(rad)]

    # MEA
    mae = mean_absolute_error(empirical_rad, rad)

    # AIC
    eps = 1e-10
    log_probs = np.log(sad[rad - 1] + eps)  # -1 for indexing
    log_likelihood = np.sum(log_probs)
    k = len(lambdas)
    aic = -2 * log_likelihood + 2 * k

    results_data = {
        'MAE': [mae],
        'AIC': [aic],
    }

    # Add lambdas to dictionary
    for i, lam in enumerate(lambdas):
        results_data[f'lambda_{i}'] = [lam]

    for constr, error in zip(['N/S', 'E/S', 'dN', 'dE'], constraint_errors):
        results_data[f'{constr}'] = error

    # Create DataFrame
    results_df = pd.DataFrame(results_data)

    return results_df, rad

if __name__ == "__main__":
    # Load data
    input = pd.read_csv(f'../../data/BCI_regression_library.csv')
    functions = get_functions()

    # Compute polynomial coefficients
    alphas, _, betas, _ = do_polynomial_regression(input)
    alphas = alphas['Coefficient'].values
    betas = betas['Coefficient'].values

    # Create list to store results
    results_list = []

    for census in input['census'].unique():
        print(f"\n Census: {census} \n")
        input_census = input[input['census'] == census]

        X = input_census[[
            'S_t', 'N_t', 'E_t',
        ]].drop_duplicates().iloc[0]

        macro_var = {
            'N/S': float(X['N_t'] / X['S_t']),
            'E/S': float(X['E_t'] / X['S_t']),
            'dN/S': input_census['dN/S'].unique()[0],
            'dE/S': input_census['dE/S'].unique()[0]
        }

        # Get empirical rank abundance distribution
        empirical_rad = get_empirical_RAD(f'../../data/BCI_regression_library.csv', census)['abundance']

        # Make initial guess
        initial_lambdas = make_initial_guess(X)
        print(f"Initial guess (theoretical): {initial_lambdas}")

        #######################################
        #####            METE             #####
        #######################################
        print(" ")
        print("----------METE----------")
        METE_lambdas = run_optimization(
            initial_lambdas[:2],
            functions[:2],
            {
                'N/S': float(X['N_t'] / X['S_t']),
                'E/S': float(X['E_t'] / X['S_t'])
            },
            X,
            alphas,
            betas
        )
        print("Optimized lambdas (METE): {}".format(METE_lambdas))
        METE_lambdas = np.append(METE_lambdas, [0, 0])
        constraint_errors = check_constraints(METE_lambdas, input_census, functions, alphas, betas)
        METE_results, METE_rad = evaluate_model(METE_lambdas, functions, X, alphas, betas, empirical_rad, constraint_errors)
        print(f"AIC: {METE_results['AIC'].values[0]}, MAE: {METE_results['MAE'].values[0]}")

        #######################################
        #####           METimE            #####
        #######################################
        print(" ")
        print("----------METimE----------")
        METimE_lambdas = run_optimization(METE_lambdas, input_census, functions, alphas, betas)
        print("Optimized lambdas: {}".format(METimE_lambdas))
        constraint_errors = check_constraints(METimE_lambdas, input_census, functions, alphas, betas)
        METimE_results, METimE_rad = evaluate_model(METimE_lambdas, functions, X, alphas, betas, empirical_rad, constraint_errors)
        print(f"AIC: {METimE_results['AIC'].values[0]}, MAE: {METimE_results['MAE'].values[0]}")

        ##########################################
        #####           Save results         #####
        ##########################################
        results_list.append({
            'census': census,
            'METE_AIC': METE_results['AIC'].values[0],
            'METE_MAE': METE_results['MAE'].values[0],
            'METimE_AIC': METimE_results['AIC'].values[0],
            'METimE_MAE': METimE_results['MAE'].values[0]
        })

        plot_RADs(empirical_rad, METE_rad, METimE_rad, f'full_census_{census}')

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(f'empirical_BCI_result_df_full.csv', index=False)
