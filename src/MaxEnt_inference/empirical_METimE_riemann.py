import os

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar, minimize
from src.MaxEnt_inference.empirical_METimE_quadrat import check_constraints, get_rank_abundance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from src.MaxEnt_inference.empirical_METimE_quadrat import get_empirical_RAD
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")


def partition_function(lambdas, func_vals, de=0.05):
    lambdas = np.asarray(lambdas).reshape(-1, 1, 1)
    exponent_matrix = np.sum(-lambdas * func_vals, axis=0)
    Z = np.exp(exponent_matrix).sum()

    if de is not None:
        Z *= de

    if np.isclose(Z, 0.0, atol=1e-12):
        print("Invalid values detected in Z")

    if np.isinf(Z):
        print("Invalid values detected in Z")
        Z = 1e10

    return Z

def ecosystem_structure_function(lambdas, func_vals, Z):
    lambdas_arr = np.asarray(lambdas).reshape(-1, 1, 1)
    exponent_matrix = np.sum(-lambdas_arr * func_vals, axis=0)
    R = np.exp(exponent_matrix) / Z

    if np.isnan(R).any():
        raise ValueError("NaN values found in R — check Z and exponent_matrix ranges.")

    return R

def entropy(lambdas, func_vals, de, scales=[1,1,1,1]):
    """
    Computes Shannon entropy: -sum R(n,e) * log(R(n,e))
    """
    # Scale back lambdas
    lambdas = lambdas * scales

    # Partition function Z
    Z = partition_function(lambdas, func_vals, de=de)

    # Ecosystem structure function R
    R = ecosystem_structure_function(lambdas, func_vals, Z)

    # Negative shannon entropy (because we minimize instead of maximize)
    H = np.sum(np.where(R > 0, R * np.log(R), 0)) * de # Only compute log(R) where R > 0 and put 0 otherwise

    # Safety check
    if np.isnan(H) or np.isinf(H):
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

def single_constraint(lambdas, func_vals, func_index, target_value, de, scales=[1,1,1,1]):
    """
    Vectorized single constraint calculation
    """
    # Scale back lambdas
    lambdas = lambdas * scales

    # Partition function Z
    Z = partition_function(lambdas, func_vals, de=de)

    # Ecosystem structure function R
    R = ecosystem_structure_function(lambdas, func_vals, Z)

    # Expected value of f_k under R
    expected_value = np.sum(R * func_vals[func_index]) * de

    # Safety check
    if np.isnan(expected_value) or np.isinf(expected_value):
        print("Invalid values detected in single constraint")
        expected_value = 1e10

    # Return deviation from target
    return expected_value - target_value
    # TODO: optional scaled down so that all constraints weigh equally

def compute_lambda_bounds(min_f, max_f, max_exp=400):
    eps = 1e-12

    # If function is basically zero, allow wide bounds
    if abs(min_f) < eps and abs(max_f) < eps:
        return (-1, 1)

    candidates = []

    for f_b in [min_f, max_f]:
        if abs(f_b) < eps:
            continue
        lower = -max_exp / f_b
        upper = max_exp / f_b

        # For negative f_b, lower will be > upper; swap them
        if lower > upper:
            lower, upper = upper, lower

        candidates.append((lower, upper))

    # Now intersect bounds: take max of lowers, min of uppers
    lowers, uppers = zip(*candidates)
    lower_bound = max(lowers)
    upper_bound = min(uppers)

    # If bounds are invalid (empty intersection), fallback to some safe range
    if lower_bound > upper_bound:
        return (-1, 1)

    return (lower_bound, upper_bound)

def run_optimization(lambdas, macro_var, X, func_vals, de):
    lambdas = np.asarray(lambdas, dtype=float)

    if len(lambdas) == 4:
        # Compute bounds (to prevent overflow in exp)
        f3_vals = func_vals[2, :, :]  # shape (N, len(e_vals))
        f4_vals = func_vals[3, :, :]
        min_f3, max_f3 = f3_vals.min(), f3_vals.max()
        min_f4, max_f4 = f4_vals.min(), f4_vals.max()
        bounds_dn = compute_lambda_bounds(min_f3, max_f3, 100)
        bounds_de = compute_lambda_bounds(min_f4, max_f4, 100)

        # Define scale factors so that parameters are roughly of the same order of magnitude
        values = np.asarray([lambdas[0], lambdas[1], bounds_dn[1], bounds_de[1]], dtype=float)

    else:
        values = np.asarray([lambdas[0], lambdas[1]], dtype=float)

    scales = np.where(values != 0,
                    10.0 ** np.floor(np.log10(np.abs(values))),
                    1.0)
    lambdas = lambdas / scales

    if len(lambdas) == 4:
        bounds = [(0, 18) / scales[0],
                  (0, 18) / scales[1],
                  bounds_dn / scales[2],
                  bounds_de / scales[3]]
    else:
        bounds = [(0, 18) / scales[0],
                  (0, 18) / scales[1]]

    # TODO: is the upper bound of 18 okay?

    # # To make sure that bounds are not violated, we need to take very small steps during minimization
    # all_bounds = list(bounds_dn) + list(bounds_de)
    # abs_bounds = [abs(b) for b in all_bounds]
    # min_abs_bound = min(abs_bounds)
    # step_size = min_abs_bound / 10

    # Collect all constraints
    constraints = [{
        'type': 'eq',
        'fun': lambda lambdas, F_k=macro_var[name]:
        single_constraint(lambdas, func_vals, idx, F_k, de, scales)
    } for idx, name in enumerate(macro_var)]

    # result = minimize(entropy,
    #                   lambdas,
    #                   args=(func_vals, de, scales),
    #                   constraints=constraints,
    #                   bounds=bounds[:len(lambdas)],
    #                   method="trust-constr",
    #                   options={'initial_tr_radius': 0.1,
    #                            'disp': True,
    #                            'verbose': 3
    #                            })

    # This doesn't change the lambda values at all, ChatGPT advises scaling again :(
    result = minimize(entropy,
                      lambdas,
                      args=(func_vals, de, scales),
                      constraints=constraints,
                      bounds=bounds[:len(lambdas)],
                      method="SLSQP",
                      options={'disp': True})

    optimized_lambdas = result.x * scales

    return optimized_lambdas

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

def get_function_values(functions, X, alphas, betas, de):
    e_vals = np.arange(0, X['E_t'] + de, de)
    n_vals = np.arange(1, int(X['N_t']) + 1)

    # Create grids
    n_grid, e_grid = np.meshgrid(n_vals, e_vals, indexing='ij')  # shape: (N, len_e_vals)

    num_funcs = len(functions)
    results = np.zeros((num_funcs, len(n_vals), len(e_vals)))

    for i, func in enumerate(functions):
        results[i] = func(n_grid, e_grid, X, alphas, betas)

    return results, e_vals

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

def evaluate_model(lambdas, X, func_vals, empirical_rad):
    Z = partition_function(lambdas, func_vals, de)
    R = ecosystem_structure_function(lambdas, func_vals, Z)

    # Compute SAD
    sad = np.sum(R, axis=1)

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


if __name__ == "__main__":
    # Use ext='' for full BCI, or ext='_quadrat_i' for quadrat i data
    for ext in ['_quadrat_1']:

        # Load data
        input = pd.read_csv(f'../../data/BCI_regression_library{ext}.csv')
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
            empirical_rad = get_empirical_RAD(f'../../data/BCI_regression_library{ext}.csv', census)['abundance']

            # Precompute functions(n, e)
            de = 1
            func_vals, _ = get_function_values(functions, X, alphas, betas, de)

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
                {
                    'N/S': float(X['N_t'] / X['S_t']),
                    'E/S': float(X['E_t'] / X['S_t'])
                },
                X,
                func_vals[:2],
                de
            )
            print("Optimized lambdas (METE): {}".format(METE_lambdas))
            METE_lambdas = np.append(METE_lambdas, [0, 0])
            constraint_errors = check_constraints(METE_lambdas, input_census, functions, alphas, betas)
            METE_results, METE_rad = evaluate_model(METE_lambdas, X, func_vals, empirical_rad)
            print(f"AIC: {METE_results['AIC'].values[0]}, MAE: {METE_results['MAE'].values[0]}")

            #######################################
            #####           METimE            #####
            #######################################
            print(" ")
            print("----------METimE----------")
            METimE_lambdas = run_optimization(METE_lambdas, macro_var, X, func_vals, de)
            print("Optimized lambdas: {}".format(METimE_lambdas))
            constraint_errors = check_constraints(METimE_lambdas, input_census, functions, alphas, betas)
            METimE_results, METimE_rad = evaluate_model(METimE_lambdas, X, func_vals, empirical_rad)
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

            plot_RADs(empirical_rad, METE_rad, METimE_rad, f'quad_{ext}_census_{census}')

        results_df = pd.DataFrame(results_list)
        results_df.to_csv(f'empirical_BCI_result_df{ext}.csv', index=False)