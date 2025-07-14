import numpy as np
import pandas as pd
from scipy.optimize import root_scalar, minimize
from scipy.integrate import quad
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import mpmath as mp

from bayes_opt import BayesianOptimization
from scipy.optimize import NonlinearConstraint



def exp_in_R(n, e, X, functions, lambdas, alphas, betas, scaling_factors=[1, 1, 1, 1]):
    """
    Compute the exponent term: -lambda1*f1 - ... - lambdak*fk
    """
    exponent = sum(-lambdas[i] * functions[i](n, e, X, alphas, betas, scaling_factors) for i in range(len(functions)))
    return exponent


def partition_function(lambdas, functions, X, alphas, betas, scaling_factors=[1, 1, 1, 1]):
    def integrand(e, n):
        exponent = sum(-lambdas[i] * functions[i](n, e, X, alphas, betas, scaling_factors) for i in range(len(functions)))
        return np.exp(exponent)

    Z = 0
    for n in range(1, int(X['N_t']) + 1):
        Z += quad(lambda e: integrand(e, n),  0, X['E_t'])[0]

    return Z


def entropy(lambdas, functions, X, alphas, betas, scaling_factors=[1, 1, 1, 1]):
    """
    Compute Shannon entropy for the given lambdas and functions.
    this function will be MAXIMIZED
    """
    def integrand(e, n, Z):
        exponent = sum(-lambdas[i] * functions[i](n, e, X, alphas, betas, scaling_factors) for i in range(len(functions)))
        return np.exp(exponent) / Z * (np.log(1/Z) + exp_in_R(n, e, X, functions, lambdas, alphas, betas, scaling_factors))

    Z = partition_function(lambdas, functions, X, alphas, betas, scaling_factors)

    H = 0
    for n in range(1, int(X['N_t']) + 1):
        H += quad(lambda e: integrand(e, n, Z), 0, X['E_t'])[0]

    return -H


def single_constraint(X, f_k, functions, lambdas, alphas, betas, scaling_factors=[1, 1, 1, 1]):
    def integrand(e, n):
        exponent = sum(-lambdas[i] * functions[i](n, e, X, alphas, betas, scaling_factors) for i in range(len(functions)))
        return np.exp(exponent) * f_k(n, e, X, alphas, betas)

    Z = partition_function(lambdas, functions, X, alphas, betas, scaling_factors)

    expected_value = 0
    for n in range(1, int(X['N_t']) + 1):
        expected_value += quad(lambda e: integrand(e, n), 0, X['E_t'])[0]
    expected_value /= Z

    return expected_value


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

    print(f"Bounds for dn: {bounds_dn}")
    print(f"Bounds for de: {bounds_de}")

    return bounds_dn, bounds_de


def f_n(n, e, X, alphas, betas, scaling_factors=[1, 1, 1, 1]):
    return n / scaling_factors[0]

def f_ne(n, e, X, alphas, betas, scaling_factors=[1, 1, 1, 1]):
    return n * e / scaling_factors[1]

def f_dn(n, e, X, alphas, betas, scaling_factors=[1, 1, 1, 1]):
    return (
    alphas[0] * e +
    alphas[1] * n +
    alphas[2] * e ** 2 +
    alphas[3] * e * X['S_t'] +
    alphas[4] * e * n +
    alphas[5] * e * X['N_t'] +
    alphas[6] * e * X['E_t'] +
    alphas[7] * X['S_t'] * n +
    alphas[8] * n ** 2 +
    alphas[9] * n * X['N_t'] +
    alphas[10] * n * X['E_t'] +
    alphas[11] * e ** 3 +
    alphas[12] * e ** 2 * X['S_t'] +
    alphas[13] * e ** 2 * n +
    alphas[14] * e ** 2 * X['N_t'] +
    alphas[15] * e ** 2 * X['E_t'] +
    alphas[16] * e * X['S_t'] ** 2 +
    alphas[17] * e * X['S_t'] * n +
    alphas[18] * e * X['S_t'] * X['N_t'] +
    alphas[19] * e * X['S_t'] * X['E_t'] +
    alphas[20] * e * n ** 2 +
    alphas[21] * e * n * X['N_t'] +
    alphas[22] * e * n * X['E_t'] +
    alphas[23] * e * X['N_t'] ** 2 +
    alphas[24] * e * X['N_t'] * X['E_t'] +
    alphas[25] * e * X['E_t'] ** 2 +
    alphas[26] * X['S_t'] ** 2 * n +
    alphas[27] * X['S_t'] * n ** 2 +
    alphas[28] * X['S_t'] * n * X['N_t'] +
    alphas[29] * X['S_t'] * n * X['E_t'] +
    alphas[30] * n ** 3 +
    alphas[31] * n ** 2 * X['N_t'] +
    alphas[32] * n ** 2 * X['E_t'] +
    alphas[33] * n * X['N_t'] ** 2 +
    alphas[34] * n * X['N_t'] * X['E_t'] +
    alphas[35] * n * X['E_t'] ** 2) / scaling_factors[2]

def f_de(n, e, X, alphas, betas, scaling_factors=[1, 1, 1, 1]):
    return (
    betas[0] * e +
    betas[1] * n +
    betas[2] * e ** 2 +
    betas[3] * e * X['S_t'] +
    betas[4] * e * n +
    betas[5] * e * X['N_t'] +
    betas[6] * e * X['E_t'] +
    betas[7] * X['S_t'] * n +
    betas[8] * n ** 2 +
    betas[9] * n * X['N_t'] +
    betas[10] * n * X['E_t'] +
    betas[11] * e ** 3 +
    betas[12] * e ** 2 * X['S_t'] +
    betas[13] * e ** 2 * n +
    betas[14] * e ** 2 * X['N_t'] +
    betas[15] * e ** 2 * X['E_t'] +
    betas[16] * e * X['S_t'] ** 2 +
    betas[17] * e * X['S_t'] * n +
    betas[18] * e * X['S_t'] * X['N_t'] +
    betas[19] * e * X['S_t'] * X['E_t'] +
    betas[20] * e * n ** 2 +
    betas[21] * e * n * X['N_t'] +
    betas[22] * e * n * X['E_t'] +
    betas[23] * e * X['N_t'] ** 2 +
    betas[24] * e * X['N_t'] * X['E_t'] +
    betas[25] * e * X['E_t'] ** 2 +
    betas[26] * X['S_t'] ** 2 * n +
    betas[27] * X['S_t'] * n ** 2 +
    betas[28] * X['S_t'] * n * X['N_t'] +
    betas[29] * X['S_t'] * n * X['E_t'] +
    betas[30] * n ** 3 +
    betas[31] * n ** 2 * X['N_t'] +
    betas[32] * n ** 2 * X['E_t'] +
    betas[33] * n * X['N_t'] ** 2 +
    betas[34] * n * X['N_t'] * X['E_t'] +
    betas[35] * n * X['E_t'] ** 2) / scaling_factors[3]

def evaluate_model(lambdas, functions, X, alphas, betas, empirical_rad, model, census, ext=""):
    # Compute SAD
    Z = partition_function(lambdas, functions, X, alphas, betas)
    sad = np.zeros(int(X['N_t']))

    if abs(Z) > 1e-300:
        for n in range(1, int(X['N_t']) + 1):
            sad[n - 1] = 1 / Z * quad(
                lambda e: np.exp(sum([-lambdas[i] * functions[i](n, e, X, alphas, betas) for i in range(4)])),
                0, X['E_t']
            )[0]
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

    # Resize to match empirical_rad length
    rad = get_rank_abundance(sad, X)
    rad = rad[:len(empirical_rad)]
    empirical_rad = empirical_rad[:len(rad)]

    # RMSE
    rmse = np.sqrt(mean_squared_error(empirical_rad, rad))

    # AIC
    eps = 1e-10
    log_probs = np.log(sad[rad - 1] + eps)  # -1 for indexing
    log_likelihood = np.sum(log_probs)
    k = len(initial_lambdas)
    aic = -2 * log_likelihood + 2 * k


    plt.figure(figsize=(8, 5))
    ranks = np.arange(1, len(empirical_rad) + 1)
    plt.plot(ranks, empirical_rad, 'o-', label='Empirical RAD', color='blue')
    plt.plot(ranks, rad, 's--', label='Predicted RAD', color='red')
    plt.xlabel('Rank')
    plt.ylabel('Abundance')
    plt.legend()

    # Annotate RMSE and AIC
    textstr = f'RMSE: {rmse:.3f}\nAIC: {aic:.2f}'
    plt.text(0.95, 0.95, textstr,
                     transform=plt.gca().transAxes,
                     fontsize=16,
                     verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    plt.tight_layout()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    # plt.show()
    plt.savefig(f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/BCI/{ext}{model}_{census}.png')

    results_data = {
        'RMSE': [rmse],
        'AIC': [aic],
    }

    # Add lambdas to dictionary
    for i, lam in enumerate(lambdas):
        results_data[f'lambda_{i}'] = [lam]

    # Create DataFrame
    results_df = pd.DataFrame(results_data)

    # Save to CSV (same name as PNG but .csv extension)
    results_df.to_csv(f'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/BCI/{ext}{model}_{census}.csv', index=False)

def constraint_function(l1, l2, l3, l4):
    return np.array([
        single_constraint(X, f_n, functions, [l1, l2, l3, l4], alphas, betas),
        single_constraint(X, f_e, functions, [l1, l2, l3, l4], alphas, betas),
        single_constraint(X, f_dn, functions, [l1, l2, l3, l4], alphas, betas),
        single_constraint(X, f_de, functions, [l1, l2, l3, l4], alphas, betas)]
    )


if __name__ == "__main__":
    # Bound region of parameter space
    pbounds = {'l1': (0, None),
               'l2': (0, None),
               'l3': (-0.05, 0.05),
               'l4': (-2.3e-06, 2.3e06)}

    # Provide constraints
    constraints_LBs = np.array([100, 10000, -0.05, -2.3e-06])
    constraints_UBs = np.array([100, 10000, 0.05, 2.3e06])
    constraint = NonlinearConstraint(constraint_function, constraints_LBs, constraints_UBs)

    optimizer = BayesianOptimization(
        f=entropy,
        constraint=constraint_function,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )

    # Make sure to visit the METE solution
    optimizer.probe(
        params = {'l1': 0.1706231298744739,
                  'l2': 0.0012045661094649538,
                  'l3': 0,
                  'l4': 0},
        lazy=True
    )

    optimizer.maximize(
        init_points=3,
        n_iter=10,
    )

    print(optimizer.max)

