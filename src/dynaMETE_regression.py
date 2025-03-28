import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error, r2_score
from METE_no_integrals import make_initial_guess, perform_optimization
import matplotlib.pyplot as plt


def load_data(data_set):

    if data_set == "birds":
        df = pd.read_csv('../data/birds_regression_library.csv')
        df = df.rename(columns={'m': 'e', 'dm': 'de'})
        df = df[df['census'] >= 2005]
        df.reset_index(drop=True, inplace=True)

    if data_set == "BCI":
        df = pd.read_csv('../data/BCI_regression_library.csv')
        print("BCI: Removing first ... censuses from data")
        df = df[df['census'] >= 5]
        df.reset_index(drop=True, inplace=True)

    return df

def f(n, e, X, p):
    return (p['b'] - p['d'] * X['E']/p['Ec']) * n / e**(1/3) + p['m'] * n / X['N']

def h(n, e, X, p):
    return (p['w'] - p['d'] * X['E'] / p['Ec']) * n * e**(2/3) - p['w1'] * n * e / np.log(1 / X['beta'])**(2/3) + p['m'] * n / X['N']

def q(n, e, X, p):
    kron = int(np.rint(n)) == 1
    #return p['m'] * np.exp(-p['mu_meta'] * X['S'] - np.euler_gamma) + (p['sigma_1'] * p['K'] / (p['K'] + X['S']) + p['sigma_2'] * p['b'] * n / e**(1/3) - kron * p['d'] / e**(1/3) * X['E'] / p['Ec']) * X['S']
    return p['m'] * np.exp(-p['mu_meta'] * X['S'] - np.euler_gamma) - kron * p['d'] / p['Ec'] * X['E'] * X['S']/e**(1/3)


def residuals(params, data):
    p = {
        'b': params[0], 'd': params[1], 'Ec': params[2], 'm': params[3],
        'w': params[4], 'w1': params[5], 'mu_meta': params[6],
        'sigma_1': params[7], 'K': params[8], 'sigma_2': params[9]
    }

    # For each row (observation), calculate the sum of squared errors for each transition function
    res = []
    for i, row in data.iterrows():
        print(f"Processing row {i}")
        n, e = row['n'], row['e']
        X = {'S': row['S_t'], 'N': row['N_t'], 'E': row['E_t'], 'beta': row['beta']}

        # Observed values
        obs_f, obs_h, obs_q = row['dn'], row['de'], row['dS']

        # Residuals
        res.append(obs_f - f(n, e, X, p))
        res.append(obs_h - h(n, e, X, p))
        res.append(obs_q - q(n, e, X, p))

    return np.array(res)


def do_least_squares(initial_guess, data, data_set):
    result = least_squares(fun = residuals, x0 = initial_guess, args=(data,))
    print("Optimization complete!")
    optimized_params = result.x

    p = {
        'b': optimized_params[0], 'd': optimized_params[1], 'Ec': optimized_params[2], 'm': optimized_params[3],
        'w': optimized_params[4], 'w1': optimized_params[5], 'mu_meta': optimized_params[6],
        'sigma_1': optimized_params[7], 'K': optimized_params[8], 'sigma_2': optimized_params[9]
    }

    # # Error on training set
    # for transition_function in ['dn', 'de', 'dS']:
    #
    #     y = data[transition_function].values
    #
    #     if transition_function == 'dn':
    #         y_pred = [f(n, e, {'S': S, 'N': N, 'E': E, 'beta': beta}, p)
    #                   for n, e, S, N, E, beta in
    #                   zip(data['n'], data['e'], data['S_t'], data['N_t'], data['E_t'], data['beta'])]
    #     elif transition_function == 'de':
    #         y_pred = [h(n, e, {'S': S, 'N': N, 'E': E, 'beta': beta}, p)
    #                   for n, e, S, N, E, beta in
    #                   zip(data['n'], data['e'], data['S_t'], data['N_t'], data['E_t'], data['beta'])]
    #     else:
    #         y_pred = [q(n, e, {'S': S, 'N': N, 'E': E, 'beta': beta}, p)
    #                   for n, e, S, N, E, beta in
    #                   zip(data['n'], data['e'], data['S_t'], data['N_t'], data['E_t'], data['beta'])]
    #
    #     rmse = (mean_squared_error(y, y_pred)) ** (1 / 2)
    #     r2 = r2_score(y, y_pred)
    #     print("f(n, e): RMSE = %.3f, \n R^2 = %.3f \n" % (rmse, r2))
    #
    #     # Scatterplot with color by YEAR
    #     census = data['census']
    #
    #     plt.figure(figsize=(8, 6))
    #     scatter = plt.scatter(y, y_pred, cmap='viridis', c=census, alpha=0.7, zorder=1)
    #     plt.colorbar(scatter, label='Census')
    #     plt.xlabel(f'Observed {transition_function}')
    #     plt.ylabel(f'Predicted {transition_function}')
    #     plt.axis('square')
    #
    #     plt.axline(xy1=(0, 0), slope=1, color='black', zorder=0, label=("r^2 = %f" % r2))  # Add 1-1 line
    #     plt.text(0.87, 0.92, "$r^2$ = %.3f" % r2, fontsize=12,
    #              horizontalalignment='right',
    #              transform=plt.gca().transAxes)
    #
    #     # Calculate the combined range
    #     max_abs = max(abs(min(y)), abs(max(y)), abs(min(y_pred)), abs(max(y_pred)))
    #     min_val, max_val = -max_abs, max_abs
    #
    #     # Set the same limits for both axes
    #     plt.xlim(min_val - 0.1 * max_val, max_val + 0.1 * max_val)
    #     plt.ylim(min_val - 0.1 * max_val, max_val + 0.1 * max_val)
    #
    #     plt.gca().set_aspect('equal', adjustable='box')
    #
    #     plt.title(transition_function)
    #     #plt.savefig('C://Users/5605407/Documents/PhD/Chapter_2/Figures/Regression/DynaMETE_{data_set}_{f}.png'.format(data_set=data_set, f=transition_function))
    #     plt.show()
    #     plt.close()

    return p


def add_beta(df, data):

    if data == "birds":
        scaling_component = 10
    else:
        scaling_component = 1e11

    beta_cache = {}

    # Iterate over unique (S_t, N_t, E_t) combinations
    for (S, N, E) in df[['S_t', 'N_t', 'E_t']].drop_duplicates().itertuples(index=False):
        # Compute theoretical guess
        theoretical_guess = make_initial_guess([S, N, E], scaling_component=scaling_component)

        # Perform optimization to find l1 and l2
        l1, l2 = perform_optimization([theoretical_guess], [S, N, E], scaling_component=scaling_component)

        # Scale results
        l1, l2 = l1 / scaling_component, l2 / scaling_component
        beta = min(l1 + l2, 0.99)  # Ensure beta does not exceed 0.99

        # Store results in the cache
        beta_cache[(S, N, E)] = (l1, l2, beta)

    # Map computed values back to the original DataFrame
    df[['l1', 'l2', 'beta']] = df.apply(
        lambda row: beta_cache[(row['S_t'], row['N_t'], row['E_t'])], axis=1, result_type="expand"
    )

    return df


if __name__ == "__main__":

    for data_set in ['BCI']:

        df = load_data(data_set)

        # For birds:
        df = df.rename(columns={'m':'e', 'next_m': 'next_e'})

        # Add beta (sum of Lagrange Multipliers that are solution to standard METE)
        df = add_beta(df, data_set)

        initial_guess = [0.2, 0.2, 30000000, 500, 1.0, 0.4096, 0.0219, 1, 250000, 1]

        params = do_least_squares(initial_guess, df, data_set)

        # p = {
        #     'b': params[0], 'd': params[1], 'Ec': params[2], 'm': params[3],
        #     'w': params[4], 'w1': params[5], 'mu_meta': params[6],
        #     'sigma_1': params[7], 'K': params[8], 'sigma_2': params[9]
        # }
        p = params

        params_df = pd.DataFrame.from_dict(p, orient='index', columns=['Value'])
        params_df.to_csv('../data/dynaMETE_parameters_{}.csv'.format(data_set))

        # TODO: can the scaling coefficient be incorporated into the METE-framework itself? Is it truly necessary?
        # TODO: