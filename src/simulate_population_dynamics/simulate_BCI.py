import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from src.MaxEnt_inference import zero_point_METE
#from mpmath import log, exp
import seaborn as sns
from scipy.integrate import odeint
import sys
import os
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def g(n, e, X, p):
    return p['w'] * e**(2/3) - p['w1'] * e


def g_wrapper(e, t, n, X, p):
    return g(n, e, X, p)


def partition_function_given_n(X, n, lambdas):
    return (np.exp(-lambdas[0] * n) - np.exp(-lambdas[0] * n - X['E'] * lambdas[1] * n)) / (lambdas[1] * n)


def get_SAD(lambdas, X):
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


def cum_SAD(lambdas, X):
    sad = get_SAD(lambdas, X)
    cum_sad = []

    p = 0
    for n in range(int(X['N'])):
        p += sad[n]
        cum_sad.append(p)

    return cum_sad


def sample_community(X):
    # Add beta value
    beta_constraint = lambda b, X: b * np.log(1 / (1 - np.exp(-b))) - X['S'] / X['N']
    beta = fsolve(beta_constraint, 0.0001, args=X)[0]
    X['beta'] = beta

    # Find Lagrange multipliers
    lambdas = zero_point_METE.make_initial_guess(X)
    lambdas, _ = zero_point_METE.find_roots(lambdas,
                                            state_variables=X,
                                            method="METE")

    # Sample species
    p = np.random.uniform(0, 1, X['S'])
    cum_sad = cum_SAD(lambdas, X)

    populations = []
    for prob in p:
        n = int(np.searchsorted(cum_sad, prob) + 1)
        populations.append(n)
    species_indices = np.cumsum(populations)
    species_indices = np.concatenate(([0], species_indices))


    # Sample individual metabolic rates
    individuals = []
    for pop in populations:
        Z_n = partition_function_given_n(X, pop, lambdas)
        CDF_inverse = lambda u: -(np.log(1 - Z_n * lambdas[1] * n * u * np.exp(lambdas[0] * n)))/(lambdas[1] * n)
        u_samples = np.random.uniform(0, 1, pop)
        samples = [float(CDF_inverse(u).real) for u in u_samples] # checked: there is no imaginary part
        if min(samples) < 0:
            u_samples = np.random.uniform(0, min(1, 1/(Z_n * lambdas[1] * n * np.exp(lambdas[0] * n))), pop)
            samples = [float(CDF_inverse(u).real) for u in u_samples]
        individuals += samples
    tree_id_list = list(range(len(individuals)))

    return individuals, species_indices, tree_id_list


# def find_n(lst, index):
#     pos = bisect.bisect_right(lst, index)
#     larger = lst[pos]
#     smaller = lst[pos - 1]
#
#     return int(larger - smaller)
#
#
# def find_species_id(lst, index):
#     pos = bisect.bisect_right(lst, index)
#     return pos - 1


# def compute_rates(metabolic_rates, species_indices, X, p):
#     event_rates = np.concatenate([
#         np.array([p['b'] * find_n(species_indices, i) * (e ** (-1 / 3)) for i, e in enumerate(metabolic_rates)]),
#         np.array([p['d'] * X['E'] / p['Ec'] * find_n(species_indices, i) * (e ** (-1 / 3)) for i, e in enumerate(metabolic_rates)]),
#         np.array([p['m'] * (species_indices[i] - species_indices[i -1]) / X['N'] for i in range(1, len(species_indices))])              # TODO: add possibility to create new species
#     ])
#     R = np.sum(event_rates)
#     return event_rates, R


# def loop_in_gillespie(q, metabolic_rates, species_indices, tree_id_list, tree_id, param, X):
#     event_rates, R = compute_rates(metabolic_rates, species_indices, X, param)
#
#     # What event happened?
#     cumulative_event_rates = np.cumsum(event_rates)
#     event_index = bisect.bisect_right(cumulative_event_rates, q * R) - 1
#
#     # Birth event
#     if event_index < X['N']:
#         print("Birth")
#         metabolic_rates = np.insert(metabolic_rates, event_index, 1)                                                    # TODO: baby has e = 1 (or metabolic_rates[event_index])
#         species_indices = [i + 1 if i > event_index else i for i in species_indices]
#         new_tree_id = tree_id
#         tree_id += 1
#         tree_id_list = np.insert(tree_id_list, event_index, new_tree_id)
#         X['N'] += 1
#         X['E'] += metabolic_rates[event_index]
#
#     # Death event
#     elif event_index < 2 * X['N']:
#         print("Death")
#         X['E'] -= metabolic_rates[event_index - int(X['N'])]
#         metabolic_rates = np.delete(metabolic_rates, event_index - int(X['N']))
#         tree_id_list = np.delete(tree_id_list, event_index - int(X['N']))
#         species_indices = [i - 1 if i > event_index - int(X['N']) else i for i in species_indices]
#         X['N'] -= 1
#
#     # Migration event
#     else:
#         print("Migration")
#         species_index = event_index - int(2 * X['N'])
#         metabolic_rates = np.insert(1, species_indices[species_index], metabolic_rates[event_index])                # germinants have e = 1
#         species_indices = [i + 1 if i > species_index else i for i in species_indices]
#         new_tree_id = tree_id
#         tree_id += 1
#         tree_id_list = np.insert(tree_id_list, event_index - int(2 * X['N']), new_tree_id)
#         X['N'] += 1
#         X['E'] += 1
#
#     X['S'] = int(np.sum(np.diff(species_indices) != 0))
#
#     return metabolic_rates, species_indices, tree_id_list, tree_id, X


def update_metabolic_rates(community, X, time_until_event, param):
    # Solve ODE for each individual's metabolic rate
    new_e = []
    for idx, row in community.iterrows():
        e = row['e']
        n = row['n']
        t_span = np.array([0, time_until_event])
        sol = odeint(g_wrapper, e, t_span, args=(
        n, X, param))
        new_e.append(float(sol[-1]))
    community['e'] = new_e
    return community


def update_event_rates(community, X, p):
    birth_rates = p['b'] * community['n'] * (community['e'] ** (-1/3))
    death_rates = p['d'] * (X['E'] / p['Ec']) * community['n'] * (community['e'] ** (-1/3))
    migration_rates = np.full(len(community), p['m'] / X['N'])
    R = birth_rates.sum() + death_rates.sum() + migration_rates.sum()
    return birth_rates, death_rates, migration_rates, R


def what_event_happened(birth_rates, death_rates, migration_rates, R, q):
    event_rates = np.concatenate([birth_rates, death_rates, migration_rates])
    cumulative_rates = np.cumsum(event_rates)
    index = np.searchsorted(cumulative_rates, q * R)

    if index < len(birth_rates):
        return ('birth', index)
    elif index < len(birth_rates) + len(death_rates):
        return ('death', index - len(birth_rates))
    else:
        return ('migration', index - len(birth_rates) - len(death_rates))


def perform_event(community, X, event_info):
    print(event_info)
    event_type, idx = event_info
    S, N, E = X['S'], X['N'], X['E']

    if (event_type == 'birth') or (event_type == 'migration'):
        # Create a new individual
        new_tree_id = community['Tree_ID'].max() + 1
        species_id = community.iloc[idx]['Species_ID']
        n = community.iloc[idx]['n']

        # Baby has e = 1
        new_individual = pd.DataFrame({
            'Tree_ID': [new_tree_id],
            'Species_ID': [species_id],
            'e': [1],
            'n': [n]
        })
        community = pd.concat([community, new_individual], ignore_index=True)

        # Update n for all individuals of the species
        community.loc[community['Species_ID'] == species_id, 'n'] += 1
        N += 1
        E += 1

    else: # event_type == 'death':
        species_id = community.iloc[idx]['Species_ID']
        e_value = community.iloc[idx]['e']

        # Remove individual
        community = community.drop(index=idx).reset_index(drop=True)

        # Update n for remaining individuals of same species
        community.loc[community['Species_ID'] == species_id, 'n'] -= 1
        N -= 1
        E -= e_value

        # If no individuals left with that species_id, species extinct
        if not (community['Species_ID'] == species_id).any():
            S -= 1

    # elif event_type == 'migration':
    #     # Create a new species (new Species_ID)
    #     new_tree_id = community['Tree_ID'].max() + 1
    #     n = community.iloc[idx]['n']
    #
    #     new_individual = pd.DataFrame({
    #         'Tree_ID': [new_tree_id],
    #         'Species_ID': [new_species_id],
    #         'e': [1],
    #         'n': [1]
    #     })
    #     community = pd.concat([community, new_individual], ignore_index=True)
    #
    #     S += 1
    #     N += 1
    #     E += 1  # assume e = 1

    return community, S, N, E


def gillespie(metabolic_rates, species_indices, tree_id_list, X, p, t_max=1e-05, obs_interval=1e-07):
    # Species_ID
    species_ids = np.zeros(species_indices[-1], dtype=int)
    for i in range(1, len(species_indices)):
        start = species_indices[i - 1]
        end = species_indices[i]
        species_ids[start:end] = i

    # Prepare saving results
    output_file = "C:/Users/5605407/Documents/PhD/Chapter_2/Results/BCI/simulated_dynaMETE_snapshots.csv"
    with open(output_file, 'w') as f:
        f.write(','.join(['Tree_ID', 'Species_ID', 'e', 'n', 'S', 'N', 'E', 't']) + '\n')

    # Get observation times
    observation_times = np.arange(0, t_max, obs_interval)
    obs_pointer = 0

    # Get df of individuals
    community = pd.DataFrame({
        'Tree_ID': tree_id_list,
        'Species_ID': species_ids,
        'e': metabolic_rates
    })
    community['n'] = community['Species_ID'].map(community['Species_ID'].value_counts())

    # Delete some things
    del(tree_id_list, species_ids, species_indices, metabolic_rates)

    S = community['Species_ID'].nunique()
    N = community['Tree_ID'].nunique()
    E = community['e'].sum()

    # Compute Birth, Death, Migration rates
    birth_rates = p['b'] * community['n'] * community['e'] ** (-1 / 3)
    death_rates = p['d'] * E / p['Ec'] * community['n'] * community['e'] ** (-1 / 3)
    migration_rates = np.full(len(community), p['m'] / N)
    R = sum(birth_rates) + sum(death_rates) + sum(migration_rates)

    # Start simulation
    t = 0
    while t < t_max:
        # Sample event time
        u = np.random.uniform(0, 1)
        time_until_event = -np.log(u) / R
        t += time_until_event

        # In case the event happens *after* the current observation time
        while obs_pointer < len(observation_times) and t > observation_times[obs_pointer]:
            # Save snapshot
            snapshot = community[['Tree_ID', 'Species_ID', 'e', 'n']].copy()
            snapshot['S'] = community['Species_ID'].nunique()
            snapshot['N'] = community['Tree_ID'].nunique()
            snapshot['E'] = community['e'].sum()
            snapshot['t'] = observation_times[obs_pointer]
            snapshot.to_csv(output_file, mode='a', header=False, index=False)

            # Progress observation time
            #print('new observation time')
            obs_pointer += 1

        if obs_pointer >= len(observation_times):
            break

        # Update individual metabolic rates
        community = update_metabolic_rates(community, {'S': S, 'N': N, 'E': E}, time_until_event, p)
        birth_rates, death_rates, migration_rates, R = update_event_rates(community, {'S': S, 'N': N, 'E': E}, p)

        # Sample event
        q = np.random.uniform(0, 1)
        event = what_event_happened(birth_rates, death_rates, migration_rates, R, q)
        community, S, N, E = perform_event(community, {'S': S, 'N': N, 'E': E}, event)

        # TODO: now we don't use a unique treeID for when trees are born


# def do_polynomial_regression(X, y, census, species, transition_function):
#     # TODO: this should be a summation over the individuals of a species,
#     #  so a summation of some of the rows in X
#
#     #model = LinearRegression()
#     model = ElasticNet()
#
#     # Remove outliers from the data set
#     threshold = 3
#     z_scores = np.abs(stats.zscore(X))
#     outliers = (z_scores > threshold)
#     outlier_indices = np.where(outliers.any(axis=1))[0]
#     #X, y = X.drop(outlier_indices, axis=0), y.drop(outlier_indices, axis=0)
#     keep_indices = np.setdiff1d(np.arange(len(X)), outlier_indices)
#     X = X.iloc[keep_indices]
#     y = y.iloc[keep_indices]
#     species = species.iloc[keep_indices]
#     print(f"Number of outliers removed: {len(outlier_indices)}")
#
#     # Error on training set
#     model.fit(X, y)
#     y_pred = model.predict(X)
#
#     return y, y_pred, species


# def set_up_regression(df):
#     # Single observation interval
#     all_census = sorted(df['census'].unique())
#     reduced_census = deepcopy(all_census)[::150]
#
#     # Filter to current censuses
#     df_filtered = df[df['census'].isin(reduced_census)].copy().reset_index(drop=True)
#
#     # Recompute dn, dN, dS
#     df_deltas = compute_deltas(df_filtered, 'LV').reset_index(drop=True)
#
#     # METimE regression
#     cols_to_exclude = ['dN', 'n_next', 'N_next', 'S_next']
#     df_for_setup = df_deltas.drop(columns=cols_to_exclude)
#     X, y, census, species = polynomial_regression.set_up_library(df_for_setup, 3, False, False, False)
#     #transition_functions = polynomial_regression.METimE(X, y, census, fig_title=f"")
#
#     return X, y, census, species


def plot_observed_vs_predicted(obs, pred, title, species=None):
    """
    Plots observed vs. predicted values, optionally colored by species.

    Parameters:
    - obs: list or array of observed values
    - pred: list or array of predicted values
    - title: str, the title of the plot
    - species: list or array of species IDs (same length as obs/pred), optional
    """
    plt.figure()

    if species is not None:
        species = np.array(species)
        unique_species = np.unique(species)
        cmap = plt.get_cmap("tab20", len(unique_species))
        for idx, sp in enumerate(unique_species):
            mask = species == sp
            plt.scatter(np.array(obs)[mask], np.array(pred)[mask],
                        label=f"Species {sp}", color=cmap(idx), alpha=0.6, edgecolors='k')
        #plt.legend(loc='best', bbox_to_anchor=(1.05, 1.0), title='Species', fontsize='small')
    else:
        plt.scatter(obs, pred, alpha=0.6, edgecolors='k')

    plt.plot([min(obs), max(obs)], [min(obs), max(obs)], 'r--', linewidth=1)  # 1:1 line
    plt.title(title)
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def k_means_clustering(df, ncluster):
    # Step 1: Load your DataFrame (assuming it's already loaded as `df`)
    cols_to_summarize = ['dn']

    # Step 2: Compute summary stats per species
    summary = df.groupby('species')[cols_to_summarize].agg(['mean', 'std', 'median']).reset_index()

    # Flatten MultiIndex columns
    summary.columns = ['species'] + [f"{col}_{stat}" for col in cols_to_summarize for stat in ['mean', 'std', 'median']]
    summary.fillna(0, inplace=True)

    # Step 3: Standardize the features
    X = summary.drop(columns='species')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 4: Run K-Means Clustering (choose 2 or 3 clusters)
    kmeans = KMeans(n_clusters=ncluster, random_state=42)
    summary['cluster'] = kmeans.fit_predict(X_scaled)

    # # Step 5: Visualize clusters (using first two PCA components or just two features)
    # sns.scatterplot(data=summary, x='n_mean', y='n_std', hue='cluster', palette='viridis')
    # plt.title('Species Clustering Based on Summary Stats')
    # plt.xlabel('Mean n')
    # plt.ylabel('Std Dev n')
    # plt.show()

    sns.scatterplot(data=summary, x='dn_mean', y='dn_std', hue='cluster', palette='viridis')
    plt.title('Species Clustering Based on Summary Stats')
    plt.xlabel('Mean dn')
    plt.ylabel('Std Dev dn')
    plt.show()

    # Step 6: Merge cluster labels back to the original DataFrame
    df_with_clusters = df.merge(summary[['species', 'cluster']], on='species', how='left')

    # Optional: Save result
    # df_with_clusters.to_csv("df_with_species_clusters.csv", index=False)
    return df_with_clusters



# def do_regression(df):
#     all_census = sorted(df['census'].unique())
#     reduced_census = deepcopy(all_census)
#     iteration = 1
#
#     while len(reduced_census) >= 2:
#         print(f"\n--- Iteration {iteration} ---")
#         print(f"Censuses used: {len(reduced_census)}")
#
#         # Filter to current censuses
#         df_filtered = df[df['census'].isin(reduced_census)].copy().reset_index(drop=True)
#
#         # Recompute dn, dN, dS
#         df_deltas = compute_deltas(df_filtered, 'BCI').reset_index(drop=True)
#
#         # METimE regression
#         cols_to_exclude = ['species', 'dN', 'dE', 'n_next', 'e_next', 'N_next', 'E_next', 'S_next']
#         df_for_setup = df_deltas.drop(columns=cols_to_exclude)
#         X, y, census = bilinear_regression.set_up_library(df_for_setup, 3, False, False, False)
#         _ = bilinear_regression.METimE(X, y, census, fig_title=f"simulated_METimE_interval_{iteration}")
#
#         # Update census list by removing every second census
#         reduced_census = reduced_census[::2]
#         iteration += 1
#
#     pass


# def fit_dynaMETE(df):
#     all_census = sorted(df['census'].unique())
#     reduced_census = deepcopy(all_census)
#     iteration = 1
#
#     while len(reduced_census) >= 2:
#         print(f"\n--- Iteration {iteration} ---")
#         print(f"Censuses used: {len(reduced_census)}")
#
#         # Filter to current censuses
#         df_filtered = df[df['census'].isin(reduced_census)].copy().reset_index(drop=True)
#
#         # Recompute dn, dN, dS
#         df_deltas = compute_deltas(df_filtered, 'LV').reset_index(drop=True)
#
#         # METimE regression
#         cols_to_exclude = ['species', 'dN', 'dE', 'n_next', 'e_next', 'N_next', 'E_next', 'S_next']
#         df = df_deltas.drop(columns=cols_to_exclude)
#
#         df = add_beta(df, 'BCI')
#
#         initial_guess = [0.2, 0.2, 30000000, 500, 1.0, 0.4096, 0.0219, 1, 250000, 1]
#
#         _ = do_least_squares(initial_guess, df, 'BCI')
#
#         # Update census list by removing every second census
#         reduced_census = reduced_census[::2]
#         iteration += 1
#
#     pass



if __name__ == '__main__':
    param = {                                                                                 # from Micahs dissertation
        'b': 0.2, 'd': 0.2, 'Ec': 2 * 10**7, 'm': 437.3,
        'w': 1.0, 'w1': 0.42, 'mu_meta': 0.0215
    }

    X = {                                                                                     # from Micahs dissertation
        'E': 2.04 * 10**7, 'N': 2.3 * 10**5, 'S': 320, 'beta': 0.0001
    }

    # Sample a community
    metabolic_rates, species_indices, tree_id_list = sample_community(X)
    X['S'], X['N'], X['E'] = len(species_indices) - 1, len(metabolic_rates), sum(metabolic_rates)

    # Generate trajectories of its state variables
    gillespie(metabolic_rates, species_indices, tree_id_list, X, param, t_max=1e-05, obs_interval=1e-06)

    # Load the CSV file
    file_path = "C:/Users/5605407/Documents/PhD/Chapter_2/Results/BCI/simulated_dynaMETE_snapshots.csv"
    df = pd.read_csv(file_path)

    # Check the column names
    print(df.columns)

    # Plot trajectories of S, N, E
    plt.figure(figsize=(10, 6))

    plt.plot(df['time'], df['S'], label='S (Species richness)', marker='o')
    plt.plot(df['time'], df['N'], label='N (Abundance)', marker='s')
    plt.plot(df['time'], df['E'], label='E (Energy)', marker='^')

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Trajectories of S, N, and E')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()