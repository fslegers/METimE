import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from scipy.optimize import fsolve
from src.MaxEnt_inference import zero_point_METE
from scipy.integrate import odeint
import sys
import os
from matplotlib.animation import FuncAnimation

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

    empirical_sad = get_SAD(lambdas, X)

    return empirical_sad


def update_metabolic_rates(community, X, time_until_event, param):
    # Solve ODE for each individual's metabolic rate
    new_e = []
    for idx, row in community.iterrows():
        e = row['e']
        n = row['n']
        t_span = np.array([0, time_until_event])
        sol = odeint(g_wrapper, e, t_span, args=(
        n, X, param))

        if np.isnan(float(sol[-1])):
            print("SOMETHING WENT WRONG")
            new_e.append(e)

        new_e.append(float(sol[-1]))
    community['e'] = new_e

    X['E'] = community['e'].sum()

    return community, X


def update_event_rates(community, X, p):
    birth_rates = p['b'] * community['n'] * (community['e'] ** (-1/3))
    death_rates = p['d'] * (X['E'] / p['Ec']) * community['n'] * (community['e'] ** (-1/3))
    migration_rate = p['m'] * 5000 / X['N'] # TODO: does this make sense?
    R = birth_rates.sum() + death_rates.sum() + migration_rate
    return birth_rates, death_rates, migration_rate, R


def what_event_happened(birth_rates, death_rates, migration_rate, R, q):
    event_rates = np.concatenate([birth_rates, death_rates, [migration_rate]])
    cumulative_rates = np.cumsum(event_rates)

    if np.isnan(cumulative_rates).any():
        print("NaN detected in cumulative_rates")

    index = np.searchsorted(cumulative_rates, q * R)

    if index < len(birth_rates):
        return ('birth', index)
    elif index < len(birth_rates) + len(death_rates):
        return ('death', index - len(birth_rates))
    else:
        return ('migration', index - len(birth_rates) - len(death_rates))


def perform_event(community, X, event_info, meta_sad):
    #print(event_info)
    event_type, idx = event_info
    S, N, E = X['S'], X['N'], X['E']

    if event_type == 'migration':
        # Sample species from meta-community
        species_id = np.random.choice(len(meta_sad), p=meta_sad)

        # Create a new individual
        new_tree_id = community['Tree_ID'].max() + 1
        if np.isnan(new_tree_id):
            new_tree_id = 1

        n = community[community['Species_ID'] == species_id].shape[0]

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

    elif event_type == 'birth':
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
        #print(f"Removed e: {e_value}")

        # Remove individual
        community = community.drop(index=idx).reset_index(drop=True)

        # Update n for remaining individuals of same species
        community.loc[community['Species_ID'] == species_id, 'n'] -= 1
        N -= 1
        E -= e_value

        # If no individuals left with that species_id, species extinct
        if not (community['Species_ID'] == species_id).any():
            S -= 1

    S = community['Species_ID'].nunique()       # TODO: Shouldn't be necessary but S is not updated correctly otherwise
    X['S'], X['N'], X['E'] = S, N, E
    return community, X


# Without live plotting:
# def gillespie(p, meta_sad, t_max=1e-04, obs_interval=1e-06):
#
#     # Start with an empty community
#     community = pd.DataFrame({
#         'Tree_ID': [],
#         'Species_ID': [],
#         'e': []
#     })
#
#     # Then, perform one migration
#     community, S, N, E = perform_event(community, {'S': 0, 'N': 0, 'E': 0}, ('migration', -1), meta_sad)
#
#     # # Species_ID
#     # species_ids = np.zeros(species_indices[-1], dtype=int)
#     # for i in range(1, len(species_indices)):
#     #     start = species_indices[i - 1]
#     #     end = species_indices[i]
#     #     species_ids[start:end] = i
#
#     # Prepare saving results
#     output_file = "C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/BCI/simulated_dynaMETE_snapshots_empty.csv"
#     with open(output_file, 'w') as f:
#         f.write(','.join(['Tree_ID', 'Species_ID', 'e', 'n', 'S', 'N', 'E', 't']) + '\n')
#
#     # Get observation times
#     observation_times = np.arange(0, t_max, obs_interval)
#     obs_pointer = 0
#
#     S = community['Species_ID'].nunique()
#     N = community['Tree_ID'].nunique()
#     E = community['e'].sum()
#
#     # Compute Birth, Death, Migration rates
#     birth_rates = p['b'] * community['n'] * community['e'] ** (-1 / 3)
#     death_rates = p['d'] * E / p['Ec'] * community['n'] * community['e'] ** (-1 / 3)
#     migration_rate = p['m'] * 5000 / X['N']
#     R = birth_rates.sum() + death_rates.sum() + migration_rate
#
#     # Start simulation
#     t = 0
#     while t < t_max:
#         # Sample event time
#         u = np.random.uniform(0, 1)
#         time_until_event = -np.log(u) / R
#         t += time_until_event
#         print(t)
#
#         # In case the event happens *after* the current observation time
#         while obs_pointer < len(observation_times) and t > observation_times[obs_pointer]:
#             # Save snapshot
#             snapshot = community[['Tree_ID', 'Species_ID', 'e', 'n']].copy()
#             snapshot['S'] = community['Species_ID'].nunique()
#             snapshot['N'] = community['Tree_ID'].nunique()
#             snapshot['E'] = community['e'].sum()
#             snapshot['t'] = observation_times[obs_pointer]
#             snapshot.to_csv(output_file, mode='a', header=False, index=False)
#
#             # Progress observation time
#             obs_pointer += 1
#
#         if obs_pointer >= len(observation_times):
#             break
#
#         # Update individual metabolic rates
#         community = update_metabolic_rates(community, {'S': S, 'N': N, 'E': E}, time_until_event, p)
#         birth_rates, death_rates, migration_rate, R = update_event_rates(community, {'S': S, 'N': N, 'E': E}, p)
#
#         # Sample event
#         q = np.random.uniform(0, 1)
#         event = what_event_happened(birth_rates, death_rates, migration_rate, R, q)
#         community, S, N, E = perform_event(community, {'S': S, 'N': N, 'E': E}, event, meta_sad)

def gillespie(p, meta_sad, t_max=0.14):
    community = pd.DataFrame({'Tree_ID': [], 'Species_ID': [], 'e': []})
    community, X = perform_event(community, {'S': 0, 'N': 0, 'E': 0}, ('migration', -1), meta_sad)

    # Setup plot
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    ax_main = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])
    ax2 = ax_main.twinx()

    ax_main.set_xlim(0, t_max)
    ax_main.set_ylim(0, 200)
    ax2.set_ylim(0, 60000)
    ax_main.set_xlabel('Time')
    ax_main.set_ylabel('N / S')
    ax2.set_ylabel('E')

    line1, = ax_main.plot([], [], 'g-', label='N')
    line2, = ax_main.plot([], [], 'b-', label='S')
    line3, = ax2.plot([], [], 'r-', label='E')
    ax_main.legend(loc='upper left')

    times, Ns, Ss, Es = [], [], [], []

    t = 0
    while t < t_max:
        birth_rates, death_rates, migration_rate, R = update_event_rates(community, X, p)
        u = np.random.uniform(0, 1)
        time_until_event = -np.log(u) / R
        dt = time_until_event
        t += dt

        # Update metabolic rates
        community, X = update_metabolic_rates(community, X, time_until_event, p)
        birth_rates, death_rates, migration_rate, R = update_event_rates(community, X, p)

        # Determine and perform event
        q = np.random.uniform(0, 1)
        event = what_event_happened(birth_rates, death_rates, migration_rate, R, q)
        community, X = perform_event(community, X, event, meta_sad)
        print(event)

        # Save stats
        times.append(t)
        Ns.append(X['N'])
        Ss.append(X['S'])
        Es.append(X['E'])

        # Update plots
        line1.set_data(times, Ns)
        line2.set_data(times, Ss)
        line3.set_data(times, Es)

        ax_main.set_xlim(0, t + 0.001)  # Extend X-axis dynamically
        ax_hist.clear()
        ax_hist.hist(community['e'], bins=30, color='gray', edgecolor='black')
        ax_hist.set_ylabel('Count')
        ax_hist.set_xlabel('Metabolic rate (e)')

        plt.pause(0.001)

    plt.show()


if __name__ == '__main__':
    param = {
        'b': 0.2, 'd': 0.2, 'Ec': 40000, 'm': 437.3,
        'w': 1, 'w1': 0.42, 'mu_meta': 0.0215
    }

    # Smaller community than BCI forest
    X = {
        'E': 46000,
        'N': 160,
        'S': 55,
        'beta': 0.0001
    }

    # Sample a meta-community
    probabilities = sample_community(X)
    species_abundances = np.random.choice(np.arange(1, len(probabilities) + 1), size=45, p=probabilities)
    meta_sad = species_abundances / np.sum(species_abundances)

    # Simulate birth-death process with migration from the meta-community
    gillespie(param, meta_sad, t_max=0.14)

    # Load the CSV file
    file_path = "C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/BCI/simulated_dynaMETE_snapshots_empty.csv"
    df = pd.read_csv(file_path)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left y-axis: N
    ax1.plot(df['t'], df['N'], color='tab:green', marker='s', label='N (Abundance)')
    ax1.plot(df['t'], df['S'], color='tab:blue', marker='o', label='S (Species)')
    ax1.set_ylabel('N (Abundance)', color='tab:green')
    ax1.tick_params(axis='y', labelcolor='tab:green')

    # Right y-axis: E
    ax2 = ax1.twinx()
    ax2.plot(df['t'], df['E'], color='tab:red', marker='^', label='E (Energy)')
    ax2.set_ylabel('E (Energy)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # X-axis and common formatting
    ax1.set_xlabel('Time')
    plt.title('Trajectories of N and E')
    ax1.grid(True)
    fig.tight_layout()
    plt.show()



