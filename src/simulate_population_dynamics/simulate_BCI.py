import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from src.MaxEnt_inference import zero_point_METE
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def partition_function_given_n(X, n, lambdas):
    return (np.exp(-lambdas[0] * n) - np.exp(-lambdas[0] * n - X['E'] * lambdas[1] * n)) / (lambdas[1] * n)


def get_SAD(lambdas, X):
    """ Calculates species abundance distribution from Lagrange multipliers """
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
    """ Samples community that follows abundance and metabolic rate distributions as predicted by METE """
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
        while np.isnan(samples).any():
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

# def solve_metabolic(e, n, X, param, t_span, dt=0.1):
#     sol = odeint(g_wrapper, e, t_span, args=(n, X, param))
#     return float(sol[-1])
#
# def update_metabolic_rates(community, X, time_until_event, param):
#     t_span = np.array([0, time_until_event])
#     e_array = community['e'].values
#     n_array = community['n'].values
#
#     args_list = [(e_array[i], n_array[i], X, param, t_span)
#                  for i in range(len(e_array))]
#
#     with ThreadPoolExecutor() as executor:
#         new_e = list(executor.map(lambda args: solve_metabolic(*args), args_list))
#
#     community['e'] = new_e
#     X['E'] = np.sum(new_e)
#     return community, X


def update_metabolic_rates(community, X, dt, param):
    """ Euler method to update metabolic rates """
    e = community['e'].values
    de_dt = np.maximum(0, param['w'] * e ** (2 / 3) - param['w1'] * e)
    new_e = e + dt * de_dt
    community['e'] = new_e
    X['E'] = np.sum(new_e)
    return community, X


def update_event_rates(community, X, p):
    """ Compute birth and death rates """
    birth_rates = p['b'] * community['n'] * (community['e'] ** (-1/3))
    death_rates = p['d'] * (X['E'] / p['Ec']) * community['n'] * (community['e'] ** (-1/3))
    R = birth_rates.sum() + death_rates.sum()
    return birth_rates, death_rates, R


def what_event_happened(birth_rates, death_rates, R, q):
    event_rates = np.concatenate([birth_rates, death_rates])
    cumulative_rates = np.cumsum(event_rates)

    index = np.searchsorted(cumulative_rates, q * R)

    if index < len(birth_rates):
        return ('birth', index)
    else:
        return ('death', index - len(birth_rates))


def perform_event(community, X, event_info):
    event_type, idx = event_info
    S, N, E = X['S'], X['N'], X['E']

    if (event_type == 'birth'):
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

    X['S'] = community['Species_ID'].nunique() # shouldn't be necessary but something went wrong with X['S']
    X['E'], X['N'] = E, N

    return community, X


def plot_community_histograms(community, X, t):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram for 'e'
    axes[0].hist(community['e'], bins=20, color='skyblue', edgecolor='black')
    axes[0].set_title('Histogram of e')
    axes[0].set_xlabel('e')
    axes[0].set_ylabel('Frequency')

    # Histogram for 'n'
    axes[1].hist(community['n'], bins=20, color='salmon', edgecolor='black')
    axes[1].set_title('Histogram of n')
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('Frequency')

    # Main title with observation time
    fig.suptitle(f'Community Metabolic State at Time {t}', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def gillespie(metabolic_rates, species_indices, tree_id_list, X, p, t_max, obs_interval):
    # some preparations
    species_ids = np.zeros(int(species_indices[-1]), dtype=int)
    for i in range(1, len(species_indices)):
        start = species_indices[i - 1]
        end = species_indices[i]
        species_ids[start:end] = i

    observation_times = np.arange(0, t_max, obs_interval)
    obs_pointer = 0

    # create df of individual trees
    community = pd.DataFrame({
        'Tree_ID': tree_id_list,
        'Species_ID': species_ids,
        'e': metabolic_rates
    })
    community['n'] = community['Species_ID'].map(community['Species_ID'].value_counts())

    # Delete some things
    del(tree_id_list, species_ids, species_indices, metabolic_rates)

    # Compute state variables and event rates
    X = {'S': community['Species_ID'].nunique(), 'N': community['Tree_ID'].nunique(), 'E': community['e'].sum()}
    birth_rates, death_rates, R = update_event_rates(community, X, p)

    # Store snapshots here
    snapshots = []

    # Start simulation
    t = 0
    while t < t_max:
        # Sample event time
        u = np.random.uniform(0, 1)
        time_until_event = -np.log(u) / R
        t += time_until_event
        print(t)

        # In case the event happens *after* the current observation time, save a snapshot of the community
        while obs_pointer < len(observation_times) and t > observation_times[obs_pointer]:
            # Save snapshot
            snapshot = community[['Tree_ID', 'Species_ID', 'e', 'n']].copy()
            snapshot['S'] = X['S']
            snapshot['N'] = X['N']
            snapshot['E'] = X['E']
            snapshot['t'] = observation_times[obs_pointer]
            snapshots.append(snapshot)
            del(snapshot)

            # Plot histogram of n and e
            #plot_community_histograms(community, X, observation_times[obs_pointer])

            # Progress observation time
            obs_pointer += 1

        if obs_pointer >= len(observation_times):
            break

        # Update individual metabolic rates
        community, X = update_metabolic_rates(community, X, time_until_event, p)

        # Update event rates based on new metabolic rates e
        birth_rates, death_rates, R = update_event_rates(community, X, p)

        # Determine what event (birth or death) happened
        q = np.random.uniform(0, 1)
        event = what_event_happened(birth_rates, death_rates, R, q)
        community, X = perform_event(community, X, event)

    output_file = "C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/BCI/simulated_dynaMETE_snapshots.csv"
    pd.concat(snapshots, ignore_index=True).to_csv(output_file, index=False)

    # TODO: generate a unique treeID for when trees are born

    return community, X


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


def start_from_METE():
    # Original values (from Micahs dissertation)
    # param = {
    #     'b': 0.2, 'd': 0.2, 'Ec': 5000 * 10 ** 6, 'm': 437.3,
    #     'w': 1.0, 'w1': 0.42, 'mu_meta': 0.0215
    # }

    # Changed 'Ec': 2 * 10**7 to 4000 * 10**6
    # and w from 1.0 to 10.0 and w1 from 0.42 to 4.2

    # X = {                                                                                     # from Micahs dissertation
    #     'E': 2.04 * 10**7, 'N': 2.3 * 10**5, 'S': 320, 'beta': 0.0001
    # }

    # Based on BCI quadrat 0 with other parameters from DynaMETE, made up Ec
    param = {
        'b': 0.2, 'd': 0.2, 'Ec': 45000, 'm': 437.3,
        'w': 1.0, 'w1': 0.42, 'mu_meta': 0.0215
    }

    X = {
        'E': 46000,
        'N': 160,
        'S': 55,
        'beta': 0.0001
    }

    metabolic_rates, species_indices, tree_id_list = sample_community(X)
    X['S'], X['N'], X['E'] = len(species_indices) - 1, len(metabolic_rates), sum(metabolic_rates)

    # Generate trajectories of its state variables
    community, X = gillespie(metabolic_rates, species_indices, tree_id_list, X, param, t_max=5, obs_interval=0.25)

    # Load the CSV file
    file_path = "C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/BCI/simulated_dynaMETE_snapshots.csv"
    df = pd.read_csv(file_path)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left y-axis: N
    ax1.plot(df['t'], df['N'], color='tab:green', marker='s', label='N (Abundance)')
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

    """
    Metabolic growth is positive only for e < (p['w'] / p['w1'])^3 = 13.5
    """

if __name__ == '__main__':
    start_from_METE()