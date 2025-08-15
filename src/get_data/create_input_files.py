import pandas as pd
import numpy as np

def load_BCI():
    path = 'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Data sets/BCI/FullMeasurementBCI.tsv'
    df = pd.read_csv(path, sep='\t', low_memory=False)

    df = df[df['Status'] == "alive"]
    df = df.drop(
        ["Mnemonic", "Subspecies", "SubspeciesID", "StemTag", "HOM", "HighHOM", "ListOfTSM", "Date", "ExactDate",
         "Status", "QuadratName", "QuadratID", 'PX', 'PY'], axis=1)
    df = df.dropna()

    # Take average of 'DBH' for duplicates
    df = df.groupby([col for col in df.columns if col not in ['DBH', 'StemID']], as_index=False).agg({'DBH': 'mean'})

    # Calculate Metabolic Rate as in "A strong test for Maximum Entropy Theory of Ecology, Xiao, 2015"
    min_DBH = min(df['DBH'])
    df['Metabolic_Rate'] = (df['DBH'] / min_DBH) ** 2

    # Select columns
    df = df.copy()[['SpeciesID', 'TreeID', 'PlotCensusNumber', 'Metabolic_Rate']]

    # # Add zero abundances so that each unique species has at least one record for each year
    #all_species = df_copy['SpeciesID'].unique()
    #all_years = range(df_copy['PlotCensusNumber'].min(), df['PlotCensusNumber'].max() + 1)
    # species_years = pd.MultiIndex.from_product([all_species, all_years], names=['SpeciesID', 'PlotCensusNumber'])
    # df_copy = df_copy.set_index(['SpeciesID', 'PlotCensusNumber']).reindex(species_years).reset_index()
    # df_copy['n'] = df_copy['n'].fillna(0)

    df.rename(columns={'SpeciesID': 'species', 'PlotCensusNumber': 'census', 'Metabolic_Rate': 'e'}, inplace=True)

    # Add State Variables to df
    df['S_t'] = df.groupby(['census'])['species'].transform('nunique')
    df['S_t'] = np.ceil(df['S_t'])
    df['n'] = df.groupby(['species', 'census'])['TreeID'].transform('nunique')
    df['N_t'] = df.groupby(['census'])['TreeID'].transform('nunique')
    df['N_t'] = np.ceil(df['N_t'])
    df['E_t'] = df.groupby(['census'])['e'].transform('sum')

    #df = add_missing_rows(df)

    # # Add population sizes to df
    # df['n_t'] = df.groupby(['PlotCensusNumber', 'SpeciesID'])['TreeID'].transform('nunique')

    # Add the values of n at the next year
    df_next = df.copy()
    df_next['census'] = df_next['census'] - 1
    df_next.rename(columns={'n': 'next_n'}, inplace=True)
    df_next.rename(columns={'e': 'next_e'}, inplace=True)
    df_next.rename(columns={'S_t': 'next_S'}, inplace=True)
    df_next.rename(columns={'N_t': 'next_N'}, inplace=True)
    df_next.rename(columns={'E_t': 'next_E'}, inplace=True)
    df_next = df_next[['species', 'census', 'TreeID', 'next_n', 'next_e', 'next_S', 'next_N', 'next_E']]
    df = df.merge(df_next, how='left', on=['species', 'census', 'TreeID'])

    df['dn'] = df['next_n'] - df['n']
    df['de'] = df['next_e'] - df['e']
    df['dS'] = df['next_S'] - df['S_t']
    df['dN/S'] = (df['next_N'] - df['N_t'])/df['S_t']
    df['dE/S'] = (df['next_E'] - df['E_t'])/df['S_t']

    df = df.drop(columns=['next_n', 'next_S', 'next_e', 'next_N', 'next_E'], axis=1)

    df = df.dropna(how='any')
    df.to_csv('../../data/BCI_regression_library.csv', index=False)



def load_BCI_quadrat(index):
    path = 'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Data sets/BCI/FullMeasurementBCI.tsv'
    df = pd.read_csv(path, sep='\t', low_memory=False)

    quadrat = df['QuadratName'].unique()[index]
    df = df[df['QuadratName'] == quadrat]
    df = df[df['Status'] == "alive"]
    df = df.drop(
        ["Mnemonic", "Subspecies", "SubspeciesID", "StemTag", "HOM", "HighHOM", "ListOfTSM", "Date", "ExactDate",
         "Status", "QuadratName", "QuadratID", 'PX', 'PY'], axis=1)
    df = df.dropna()

    # Take average of 'DBH' for duplicates
    df = df.groupby([col for col in df.columns if col not in ['DBH', 'StemID']], as_index=False).agg({'DBH': 'mean'})

    # Calculate Metabolic Rate as in "A strong test for Maximum Entropy Theory of Ecology, Xiao, 2015"
    min_DBH = min(df['DBH'])
    df['Metabolic_Rate'] = (df['DBH'] / min_DBH) ** 2

    # Select columns
    df = df.copy()[['SpeciesID', 'TreeID', 'PlotCensusNumber', 'Metabolic_Rate']]

    df.rename(columns={'SpeciesID': 'species', 'PlotCensusNumber': 'census', 'Metabolic_Rate': 'e'}, inplace=True)

    # Add State Variables to df
    df['S_t'] = df.groupby(['census'])['species'].transform('nunique')
    df['S_t'] = np.ceil(df['S_t'])
    df['n'] = df.groupby(['species', 'census'])['TreeID'].transform('nunique')
    df['N_t'] = df.groupby(['census'])['TreeID'].transform('nunique')
    df['N_t'] = np.ceil(df['N_t'])
    df['E_t'] = df.groupby(['census'])['e'].transform('sum')

    # Add the values of n at the next year
    df_next = df.copy()
    df_next['census'] = df_next['census'] - 1
    df_next.rename(columns={'n': 'next_n'}, inplace=True)
    df_next.rename(columns={'e': 'next_e'}, inplace=True)
    df_next.rename(columns={'S_t': 'next_S'}, inplace=True)
    df_next.rename(columns={'N_t': 'next_N'}, inplace=True)
    df_next.rename(columns={'E_t': 'next_E'}, inplace=True)
    df_next = df_next[['species', 'census', 'TreeID', 'next_n', 'next_e', 'next_S', 'next_N', 'next_E']]
    df = df.merge(df_next, how='left', on=['species', 'census', 'TreeID'])

    df['dn'] = df['next_n'] - df['n']
    df['de'] = df['next_e'] - df['e']
    #df['dS'] = df['next_S'] - df['S_t']
    df['dN/S'] = (df['next_N'] - df['N_t'])/df['S_t']
    df['dE/S'] = (df['next_E'] - df['E_t'])/df['S_t']

    df = df.drop(columns=['next_n', 'next_S', 'next_e', 'next_N', 'next_E'], axis=1)

    df = df.dropna(how='any')
    df.to_csv(f'../../data/BCI_regression_library_quadrat_{index}.csv', index=False)


def add_missing_rows(df):
    # Create an empty list to store new rows
    new_rows = []

    # Loop over each unique combination of TreeID and census
    for (tree_id, census), group in df.groupby(['TreeID', 'census']):
        # Check for missing rows in the range of census numbers
        existing_census = set(group['census'])
        max_census = max(existing_census)

        # Check if census records are missing and add them
        for c in range(1, max_census + 1):
            if c not in existing_census:
                # For this missing census, find the next available census record'diag 'e'
                next_census_record = group[group['census'] > c].iloc[0]

                # Get the 'n' value from the next available census record (where census == c + 1)
                next_n = df[(df['species'] == next_census_record['species']) & (df['census'] == c + 1)]['n'].values[0]

                # Create the new row
                new_row = {
                    'species': next_census_record['species'],
                    'TreeID': tree_id,
                    'census': c,
                    'e': next_census_record['e'],  # Get the 'e' value from next available census record
                    'N_t': next_census_record['N_t'],
                    'S_t': next_census_record['S_t'],
                    'E_t': next_census_record['E_t'],
                    'dn': next_n,  # Use 'n' value from next available census record
                    'de': 0  # Set de to 0
                }

                new_rows.append(new_row)

    # Convert the new rows into a DataFrame
    new_rows_df = pd.DataFrame(new_rows)

    # Append the new rows to the original DataFrame
    df = pd.concat([df, new_rows_df], ignore_index=True)

    return df



if __name__ == '__main__':
    #load_BCI()
    for i in range(10, 12):
        load_BCI_quadrat(i) # index can be 0, 1, ..., 1251

        # There seems to be a problem with quadrat 9, so we exclude it
        # We have ran it for all other quadrats from 0 to 11


