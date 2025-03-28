import pandas as pd
import numpy as np

def load_fish():

    # Load fish population data
    df_abundance = pd.read_csv('../../data/fish_abundance.csv', index_col=0)

    # Add State Variables to df
    df_mete = pd.DataFrame()
    df_mete['S_t'] = np.ceil(df_abundance[df_abundance['ABUNDANCE'] > 0]
                             .groupby(['YEAR'])['GENUS_SPECIES']
                             .nunique()
                             ).reset_index(drop=True)
    df_mete['next_S'] = df_mete['S_t'].shift(-1)
    df_mete['N_t'] = np.ceil(df_abundance.groupby(['YEAR'])['ABUNDANCE'].sum()).reset_index(drop=True)
    df_mete['next_N'] = df_mete['N_t'].shift(-1)
    df_mete['YEAR'] = df_abundance.groupby(['YEAR']).sum().index
    df_mete.rename(columns={'YEAR': 'census'}, inplace=True)

    # Sum abundance over stations
    df_dyna = df_abundance.groupby(['YEAR', 'GENUS_SPECIES'])['ABUNDANCE'].sum().reset_index()
    df_dyna = df_dyna.sort_values(by=['GENUS_SPECIES', 'YEAR'])

    # Add zero abundances so that each unique species has at least one record for each year
    all_species = df_dyna['GENUS_SPECIES'].unique()
    all_years = range(df_dyna['YEAR'].min(), df_dyna['YEAR'].max() + 1)
    species_years = pd.MultiIndex.from_product([all_species, all_years], names=['GENUS_SPECIES', 'YEAR'])
    df_dyna = df_dyna.set_index(['GENUS_SPECIES', 'YEAR']).reindex(species_years).reset_index()
    df_dyna['ABUNDANCE'] = df_dyna['ABUNDANCE'].fillna(0)

    # Create the 'next_n' column by shifting the 'next_n' values within each species group
    df_dyna['next_n'] = df_dyna.groupby('GENUS_SPECIES')['ABUNDANCE'].shift(-1)
    df_dyna.columns = df_dyna.columns.str.strip().str.upper()
    df_dyna.rename(columns={'ABUNDANCE': 'n', 'YEAR': 'census', 'GENUS_SPECIES': 'species', 'NEXT_N':'next_n'}, inplace=True)
    df_dyna = df_dyna.merge(df_mete, on='census', how='left') # add state variables

    df_dyna['dn'] = df_dyna['next_n'] - df_dyna['n']
    df_dyna['dS'] = df_dyna['next_S'] - df_dyna['S_t']
    df_dyna['dN'] = df_dyna['next_N'] - df_dyna['N_t']
    df_dyna = df_dyna.drop(columns=['next_n', 'next_S', 'next_N'], axis=1)

    df_dyna = df_dyna.dropna(how='any')
    df_dyna.to_csv('../../data/fish_regression_library.csv', index=False)

    df_METimE = pd.DataFrame()
    df_METimE['S'] = df_mete['S_t']
    df_METimE['N'] = df_mete['N_t']
    df_METimE['census'] = df_mete['census']
    df_METimE['N/S'] = df_mete['N_t']/df_mete['S_t']
    df_METimE['dN/S'] = (df_mete['next_N'] - df_mete['N_t']) / df_mete['S_t']
    df_METimE['dS'] = df_mete['next_S'] - df_mete['S_t']

    # Save empirical SADs
    df_SAD_dict = {}
    grouped = df_abundance.groupby(["YEAR"], sort=True)

    for year, group in grouped:
        sad_year = []

        # Count species abundances
        abundance_counts = group.groupby("GENUS_SPECIES")["ABUNDANCE"].sum()

        # Determine max abundance (S)
        S = group["GENUS_SPECIES"].nunique()
        N = int(group["ABUNDANCE"].sum())

        # Compute species count for each n
        species_count = abundance_counts.value_counts().to_dict()

        # Compute relative abundance for each n
        for n in range(1, N - S + 1):
            relative_abundance = species_count.get(n, 0)
            sad_year.append(relative_abundance)

        df_SAD_dict[year] = sad_year

    df_SAD = pd.DataFrame({
        "census": [year if isinstance(year, int) else year[0] for year in df_SAD_dict.keys()],  # Extract integer years
        "SAD": list(df_SAD_dict.values())  # Ensure SAD lists are properly stored
    })

    df_rank_dict = {}
    for year, group in grouped:
        # Count species abundances
        abundance_counts = group.groupby("GENUS_SPECIES")["ABUNDANCE"].sum()

        # Sort species by abundance (high to low)
        sorted_abundances = abundance_counts.sort_values(ascending=False).tolist()

        # Store in dictionary
        df_rank_dict[year] = [i for i in sorted_abundances if i > 0]

    df_rank = pd.DataFrame({
        "census": [year if isinstance(year, int) else year[0] for year in df_rank_dict.keys()],
        "rank_SAD": list(df_rank_dict.values())
    })

    # Merge explicitly on "census" to ensure alignment
    df_METimE = df_METimE.merge(df_SAD, on="census", how="left")
    df_METimE = df_METimE.merge(df_rank, on="census", how="left")

    # Save the final dataframe
    df_METimE.to_csv('../../data/fish_METimE_values.csv', index=False)


def load_birds():

    # # Create fish_abundance.csv
    # path = 'C:/Users/5605407/Documents/PhD/Chapter_2/Data sets/BioTIME/Bird_Hubbard_Brook_39/raw_data_39.csv'
    # df = pd.read_csv(path)
    #
    # df = df[['ABUNDANCE', 'GENUS_SPECIES', 'YEAR']]
    # df.to_csv('../data/birds_abundance.csv')

    # Load fish population data
    df_abundance = pd.read_csv('../../data/birds_abundance.csv', index_col=0)
    df_abundance = df_abundance[df_abundance['YEAR'] > 1985]

    # Add zero abundances so that each unique species has at least one record for each year
    all_species = df_abundance['GENUS_SPECIES'].unique()
    all_years = range(df_abundance['YEAR'].min(), df_abundance['YEAR'].max() + 1)
    species_years = pd.MultiIndex.from_product([all_species, all_years], names=['GENUS_SPECIES', 'YEAR'])
    df_abundance = df_abundance.set_index(['GENUS_SPECIES', 'YEAR']).reindex(species_years).reset_index()
    df_abundance['ABUNDANCE'] = df_abundance['ABUNDANCE'].fillna(0)

    # Load functional data (body mass)
    trait_path = 'C:/Users/5605407/Documents/PhD/Chapter_2/Data sets/elton_trait_database/BirdFuncDat.txt'
    df_bird_biomass = pd.read_csv(trait_path, delimiter="\t", encoding="ISO-8859-1")[["Scientific", "BodyMass-Value"]]
    df_bird_biomass.rename(columns={"Scientific": "GENUS_SPECIES"}, inplace=True)

    new_data = {
        'GENUS_SPECIES': [
            'Blue jay', 'Black-capped chickadee', 'Downy woodpecker', 'Setophaga fusca', 'Wood thrush',
            'Solitary Vireo',
            'Winter wren', 'Yellow-throated Warbler', 'Red-breasted nuthatch', 'White-breasted nuthatch',
            'Hairy woodpecker',
            'Rose-breasted grosbeak', 'Swainsons Thrush', 'Brown creeper', 'Yellow-bellied Sapsucker', 'Hermit Thrush',
            'Dark-eyed junco', 'Ovenbird', 'Setophaga caerulescens', 'Setophaga virens', 'Red-eyed vireo'
        ],
        'BodyMass-Value': [
            92.4, 12, 27, 10.5, 45, 16, 10, 9.7, 10, 20, 70, 45, 28, 8.4, 45, 27.5, 25, 22, 9.5, 8.6, 17
        ]
    }

    # Converting dictionary to DataFrame and appending to the existing df_bird_biomass
    df_new = pd.DataFrame(new_data)

    # Appending the new data to the existing DataFrame
    df_bird_biomass = pd.concat([df_bird_biomass, df_new], ignore_index=True)

    # Transform into metabolic rates
    df_bird_biomass["m"] = 0.7725 * df_bird_biomass["BodyMass-Value"] ** 0.7050
    df = df_abundance.merge(df_bird_biomass[['GENUS_SPECIES', 'm']], on="GENUS_SPECIES", how="left")

    # df_bird_biomass.append{
    #     'Blue jay': 92.4,
    #     'Black-capped chickadee': 12,
    #     'Downy woodpecker': 27,
    #     'Setophaga fusca': 10.5,
    #     'Wood thrush': 45,
    #     'Solitary Vireo': 16,
    #     'Winter wren': 10,
    #     'Yellow-throated Warbler': 9.7,
    #     'Red-breasted nuthatch': 10,
    #     'White-breasted nuthatch': 20,
    #     'Hairy woodpecker': 70,
    #     'Rose-breasted grosbeak': 45,
    #     'Swainsons Thrush': 28,
    #     'Brown creeper': 8.4,
    #     'Yellow-bellied Sapsucker': 45,
    #     'Hermit Thrush': 27.5,
    #     'Dark-eyed junco': 25,
    #     'Ovenbird': 22,
    #     'Setophaga caerulescens': 9.5,
    #     'Setophaga virens': 8.6,
    #     'Red-eyed vireo': 17
    # }

    # Set 'm' to 0 where 'ABUNDANCE' is 0
    # df['m'] = df['m'].where(df['ABUNDANCE'] != 0, 0)
    df['Total_m_species'] = df['m'] * df['ABUNDANCE']

    # Add State Variables to df
    df_mete = pd.DataFrame()
    df_mete['S_t'] = (
        df_abundance[df_abundance['ABUNDANCE'] > 0]
        .groupby('YEAR')['GENUS_SPECIES']
        .nunique()
        .reset_index(drop=True)
        .apply(np.ceil)
    )
    df_mete['next_S'] = df_mete['S_t'].shift(-1)
    df_mete['N_t'] = np.ceil(df_abundance.groupby(['YEAR'])['ABUNDANCE'].sum()).reset_index(drop=True)
    df_mete['next_N'] = df_mete['N_t'].shift(-1)
    df_mete['E_t'] = df.groupby(['YEAR'])['Total_m_species'].transform('sum')
    df_mete['next_E'] = df_mete['E_t'].shift(-1)
    df_mete['YEAR'] = df_abundance.groupby(['YEAR']).sum().index
    df_mete.rename(columns={'YEAR': 'census'}, inplace=True)

    # Sum abundance over stations
    df_dyna = df.copy()
    df_dyna = df_dyna.drop(columns={'Total_m_species'})

    # Create the 'next_n' column by shifting the 'next_n' values within each species group
    df_dyna['next_n'] = df_dyna.groupby('GENUS_SPECIES')['ABUNDANCE'].shift(-1)
    df_dyna['next_m'] = df_dyna.groupby('GENUS_SPECIES')['m'].shift(-1)

    df_dyna.columns = df_dyna.columns.str.strip().str.upper()
    df_dyna.rename(columns={'ABUNDANCE': 'n', 'M': 'm', 'YEAR': 'census', 'GENUS_SPECIES': 'species', 'NEXT_N':'next_n', 'NEXT_M': 'next_m'}, inplace=True)
    df_dyna = df_dyna.merge(df_mete, on='census', how='left') # add state variables

    df_dyna['dn'] = df_dyna['next_n'] - df_dyna['n']
    df_dyna['dm'] = df_dyna['next_m'] - df_dyna['m']
    df_dyna['dS'] = df_dyna['next_S'] - df_dyna['S_t']
    df_dyna['dN'] = df_dyna['next_N'] - df_dyna['N_t']
    df_dyna['dE'] = df_dyna['next_E'] - df_dyna['E_t']

    df_dyna = df_dyna.drop(columns=['next_n', 'next_S', 'next_m', 'next_N', 'next_E'], axis=1)

    df_dyna = df_dyna.dropna(how='any')
    df_dyna.to_csv('../../data/birds_regression_library.csv', index=False)

    df_METimE = pd.DataFrame()
    df_METimE['census'] = df_mete['census']
    df_METimE['S'] = df_mete['S_t']
    df_METimE['N'] = df_mete['N_t']
    df_METimE['E'] = df_mete['E_t']
    df_METimE['N/S'] = df_dyna['N_t']/df_dyna['S_t']
    df_METimE['E/S'] = df_dyna['E_t'] / df_dyna['S_t']
    df_METimE['dN/S'] = df_dyna['dN'] / df_dyna['S_t']
    df_METimE['dE/S'] = df_dyna['dE'] / df_dyna['S_t']
    df_METimE['dS'] = df_dyna['dS']

    # Save empirical SADs
    df_SAD_dict = {}
    grouped = df_abundance.groupby(["YEAR"], sort=True)

    for year, group in grouped:
        sad_year = []

        # Count species abundances
        abundance_counts = group.groupby("GENUS_SPECIES")["ABUNDANCE"].sum()

        # Determine max abundance (S)
        S = group["GENUS_SPECIES"].nunique()
        N = int(group["ABUNDANCE"].sum())

        # Compute species count for each n
        species_count = abundance_counts.value_counts().to_dict()

        for n in range(1, N - S + 1):
            abundance = species_count.get(n, 0)
            sad_year.append(abundance)

        df_SAD_dict[year] = sad_year  # Store SAD with YEAR as key

    df_SAD = pd.DataFrame({
        "census": [year if isinstance(year, int) else year[0] for year in df_SAD_dict.keys()],  # Extract integer years
        "SAD": list(df_SAD_dict.values())  # Ensure SAD lists are properly stored
    })

    df_rank_dict = {}
    for year, group in grouped:
        # Count species abundances
        abundance_counts = group.groupby("GENUS_SPECIES")["ABUNDANCE"].sum()

        # Sort species by abundance (high to low)
        sorted_abundances = abundance_counts.sort_values(ascending=False).tolist()

        # Store in dictionary
        df_rank_dict[year] = [i for i in sorted_abundances if i > 0]

    df_rank = pd.DataFrame({
        "census": [year if isinstance(year, int) else year[0] for year in df_rank_dict.keys()],
        "rank_SAD": list(df_rank_dict.values())
    })

    # Merge explicitly on "census" to ensure alignment
    df_METimE = df_METimE.merge(df_SAD, on="census", how="left")
    df_METimE = df_METimE.merge(df_rank, on="census", how="left")

    # Save the final dataframe
    df_METimE.to_csv('../../data/birds_METimE_values.csv', index=False)

def load_BCI():
    path = 'C:/Users/5605407/Documents/PhD/Chapter_2/Data sets/BCI/FullMeasurementBCI.tsv'
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
    df['dN'] = df['next_N'] - df['N_t']
    df['dE'] = df['next_E'] - df['E_t']

    df = df.drop(columns=['next_n', 'next_S', 'next_e', 'next_N', 'next_E'], axis=1)

    df = df.dropna(how='any')
    df.to_csv('../../data/BCI_regression_library.csv', index=False)

    df_METimE = pd.DataFrame()
    df_METimE['census'] = df['census']
    df_METimE['S'] = df['S_t']
    df_METimE['N'] = df['N_t']
    df_METimE['E'] = df['E_t']
    df_METimE['N/S'] = df['N_t']/df['S_t']
    df_METimE['E/S'] = df['E_t'] / df['S_t']
    df_METimE['dN/S'] = df['dN'] / df['S_t']
    df_METimE['dE/S'] = df['dE'] / df['S_t']
    df_METimE['dS'] = df['dS']

    df_METimE = df_METimE.drop_duplicates()

    # Save empirical SADs
    df_SAD_dict = {}
    df_rank_dict = {}
    grouped = df.groupby(["census"], sort=True)

    for year, group in grouped:
        sad_year = []
        rank_year = []

        # Remove duplicate rows (for multiple treeIDs)
        group = group.drop(columns=['TreeID', 'e', 'de'], errors='ignore').drop_duplicates()

        # Count species abundances
        abundance_counts = group.groupby("species")["n"].sum()

        # Determine max abundance (S)
        S = group["species"].nunique()
        N = int(group["n"].sum())

        # Compute species count for each n
        species_count = abundance_counts.value_counts().to_dict()

        for n in range(1, N - S + 1):
            abundance = species_count.get(n, 0)
            sad_year.append(abundance)

        for n in range(N - S, 0, -1):
            count = species_count.get(n, 0)
            for i in range(count):
                rank_year.append(n)

        df_SAD_dict[year] = sad_year
        df_rank_dict[year] = rank_year

    df_SAD = pd.DataFrame({
        "census": [year if isinstance(year, int) else year[0] for year in df_SAD_dict.keys()],  # Extract integer years
        "SAD": list(df_SAD_dict.values())  # Ensure SAD lists are properly stored
    })

    df_rank = pd.DataFrame({
        "census": [year if isinstance(year, int) else year[0] for year in df_rank_dict.keys()],
        "rank_SAD": list(df_rank_dict.values())
    })

    # Merge explicitly on "census" to ensure alignment
    df_METimE = df_METimE.merge(df_SAD, on="census", how="left")
    df_METimE = df_METimE.merge(df_rank, on="census", how="left")

    # Save the final dataframe
    df_METimE.to_csv('../../data/BCI_METimE_values.csv', index=False)


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
                # For this missing census, find the next available census record's 'e'
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
    #load_fish()
    #load_birds()
    load_BCI()


