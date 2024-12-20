import os
import pandas as pd
import numpy as np


def calculate_SAD(group):
    species_abundance = group.groupby('GENUS_SPECIES')['ABUNDANCE'].sum()
    ranked_abundance = species_abundance.sort_values(ascending=False).reset_index()
    return ranked_abundance['ABUNDANCE'].tolist()


if __name__ == "__main__":

    path = 'C:/Users/5605407/Documents/PhD/Chapter_2/Data sets/BioTIME'

    for root, _, files in os.walk(path):

        for file in files:
            if file.startswith("raw_data_"):

                # Extract study ID from file name
                study_ID = file.split("raw_data_")[1].split(".csv")[0]

                # Create the full file path
                file_path = os.path.join(root, file)

                # Load the CSV file as a DataFrame
                df = pd.read_csv(file_path)

                # Birds
                if study_ID == "39":

                    # Load bird functional data (body mass)
                    trait_path = 'C:/Users/5605407/Documents/PhD/Chapter_2/Data sets/elton_trait_database/BirdFuncDat.txt'
                    df_bird_biomass = pd.read_csv(trait_path, delimiter="\t", encoding="ISO-8859-1")[["Scientific", "BodyMass-Value"]]
                    df_bird_biomass.rename(columns={"Scientific": "GENUS_SPECIES"}, inplace=True)

                    # Estimate metabolic rate from body mass
                    df_bird_biomass["Metabolic_Rate"] = 0.7725 * df_bird_biomass["BodyMass-Value"]**0.7050
                    df = df.merge(df_bird_biomass[['GENUS_SPECIES', 'Metabolic_Rate']], on = "GENUS_SPECIES", how = "left")

                    # Transform individual metabolic rates to reflect Abundance
                    df['Metabolic_Rate'] = df['Metabolic_Rate'] * df['ABUNDANCE']


                # Fish
                if study_ID == "428":
                    # 59 species
                    break

                # BCI
                if study_ID == "60":
                    # 324 species
                    break

                # Phytoplankton
                if study_ID == "354":
                    # 734 species
                    # Not absolute abundance
                    # Some biomass information missing
                    break

                # Calculate METE's state variables
                df_METE = (
                    df.groupby(['DAY', 'MONTH', 'YEAR', 'LATITUDE', 'LONGITUDE'])
                    .agg(
                        S=('GENUS_SPECIES', 'nunique'),
                        N=('ABUNDANCE', 'sum'),
                        E=('Metabolic_Rate', 'sum'),
                    )
                    .reset_index()
                )

                # Ensure that total number of species/individuals are integers
                df_METE['S'] = np.ceil(df_METE['S'])
                df_METE['N'] = np.ceil(df_METE['N'])

                # Calculate the empirical SAD
                df_SAD = (
                    df.groupby(['DAY', 'MONTH', 'YEAR', 'LATITUDE', 'LONGITUDE'], group_keys=False)
                    .apply(calculate_SAD)
                    .reset_index(name='SAD')
                )

                df_METE = df_METE.merge(df_SAD, how='left', on=['DAY', 'MONTH', 'YEAR', 'LATITUDE', 'LONGITUDE'])

                filename = 'C:/Users/5605407/Documents/PhD/Chapter_2/Data sets/BioTIME/METE_Input_' + str(study_ID) + '.csv'
                df_METE.to_csv(filename)

                # Add derivatives of state variables
                df_METE['next_N'] = df_METE['N'].shift(-1)
                df_METE['next_S'] = df_METE['S'].shift(-1)
                df_METE['next_E'] = df_METE['E'].shift(-1)

                df_METE.dropna()

                df_METE['dN'] = df_METE['next_N'] - df_METE['N']
                df_METE['dS'] = df_METE['next_S'] - df_METE['S']
                df_METE['dE'] = df_METE['next_E'] - df_METE['E']

                filename = 'C:/Users/5605407/Documents/PhD/Chapter_2/Data sets/BioTIME/dynaMETE_Input_' + str(
                    study_ID) + '.csv'
                df_METE.to_csv(filename)





