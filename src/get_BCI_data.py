import os
import pandas as pd
import numpy as np


def calculate_SAD(group):
    species_abundance = group.groupby('SpeciesID').size().reset_index(name='Count')
    ranked_abundance = species_abundance.sort_values(by='Count', ascending=False).reset_index()
    return ranked_abundance['Count'].tolist()


if __name__ == "__main__":

    path = 'C:/Users/5605407/Documents/PhD/Chapter_2/Data sets/BCI/FullMeasurementBCI.tsv'
    df = pd.read_csv(path, sep='\t', low_memory=False)

    df = df[df['Status'] == "alive"]
    df = df.drop(["Mnemonic", "Subspecies", "SubspeciesID", "StemTag", "HOM", "HighHOM", "ListOfTSM", "Date", "ExactDate", "Status", "QuadratName", "QuadratID", 'PX', 'PY'], axis = 1)
    df = df.dropna()


    # Take average of 'DBH' for duplicates
    df = df.groupby([col for col in df.columns if col != 'DBH'], as_index=False).agg({'DBH': 'mean'})


    # Calculate Metabolic Rate as in "A strong test for Maximum Entropy Theory of Ecology, Xiao, 2015"
    min_DBH = min(df['DBH'])
    df['MetabolicRate'] = (df['DBH'] / min_DBH)**2


    # Calculate METE's state variables
    df_METE = (
        df.groupby(['PlotCensusNumber'])
        .agg(
            S=('SpeciesID', 'nunique'),
            N=('SpeciesID', 'size'),
            E=('MetabolicRate', 'sum')
        )
        .reset_index()
    )


    # Calculate the empirical SAD
    df_SAD = (
        df.groupby(['PlotCensusNumber'], group_keys=False)
        .apply(calculate_SAD, include_groups=False)
        .reset_index(name='SAD')
    )


    # Merge the two
    df_METE = df_METE.merge(df_SAD, how='left', on=['PlotCensusNumber'])

    filename = 'C:/Users/5605407/Documents/PhD/Chapter_2/Data sets/BCI/METE_Input_BCI.csv'
    df_METE.to_csv(filename, index=False)





