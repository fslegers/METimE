import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyexpat import model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, LinearRegression


def load_data(data_set):

    if data_set == "birds":

        # Load bird population data
        path = 'C:/Users/5605407/Documents/PhD/Chapter_2/Data sets/BioTIME/Bird_Hubbard_Brook_39/raw_data_39.csv'
        df = pd.read_csv(path)


        # Load functional data (body mass)
        trait_path = 'C:/Users/5605407/Documents/PhD/Chapter_2/Data sets/elton_trait_database/BirdFuncDat.txt'
        df_bird_biomass = pd.read_csv(trait_path, delimiter="\t", encoding="ISO-8859-1")[["Scientific", "BodyMass-Value"]]
        df_bird_biomass.rename(columns={"Scientific": "GENUS_SPECIES"}, inplace=True)


        # Transform into metabolic rates
        df_bird_biomass["Metabolic_Rate"] = 0.7725 * df_bird_biomass["BodyMass-Value"] ** 0.7050
        df = df.merge(df_bird_biomass[['GENUS_SPECIES', 'Metabolic_Rate']], on="GENUS_SPECIES", how="left")
        df['Metabolic_Rate'] = df['Metabolic_Rate'] * df['ABUNDANCE']


        # Add State Variables to df
        df['S_t'] = df.groupby(['YEAR'])['GENUS_SPECIES'].transform('nunique')
        df['S_t'] = np.ceil(df['S_t'])
        df['N_t'] = df.groupby(['YEAR'])['ABUNDANCE'].transform('sum')
        df['N_t'] = np.ceil(df['N_t'])
        df['E_t'] = df.groupby(['YEAR'])['Metabolic_Rate'].transform('sum')


        # Select columns
        df = df.copy()[['GENUS_SPECIES', 'YEAR', 'ABUNDANCE', 'Metabolic_Rate', 'S_t', 'N_t', 'E_t']]
        df.rename(columns={'GENUS_SPECIES': 'species', 'YEAR': 'census', 'ABUNDANCE': 'n_t', 'Metabolic_Rate': 'e_t'}, inplace=True)


        # Add the values of n at the next year
        df_next = df.copy()
        df_next['census'] = df_next['census'] - 1
        df_next.rename(columns={'n_t': 'n_t+1'}, inplace=True)
        df_next = df_next[['species', 'census', 'n_t+1']]
        df = df.merge(df_next, how='left', on=['species', 'census'])
        df = df.dropna()


    elif data_set == "BCI":
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


        # Add State Variables to df
        df['S_t'] = df.groupby(['PlotCensusNumber'])['SpeciesID'].transform('nunique')
        df['S_t'] = np.ceil(df['S_t'])
        df['N_t'] = df.groupby(['PlotCensusNumber'])['TreeID'].transform('nunique')
        df['N_t'] = np.ceil(df['N_t'])
        df['E_t'] = df.groupby(['PlotCensusNumber'])['Metabolic_Rate'].transform('sum')


        # Add population sizes to df
        df['n_t'] = df.groupby(['PlotCensusNumber', 'SpeciesID'])['TreeID'].transform('nunique')
        df.rename(columns={'SpeciesID': 'species', 'PlotCensusNumber': 'census', 'Metabolic_Rate': 'e_t'}, inplace=True)


        # Add the values of n at the next year
        df_next = df.copy()
        df_next['census'] = df_next['census'] - 1
        df_next.rename(columns={'n_t': 'n_t+1'}, inplace=True)
        df_next = df_next[['species', 'census', 'TreeID', 'n_t+1']]
        df = df.merge(df_next, how='left', on=['species', 'census', 'TreeID'])
        df = df.drop(['TreeID'], axis=1)
        df = df.dropna()


    return df


def set_up_library(df, lib):
    """
    Sets up the library of predictor variables for the bilinear regression.
    :param df: data frame with columns that are used to form the library
    ('species', 'census', 'n_t', 'e_t', 'S_t', 'N_t', 'E_t', and 'n_t+1')
    :param lib: character strings that represent which predictor variables should be included in the regression library.
    Possible terms to include are: 'C', 'n', 'ne', 'e', 'n/N', and 'e/E'
    :return: a dataframe with columns of predictor variables and n_{t + 1}
    """
    if ('n' not in lib) or ('e' not in lib):
        print('')
    if 'e/E' in lib:
        df['e/E'] = df['e_t'] / df['E_t']
    if 'n/N' in lib:
        df['n/N'] = df['n_t'] / df['N_t']
    if 'ne' in lib:
        df['ne'] = df['n_t'] * df['e_t']
    if 'C' in lib:
        df['C'] = 1

    df.drop(['S_t', 'N_t', 'E_t'], axis=1, inplace=True)
    return df


def do_regression(df):

    model = LinearRegression()
    #model = Ridge(alpha=0.5)

    X = df.drop(['n_t+1', 'species', 'census'], axis=1)
    y = df['n_t+1']
    model.fit(X, y)


    # Error on training set
    y_pred = model.predict(X)
    rmse = (mean_squared_error(y, y_pred))**(1/2)
    r2 = r2_score(y, y_pred)
    print("RMSE = %.3f, \n R^2 = %.3f \n" % (rmse, r2))


    # Add year column back to X for coloring the scatterplot
    df['y_pred'] = y_pred
    df['y_true'] = y
    df['census'] = df['census'] + 1 # we want to color by t+1 instead of t


    # Scatterplot with color by YEAR
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df['y_true'], df['y_pred'], c=df['census'], cmap='viridis', alpha=0.7, zorder=1)
    plt.colorbar(scatter, label='Census')
    plt.xlabel('Observed ($n_{t+1}$)')
    plt.ylabel('Predicted ($n_{t+1}$)')
    plt.axis('square')

    plt.axline(xy1 = (0,0), slope=1, color='black', zorder=0, label=("r^2 = %f" % r2)) # Add 1-1 line
    plt.text(0.87, 0.92, "$r^2$ = %.3f" % r2, fontsize=12,
             horizontalalignment='right',
             transform=plt.gca().transAxes)

    # Calculate the combined range
    min_val = min(min(df['y_true']), min(df['y_pred'])) *0.9
    max_val = max(max(df['y_true']), max(df['y_pred'])) *1.1

    # Set the same limits for both axes
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()


    coefficients = dict(zip(model.feature_names_in_, model.coef_))
    print("Coefficients:")
    for key, value in coefficients.items():
        print(f"{key} = {value:.6f}")


    return coefficients


if __name__ == "__main__":
    data_set = 'birds'
    #data_set = 'BCI'
    df = load_data(data_set)

    df = set_up_library(df, lib=['C', 'n', 'ne', 'e'])
    coefficients = do_regression(df)

    # TODO: save coefficients
    # TODO: include test set

    # NOTES:
    # For the BCI data set, we don't have e_t per species, but per individual. So the data that goes into the
    # regression model is different from the bird data, where e_t was the same for individuals within a
    # single species.


    # For the BCI data set, there are multiple Stem IDs per Tree ID, which causes duplicates. For now, I just took
    # the mean over the Stem IDs.


