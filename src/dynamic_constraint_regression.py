import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge

if __name__ == "__main__":

    # Load bird data
    path = 'C:/Users/5605407/Documents/PhD/Chapter_2/Data sets/BioTIME/Bird_Hubbard_Brook_39/raw_data_39.csv'
    df = pd.read_csv(path)


    # Load functional data (biomass)
    trait_path = 'C:/Users/5605407/Documents/PhD/Chapter_2/Data sets/elton_trait_database/BirdFuncDat.txt'
    df_bird_biomass = pd.read_csv(trait_path, delimiter="\t", encoding="ISO-8859-1")[["Scientific", "BodyMass-Value"]]
    df_bird_biomass.rename(columns={"Scientific": "GENUS_SPECIES"}, inplace=True)


    # Transform into metabolic rates
    df_bird_biomass["Metabolic_Rate"] = 70 * df_bird_biomass["BodyMass-Value"] ** 0.75
    df = df.merge(df_bird_biomass[['GENUS_SPECIES', 'Metabolic_Rate']], on="GENUS_SPECIES", how="left")
    df['Metabolic_Rate'] = df['Metabolic_Rate'] * df['ABUNDANCE']


    # Add State variables to df
    df['N_t'] = df.groupby(['YEAR'])['ABUNDANCE'].transform('sum')
    #df['E_t'] = df.groupby(['YEAR'])['Metabolic_Rate'].transform('sum')
    #df['S_t'] = df.groupby(['YEAR'])['GENUS_SPECIES'].transform('nunique')
    df_n = df.copy()[['GENUS_SPECIES', 'YEAR', 'ABUNDANCE', 'Metabolic_Rate', 'N_t']]
    df_n.rename(columns={'ABUNDANCE': 'n_t', 'Metabolic_Rate': 'e_t'}, inplace=True)


    # Add the values of n at the next year
    df_n_next = df_n.copy()
    df_n_next['YEAR'] = df_n_next['YEAR'] - 1
    df_n_next.rename(columns={'n_t': 'n_t+1'}, inplace=True)
    df_n_next = df_n_next[['GENUS_SPECIES', 'YEAR', 'n_t+1']]
    df_n = df_n.merge(df_n_next, how='left', on=['GENUS_SPECIES', 'YEAR'])
    df_n = df_n.drop(['GENUS_SPECIES', 'YEAR'], axis=1)
    df_n = df_n.dropna()


    #TODO: delete
    df_n['e_t'] = df_n['e_t'] / 7692.0


    # add interaction coefficients
    df_n['ne'] = df_n['n_t'] * df_n['e_t']
    df_n['n_over_N'] = df_n['n_t'] / df_n['N_t']
    df_n['constant'] = 1


    # Fit a function
    #model = LinearRegression()
    model = Ridge(alpha=1.0)
    X = df_n.drop(['n_t+1', 'N_t'], axis=1)
    y = df_n['n_t+1']
    model.fit(X, y)


    # Error on training set
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    plt.scatter(y, y_pred)
    plt.axis('square')
    plt.show()

    coefficients = dict(zip(model.feature_names_in_, model.coef_))
    print(coefficients)