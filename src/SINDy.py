import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pysindy as ps
from pysindy.feature_library import CustomLibrary, GeneralizedLibrary, PolynomialLibrary, FourierLibrary
from scipy import stats
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def load_data(data_set):

    if data_set == "fish":
        df = pd.read_csv('../data/fish_regression_library.csv')
        #df = df[df['census'] >= 2007]
        df.reset_index(drop=True, inplace=True)

    if data_set == "birds":
        df = pd.read_csv('../data/birds_regression_library.csv')
        #df = df[df['census'] >= 2014]
        df.reset_index(drop=True, inplace=True)

    if data_set == "BCI":
        df = pd.read_csv('../data/BCI_regression_library.csv')
        #df = df[df['census'] >= 7]
        df.reset_index(drop=True, inplace=True)

    return df


def sigmoid(x, k=5, x0=0):
    return 1 / (1 + np.exp(-k * (x - x0)))


def SINDy_METiME(df):
    response_vars = ['dn', 'de', 'dS'] if 'e' in df.columns else ['dn', 'dS']
    df_y = df[response_vars]
    t = df['census']
    X = df.drop(response_vars, axis=1, errors='ignore')
    X = X.drop(['census', 'species', 'dN', 'dE'], axis=1, errors='ignore')

    poly_lib = PolynomialLibrary(degree=2)
    fourier_lib = FourierLibrary(n_frequencies=2)
    sigmoid_lib = CustomLibrary(
        library_functions=[
            lambda x: sigmoid(x, k=5, x0=0),
            lambda x: sigmoid(x, k=10, x0=1)
        ],
        function_names=[lambda x: "sigmoid5(x)", lambda x: "sigmoid10(x-1)"]
    )

    # Combine all feature libraries
    feature_lib = GeneralizedLibrary([poly_lib, fourier_lib, sigmoid_lib])

    # Fit SINDy model
    for pred in response_vars:
        y = df_y[[pred]]
        sindy_model = ps.SINDy(feature_library=feature_lib)
        sindy_model.fit(X, y, t=t)

        # Print discovered equations
        y_pred = sindy_model.predict(X)

        plt.scatter(y, y_pred)
        plt.show()
        sindy_model.print()

    pass




if __name__ == "__main__":

    for data_set in ['fish', 'birds', 'BCI']:
        print("-----" + data_set + "-----")

        df = load_data(data_set)
        SINDy_METiME(df)






