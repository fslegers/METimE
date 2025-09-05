import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV, Lasso
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import pysindy as sindy
import re

def remove_outliers(df):
    df_clean = df.copy()

    for col in ['n', 'dn']:
        Q1 = np.percentile(df_clean[col], 15, method='midpoint')
        Q3 = np.percentile(df_clean[col], 85, method='midpoint')
        IQR = Q3 - Q1

        upper = Q3 + 1.5 * IQR
        lower = Q1 - 1.5 * IQR

        # Keep only values within the bounds
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

    return df_clean

def do_polynomial_regression(df, lv_ratio=0.6):

    # Make base nonlinear transformations of e and n
    df = df.copy()

    # Remove outliers
    df = remove_outliers(df)

    # Protect against zero/negative values for logs and inverses
    eps = 1e-12
    e = df['e'].clip(lower=eps)

    df['(n ** (1/2))'] = df['n'] ** (1 / 2)
    df['(n)'] = df['n']
    df['(n ** (3/4))'] = df['n'] ** (3 / 4)
    df['(n ** (3/2))'] = df['n'] ** (3 / 2)

    df['(np.log(n))'] = np.log(df['n'])

    df['(1/N_t)'] = 1.0 / df['N_t']

    df.rename(columns={'N_t': '(N_t)', 'S_t': '(S_t)'}, inplace=True)

    # These are the columns that will be used to create polynomial features
    poly_cols = ['(n ** (1/2))', '(n)', '(n ** (3/4))', '(n ** (3/2))',
                 '(np.log(n))',
                 '(N_t)', '(1/N_t)']

    # Generate polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=True)
    poly_features = poly.fit_transform(df[poly_cols])

    # Create a new DataFrame with polynomial features
    poly_feature_names = poly.get_feature_names_out(poly_cols)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)

    # Concatenate polynomial features back to the original DataFrame
    df = pd.concat([df.drop(columns=poly_cols), poly_df], axis=1)

    # Drop 'tree_id' and dN/S and dE/S columns
    df = df.drop(columns=['TreeID', 'dN/S', 'dS'], errors='ignore')

    # Here, we take averages per species over features calculated per tree
    # Group by (t, species_id) and sum all features
    df_grouped = df.groupby(['census', 'species']).sum().reset_index()

    # Run STLSQ
    coef_dn, r2_dn, scaler = stepwise_sparse_regression(df_grouped, lv_ratio)
    print(f"R2 dn: {r2_dn}")

    return coef_dn, r2_dn, scaler

def stepwise_sparse_regression(df_grouped, alpha=0.01):
    # Split into target and features
    dn_obs = df_grouped['dn'].values
    X = df_grouped.drop(columns=['census', 'species', 'dn'], errors='ignore')

    feature_names = X.columns.tolist()

    # Standardize features
    scaler = StandardScaler(with_mean=True)
    X_scaled = scaler.fit_transform(X)

    results = []
    r2s = []

    for y_obs, target_name in [(dn_obs, 'dn')]:

        # initial Lasso fit
        model = Lasso(alpha=alpha, fit_intercept=False)
        model.fit(X_scaled, y_obs)
        y_pred = model.predict(X_scaled)
        prev_r2 = r2_score(y_obs, y_pred)
        best_coef = model.coef_.copy()

        for _ in range(len(feature_names) - 1):

            # Identify the index of the smallest non-zero coefficient
            non_zero_idx = np.where(best_coef != 0)[0]
            if len(non_zero_idx) <= 1:
                break  # all coefficients eliminated

            smallest_idx = non_zero_idx[np.argmin(np.abs(best_coef[non_zero_idx]))]

            # Zero out the smallest coefficient
            new_coef = best_coef.copy()
            new_coef[smallest_idx] = 0

            # Refit only on remaining non-zero features
            mask = new_coef != 0
            if not np.any(mask):
                break  # nothing left to fit

            model = Lasso(alpha=alpha, fit_intercept=False)
            model.fit(X_scaled[:, mask], y_obs)

            # Update coefficients
            new_coef = np.zeros_like(new_coef)
            new_coef[mask] = model.coef_

            # compute R²
            y_pred = model.predict(X_scaled[:, mask])
            r2 = r2_score(y_obs, y_pred)

            # stop if prediction accuracy decreases too much
            if prev_r2 - r2 > 1e-4:
                break

            prev_r2 = r2
            best_coef = new_coef.copy()

        # Recalculate non-zero coefficients (without l1 norm regularization)
        mask = best_coef != 0
        model = ElasticNetCV(l1_ratio=0.6, alphas=np.logspace(-3, 1, 50), fit_intercept=False)
        model.fit(X_scaled[:, mask], y_obs)
        coef_new = np.zeros_like(best_coef)
        coef_new[mask] = model.coef_
        best_coef = coef_new
        y_pred = model.predict(X_scaled[:, mask])
        r2 = r2_score(y_obs, y_pred)

        # store results
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'Coefficient': best_coef.tolist()
        })
        results.append(coef_df)
        r2s.append(r2)

    coef_dn, coef_de = results
    r2_dn, r2_de = r2s

    return coef_dn, r2_dn, scaler

def f_n(n, e, X, alphas, betas, scaler):
    return n

def f_dn(n, e, X, alphas, betas, scaler):
    return 0

def get_functions():
    return [f_n, f_dn]

