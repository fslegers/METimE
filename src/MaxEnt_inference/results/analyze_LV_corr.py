import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, t


def remove_outliers(df, cols):
    """Removes rows from df where any of the specified columns have outliers using the IQR method."""
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask = (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
        df = df[mask]
    return df


def spearman_summary(x, y, x_name, y_name):
    rho, pval = spearmanr(x, y)
    print(f"Spearman correlation between '{x_name}' and '{y_name}':")
    print(f"  ρ = {rho:.4f}, p = {pval:.4e}\n")
    return rho, pval


def scatter_and_spearman(df, x_col, y_col, x_label=None, y_label=None,
                         alpha=0.1, xlim=None, ylim=None,
                         color_split_y=False, above_color='blue', below_color='red'):
    """
    Removes outliers, creates a scatter plot (with optional color split), and prints Spearman correlation.

    Args:
        df: DataFrame
        x_col, y_col: column names for x and y axes
        x_label, y_label: optional axis labels
        alpha: point transparency
        xlim, ylim: optional limits for axes
        color_split_y: if True, colors points differently for y >= 0 vs y < 0
        above_color, below_color: colors for y >= 0 and y < 0 points
    """
    x_label = x_label or x_col
    y_label = y_label or y_col

    tmp = remove_outliers(df.copy(), [x_col, y_col])

    if color_split_y:
        above = tmp[tmp[y_col] >= 0]
        below = tmp[tmp[y_col] < 0]
        plt.scatter(above[x_col], above[y_col], alpha=alpha, color=above_color, label=f"{y_col} ≥ 0")
        plt.scatter(below[x_col], below[y_col], alpha=alpha, color=below_color, label=f"{y_col} < 0")
        plt.legend()
    else:
        plt.scatter(tmp[x_col], tmp[y_col], alpha=alpha)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    plt.show()

    spearman_summary(tmp[x_col], tmp[y_col], x_label, y_label)


if __name__ == '__main__':
    df = pd.read_csv("METimE_vs_METE_summary.csv")
    df['diff_AIC'] = df['AIC_mete'] - df['AIC_metime']
    df['diff_RMSE'] = df['rmse_mete'] - df['rmse_metime']
    df['ratio_RMSE'] = df['rmse_mete'] / df['rmse_metime']
    df['ratio'] = df['dN'] / df['N/S']

    #scatter_and_spearman(df, 'r^2_transition', 'ratio_RMSE', "R^2_transition", "RMSE METE / RMSE METimE",  color_split_y=True)
    #scatter_and_spearman(df, 'r^2_transition', 'rmse_metime', "R^2_transition", "RMSE METimE")
    #scatter_and_spearman(df, 'ratio', 'diff_RMSE', "dN / N/S", "RMSE METE - RMSE METIME", color_split_y=True)
    #scatter_and_spearman(df, 'ratio', 'ratio_RMSE', "dN / N/S", "RMSE METE / RMSE METIME")
    #scatter_and_spearman(df, 'ratio', 'rmse_metime', "dN / N/S", "RMSE METimE")
    #scatter_and_spearman(df, 'ratio', 'rmse_mete', "dN / N/S", "RMSE METE")
    #scatter_and_spearman(df, 'ratio', 'r^2_transition', "dN / N/S", "R^2 transition")

    # For each unique model and variance combination, calculate the mean and 95 CI interval of R^2_transition
    # And save as Pandas DataFrame
    # with columns: inter-genus variance, model a, model b, ..., model f
    # with entries: mean R^2 \n [5%, 95%]

    # Group by 'var' and 'model'
    grouped = df.groupby(['var', 'model'])['r^2_transition']

    # Compute statistics
    summary = grouped.agg(['mean', 'count', 'std']).reset_index()

    # Compute 95% CI using t-distribution
    confidence = 0.95
    summary['ci_half_width'] = (
            summary['std'] / np.sqrt(summary['count']) *
            t.ppf((1 + confidence) / 2., summary['count'] - 1)
    )

    # Format string: mean R^2\n[5%, 95%]
    summary['formatted'] = summary.apply(
        lambda
            row: f"{row['mean']:.3f}\n[{(row['mean'] - row['ci_half_width']):.3f}, {(row['mean'] + row['ci_half_width']):.3f}]",
        axis=1
    )

    # Pivot table: rows = var, columns = model, values = formatted
    result_df = summary.pivot(index='var', columns='model', values='formatted')

    # Optional: sort columns and index for readability
    result_df = result_df.sort_index().sort_index(axis=1)

    # Save result as latex table
    result_df.to_latex("r2_summary_table.tex",
                       index=True,
                       index_names=True,
                       bold_rows=True,
                       header=['(a)', '(b)', '(c)', '(d)', '(e)', '(f)'],
                       float_format="%.3f",
                       escape=False)

