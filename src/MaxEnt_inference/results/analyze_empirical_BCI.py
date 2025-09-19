import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# --- Format numbers ---
def format_value(x, col=None):
    if isinstance(x, (int, float)):
        if col == "slack_weight":
            return f"{x:.1e}"  # scientific notation, 1 decimal
        else:
            return f"{x:.2f}".rstrip("0").rstrip(".")  # normal 2-decimal format
    return x

def metrics_per_slack_weight(df, quad):
    # Calculate group averages
    df_mean = df.groupby("slack_weight")[
        ["METimE_AIC", "METimE_MAE", "METimE_RMSE", "METE_AIC", "METE_MAE", "METE_RMSE"]
    ].mean().reset_index()

    # Create figure with shared x-axis
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    # Colormap for consistent census colors
    cmap = plt.get_cmap("tab10", df["census"].nunique())
    census_colors = {c: cmap(i) for i, c in enumerate(sorted(df["census"].unique()))}

    # --- Top subplot: MAE ---
    for census, group in df.groupby("census"):
        color = census_colors[census]
        axes[0].plot(group["slack_weight"], group["METimE_MAE"], color=color, label=f"Census {census}")
        axes[0].plot(group["slack_weight"], group["METE_MAE"], color=color, linestyle="--")

    # Average lines
    axes[0].plot(df_mean["slack_weight"], df_mean["METimE_MAE"],
                 color="black", linewidth=2)
    axes[0].plot(df_mean["slack_weight"], df_mean["METE_MAE"],
                 color="black", linewidth=2, linestyle="--")

    axes[0].set_ylabel("MAE")
    axes[0].grid(False)
    axes[0].legend(ncol=2)

    # --- Bottom subplot: RMSE ---
    for census, group in df.groupby("census"):
        color = census_colors[census]
        axes[1].plot(group["slack_weight"], group["METimE_RMSE"], color=color, label=f"Census {census}")
        axes[1].plot(group["slack_weight"], group["METE_RMSE"], color=color, linestyle="--")

    # Average lines
    axes[1].plot(df_mean["slack_weight"], df_mean["METimE_RMSE"],
                 color="black", linewidth=2)
    axes[1].plot(df_mean["slack_weight"], df_mean["METE_RMSE"],
                 color="black", linewidth=2, linestyle="--")

    axes[1].set_xlabel("Slack weight")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(False)
    axes[1].legend(ncol=2)

    plt.xscale('log')

    # --- Bottom subplot: AIC ---
    for census, group in df.groupby("census"):
        color = census_colors[census]
        axes[2].plot(group["slack_weight"], group["METimE_AIC"], color=color, label=f"Census {census}")
        axes[2].plot(group["slack_weight"], group["METE_AIC"], color=color, linestyle="--")

    # Average lines
    axes[2].plot(df_mean["slack_weight"], df_mean["METimE_AIC"],
                 color="black", linewidth=2)
    axes[2].plot(df_mean["slack_weight"], df_mean["METE_AIC"],
                 color="black", linewidth=2, linestyle="--")

    axes[2].set_ylabel("AIC")
    axes[2].grid(False)
    axes[2].legend(ncol=2)

    plt.tight_layout()
    plt.show()

def fill_latex_table(df):
    # --- Clean quad column ---
    df["quad"] = df["quad"].str.replace("_quadrat_", "", regex=False)

    # Apply formatting column by column
    for col in df.columns:
        df[col] = df[col].apply(lambda x: format_value(x, col))

    # --- Reorder columns (added slack_weight after quad and census) ---
    cols = [
        "quad", "census", "r2_dn", "r2_de", "METE_AIC", "METE_MAE", "METE_RMSE",
        "METimE_AIC", "METimE_MAE", "METimE_RMSE"
    ]
    df = df[cols]

    # --- Convert to LaTeX without headers ---
    latex_table = df.to_latex(
        index=False,
        header=False,
        column_format="cc|cc|ccc|ccc",
        escape=False
    )

    # --- Build custom header (added slack_weight) ---
    custom_header = (
        "\\toprule\n"
        " & & & &  \\multicolumn{3}{c|}{METE} & \\multicolumn{3}{c}{METimE} \\\\\n"
        "Quadrat & Census & r2_dn & r2_de & AIC & MAE & RMSE & AIC & MAE & RMSE \\\\\n"
        "\\midrule\n"
    )

    # --- Insert header ---
    latex_table = latex_table.replace("\\toprule", custom_header, 1)

    # --- Save to file ---
    with open("table.tex", "w") as f:
        f.write(latex_table)

    print(latex_table)

def select_best_slack_weight(df, metric="MAE"):
    results = []

    for quad in df['quad'].unique():
        for census in df['census'].unique():
            # filter by both quad and census
            df_subset = df[(df['quad'] == quad) & (df['census'] == census)]

            # find row that minimizes MAE
            best_idx = df_subset[f'METimE_{metric}'].idxmin()
            best_row = df_subset.loc[best_idx]

            results.append(best_row)

    # return a DataFrame of the selected best rows
    return pd.DataFrame(results).reset_index(drop=True)

def print_additional_metrics(df):
    n = len(df)

    # Calculate how often METimE outperforms METE
    better_AIC   = (df["METE_AIC"]   > df["METimE_AIC"]).sum()   / n * 100
    better_MAE   = (df["METE_MAE"]   > df["METimE_MAE"]).sum()   / n * 100
    better_RMSE  = (df["METE_RMSE"]  > df["METimE_RMSE"]).sum()  / n * 100
    better_NS    = (df["METE_error_N/S"]  > df["METimE_error_N/S"]).sum() / n * 100
    better_ES    = (df["METE_error_E/S"]  > df["METimE_error_E/S"]).sum() / n * 100
    better_NoverS= (df["METE_error_dN/S"] > df["METimE_error_dN/S"]).sum() / n * 100
    better_EoverS= (df["METE_error_dE/S"] > df["METimE_error_dE/S"]).sum() / n * 100

    # Calculate how often they're equal
    equal_AIC   = (df["METE_AIC"]   == df["METimE_AIC"]).sum()   / n * 100
    equal_MAE   = (df["METE_MAE"]   == df["METimE_MAE"]).sum()   / n * 100
    equal_RMSE  = (df["METE_RMSE"]  == df["METimE_RMSE"]).sum()  / n * 100
    equal_NS    = (df["METE_error_N/S"]  == df["METimE_error_N/S"]).sum() / n * 100
    equal_ES    = (df["METE_error_E/S"]  == df["METimE_error_E/S"]).sum() / n * 100
    equal_NoverS= (df["METE_error_dN/S"] == df["METimE_error_dN/S"]).sum() / n * 100
    equal_EoverS= (df["METE_error_dE/S"] == df["METimE_error_dE/S"]).sum() / n * 100

    # Build summary table with LaTeX-friendly labels
    summary = pd.DataFrame({
        "Metric": [
            "AIC", "MAE", "RMSE", r"$N/S$ error", r"$E/S$ error",
            r"$\Delta N/S$ error", r"$\Delta E/S$ error"
        ],
        "METimE better than METE (\\%)": [
            better_AIC, better_MAE, better_RMSE, better_NS, better_ES, better_NoverS, better_EoverS
        ],
        "METE as good as METimE (\\%)": [
            equal_AIC, equal_MAE, equal_RMSE, equal_NS, equal_ES, equal_NoverS, equal_EoverS
        ]
    })

    # Format as LaTeX table
    latex_table = summary.to_latex(
        index=False,
        escape=False,  # keep LaTeX math symbols
        float_format="%.2f"
    )

    print(latex_table)

def how_much_difference(df):
    #sns.set_theme(style="white")
    #custom_params = {"axes.spines.right": False, "axes.spines.top": False, "axes.spines.bottom": False}
    #sns.set_theme(style="ticks", rc=custom_params)

    # Colors
    blueish = "#67a9cf"
    greyish = "#4c4c4c"
    orangy = "#ef8a62"

    # Ensure numeric
    for col in ['METE_AIC', 'METimE_AIC', 'METE_MAE', 'METimE_MAE', 'METE_RMSE', 'METimE_RMSE', 'METE_error_N/S', 'METimE_error_N/S', 'METE_error_E/S', 'METimE_error_E/S', 'METE_error_dN/S', 'METimE_error_dN/S', 'METE_error_dE/S', 'METimE_error_dE/S']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Compute differences
    df_diff = pd.DataFrame({
        'AIC': df['METE_AIC'] - df['METimE_AIC'],
        'MAE': df['METE_MAE'] - df['METimE_MAE'],
        'RMSE': df['METE_RMSE'] - df['METimE_RMSE'],
        'N/S error': df['METE_error_N/S'] - df['METimE_error_N/S'],
        'E/S error': df['METE_error_E/S'] - df['METimE_error_E/S'],
        'dN/S error': df['METE_error_dN/S'] - df['METimE_error_dN/S'],
        'dE/S error': df['METE_error_dE/S'] - df['METimE_error_dE/S']
    })

    metrics = ['AIC', 'MAE', 'RMSE', 'N/S error', 'E/S error', 'dN/S error', 'dE/S error']

    fig, axes = plt.subplots(1, 7, figsize=(24, 5), sharey=False)

    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        values = df_diff[metric].dropna()

        # Violin plot without inner box
        sns.violinplot(
            y=values, ax=ax, color=blueish, alpha=0.8,
            inner=None, bw_adjust=0.5, cut=0, zorder=1
        )

        # Overlay custom boxplot (smaller width, rounded, black fill, white median)
        sns.boxplot(
            y=values, ax=ax, width=0.1, showcaps=False, showfliers=True,
            boxprops=dict(facecolor=greyish, edgecolor=greyish, linewidth=1.2),
            whiskerprops=dict(color=greyish, linewidth=1.0),
            capprops=dict(color=greyish, linewidth=1.0),
            medianprops=dict(color="white", linewidth=3),
            flierprops=dict(markerfacecolor=greyish, markersize=5, alpha=0.5)
        )

        # Strong horizontal line at 0
        ax.axhline(0, color=orangy, linewidth=4, linestyle="-", zorder=0)

        # Get y-limits after plotting
        ylim_min, ylim_max = ax.get_ylim()
        #ax.axhspan(0, ylim_max, facecolor='lightgreen', alpha=0.3, zorder=0)

        # Compute percentage above 0
        total = len(values)
        above = (values > 0).sum()
        perc_above = above / total * 100 if total > 0 else 0

        # Annotate percentages
        ax.text(0.5, 0.98, f"{perc_above:.1f}% > 0",
                ha='center', va='top', transform=ax.transAxes,
                fontsize=14)

        ax.set_title(metric, fontsize=16)

        if i == 0:
            ax.set_ylabel("Difference (METE - METimE)", fontsize=14)
        else:
            ax.set_ylabel("")

        ax.tick_params(axis='both', which='major', labelsize=12)

        # Inset boxplot on the right, centered vertically
        inset_ax = inset_axes(ax, width="25%", height="40%", loc="lower right",
                              borderpad=1.2)

        sns.boxplot(
            y=values, ax=inset_ax, width=0.2, showcaps=True, showfliers=False,
            boxprops=dict(facecolor=greyish, edgecolor="black", linewidth=1.2),
            whiskerprops=dict(color=greyish, linewidth=1.0),
            capprops=dict(color=greyish, linewidth=1.0),
            medianprops=dict(color="white", linewidth=3.0)
        )

        inset_ax.axhline(0, color=orangy, linewidth=4, linestyle="-")
        inset_ax.set_xticks([])
        inset_ax.set_xlabel("")
        inset_ax.set_ylabel("")
        inset_ax.tick_params(axis='y', labelsize=8)

        ylim_min, ylim_max = ax.get_ylim()
        ylim_max *= 1.80  # scale max by 10%
        ax.set_ylim(ylim_min, ylim_max)

    plt.tight_layout()
    plt.show()

def cleaner_look(df):
    # Colors
    blueish = "#67a9cf"
    greyish = "#4c4c4c"
    orangy = "#ef8a62"

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    # Ensure numeric
    numeric_cols = [
        'METE_AIC', 'METimE_AIC', 'METE_MAE', 'METimE_MAE',
        'METE_RMSE', 'METimE_RMSE', 'METE_error_N/S', 'METimE_error_N/S',
        'METE_error_E/S', 'METimE_error_E/S', 'METE_error_dN/S', 'METimE_error_dN/S',
        'METE_error_dE/S', 'METimE_error_dE/S'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Compute differences
    df_diff = pd.DataFrame({
        'AIC': df['METE_AIC'] - df['METimE_AIC'],
        'MAE': df['METE_MAE'] - df['METimE_MAE'],
        'RMSE': df['METE_RMSE'] - df['METimE_RMSE'],
        'N/S error': df['METE_error_N/S'] - df['METimE_error_N/S'],
        'E/S error': df['METE_error_E/S'] - df['METimE_error_E/S'],
        'dN/S error': df['METE_error_dN/S'] - df['METimE_error_dN/S'],
        'dE/S error': df['METE_error_dE/S'] - df['METimE_error_dE/S']
    })

    metrics = ['AIC', 'MAE', 'RMSE', 'N/S error', 'E/S error', 'dN/S error', 'dE/S error']

    fig, axes = plt.subplots(1, 7, figsize=(16, 5), sharey=False)

    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        values = df_diff[metric].dropna()

        # Overlay boxplot without outliers
        sns.boxplot(
            y=values, ax=ax, width=0.5, showcaps=True, showfliers=False,
            boxprops=dict(facecolor=blueish, edgecolor=greyish, linewidth=1.2, alpha=0.8),
            whiskerprops=dict(color=greyish, linewidth=2.0),
            capprops=dict(color=greyish, linewidth=2.0),
            medianprops=dict(color=greyish, linewidth=4)
        )

        # Strong horizontal line at 0
        ax.axhline(0, color=orangy, linewidth=4, linestyle="-", zorder=0)

        # Compute percentage above 0
        total = len(values)
        above = (values > 0).sum()
        perc_above = above / total * 100 if total > 0 else 0

        # # Annotate percentages (centered)
        # ax.text(0.5, 0.98, f"{perc_above:.1f}% > 0",
        #         ha='center', va='top', transform=ax.transAxes,
        #         fontsize=14)

        #ax.set_title(f"{metric}\n{perc_above:.1f}% > 0", fontsize=16, pad=20, linespacing=2.0, fontweight='bold')
        ax.set_title(f"{perc_above:.1f}% > 0", fontsize=16, pad=10)

        if i == 0:
            ax.set_ylabel("Difference\n(METE - METimE)", fontsize=16, linespacing=1.5)
        else:
            ax.set_ylabel("")
        ax.set_xlabel(metric, fontsize=16)

        ax.tick_params(axis='both', which='major', labelsize=12)

    fig.text(0.5, 0.05, 'Metric', ha='center', fontsize=20)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()

def simple_violin(df):
    """
    Make a single figure with horizontal subplots:
    - Each subplot shows half-violin comparisons (METE vs METimE) for one metric.
    - Each subplot keeps its own y-axis.
    """

    # Colors
    blueish = "#67a9cf"
    orangy = "#ef8a62"

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    metrics = [
        'AIC', 'MAE', 'RMSE', 'error_N/S', 'error_E/S',
        'error_dN/S', 'error_dE/S'
    ]

    # Filter to metrics present in the DataFrame
    valid_metrics = []
    for m in metrics:
        if f"METE_{m}" in df.columns and f"METimE_{m}" in df.columns:
            valid_metrics.append(m)
    n_metrics = len(valid_metrics)
    if n_metrics == 0:
        print("No valid metrics found.")
        return

    # Create a wide figure with one column per metric
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(4 * n_metrics, 6),
        sharey=False  # independent y-axis for each metric
    )
    # axes is an array even if n_metrics == 1
    if n_metrics == 1:
        axes = [axes]

    for ax, m in zip(axes, valid_metrics):
        mete_col = f"METE_{m}"
        metime_col = f"METimE_{m}"

        # Tidy dataframe for this metric
        plot_df = pd.concat([
            pd.DataFrame({'Value': pd.to_numeric(df[mete_col], errors='coerce'),
                          'Model': 'METE',
                          'Metric': m}),
            pd.DataFrame({'Value': pd.to_numeric(df[metime_col], errors='coerce'),
                          'Model': 'METimE',
                          'Metric': m})
        ], ignore_index=True)

        sns.violinplot(
            data=plot_df,
            x="Metric",
            y="Value",
            split=True,
            inner="quart",
            hue="Model",
            palette={"METE": blueish, "METimE": orangy},
            ax=ax,
            density_norm="area",
            inner_kws=dict(linewidth=2.5)
        )

        ax.set_title(m, fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("Value", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    # Load data
    path = r'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/BCI/empirical_BCI_df'
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    # # For each quadrat, plot the AIC, MAE and RMSE per slack_weight
    # for quad in df['quad'].unique():
    #     df_quad = df[df['quad'] == quad]
    #     metrics_per_slack_weight(df_quad, quad)

    # Calculate something and turn it into a latex table
    df = select_best_slack_weight(df, 'MAE')

    # Remove the one outlier
    # Find the one with extreme METimE_MAE
    outliers = df[df['METimE_MAE'] - df['METE_MAE'] > 100]
    print(outliers)
    df = df[df['METimE_MAE'] - df['METE_MAE'] <= 100]

    # Find the best and worst METimE_MAE
    diff = (df['METimE_MAE'] - df['METE_MAE'])/df['METE_MAE']

    best_idx = diff.idxmin()  # index of the minimum difference
    worst_idx = diff.idxmax()  # index of the maximum difference

    best = df.loc[best_idx]
    worst = df.loc[worst_idx]

    print(f"Best:\n{best}\n")
    print(f"Worst:\n{worst}\n")

    fill_latex_table(df)
    print_additional_metrics(df)

    # Also report how much better METimE predicts than METE on average, or vise versa, also reporting outliers
    #how_much_difference(df)
    simple_violin(df)
    cleaner_look(df)
