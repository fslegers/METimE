import os
import glob
import numpy as np
import pandas as pd

# --- Format numbers ---
def format_value(x, col=None):
    if isinstance(x, (int, float)):
        if col == "slack_weight":
            return f"{x:.1e}"  # scientific notation, 1 decimal
        else:
            return f"{x:.2f}".rstrip("0").rstrip(".")  # normal 2-decimal format
    return x

path = r'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/BCI/empirical_BCI_df'
all_files = glob.glob(os.path.join(path, "*.csv"))
df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

# Find order of magnitude difference
df.loc[:, 'min'] = df[['N/S', 'E/S', 'dN/S', 'dE/S']].abs().min(axis=1)
df.loc[:, 'max'] = df[['N/S', 'E/S', 'dN/S', 'dE/S']].abs().max(axis=1)
df.loc[:, 'order_of_magnitude'] = np.log10(df['max'] / df['min'])

# --- Clean quad column ---
df["quad"] = df["quad"].str.replace("_quadrat_", "", regex=False)

# # --- Format numbers with 2 decimals but without trailing zeros ---
# df = df.applymap(lambda x: f"{x:.2f}".rstrip("0").rstrip(".") if isinstance(x, (int, float)) else x)

# Apply formatting column by column
for col in df.columns:
    df[col] = df[col].apply(lambda x: format_value(x, col))

# --- Reorder columns (added slack_weight after quad and census) ---
cols = [
    "quad", "census", "slack_weight", "N/S", "E/S", "dN/S", "dE/S", "r2_dn", "r2_de",
    "METE_AIC", "METE_MAE", "METE_RMSE",
    "METimE_AIC", "METimE_MAE", "METimE_RMSE"
]
df = df[cols]

# --- Convert to LaTeX without headers ---
latex_table = df.to_latex(
    index=False,
    header=False,
    column_format="ccccccccc|ccc|ccc",  # one extra 'c' for slack_weight
    escape=False
)

# --- Build custom header (added slack_weight) ---
custom_header = (
    "\\toprule\n"
    " & & & & & & & & & \\multicolumn{3}{c|}{METE} & \\multicolumn{3}{c}{METimE} \\\\\n"
    "quad & census & slack\\_weight & N/S & E/S & dN/S & dE/S & r2_dn & r2_de & AIC & MAE & RMSE & AIC & MAE & RMSE \\\\\n"
    "\\midrule\n"
)

# --- Insert header ---
latex_table = latex_table.replace("\\toprule", custom_header, 1)

# --- Save to file ---
with open("table.tex", "w") as f:
    f.write(latex_table)

print(latex_table)

# --- Aggregated table: mean per slack_weight (averaged over quads & censuses) ---

# columns to average
metrics = [
    "N/S", "E/S", "dN/S", "dE/S", "r2_dn", "r2_de",
    "METE_AIC", "METE_MAE", "METE_RMSE",
    "METimE_AIC", "METimE_MAE", "METimE_RMSE"
]

# work on a numeric copy for grouping
gdf = df.copy()
gdf["slack_weight"] = pd.to_numeric(gdf["slack_weight"], errors="coerce")
for m in metrics:
    gdf[m] = pd.to_numeric(gdf[m], errors="coerce")

# group by slack_weight and take means
agg = (
    gdf.groupby("slack_weight", dropna=False)[metrics]
       .mean()
       .reset_index()
       .sort_values("slack_weight")
)

# add placeholder 'All' for quad/census and reorder columns
agg.insert(0, "quad", "All")
agg.insert(1, "census", "All")
agg = agg[cols]  # same cols list you already defined

# format for display (scientific only for slack_weight)
agg_display = agg.copy()
for c in agg_display.columns:
    agg_display[c] = agg_display[c].apply(lambda x: format_value(x, c))

# to LaTeX
latex_table_agg = agg_display.to_latex(
    index=False,
    header=False,
    column_format="ccccccccc|ccc|ccc",
    escape=False
)

# header (same as before; includes slack\_weight)
latex_table_agg = latex_table_agg.replace("\\toprule", custom_header, 1)

with open("table_aggregated_by_slack.tex", "w") as f:
    f.write(latex_table_agg)

print(latex_table_agg)