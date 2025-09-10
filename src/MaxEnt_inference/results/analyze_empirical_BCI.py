import pandas as pd
import matplotlib.pyplot as plt


# Load your CSV (comma delimiter, dot decimal)
df = pd.read_csv("C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/BCI/results_per_slack_weight.csv", delimiter=",", decimal=".")

# (Optional) check the first rows
print(df.head())

# Compute mean per slack_weight per census
grouped = df.groupby(["slack_weight", "census"]).agg(
    mean_MAE=("MAE", "mean"),
    mean_RMSE=("RMSE", "mean")
).reset_index()

# Compute overall mean per slack_weight
overall = grouped.groupby("slack_weight").agg(
    mean_MAE=("mean_MAE", "mean"),
    mean_RMSE=("mean_RMSE", "mean")
).reset_index()

# --- Plot MAE ---
plt.figure(figsize=(8, 5))

# Unique colors automatically assigned per census
for census, g in df.groupby("census"):
    # Solid line: METimE_MAE
    line, = plt.plot(g["slack_weight"], g["MAE"], label=f"Census {census}")
    color = line.get_color()


plt.xscale("log")
plt.title("MAE per Slack Weight (solid) with METE baseline (dotted)")
plt.xlabel("Slack Weight")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.show()

# --- Plot RMSE ---
plt.figure(figsize=(8, 5))

for census, g in df.groupby("census"):
    line, = plt.plot(g["slack_weight"], g["RMSE"], label=f"Census {census}")
    color = line.get_color()

plt.xscale("log")
plt.title("RMSE per Slack Weight (solid) with METE baseline (dotted)")
plt.xlabel("Slack Weight")
plt.ylabel("RMSE")
plt.legend()
plt.grid(True)
plt.show()

#
# # --- Format numbers ---
# def format_value(x, col=None):
#     if isinstance(x, (int, float)):
#         if col == "slack_weight":
#             return f"{x:.1e}"  # scientific notation, 1 decimal
#         else:
#             return f"{x:.2f}".rstrip("0").rstrip(".")  # normal 2-decimal format
#     return x
#
# path = r'C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/BCI/empirical_BCI_df'
# all_files = glob.glob(os.path.join(path, "*.csv"))
# df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
#
# # Find order of magnitude difference
# df.loc[:, 'min'] = df[['N/S', 'E/S', 'dN/S', 'dE/S']].abs().min(axis=1)
# df.loc[:, 'max'] = df[['N/S', 'E/S', 'dN/S', 'dE/S']].abs().max(axis=1)
# df.loc[:, 'order_of_magnitude'] = np.log10(df['max'] / df['min'])
#
# # --- Clean quad column ---
# df["quad"] = df["quad"].str.replace("_quadrat_", "", regex=False)
#
# # # --- Format numbers with 2 decimals but without trailing zeros ---
# # df = df.applymap(lambda x: f"{x:.2f}".rstrip("0").rstrip(".") if isinstance(x, (int, float)) else x)
#
# # Apply formatting column by column
# for col in df.columns:
#     df[col] = df[col].apply(lambda x: format_value(x, col))
#
# # --- Reorder columns (added slack_weight after quad and census) ---
# cols = [
#     "quad", "census", "slack_weight", "N/S", "E/S", "dN/S", "dE/S", "r2_dn", "r2_de",
#     "METE_AIC", "METE_MAE", "METE_RMSE",
#     "METimE_AIC", "METimE_MAE", "METimE_RMSE"
# ]
# df = df[cols]
#
# # --- Convert to LaTeX without headers ---
# latex_table = df.to_latex(
#     index=False,
#     header=False,
#     column_format="ccccccccc|ccc|ccc",  # one extra 'c' for slack_weight
#     escape=False
# )
#
# # --- Build custom header (added slack_weight) ---
# custom_header = (
#     "\\toprule\n"
#     " & & & & & & & & & \\multicolumn{3}{c|}{METE} & \\multicolumn{3}{c}{METimE} \\\\\n"
#     "quad & census & slack\\_weight & N/S & E/S & dN/S & dE/S & r2_dn & r2_de & AIC & MAE & RMSE & AIC & MAE & RMSE \\\\\n"
#     "\\midrule\n"
# )
#
# # --- Insert header ---
# latex_table = latex_table.replace("\\toprule", custom_header, 1)
#
# # --- Save to file ---
# with open("table.tex", "w") as f:
#     f.write(latex_table)
#
# print(latex_table)
#
# # --- Aggregated table: mean per slack_weight (averaged over quads & censuses) ---
#
# # columns to average
# metrics = [
#     "N/S", "E/S", "dN/S", "dE/S", "r2_dn", "r2_de",
#     "METE_AIC", "METE_MAE", "METE_RMSE",
#     "METimE_AIC", "METimE_MAE", "METimE_RMSE"
# ]
#
# # work on a numeric copy for grouping
# gdf = df.copy()
# gdf["slack_weight"] = pd.to_numeric(gdf["slack_weight"], errors="coerce")
# for m in metrics:
#     gdf[m] = pd.to_numeric(gdf[m], errors="coerce")
#
# # group by slack_weight and take means
# agg = (
#     gdf.groupby("slack_weight", dropna=False)[metrics]
#        .mean()
#        .reset_index()
#        .sort_values("slack_weight")
# )
#
# # add placeholder 'All' for quad/census and reorder columns
# agg.insert(0, "quad", "All")
# agg.insert(1, "census", "All")
# agg = agg[cols]  # same cols list you already defined
#
# # format for display (scientific only for slack_weight)
# agg_display = agg.copy()
# for c in agg_display.columns:
#     agg_display[c] = agg_display[c].apply(lambda x: format_value(x, c))
#
# # to LaTeX
# latex_table_agg = agg_display.to_latex(
#     index=False,
#     header=False,
#     column_format="ccccccccc|ccc|ccc",
#     escape=False
# )
#
# # header (same as before; includes slack\_weight)
# latex_table_agg = latex_table_agg.replace("\\toprule", custom_header, 1)
#
# with open("table_aggregated_by_slack.tex", "w") as f:
#     f.write(latex_table_agg)
#
# print(latex_table_agg)