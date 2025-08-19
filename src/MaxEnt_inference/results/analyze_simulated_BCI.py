import pandas as pd
import glob
import os

# Step 1: Read all CSVs into a list
dfs = []
for file in glob.glob("C:/Users/5605407/OneDrive - Universiteit Utrecht/Documents/PhD/Chapter_2/Results/BCI/simulated_BCI_df/*.csv"):
    filename = os.path.splitext(os.path.basename(file))[0]
    frac_removed = filename.replace("simulated_BCI_", "")
    df = pd.read_csv(file)
    df["fraction_removed"] = frac_removed
    dfs.append(df)

# Step 2: Concatenate
combined = pd.concat(dfs, ignore_index=True)

# Step 3: Drop iter column if present
if "iter" in combined.columns:
    combined = combined.drop(columns=["iter"])

# Step 4: Drop census and average over it
combined = combined.drop(columns=["census"]).groupby("fraction_removed", as_index=False).mean()

# Step 5: Reorder columns (including r2_dn and r2_de)
cols = ["fraction_removed", "METE_AIC", "METE_MAE", "METE_RMSE",
        "METimE_AIC", "METimE_MAE", "METimE_RMSE",
        "r2_dn", "r2_de"]
combined = combined[cols]

# Step 6: Convert to LaTeX with multicolumn headers
latex_table = combined.to_latex(
    index=False,
    float_format="{:.3f}".format,
    column_format="lccc ccc cc",
    header=["Fraction Removed", "AIC", "MAE", "RMSE", "AIC", "MAE", "RMSE", "R2_DN", "R2_DE"]
)

# Insert multicolumn lines manually
lines = latex_table.splitlines()
header_line = " \\multicolumn{1}{c}{} & \\multicolumn{3}{c}{METE} & \\multicolumn{3}{c}{METimE} & \\multicolumn{2}{c}{R2} \\\\"
cline_line = " \\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-9}"

# Rebuild table with added header rows
latex_table = "\n".join([lines[0], header_line, cline_line] + lines[1:])

with open("combined_table.tex", "w") as f:
    f.write(latex_table)

print(latex_table)