import os
import pandas as pd
from tabulate import tabulate



def process_csv_files(folder_path, output_folder):
    # Define the prefixes to search for
    data_sets = ['birds', 'fish', 'BCI']
    scaling = ['Scaled', 'Unscaled']

    # Create a dictionary to store dataframes based on (dataset, scaling)
    dfs_dict = {(ds, s): [] for ds in data_sets for s in scaling}

    # List all files in the folder that match the required prefixes
    selected_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    for file in selected_files:
        for ds in data_sets:
            for s in scaling:
                prefix = f"{ds}_{s}_"
                if file.startswith(prefix):
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_csv(file_path)

                    # Ensure the "Method" column exists
                    if "Method" not in df.columns:
                        print(f"Skipping {file}, missing 'Method' column")
                        continue

                    # Store in the corresponding list
                    dfs_dict[(ds, s)].append(df)
                    break  # No need to check other prefixes once matched

    # Process and compute means
    processed_dfs = {}
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    for (dataset, scaling), dfs in dfs_dict.items():
        if dfs:  # Ensure there are files for this (dataset, scaling) pair
            combined_df = pd.concat(dfs)
            result_df = combined_df.groupby("Method").mean().reset_index()
            processed_dfs[(dataset, scaling)] = result_df  # Store the processed DataFrame

            # Save to CSV
            output_filename = f"{dataset}_{scaling}_aggregated_results.csv"
            output_path = os.path.join(output_folder, output_filename)
            result_df.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")

    return processed_dfs  # Return dictionary of DataFrames

# # # Usage example
# # folder_path = "../../results"
# output_folder = "../../aggregated_results"
# # result_dict = process_csv_files(folder_path, output_folder)
# #
# # # Print each processed DataFrame
# # for (dataset, scaling), df in result_dict.items():
# #     print(f"\nProcessed Data for {dataset} - {scaling}:\n")
# #     print(df)
#
# df = pd.read_csv(output_folder + "/fish_Scaled_aggregated_results.csv", decimal='.')
# df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
# latex_table = tabulate(df, headers='keys', tablefmt='latex', showindex=False)
#
# # with open("table.tex", "w") as f:
# #     f.write(latex_table)
#
# print(latex_table)

def process_csv_files_to_excel(folder_path, output_file):
    """Processes CSV files and saves aggregated results into an Excel file with multiple sheets."""

    # Define the prefixes to search for
    data_sets = ['birds', 'fish', 'BCI']
    scaling = ['Scaled', 'Unscaled']

    # Create a dictionary to store dataframes based on (dataset, scaling)
    dfs_dict = {(ds, s): [] for ds in data_sets for s in scaling}

    # List all files in the folder that match the required prefixes
    selected_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    for file in selected_files:
        for ds in data_sets:
            for s in scaling:
                prefix = f"{ds}_{s}_"
                if file.startswith(prefix):
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_csv(file_path, decimal=".")

                    # Store in the corresponding list
                    dfs_dict[(ds, s)].append(df)
                    break

    # Process and compute means
    processed_dfs = {}

    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        for (dataset, scaling), dfs in dfs_dict.items():
            if dfs:  # Ensure there are files for this (dataset, scaling) pair
                combined_df = pd.concat(dfs)
                result_df = combined_df.groupby("Method").mean().reset_index()

                # Remove unnamed columns
                result_df = result_df.loc[:, ~result_df.columns.str.contains('^Unnamed')]

                # Store the processed DataFrame
                processed_dfs[(dataset, scaling)] = result_df

                # Write to an Excel sheet
                sheet_name = f"{dataset}_{scaling}"
                result_df.to_excel(writer, float_format="%.3f", sheet_name=sheet_name, index=False)

                print(f"Saved to Excel: {sheet_name}")

    return processed_dfs  # Return dictionary of DataFrames


# Example Usage
folder_path = "../../results"
output_excel = "../../aggregated_results.xlsx"
processed_results = process_csv_files_to_excel(folder_path, output_excel)

print(f"Excel file saved at: {output_excel}")