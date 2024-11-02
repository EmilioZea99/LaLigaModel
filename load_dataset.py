import os

import pandas as pd

# Define the path to the dataset folder
dataset_folder = "la-liga-results-19952020"

# Verify that the folder exists
if os.path.exists(dataset_folder):
    # List all files in the folder to find the CSV file
    files = os.listdir(dataset_folder)
    csv_files = [file for file in files if file.endswith(".csv")]

    if csv_files:
        # Load the first CSV file found
        csv_file_path = os.path.join(dataset_folder, csv_files[0])
        df = pd.read_csv(csv_file_path)

        # Print the first few rows of the dataset
        print(df.head())
    else:
        print("No CSV files found in the dataset folder.")
else:
    print("Dataset folder not found. Please check the folder name.")
