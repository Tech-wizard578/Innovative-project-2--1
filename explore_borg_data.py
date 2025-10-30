import pandas as pd
import os

# Define the path to your downloaded file
file_path = os.path.join("data", "borg_traces_data.csv") # Assumes it's in the 'data' subfolder

print(f"Attempting to load data from: {os.path.abspath(file_path)}")

try:
    # --- Load the CSV ---
    # Note: This might take a minute or two because the file is large.
    # If it takes too long or uses too much memory, we can load just a sample later.
    df = pd.read_csv(file_path)

    # --- Display Basic Info ---
    print("\n" + "="*50)
    print("First 5 Rows:")
    print("="*50)
    print(df.head()) # Shows the first few rows and column names

    print("\n" + "="*50)
    print("Column Names and Data Types:")
    print("="*50)
    df.info(verbose=True, show_counts=True) # Shows all columns, types, and non-null counts

    # Optional: Get a statistical summary of numerical columns
    # print("\n" + "="*50)
    # print("Statistical Summary:")
    # print("="*50)
    # print(df.describe())

except FileNotFoundError:
    print(f"\nError: File not found at '{file_path}'.")
    print("Please ensure 'borg_traces_data.csv' is inside the 'data' folder within your project.")
except Exception as e:
    print(f"\nAn error occurred while loading or inspecting the data: {e}")