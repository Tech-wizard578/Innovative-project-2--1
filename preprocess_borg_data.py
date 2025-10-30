import pandas as pd
import numpy as np
import os
import json # For parsing stringified dictionary columns

# --- Configuration ---
INPUT_FILE = os.path.join("data", "borg_traces_data.csv")
OUTPUT_FILE = os.path.join("data", "processed_borg_data_v1.csv") # Save processed data here
# Define the features your models expect
EXPECTED_FEATURES = [
    'process_size', 'priority', 'arrival_time',
    'process_type', 'time_of_day'
]
TARGET_VARIABLE = 'burst_time'

print(f"Loading data from: {os.path.abspath(INPUT_FILE)}")

try:
    # --- Load Data ---
    # Load only necessary columns initially to save memory
    columns_to_load = [
        'time', 'priority', 'start_time', 'end_time',
        'resource_request', 'average_usage', 'assigned_memory',
        'scheduling_class', 'event' # Needed for filtering
    ]
    df = pd.read_csv(INPUT_FILE, usecols=columns_to_load)
    print(f"Loaded {len(df)} rows.")

    # --- Preprocessing Steps ---

    # 1. Filter for relevant events: We only care about completed tasks ('FINISH')
    #    to calculate their duration.
    df_finished = df[df['event'] == 'FINISH'].copy()
    print(f"Filtered down to {len(df_finished)} FINISH events.")
    if len(df_finished) == 0:
        raise ValueError("No 'FINISH' events found. Cannot calculate burst times.")

    # 2. Calculate Burst Time (Duration)
    #    Assuming start_time and end_time are timestamps (e.g., microseconds)
    #    Convert to a more manageable unit if needed (e.g., seconds or milliseconds)
    #    Handle potential cases where end_time <= start_time
    df_finished['duration'] = df_finished['end_time'] - df_finished['start_time']
    # Filter out invalid durations (e.g., duration <= 0)
    df_finished = df_finished[df_finished['duration'] > 0]
    # Convert duration to seconds (assuming original is microseconds) - ADJUST IF NEEDED
    df_finished[TARGET_VARIABLE] = df_finished['duration'] / 1_000_000
    print(f"Calculated burst_time (duration). Kept {len(df_finished)} rows with valid duration > 0.")

    # 3. Handle 'arrival_time'
    #    Let's use the 'time' column associated with the FINISH event for now.
    #    Alternatively, one could try to find the corresponding SUBMIT event time.
    df_finished['arrival_time'] = df_finished['time'] # Using FINISH event time as a proxy

    # 4. Handle 'time_of_day'
    #    Requires converting the 'arrival_time' timestamp to a datetime object
    #    Assuming the timestamp is microseconds since epoch - ADJUST IF NEEDED
    try:
        timestamps_s = df_finished['arrival_time'] / 1_000_000 # Convert to seconds
        datetimes = pd.to_datetime(timestamps_s, unit='s')
        df_finished['time_of_day'] = datetimes.dt.hour # Extract hour of the day (0-23)
    except Exception as e:
        print(f"Warning: Could not parse time_of_day from 'time'. Setting to 0. Error: {e}")
        df_finished['time_of_day'] = 0


    # 5. Handle 'priority'
    #    The 'priority' column seems directly usable.
    #    Make sure values are within a reasonable range if needed by your model.
    df_finished['priority'] = df_finished['priority'] # Already exists

    # 6. Handle 'process_size'
    #    Let's use 'assigned_memory' as a proxy.
    #    Handle potential NaN values (if any). Fill with 0 or mean/median.
    df_finished['process_size'] = df_finished['assigned_memory'].fillna(0)


    # 7. Infer 'process_type' (Example - needs refinement)
    #    This is complex. Let's use 'scheduling_class' as a simple proxy for now.
    #    You might need a more sophisticated method based on resource usage.
    #    Mapping: 0=CPU, 1=IO, 2=Mixed, 3=Interactive (adjust as needed)
    df_finished['process_type'] = df_finished['scheduling_class'] % 4 # Simple modulo mapping
    print("Inferred process_type based on scheduling_class (simple mapping).")

    # --- Final Selection and Cleaning ---

    # Keep only the required features and the target variable
    final_columns = EXPECTED_FEATURES + [TARGET_VARIABLE]
    df_final = df_finished[final_columns].copy()

    # Drop rows with any remaining NaN values in critical columns
    df_final.dropna(subset=final_columns, inplace=True)
    print(f"Kept {len(df_final)} rows after final NaN drop.")

    # Optional: Clip burst_time to a reasonable range if needed
    # df_final[TARGET_VARIABLE] = df_final[TARGET_VARIABLE].clip(lower=0.1) # e.g., minimum 0.1 seconds

    # --- Save Processed Data ---
    if not df_final.empty:
        df_final.to_csv(OUTPUT_FILE, index=False)
        print(f"\nProcessed data saved successfully to: {os.path.abspath(OUTPUT_FILE)}")
        print("\nFinal Data Head:")
        print(df_final.head())
        print("\nFinal Data Info:")
        df_final.info()
    else:
        print("\nError: No data remaining after preprocessing.")


except FileNotFoundError:
    print(f"\nError: Input file not found at '{INPUT_FILE}'.")
    print("Please ensure 'borg_traces_data.csv' is inside the 'data' folder.")
except KeyError as e:
    print(f"\nError: A required column is missing from the CSV: {e}")
except Exception as e:
    print(f"\nAn unexpected error occurred during preprocessing: {e}")