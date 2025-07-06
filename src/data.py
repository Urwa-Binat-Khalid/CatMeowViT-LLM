# data.py

import os
import pandas as pd
from sklearn.utils import resample

from config import DATA_DIR, CSV_PATH

def build_dataframe():
    """
    Build a DataFrame from audio files in the specified DATA_DIR.
    It scans the directory for .wav files and assigns a label based on the filename prefix.
    """
    filepaths, labels = [], []

    # Iterate through files in the data directory
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".wav"):
            path = os.path.join(DATA_DIR, fname)
            filepaths.append(path)

            # Assign label based on filename prefix
            if fname.startswith("B"):
                labels.append("Brushing")
            elif fname.startswith("F"):
                labels.append("WaitingForFood")
            elif fname.startswith("I"):
                labels.append("Isolation")
            else:
                labels.append("Unknown")

    # Create a DataFrame with file paths and labels
    df = pd.DataFrame({"file_path": filepaths, "label": labels})
    return df

def balance_dataframe(df):
    """
    Balance the dataset by downsampling classes so that each class
    has the same number of samples as the 'WaitingForFood' class.
    The balanced DataFrame is saved as a CSV.
    """
    # Separate DataFrame by label
    df_iso = df[df['label'] == 'Isolation']
    df_brush = df[df['label'] == 'Brushing']
    df_food = df[df['label'] == 'WaitingForFood']

    # Use the size of the 'WaitingForFood' class as target count
    target_count = len(df_food)

    # Downsample 'Isolation' and 'Brushing' to match target count
    df_iso_down = resample(df_iso, replace=False, n_samples=target_count, random_state=42)
    df_brush_down = resample(df_brush, replace=False, n_samples=target_count, random_state=42)

    # Combine balanced DataFrames and shuffle
    df_balanced = pd.concat([df_iso_down, df_brush_down, df_food])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save balanced DataFrame to CSV
    df_balanced.to_csv(CSV_PATH, index=False)
    return df_balanced
