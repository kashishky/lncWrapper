# preprocessing/load_preprocessed.py

import os
import pandas as pd
import numpy as np

def load_summary():
    """
    Load 'data/preprocessed/summary.csv' into a pandas DataFrame.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    summary_path = os.path.join(project_root, 'data', 'preprocessed', 'summary.csv')
    return pd.read_csv(summary_path)

def load_base_array(transcript_id):
    """
    Load 'data/preprocessed/base_arrays/{transcript_id}.npz' and return its arrays.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(
        project_root, 'data', 'preprocessed', 'base_arrays', f"{transcript_id}.npz"
    )
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No base array file found for '{transcript_id}'")
    return np.load(file_path)
