"""Utility functions for flow cytometry analysis."""

import pandas as pd

def parse_plate_map(file_path):
    """
    Parse a spreadsheet in 96-well plate format and return a dictionary
    with well names (A1-H12) as keys and sample names as values.

    Args:
        file_path (str): Path to the spreadsheet file.

    Returns:
        dict: Dictionary with well names as keys and sample names as values.
    """
    plate_df = pd.read_csv(file_path, index_col=0)
    return {f"{row}{col}": plate_df.loc[row, col]
            for row in plate_df.index
            for col in plate_df.columns
            if not pd.isna(plate_df.loc[row, col])}
