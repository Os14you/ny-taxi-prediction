import pandas as pd
import numpy as np

def load_data(csv_path: str, apply_date_conv: bool = False) -> pd.DataFrame:
    """Load raw datasets and return a pandas dataframe."""
    df = pd.read_csv(csv_path)
    if apply_date_conv:
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    return df

def clean_data(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Remove passenger outliers and filter trip duration based on log limits."""
    
    outliers = params['cleaning']['passenger_outliers']
    df = df[~df['passenger_count'].isin(outliers)].copy()

    df['trip_duration'] = df['trip_duration'].apply(np.log1p)
    min_dur = params['cleaning']['trip_duration_log_min']
    max_dur = params['cleaning']['trip_duration_log_max']

    filter_normd = (df['trip_duration'] >= min_dur) & (df['trip_duration'] <= max_dur)
    df = df[filter_normd]

    return df

if __name__ == '__main__':
    df = load_data('data/raw/train.csv', True)
    print("-> Note <-: testing the rest of the methods in this file is not available right now")