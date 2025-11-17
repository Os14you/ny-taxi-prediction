import pandas as pd
from haversine import haversine

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract day of year, week, and hour."""

    df['pickup_day_of_year'] = df['pickup_datetime'].dt.dayofyear
    df['pickup_day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['pickup_hour_of_day'] = df['pickup_datetime'].dt.hour

    return df

def add_anomaly_feature(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Identify high/low traffic volume days."""

    df['pickup_date'] = df['pickup_datetime'].dt.date
    daily_trips = df.groupby('pickup_date').size().reset_index(name='trip_count')

    mean = daily_trips['trip_count'].mean()
    std = daily_trips['trip_count'].std()

    upper = mean + threshold * std
    lower = mean - threshold * std

    anomalies = daily_trips[
        (daily_trips['trip_count'] > upper) | 
        (daily_trips['trip_count'] < lower)
    ]['pickup_date']

    df['is_anomaly'] = 0
    df.loc[df['pickup_date'].isin(anomalies), 'is_anomaly'] = 1
    
    df.drop(columns=['pickup_date'], inplace=True)
    
    return df

def add_trip_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the Haversine distance between pickup and dropoff points."""

    def _calc_dist(row):
        pickup = (row['pickup_latitude'], row['pickup_longitude'])
        dropoff= (row['dropoff_latitude'], row['dropoff_longitude'])
        return haversine(pickup, dropoff)
    
    df['trip_distance'] = df.apply(_calc_dist, axis=1)
    return df

def add_traffic_congestion(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a feature of traffic congestion based on the notebook logic."""
    df['traffic_congestion'] = 0

    cong_mask = (
        (df['pickup_day_of_week'] >= 1) &
        (df['pickup_day_of_week'] <= 4) &
        (df['pickup_hour_of_day'] >= 8) &
        (df['pickup_hour_of_day'] <= 18)
    )

    df.loc[cong_mask, 'traffic_congestion'] = 1
    return df

def add_airport_features(df: pd.DataFrame, airport_config: dict, radius: float) -> pd.DataFrame:
    """Creates a flag for pickup/dropoff proximity to airports."""

    def _is_near_any_airport(lat, lon):
        for airport, coords in airport_config.items():
            airport_coords = tuple(coords)
            if haversine((lat, lon), airport_coords) <= radius:
                return 1
        return 0
    
    df['pickup_near_airport'] = df.apply(
        lambda x: _is_near_any_airport(x['pickup_latitude'], x['pickup_longitude']), axis=1
    )

    df['dropoff_near_airport'] = df.apply(
        lambda x: _is_near_any_airport(x['dropoff_latitude'], x['dropoff_longitude']), axis=1
    )

    return df