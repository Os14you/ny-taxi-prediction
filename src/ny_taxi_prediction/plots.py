import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import folium
from pathlib import Path

sns.set_theme()

def save_plot(fig, output_dir: Path, filename: str):
    """Helper to save matplotlib figures to the figures directory."""

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(figures_dir / filename)
    plt.close(fig)

def save_map(m, output_dir: Path, filename: str):
    """Helper to save folium maps to the maps directory."""
    maps_dir = output_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    m.save(maps_dir / filename)

def plot_daily_trip_counts(df: pd.DataFrame, output_dir: Path, threshold: float = 1.6):
    """Generates a time-series plot of daily trip counts with anomaly thresholds."""

    daily_trips = df.groupby(df['pickup_datetime'].dt.date).size().reset_index(name='trip_count')
    daily_trips['pickup_date'] = pd.to_datetime(daily_trips['pickup_datetime'])
    
    mean_trips = daily_trips['trip_count'].mean()
    std_trips = daily_trips['trip_count'].std()
    
    upper_bound = mean_trips + (threshold * std_trips)
    lower_bound = mean_trips - (threshold * std_trips)
    
    anomalies = daily_trips[
        (daily_trips['trip_count'] > upper_bound) | 
        (daily_trips['trip_count'] < lower_bound)
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_trips['pickup_date'], daily_trips['trip_count'], label='Daily trip count')
    ax.axhline(y=lower_bound, color='g', linestyle='--', label='Lower threshold')
    ax.axhline(y=upper_bound, color='g', linestyle='--', label='Upper threshold')
    ax.scatter(anomalies['pickup_date'], anomalies['trip_count'], color='red', label='Anomalies', zorder=5)
    
    ax.set_title('Daily Trip Count Anomalies')
    ax.set_xlabel('Date')
    ax.set_ylabel('Trip Count')
    ax.legend()
    
    save_plot(fig, output_dir, "daily_trip_anomalies.png")

def plot_traffic_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Calculates average speed per Day/Hour and plots a heatmap."""

    viz_df = df.copy()
    viz_df['speed'] = viz_df['trip_distance'] / (viz_df['trip_duration'] / 3600)
    
    viz_df = viz_df[viz_df['speed'] < 100]

    heatmap_data = viz_df.groupby(['pickup_day_of_week', 'pickup_hour_of_day'])['speed'].mean().unstack()
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='viridis', annot=False)
    plt.title('Average Traffic Speed (km/h) by Day and Hour')
    plt.ylabel('Day of Week (0=Mon, 6=Sun)')
    plt.xlabel('Hour of Day')
    
    output_path = output_dir / "traffic_heatmap.png"
    plt.savefig(output_path)
    plt.close()

def plot_duration_distribution(df: pd.DataFrame, output_dir: Path):
    """Plots the histogram of trip durations (log-scale)."""

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df['trip_duration'], bins=100)
    ax.set_title('Log(Trip Duration) Distribution')
    ax.set_xlabel('Log(Seconds)')
    ax.set_ylabel('Frequency')
    
    save_plot(fig, output_dir, "trip_duration_log_hist.png")

def create_folium_map(df: pd.DataFrame, output_dir: Path, sample_size: int = 1000):
    """Creates a Folium map with a sample of pickup locations."""

    if len(df) > sample_size:
        sample = df.sample(sample_size)
    else:
        sample = df
        
    lat_col, lon_col = 'pickup_latitude', 'pickup_longitude'
    
    center_lat = sample[lat_col].mean()
    center_lon = sample[lon_col].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    
    for _, row in sample.iterrows():
        folium.Circle(
            radius=50,
            location=(row[lat_col], row[lon_col]),
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(m)
        
    save_map(m, output_dir, "pickup_locations_map.html")