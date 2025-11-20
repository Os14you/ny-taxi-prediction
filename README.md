# NYC Taxi Trip Duration Prediction

An end-to-end Machine Learning project designed to predict the duration of taxi trips in New York City. This repository demonstrates a production-ready pipeline structure, transitioning from Jupyter Notebooks to modular Python code.

## All Features

* **Modular Pipeline**: organized into distinct stages for data loading, cleaning, feature engineering, and modeling.
* **Geospatial Engineering**: Uses **Haversine** distance to calculate trip lengths and identifies proximity to major NYC airports (JFK, LGA, EWR).
* **Traffic Insights**: Features logic to detect traffic congestion based on day-of-week and hour-of-day.
* **Config-Driven**: All parameters (paths, hyperparameters, feature selection) are managed centrally via `params.yaml`.
* **Reproducible Environment**: Built using **uv** for fast and reliable dependency management.

## Project Structure

* `src/ny_taxi_prediction/`: Core source code.
  * `data.py`: Data loading and cleaning (outlier removal, log-transform targets).
  * `features.py`: Temporal extraction, holiday detection, and coordinate-based features.
  * `modeling.py`: Training and evaluating the Decision Tree Regressor.
  * `plots.py`: Utilities for generating traffic heatmaps and anomaly detection plots.
* `params.yaml`: Central configuration file for reproducible runs.

## Tech Stack

* **Python 3.12+**
* **Scikit-Learn** (Decision Tree Regressor, Pipeline Transformers)
* **Pandas & NumPy** (Data Manipulation)
* **Haversine** (Geospatial calculations)
* **Joblib** (Model persistence)
