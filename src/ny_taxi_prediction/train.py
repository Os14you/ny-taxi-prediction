import joblib
from ny_taxi_prediction.utils import load_config
from ny_taxi_prediction.data import load_data, clean_data
from ny_taxi_prediction.features import (
    add_temporal_features, add_trip_distance, 
    add_traffic_congestion, add_airport_features, add_anomaly_feature
)
from ny_taxi_prediction.modeling import split_data, preprocess_features, train_model, evaluate_model

def build_pipeline():
    """The entre point of the project, where we clean and process the data and then train the DT model."""

    params = load_config("params.yaml")

    df = load_data(params['data']['raw_path'], apply_date_conv=True)
    df = clean_data(df, params)

    df = add_temporal_features(df)
    df = add_trip_distance(df)
    df = add_traffic_congestion(df)
    df = add_airport_features(df, params['features']['airports'], params['features']['airport_radius'])
    df = add_anomaly_feature(df, params['cleaning']['anomaly_threshold'])

    X_train, X_test, y_train, y_test = split_data(df, params)
    X_train_proc, X_test_proc, preprocessor = preprocess_features(X_train, X_test, params)

    model = train_model(X_train_proc, y_train, params)
    
    metrics = evaluate_model(model, X_test_proc, y_test)
    print(f"Model Metrics: {metrics}")

    joblib.dump(model, "models/model.joblib")
    joblib.dump(preprocessor, "models/preprocessor.joblib")

if __name__ == "__main__":
    build_pipeline()