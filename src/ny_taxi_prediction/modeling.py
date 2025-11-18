import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from typing import Tuple, Dict

def split_data(df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the dataframe into training and testing sets based on params."""

    target_col = params['modeling']['target']
    test_size = params['data']['test_size']
    random_state = params['preprocessing']['random_state']

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)

    return X_train, X_test, y_train, y_test

def preprocess_features(X_train: pd.DataFrame, X_test: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """
    Applies StandardScaler and OneHotEncoding to features.
    Renames columns to match the notebook's naming convention (e.g., 'pickup_on_Monday').
    Filters columns to keep only the 'selected_features' defined in params.
    """

    numerical_cols = params['preprocessing']['numerical_cols']
    categorical_cols = params['preprocessing']['categorical_cols']
    selected_features = params['modeling']['selected_features']

    preprocessor = ColumnTransformer(
        transformers=[
            ('scaling', StandardScaler(), numerical_cols),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        verbose_feature_names_out=False
    )
    
    preprocessor.set_output(transform="pandas")

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    day_mapping = {
        'pickup_day_of_week_0': 'pickup_on_Monday',
        'pickup_day_of_week_1': 'pickup_on_Tuesday',
        'pickup_day_of_week_2': 'pickup_on_Wednesday',
        'pickup_day_of_week_3': 'pickup_on_Thursday',
        'pickup_day_of_week_4': 'pickup_on_Friday',
        'pickup_day_of_week_5': 'pickup_on_Saturday',
        'pickup_day_of_week_6': 'pickup_on_Sunday'
    }
    
    X_train_processed = X_train_processed.rename(columns=day_mapping)
    X_test_processed = X_test_processed.rename(columns=day_mapping)

    final_cols = [c for c in selected_features if c in X_train_processed.columns]
    return X_train_processed[final_cols], X_test_processed[final_cols], preprocessor

def train_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> DecisionTreeRegressor:
    """Trains a Decision Tree Regressor using parameters from params.yaml."""

    dt_params = params['model_params']
    
    model = DecisionTreeRegressor(
        max_depth=dt_params['max_depth'],
        max_features=dt_params['max_features'],
        min_samples_leaf=dt_params['min_samples_leaf'],
        random_state=dt_params['random_state']
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: DecisionTreeRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluates the model and returns performance metrics (MSE, RMSE, R2)."""

    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }