from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from data_loader import load_data
from preprocessing import clean_data, create_features


# Features used to predict electricity consumption
FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "previous_minute",
    "previous_hour",
    "previous_day",
    "rolling_mean_1hr",
    "rolling_std_1hr"
]

# Variable the model will predict
TARGET = "Global_active_power"


def prepare_data(data):
    """
    Input:
        data: Preprocessed electricity dataset.
    Process:
        Separate input features from the prediction target.
    Output:
        X: Feature dataset.
        y: Target values.
    """

    X = data[FEATURES]
    y = data[TARGET]

    return X, y


def split_data(X, y):
    """
    Input:
        X: Feature dataset.
        y: Target values.
    Process:
        Split data chronologically into 80% training
        and 20% testing.
    Output:
        X_train, X_test, y_train, y_test.
    """

    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Input:
        X_train: Training features.
        y_train: Training target values.
    Process:
        Train a Random Forest regression model.
    Output:
        Trained machine learning model.
    """

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Input:
        model: Trained regression model.
        X_test: Testing features.
        y_test: Actual electricity consumption.
    Process:
        Generate predictions and calculate evaluation metrics.
    Output:
        Predicted electricity consumption values.
    """

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(y_test, predictions)

    print("\n" + "=" * 50)
    print("ECOSENSE AI - MODEL PERFORMANCE")
    print("=" * 50)

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    return predictions


if __name__ == "__main__":

    print("Loading dataset...")
    data, _ = load_data()

    print("Cleaning dataset...")
    data = clean_data(data)

    print("Creating features...")
    data = create_features(data)

    print(f"Final dataset shape: {data.shape}")

    print("\nPreparing model inputs...")
    X, y = prepare_data(data)

    print("\nSplitting dataset...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples: {len(X_test):,}")

    print("\nTraining Random Forest model...")

    model = train_model(X_train, y_train)

    print("Model training completed!")

    predictions = evaluate_model(
        model,
        X_test,
        y_test
    )