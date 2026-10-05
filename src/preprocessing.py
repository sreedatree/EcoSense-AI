import pandas as pd


NUMERIC_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def clean_data(data):
    """
    Clean the raw household electricity dataset.
    """

    data = data.copy()

    # Convert electricity measurements to numeric values
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Combine Date and Time into one datetime column
    data["datetime"] = pd.to_datetime(
        data["Date"] + " " + data["Time"],
        dayfirst=True,
        errors="coerce"
    )

    # Sort chronologically
    data = data.sort_values("datetime")

    # Remove rows where datetime could not be created
    data = data.dropna(subset=["datetime"])

    # Fill missing Sub_metering_3 values using interpolation
    data["Sub_metering_3"] = data["Sub_metering_3"].interpolate()

    # Remove any remaining missing values
    data = data.dropna()

    return data


def create_features(data):
    """
    Create time-based and historical features for machine learning.
    """

    data = data.copy()

    # Time-based features
    data["hour"] = data["datetime"].dt.hour
    data["day_of_week"] = data["datetime"].dt.dayofweek
    data["month"] = data["datetime"].dt.month
    data["day_of_year"] = data["datetime"].dt.dayofyear

    # Historical electricity usage
    data["previous_hour"] = data["Global_active_power"].shift(60)

    data["previous_day"] = data["Global_active_power"].shift(1440)

    # Rolling statistics
    data["rolling_mean_1hr"] = (
        data["Global_active_power"]
        .rolling(window=60)
        .mean()
    )

    data["rolling_std_1hr"] = (
        data["Global_active_power"]
        .rolling(window=60)
        .std()
    )

    # Remove rows created by lag/rolling calculations
    data = data.dropna()

    return data


if __name__ == "__main__":
    from data_loader import load_data

    print("Loading dataset...")
    data, _ = load_data()

    print(f"Raw dataset shape: {data.shape}")

    print("\nCleaning data...")
    data = clean_data(data)

    print(f"Cleaned dataset shape: {data.shape}")

    print("\nCreating features...")
    data = create_features(data)

    print(f"Final dataset shape: {data.shape}")

    print("\nFeatures:")
    print(data.columns.tolist())

    print("\nSample:")
    print(data.head())

    print("\nData types:")
    print(data.dtypes)