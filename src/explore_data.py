from ucimlrepo import fetch_ucirepo
import pandas as pd


def load_data():
    """Load the UCI Individual Household Electric Power Consumption dataset."""
    dataset = fetch_ucirepo(id=235)

    data = dataset.data.features.copy()

    return data


def explore_data(data):
    """Print basic information about the dataset."""

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"\nNumber of rows: {len(data):,}")
    print(f"Number of columns: {len(data.columns)}")

    print("\nColumns:")
    for column in data.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(data.dtypes)

    print("\nMissing values:")
    missing = data.isnull().sum()
    print(missing)

    print("\nFirst 5 rows:")
    print(data.head())

    print("\nNumerical summary:")
    print(data.describe())


if __name__ == "__main__":
    data = load_data()
    explore_data(data)