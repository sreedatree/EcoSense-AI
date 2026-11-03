from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CACHE_FILE = RAW_DATA_DIR / "household_power_consumption.parquet"


def load_data(cache_path=CACHE_FILE):
    """
    Input:
        cache_path (Path, optional): Local file path for cached parquet dataset.
    Processing:
        Check if a local cached copy exists to avoid repeated large downloads.
        If available, load directly from disk.
        Otherwise, fetch from the UCI Machine Learning Repository (id=235),
        cache the raw features locally as parquet, and return.
    Output:
        X (pd.DataFrame): Raw electricity measurements dataset.
        y (None or pd.DataFrame): Target values (if provided by repository).
    """
    if cache_path.exists():
        print(f"Loading cached dataset from {cache_path}...")
        X = pd.read_parquet(cache_path)
        y = None
        return X, y

    print("Downloading dataset from UCI Machine Learning Repository (id=235)...")
    dataset = fetch_ucirepo(id=235)

    X = dataset.data.features
    y = dataset.data.targets

    # Cache locally to accelerate future runs
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    X.astype(str).to_parquet(cache_path, index=False)
    print(f"Dataset successfully cached to {cache_path}")

    return X, y


if __name__ == "__main__":
    X, y = load_data()

    print("Features:")
    print(X.head())

    print("\nShape:")
    print(X.shape)

    print("\nColumns:")
    print(X.columns.tolist())
