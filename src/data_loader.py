from ucimlrepo import fetch_ucirepo


def load_data():
    dataset = fetch_ucirepo(id=235)

    X = dataset.data.features
    y = dataset.data.targets

    return X, y


if __name__ == "__main__":
    X, y = load_data()

    print("Features:")
    print(X.head())

    print("\nShape:")
    print(X.shape)

    print("\nColumns:")
    print(X.columns.tolist())