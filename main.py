from pathlib import Path

import matplotlib.pyplot as plt
import names
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from rich import print as rprint
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def generate_random_distance_matrix(n):
    """
    Generates n random points in a 3D space and computes their distance matrix.

    Parameters:
    - n: The number of points to generate.

    Returns:
    - points: An nx3 numpy array where each row represents a point in 3D space.
    - distance_matrix: An nxn numpy array representing the Euclidean distance matrix.
    """
    # Generate n random points in 3D space
    points = np.random.rand(n, 3)

    # Compute the distance matrix
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    distance_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    return distance_matrix


def generate_random_dataset(n_stocks=10):
    np.random.seed(42)  # For reproducibility

    # Initialize empty lists to store the data
    volatility, volume_of_trades, returns, number_of_mentions_in_twitter = (
        [],
        [],
        [],
        [],
    )
    names_list = [names.get_full_name() for _ in range(n_stocks)]
    names_list[0] = "Bitcoin"

    # Categorical feature (e.g., tech, finance, health)
    stock_types = ["tech", "finance", "health"]
    types = np.random.choice(stock_types, size=n_stocks)

    # Base values for each type
    base_values = {
        "tech": {
            "volatility": 0.3,
            "volumeoftrades": 5000,
            "return": 0.02,
            "numberofmentionsintwitter": 250,
        },
        "finance": {
            "volatility": 0.2,
            "volumeoftrades": 3000,
            "return": 0.01,
            "numberofmentionsintwitter": 150,
        },
        "health": {
            "volatility": 0.4,
            "volumeoftrades": 4000,
            "return": 0.03,
            "numberofmentionsintwitter": 200,
        },
    }

    # Generate features with variations based on type
    for stock_type in types:
        base = base_values[stock_type]
        volatility.append(
            base["volatility"] + np.random.rand() * 0.1 - 0.05
        )  # +/- 0.05 variation
        volume_of_trades.append(
            int(base["volumeoftrades"] + np.random.randint(-1000, 1000))
        )
        returns.append(
            base["return"] + np.random.randn() * 0.01
        )  # +/- 0.01 variation
        number_of_mentions_in_twitter.append(
            int(
                base["numberofmentionsintwitter"]
                + np.random.randint(-100, 100)
            )
        )

    # Create the DataFrame
    stocks_data = pd.DataFrame(
        {
            "name": names_list,
            "volatility": volatility,
            "volumeoftrades": volume_of_trades,
            "return": returns,
            "type": types,
            "numberofmentionsintwitter": number_of_mentions_in_twitter,
        }
    )

    return stocks_data


def compute_row_distance(row1, row2, categorical_columns):
    """
    Computes the distance between two rows of a dataset, considering both numerical and categorical features.

    Parameters:
    - row1: The first row of data (as a Pandas Series or array-like).
    - row2: The second row of data (as a Pandas Series or array-like).
    - categorical_columns: A list of column names that are categorical.

    Returns:
    - distance: The computed distance between the two rows.
    """
    # Initialize distance
    distance = 0

    # Loop through each column to compute distance
    for column in row1.index:
        if column in categorical_columns:
            # Categorical: 0 if same, 1 otherwise
            distance += 0 if row1[column] == row2[column] else 1
        else:
            # Numerical: Euclidean distance component
            distance += (row1[column] - row2[column]) ** 2

    # Return the square root of the summed squares for numerical components
    return np.sqrt(distance)


def generate_distance_matrix(data, categorical_columns):
    """
    Generates a distance matrix for a given dataset, considering both numerical and categorical features.

    Parameters:
    - data: The dataset as a Pandas DataFrame.
    - categorical_columns: A list of column names that are categorical.

    Returns:
    - distance_matrix: A numpy array representing the distance matrix.
    """
    n_rows = data.shape[0]
    distance_matrix = np.zeros((n_rows, n_rows))

    for i in range(n_rows):
        for j in range(
            i + 1, n_rows
        ):  # No need to compute when j <= i due to symmetry
            distance = compute_row_distance(
                data.iloc[i], data.iloc[j], categorical_columns
            )
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Symmetry

    return distance_matrix


def load_seedtag_dataset():

    path = Path("dataset.csv")
    dataset = pd.read_csv(path, sep=",", index_col=0)
    dataset = dataset.dropna(subset="WSFJ Score")
    rprint(dataset)

    dataset["LoE"] = dataset.filter(regex="LoE*").sum(axis=1)
    dataset = dataset.drop(columns=["LoE Backend", "LoE Data", "LoE MLOps"])

    columns = dataset.columns
    rprint(columns)
    rprint(dataset)

    return dataset


if __name__ == "__main__":

    # Create dataset
    rprint("[bold]Dataset[/bold]")

    # n_stocks = 25
    # dataset = generate_random_dataset(n_stocks)
    # rprint("Stocks Data:\n", dataset)

    # # Example usage with our stocks data
    # numerical_columns = [
    #     "volatility",
    #     "volumeoftrades",
    #     "return",
    #     "numberofmentionsintwitter",
    # ]

    dataset = load_seedtag_dataset()

    numerical_columns = [
        "Business Value",
        "Time Criticality",
        "Risk Reduction",
        "WSFJ Score",
        "LoE",
    ]

    normalised_dataset = dataset.copy()
    scaler = MinMaxScaler()
    normalised_dataset[numerical_columns] = scaler.fit_transform(
        dataset[numerical_columns]
    )

    categorical_columns = []
    _dataset = normalised_dataset.drop(columns=["Description"])
    distance_matrix = generate_distance_matrix(
        _dataset,
        categorical_columns,
    )
    rprint(distance_matrix)

    # Example usage
    # rprint("[bold]Distance Matrix[/bold]")
    # n = 100
    # distance_matrix = generate_random_distance_matrix(n)
    # rprint("Distance Matrix:\n", distance_matrix)

    rprint("[bold]MDS projection[/bold]")
    embedding = MDS(
        n_components=2,
        normalized_stress="auto",
        dissimilarity="precomputed",
    )
    points = embedding.fit_transform(distance_matrix)

    rprint("[bold]Plotting[/bold]")
    # Translate points so that Bitcoin point (index 0) is at the origin
    # bitcoin_point = points[0]  # Coordinates of the Bitcoin point
    # points = points - bitcoin_point  # Translate all points

    fig, ax = plt.subplots()

    ax.scatter(points[:, 0], points[:, 1])

    # Include the name
    point_names = dataset.index
    texts = []
    for i, name in enumerate(point_names):

        text = plt.text(
            points[i, 0], points[i, 1], name, ha="center", va="center"
        )

        texts.append(text)

    ax.grid(True)
    ax.set_title("Martell Opportunities (WSFJ Methodology)")
    plt.tight_layout()
    adjust_text(texts, pull_threshold=10)

    plt.savefig("output.png")
