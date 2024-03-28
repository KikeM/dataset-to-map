import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich import print as rprint
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
import names


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
    # Set the random seed for reproducibility
    np.random.seed(42)

    # Numerical features
    volatility = np.random.rand(n_stocks)  # Volatility as a fraction
    volume_of_trades = np.random.randint(
        1000, 10000, size=n_stocks
    )  # Volume of trades
    returns = (
        np.random.randn(n_stocks) * 0.05
    )  # Returns, centered around 0 with some deviation
    number_of_mentions_in_twitter = np.random.randint(
        0, 500, size=n_stocks
    )  # Number of mentions in Twitter
    names_list = [names.get_full_name() for _ in range(n_stocks)]
    names_list[0] = "Bitcoin"

    # Categorical feature (e.g., tech, finance, health)
    stock_types = ["tech", "finance", "health"]
    types = np.random.choice(stock_types, size=n_stocks)

    # Create a DataFrame
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


if __name__ == "__main__":

    # Create dataset
    rprint("[bold]Stocks Dataset[/bold]")

    n_stocks = 40
    stocks_data = generate_random_dataset(n_stocks)
    rprint("Stocks Data:\n", stocks_data)

    # Example usage with our stocks data
    numerical_columns = [
        "volatility",
        "volumeoftrades",
        "return",
        "numberofmentionsintwitter",
    ]

    normalised_stocks_data = stocks_data.copy()
    scaler = StandardScaler()
    normalised_stocks_data[numerical_columns] = scaler.fit_transform(
        stocks_data[numerical_columns]
    )

    categorical_columns = ["type"]
    _stocks_data = normalised_stocks_data.drop(columns=["name"])
    distance_matrix = generate_distance_matrix(
        _stocks_data,
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
    bitcoin_point = points[0]  # Coordinates of the Bitcoin point
    points = points - bitcoin_point  # Translate all points

    fig, ax = plt.subplots()

    ax.scatter(points[:, 0], points[:, 1])

    ax.grid(True)

    plt.savefig("output.png")
