from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from adjustText import adjust_text
from rich import print as rprint
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from thefuzz.process import extract

app = typer.Typer(no_args_is_help=True)

CHOICES = [
    "businessValue",
    "timeCriticality",
    "riskReduction",
    "wsfj",
    "levelOfEffort",
    "description",
    "notes",
    "observations",
    "name",
]

DIMENSIONS = [
    "name",
    "businessValue",
    "timeCriticality",
    "riskReduction",
    "levelOfEffort",
    "wsfj",
]


def process_string(string: str):
    # to lower case
    string = string.lower()

    # remove special characters
    string = string.replace(" ", "")

    return string


def rename_columns(dataset: pd.DataFrame, threshold: int):
    mapping = {}
    for column in dataset.columns:
        _input = process_string(column)
        new_column = extract(_input, CHOICES, limit=1)
        value, score = new_column[0]
        if score < threshold:
            rprint(
                f"For column '{column}' current value is '{value}' with score ({score}), keeping it as is."
            )
            value = column
        mapping[column] = value

    rprint("Mapping:")
    rprint(mapping)

    dataset = dataset.rename(columns=mapping)

    return dataset


def drop_rows_cols(dataset: pd.DataFrame):

    columns = dataset.columns
    assert "wsfj" in columns, "Column 'wsfj' is required."
    assert "name" in columns, "Column 'name' is required."

    dataset = dataset.dropna(subset=["wsfj", "name"])
    dataset = dataset.drop(
        columns=[
            "observations",
            "notes",
            "description",
        ],
        errors="ignore",
    )
    return dataset


def enforce_csv_extension(path: Path):
    if path.suffix != ".csv":
        path = path.with_suffix(".csv")
    return path


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


def generate_distance_matrix(data, categorical_columns=[]):
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


def load_dataset(path) -> pd.DataFrame:

    dataset = pd.read_csv(path, sep=",")
    rprint(dataset)

    # Populate domain if not present
    if "domain" not in dataset.columns:
        dataset["domain"] = "squad"

    return dataset


@app.command()
def dataset(
    path: Path,
    threshold: int = 70,
    output: Path = "dataset.csv",
):

    # Create dataset
    rprint("\n[bold]Dataset Cleaning[/bold]\n")

    rprint(f"Reading from: {path}")

    dataset = load_dataset(path=path)
    dataset = rename_columns(dataset, threshold=threshold)
    dataset = drop_rows_cols(dataset)

    dataset = dataset.sort_values(by="wsfj", ascending=False)

    dataset = dataset[DIMENSIONS]

    rprint("\n[bold]Dataset Clean[/bold]\n")
    rprint(dataset)

    output = enforce_csv_extension(output)
    rprint(f"Saving to: {output}")
    dataset.to_csv(output, index=False)


@app.command()
def represent(
    path: Path,
    png_name: str = "output.png",
    index: str = "name",
):

    # Create dataset
    rprint("\n[bold]Dataset[/bold]\n")
    dataset = load_dataset(path)
    dataset = dataset.set_index(index)

    scaler = MinMaxScaler()

    for col in dataset.columns:
        if col == "domain":
            continue
        dataset[col] = scaler.fit_transform(dataset[[col]])

    rprint("\n[bold]Normalised Dataset[/bold]\n")
    rprint(dataset)
    distance_matrix = generate_distance_matrix(
        dataset, categorical_columns=["domain"]
    )

    rprint("\n[bold]MDS projection[/bold]\n")
    embedding = MDS(
        n_components=2,
        normalized_stress="auto",
        dissimilarity="precomputed",
    )
    points = embedding.fit_transform(distance_matrix)

    rprint("\n[bold]Plotting[/bold]\n")
    _, ax = plt.subplots()

    ax.scatter(points[:, 0], points[:, 1])

    # Include the name
    point_names = dataset.index
    texts = []
    for i, name in enumerate(point_names):

        # Pick randomly between left or right
        choices = ["left", "right"]
        ha = np.random.choice(choices)

        text = plt.text(
            points[i, 0],
            points[i, 1],
            name,
            ha=ha,
            va="baseline",
            fontsize=7,
        )

        texts.append(text)

    which = path.stem
    ax.set_title(f"{which.title()} Opportunities (WSFJ Methodology)")

    # Make the x and y axes 20% larger
    margin = 0.2
    ax.set_xlim(ax.get_xlim()[0] - margin, ax.get_xlim()[1] + margin)
    ax.set_ylim(ax.get_ylim()[0] - margin, ax.get_ylim()[1] + margin)

    # Remove numbers from the axes, keep the grid
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)

    plt.tight_layout()
    adjust_text(texts, pull_threshold=1)

    plt.savefig(png_name)
