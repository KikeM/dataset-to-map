from pathlib import Path

import pandas as pd
from rich import print as rprint
from thefuzz.process import extract

choices = [
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
        new_column = extract(_input, choices, limit=1)
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


def load_dataset(path):

    path = Path(path)
    dataset = pd.read_csv(path, sep=",")
    rprint("\n[bold]Dataset Raw[/bold]\n")
    rprint(dataset)

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


def main(
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

    rprint("\n[bold]Dataset Clean[/bold]\n")
    rprint(dataset)

    output = enforce_csv_extension(output)
    rprint(f"Saving to: {output}")
    dataset.to_csv(output, index=False)
