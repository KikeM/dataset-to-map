from hmac import new
from pathlib import Path

import pandas as pd
from rich import print as rprint
from thefuzz.process import extract
import typer

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


def rename_columns(dataset: pd.DataFrame):
    mapping = {}
    for column in dataset.columns:
        new_column = extract(column, choices, limit=1)
        value, score = new_column[0]
        if score < 80:
            rprint(
                f"For column '{column}' current value is '{value}' with score ({score}), keeping it as is."
            )
            value = column
        mapping[column] = value

    rprint("Mapping: ", mapping)

    dataset = dataset.rename(columns=mapping)

    return dataset


def load_dataset(path):

    path = Path(path)
    dataset = pd.read_csv(path, sep=",")
    rprint("\n[bold]Dataset Raw[/bold]\n")
    rprint(dataset)

    dataset = rename_columns(dataset)

    dataset = dataset.dropna(subset=["wsfj"])
    dataset = dataset.drop(
        columns=["observations", "notes", "description"],
        errors="ignore",
    )
    rprint("\n[bold]Dataset Clean[/bold]\n")
    rprint(dataset)

    return dataset


def main(
    path: Path,
    output: Path = "output.png",
):

    # Create dataset
    rprint("\n[bold]Dataset Cleaning[/bold]\n")

    rprint(f"Reading from: {path}")

    load_dataset(path=path)
