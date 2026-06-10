from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]


def convert_online_shoppers(
    input_path: Path,
    output_path: Path,
    balanced_per_class: int | None = None,
    seed: int = 205,
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"rows": 0, "negative": 0, "positive": 0}

    with input_path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)

    if balanced_per_class is not None:
        negative_rows = [row for row in rows if row["Revenue"].strip().upper() == "FALSE"]
        positive_rows = [row for row in rows if row["Revenue"].strip().upper() == "TRUE"]
        if balanced_per_class > min(len(negative_rows), len(positive_rows)):
            raise ValueError("Requested more rows per class than the dataset contains")

        rng = random.Random(seed)
        rows = (
            rng.sample(negative_rows, balanced_per_class)
            + rng.sample(positive_rows, balanced_per_class)
        )
        rng.shuffle(rows)

    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for row in rows:
            revenue = row["Revenue"].strip().upper()
            if revenue == "TRUE":
                label = "2"
                counts["positive"] += 1
            elif revenue == "FALSE":
                label = "1"
                counts["negative"] += 1
            else:
                raise ValueError(f"Unexpected Revenue value: {row['Revenue']}")

            values = [label]
            values.extend(str(float(row[feature])) for feature in NUMERIC_FEATURES)
            output_file.write(" ".join(values) + "\n")
            counts["rows"] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Online Shoppers CSV to the CS 205 class-first numeric format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/real/online_shoppers_intention.csv"),
        help="Path to the raw Online Shoppers CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/real/online_shoppers_numeric.txt"),
        help="Path for the converted numeric dataset.",
    )
    parser.add_argument(
        "--balanced-per-class",
        type=int,
        default=None,
        help="Optionally write a deterministic class-balanced subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=205,
        help="Random seed used with --balanced-per-class.",
    )
    args = parser.parse_args()

    counts = convert_online_shoppers(
        args.input,
        args.output,
        balanced_per_class=args.balanced_per_class,
        seed=args.seed,
    )
    print(f"Wrote {counts['rows']} rows to {args.output}")
    print(f"Class 1 no revenue: {counts['negative']}")
    print(f"Class 2 revenue: {counts['positive']}")
    print("Features:")
    for index, feature in enumerate(NUMERIC_FEATURES, start=1):
        print(f"{index}. {feature}")


if __name__ == "__main__":
    main()
