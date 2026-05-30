from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    labels: list[int]
    features: list[list[float]]

    @property
    def num_instances(self) -> int:
        return len(self.labels)

    @property
    def num_features(self) -> int:
        return len(self.features[0]) if self.features else 0


def load_dataset(path: str | Path) -> Dataset:
    """Load class labels and continuous features from a numeric text file."""
    rows: list[list[float]] = []

    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.replace(",", " ").split()
        if len(parts) < 2:
            raise ValueError(f"Line {line_number} must contain a class and at least one feature")

        try:
            rows.append([float(part) for part in parts])
        except ValueError as error:
            raise ValueError(f"Line {line_number} contains a non-numeric value") from error

    if not rows:
        raise ValueError("Dataset is empty")

    expected_columns = len(rows[0])
    for row_number, row in enumerate(rows, start=1):
        if len(row) != expected_columns:
            raise ValueError(
                f"Line {row_number} has {len(row)} columns; expected {expected_columns}"
            )

    labels = [_parse_label(row[0]) for row in rows]
    features = [row[1:] for row in rows]
    return Dataset(labels=labels, features=features)


def z_normalize(features: list[list[float]]) -> list[list[float]]:
    """Return a z-normalized copy of the feature matrix."""
    if not features:
        return []

    num_features = len(features[0])
    for row_number, row in enumerate(features, start=1):
        if len(row) != num_features:
            raise ValueError(
                f"Feature row {row_number} has {len(row)} values; expected {num_features}"
            )

    columns = list(zip(*features))
    means = [sum(column) / len(column) for column in columns]
    std_devs = []

    for column, mean in zip(columns, means):
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        std_devs.append(variance**0.5)

    normalized: list[list[float]] = []
    for row in features:
        normalized_row = []
        for value, mean, std_dev in zip(row, means, std_devs):
            if std_dev == 0:
                normalized_row.append(0.0)
            else:
                normalized_row.append((value - mean) / std_dev)
        normalized.append(normalized_row)

    return normalized


def _parse_label(value: float) -> int:
    label = int(value)
    if label != value:
        raise ValueError(f"Class label {value} is not an integer value")
    return label
