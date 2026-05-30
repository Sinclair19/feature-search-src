from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
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


@dataclass(frozen=True)
class SearchResult:
    selected_features: list[int]
    accuracy: float


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


def euclidean_distance(
    features: list[list[float]],
    first_index: int,
    second_index: int,
    selected_features: list[int] | tuple[int, ...],
) -> float:
    """Calculate Euclidean distance between two rows over selected features."""
    squared_distance = 0.0
    for feature_index in selected_features:
        difference = features[first_index][feature_index] - features[second_index][feature_index]
        squared_distance += difference * difference

    return sqrt(squared_distance)


def nearest_neighbor_predict(
    labels: list[int],
    features: list[list[float]],
    test_index: int,
    selected_features: list[int] | tuple[int, ...],
) -> int:
    """Predict one row's label using every other row as possible training data."""
    best_label: int | None = None
    best_distance: float | None = None

    for candidate_index in range(len(features)):
        if candidate_index == test_index:
            continue

        distance = euclidean_distance(features, test_index, candidate_index, selected_features)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_label = labels[candidate_index]

    if best_label is None:
        raise ValueError("Nearest neighbor requires at least two instances")

    return best_label


def leave_one_out_accuracy(
    labels: list[int],
    features: list[list[float]],
    selected_features: list[int] | tuple[int, ...],
) -> float:
    """Return leave-one-out nearest-neighbor accuracy as a percentage."""
    correct = 0
    for test_index, actual_label in enumerate(labels):
        predicted_label = nearest_neighbor_predict(labels, features, test_index, selected_features)
        if predicted_label == actual_label:
            correct += 1

    return correct / len(labels) * 100.0


def forward_selection(labels: list[int], features: list[list[float]]) -> SearchResult:
    """Run forward feature selection using leave-one-out nearest neighbor accuracy."""
    selected_features: list[int] = []
    best_features: list[int] = []
    best_accuracy = -1.0
    num_features = len(features[0]) if features else 0

    print("Beginning forward selection search.")

    for _ in range(num_features):
        feature_to_add = -1
        level_best_accuracy = -1.0

        for feature_index in range(num_features):
            if feature_index in selected_features:
                continue

            candidate_features = selected_features + [feature_index]
            accuracy = leave_one_out_accuracy(labels, features, candidate_features)
            print(
                f"Using feature(s) {_format_feature_set(candidate_features)} "
                f"accuracy is {accuracy:.1f}%"
            )

            if accuracy > level_best_accuracy:
                level_best_accuracy = accuracy
                feature_to_add = feature_index

        selected_features.append(feature_to_add)
        print(
            f"Feature set {_format_feature_set(selected_features)} was best, "
            f"accuracy is {level_best_accuracy:.1f}%\n"
        )

        if level_best_accuracy > best_accuracy:
            best_accuracy = level_best_accuracy
            best_features = selected_features.copy()

    print(
        f"Finished search. The best feature subset is {_format_feature_set(best_features)}, "
        f"accuracy is {best_accuracy:.1f}%"
    )

    return SearchResult(selected_features=best_features, accuracy=best_accuracy)


def backward_elimination(labels: list[int], features: list[list[float]]) -> SearchResult:
    """Run backward feature elimination using leave-one-out nearest neighbor accuracy."""
    selected_features = list(range(len(features[0]) if features else 0))
    best_features = selected_features.copy()
    best_accuracy = leave_one_out_accuracy(labels, features, selected_features)

    print("Beginning backward elimination search.")
    print(
        f"Using feature(s) {_format_feature_set(selected_features)} "
        f"accuracy is {best_accuracy:.1f}%\n"
    )

    while len(selected_features) > 1:
        feature_to_remove = -1
        level_best_features: list[int] = []
        level_best_accuracy = -1.0

        for feature_index in selected_features:
            candidate_features = [
                current_feature
                for current_feature in selected_features
                if current_feature != feature_index
            ]
            accuracy = leave_one_out_accuracy(labels, features, candidate_features)
            print(
                f"Using feature(s) {_format_feature_set(candidate_features)} "
                f"accuracy is {accuracy:.1f}%"
            )

            if accuracy > level_best_accuracy:
                level_best_accuracy = accuracy
                level_best_features = candidate_features
                feature_to_remove = feature_index

        selected_features = level_best_features
        print(
            f"Feature set {_format_feature_set(selected_features)} was best, "
            f"removed feature {feature_to_remove + 1}, accuracy is {level_best_accuracy:.1f}%\n"
        )

        if level_best_accuracy > best_accuracy:
            best_accuracy = level_best_accuracy
            best_features = selected_features.copy()

    print(
        f"Finished search. The best feature subset is {_format_feature_set(best_features)}, "
        f"accuracy is {best_accuracy:.1f}%"
    )

    return SearchResult(selected_features=best_features, accuracy=best_accuracy)


def _parse_label(value: float) -> int:
    label = int(value)
    if label != value:
        raise ValueError(f"Class label {value} is not an integer value")
    return label


def _format_feature_set(feature_indexes: list[int] | tuple[int, ...]) -> str:
    display_indexes = [str(feature_index + 1) for feature_index in feature_indexes]
    return "{" + ", ".join(display_indexes) + "}"
