from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


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

    # The project format always stores the class label first.
    labels = [_parse_label(row[0]) for row in rows]
    features = [row[1:] for row in rows]
    return Dataset(labels=labels, features=features)


def forward_selection(labels: list[int], features: list[list[float]]) -> SearchResult:
    features = _prepare_features(features)
    # Feature indexes are 0-based in the code and printed as 1-based for the trace.
    selected_features: list[int] = []
    best_features: list[int] = []
    best_accuracy = -1.0
    num_features = _num_features(features)
    # The same subset can be checked more than once, so remember its accuracy.
    accuracy_cache: dict[tuple[int, ...], float] = {}

    print("Beginning forward selection search.")

    for _ in range(num_features):
        feature_to_add = -1
        level_best_accuracy = -1.0

        for feature_index in range(num_features):
            if feature_index in selected_features:
                continue

            candidate_features = selected_features + [feature_index]
            accuracy = cached_leave_one_out_accuracy(
                labels,
                features,
                candidate_features,
                accuracy_cache,
            )
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
    features = _prepare_features(features)
    # Start with every feature, then remove one feature at each level.
    selected_features = list(range(_num_features(features)))
    # Cache keys are sorted tuples of feature indexes, for example (3, 4, 7).
    accuracy_cache: dict[tuple[int, ...], float] = {}
    best_features = selected_features.copy()
    best_accuracy = cached_leave_one_out_accuracy(
        labels,
        features,
        selected_features,
        accuracy_cache,
    )

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
            accuracy = cached_leave_one_out_accuracy(
                labels,
                features,
                candidate_features,
                accuracy_cache,
            )
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


def cached_leave_one_out_accuracy(
    labels: list[int],
    features: list[list[float]],
    selected_features: list[int] | tuple[int, ...],
    accuracy_cache: dict[tuple[int, ...], float],
) -> float:
    # Sorting makes [4, 8] and [8, 4] share the same cached result.
    key = tuple(sorted(selected_features))
    if key not in accuracy_cache:
        accuracy_cache[key] = leave_one_out_accuracy(labels, features, selected_features)
    return accuracy_cache[key]


def leave_one_out_accuracy(
    labels: list[int],
    features: list[list[float]],
    selected_features: list[int] | tuple[int, ...],
) -> float:
    if _is_numpy_matrix(features):
        return numpy_leave_one_out_accuracy(labels, features, selected_features)

    correct = 0
    for test_index, actual_label in enumerate(labels):
        # Keep this loop here instead of calling nearest_neighbor_predict:
        # this function runs many times during search, so avoiding extra calls helps.
        predicted_label = None
        best_distance = None

        for candidate_index, candidate_label in enumerate(labels):
            # Leave-one-out means the test row cannot be its own nearest neighbor.
            if candidate_index == test_index:
                continue

            distance = squared_distance(features, test_index, candidate_index, selected_features)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                predicted_label = candidate_label

        if predicted_label == actual_label:
            correct += 1

    return correct / len(labels) * 100.0


def numpy_leave_one_out_accuracy(
    labels: list[int],
    features,
    selected_features: list[int] | tuple[int, ...],
) -> float:
    labels_array = np.asarray(labels)
    selected = np.asarray(selected_features)
    correct = 0

    for test_index, actual_label in enumerate(labels_array):
        # NumPy computes every distance from this test row in one vectorized step.
        differences = features[:, selected] - features[test_index, selected]
        distances = (differences * differences).sum(axis=1)
        distances[test_index] = float("inf")

        nearest_index = int(distances.argmin())
        if labels_array[nearest_index] == actual_label:
            correct += 1

    return correct / len(labels_array) * 100.0


def nearest_neighbor_predict(
    labels: list[int],
    features: list[list[float]],
    test_index: int,
    selected_features: list[int] | tuple[int, ...],
) -> int:
    best_label: int | None = None
    best_distance: float | None = None

    for candidate_index in range(len(features)):
        if candidate_index == test_index:
            continue

        distance = squared_distance(features, test_index, candidate_index, selected_features)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_label = labels[candidate_index]

    if best_label is None:
        raise ValueError("Nearest neighbor requires at least two instances")

    return best_label


def squared_distance(
    features: list[list[float]],
    first_index: int,
    second_index: int,
    selected_features: list[int] | tuple[int, ...],
) -> float:
    # No sqrt is needed because squared distance has the same ordering as distance.
    squared_distance = 0.0
    for feature_index in selected_features:
        difference = features[first_index][feature_index] - features[second_index][feature_index]
        squared_distance += difference * difference

    return squared_distance


def z_normalize(features: list[list[float]]) -> list[list[float]]:
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
                # A constant column has no distance information after normalization.
                normalized_row.append(0.0)
            else:
                normalized_row.append((value - mean) / std_dev)
        normalized.append(normalized_row)

    return normalized


def run_search(
    dataset_path: str | Path,
    search_mode: str,
    use_normalization: bool = False,
    show_baseline: bool = False,
) -> None:
    dataset = load_dataset(dataset_path)
    # The CS170 synthetic files are already scaled, so normalization is opt-in.
    features = z_normalize(dataset.features) if use_normalization else dataset.features
    if np is not None:
        features = _prepare_features(features)

    print(f"Dataset: {dataset_path}")
    print(f"Instances: {dataset.num_instances}")
    print(f"Features: {dataset.num_features}")
    print(f"Normalization: {'z-normalization' if use_normalization else 'none'}")
    print(f"Distance engine: {'NumPy' if _is_numpy_matrix(features) else 'pure Python'}")

    if show_baseline:
        all_features = list(range(dataset.num_features))
        accuracy = leave_one_out_accuracy(dataset.labels, features, all_features)
        print(f"All-feature baseline accuracy: {accuracy:.1f}%\n")

    if search_mode in ("forward", "both"):
        forward_selection(dataset.labels, features)

    if search_mode == "both":
        print()

    if search_mode in ("backward", "both"):
        backward_elimination(dataset.labels, features)


def interactive_main() -> None:
    dataset_path = input("Dataset path: ").strip()
    search_mode = _prompt_choice("Search mode", ("forward", "backward", "both"))
    use_normalization = _prompt_yes_no("Use z-normalization? [y/N]: ")
    show_baseline = _prompt_yes_no("Show all-feature baseline? [y/N]: ")

    run_search(
        dataset_path=dataset_path,
        search_mode=search_mode,
        use_normalization=use_normalization,
        show_baseline=show_baseline,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run nearest-neighbor feature selection on a project-format dataset."
    )
    parser.add_argument("dataset", nargs="?", help="Path to the dataset file.")
    parser.add_argument(
        "--search",
        choices=("forward", "backward", "both"),
        default="both",
        help="Feature-search mode to run.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply z-normalization before nearest-neighbor search.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Print leave-one-out accuracy using all features before search.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for dataset path and run options.",
    )
    args = parser.parse_args()

    if args.interactive or not args.dataset:
        interactive_main()
        return

    run_search(
        dataset_path=args.dataset,
        search_mode=args.search,
        use_normalization=args.normalize,
        show_baseline=args.baseline,
    )


def _parse_label(value: float) -> int:
    label = int(value)
    if label != value:
        raise ValueError(f"Class label {value} is not an integer value")
    return label


def _prepare_features(features):
    if np is None or _is_numpy_matrix(features):
        return features
    return np.asarray(features, dtype=float)


def _is_numpy_matrix(features) -> bool:
    return np is not None and isinstance(features, np.ndarray)


def _num_features(features) -> int:
    if _is_numpy_matrix(features):
        return features.shape[1] if features.size else 0
    return len(features[0]) if features else 0


def _format_feature_set(feature_indexes: list[int] | tuple[int, ...]) -> str:
    display_indexes = [str(feature_index + 1) for feature_index in feature_indexes]
    return "{" + ", ".join(display_indexes) + "}"


def _prompt_choice(prompt: str, choices: tuple[str, ...]) -> str:
    choice_text = "/".join(choices)
    while True:
        value = input(f"{prompt} ({choice_text}): ").strip().lower()
        if value in choices:
            return value
        print(f"Please enter one of: {choice_text}")


def _prompt_yes_no(prompt: str) -> bool:
    value = input(prompt).strip().lower()
    return value in ("y", "yes")


if __name__ == "__main__":
    main()
