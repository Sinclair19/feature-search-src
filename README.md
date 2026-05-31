# CS 205 Project 2

This repository contains the work for CS 205 Project 2: feature selection with a nearest-neighbor classifier, experiments on synthetic and real datasets, and a dendrogram clustering analysis.

## Dataset Format

Input datasets should be plain numeric text files. Each row is one instance.

```text
class feature_1 feature_2 ... feature_n
```

- The first column is the class label.
- Class labels are expected to be numeric.
- All remaining columns are continuous feature values.
- Feature values are not assumed to be normalized in the input file.

## Command Line Usage

Run the feature-search program with Python:

```bash
python src/feature_search.py <dataset_path> [options]
```

Options:

- `--search forward`: run forward selection.
- `--search backward`: run backward elimination.
- `--search both`: run both searches. This is the default.
- `--baseline`: print leave-one-out nearest-neighbor accuracy using all features before search.
- `--normalize`: apply z-normalization before running nearest neighbor.
- `--interactive`: prompt for the dataset path and options.

Examples:

```bash
python src/feature_search.py data/synthetic/CS170_Small_DataSet__32.txt --search forward
python src/feature_search.py data/synthetic/CS170_Small_DataSet__32.txt --search backward --baseline
python src/feature_search.py data/synthetic/CS170_Small_DataSet__32.txt --search both --baseline
python src/feature_search.py data/synthetic/CS170_Small_DataSet__32.txt --search both --normalize --baseline
python src/feature_search.py --interactive
```

The program prints feature numbers using 1-based indexing. For example, `{4, 8}` means the fourth and eighth feature columns after the class-label column.

## Project Layout

- `src/`: feature-search implementation.
- `tests/`: tests for loading, normalization, nearest neighbor, and search.
- `data/synthetic/`: instructor-provided synthetic datasets.
- `data/real/`: real-world datasets used for experiments.
- `results/synthetic/`: saved traces and results for synthetic datasets.
- `results/real/`: saved traces and results for real datasets.
- `results/clustering/`: dendrogram figures and clustering outputs.
- `report/`: report notes, references, and draft materials.
- `slides/`: presentation outline and slide materials.
