import os
from pathlib import Path

import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/computer-hardware/machine.csv")
OUTPUT_DIR = Path("outputs/computer_hardware_clean")
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / "matplotlib_config"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / "cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

LABEL_COLUMNS = ["vendor_name", "Model"]
CYCLE_TIME_COLUMNS = ["MYCT"]
HARDWARE_COLUMNS = ["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX"]
MEMORY_IO_COLUMNS = ["MMIN", "MMAX", "CACH", "CHMIN", "CHMAX"]
PERFORMANCE_COLUMNS = ["PRP", "ERP"]
ALL_NUMERIC_COLUMNS = HARDWARE_COLUMNS + PERFORMANCE_COLUMNS
LINKAGE_METHODS = ["single", "complete", "average", "ward"]
CLUSTER_COUNTS = (3, 4)


def load_hardware_data(data_path=DATA_PATH):
    """Load the UCI computer hardware data and validate the expected schema."""
    df = pd.read_csv(data_path)
    expected_columns = LABEL_COLUMNS + ALL_NUMERIC_COLUMNS
    missing_columns = [column for column in expected_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    clean_df = df.loc[:, expected_columns].copy()
    for column in ALL_NUMERIC_COLUMNS:
        clean_df[column] = pd.to_numeric(clean_df[column], errors="raise")

    clean_df["machine_label"] = (
        clean_df["vendor_name"].astype(str) + " " + clean_df["Model"].astype(str)
    )
    return clean_df


def scale_numeric_features(df, columns):
    values = df.loc[:, columns].copy()
    return pd.DataFrame(
        StandardScaler().fit_transform(values),
        columns=columns,
        index=df.index,
    )


def save_basic_summaries(df, output_dir=OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

    df.loc[:, LABEL_COLUMNS + ["machine_label"] + ALL_NUMERIC_COLUMNS].to_csv(
        output_dir / "computer_hardware_processed.csv",
        index=False,
    )

    numeric_summary = df.loc[:, ALL_NUMERIC_COLUMNS].describe().round(2).T
    numeric_summary.to_csv(output_dir / "numeric_column_summary.csv")

    vendor_summary = (
        df.groupby("vendor_name")
        .agg(
            machines=("Model", "count"),
            prp_mean=("PRP", "mean"),
            erp_mean=("ERP", "mean"),
            myct_mean=("MYCT", "mean"),
            mmax_mean=("MMAX", "mean"),
        )
        .round(2)
        .sort_values(["machines", "prp_mean"], ascending=[False, False])
        .reset_index()
    )
    vendor_summary.to_csv(output_dir / "vendor_summary.csv", index=False)


def make_linkage_matrix(df, columns, method):
    scaled = scale_numeric_features(df, columns)
    return linkage(scaled, method=method)


def plot_machine_dendrogram(df, linkage_matrix, title, output_path, truncate=True):
    plt.figure(figsize=(18, 9))
    dendrogram_kwargs = {
        "leaf_rotation": 90,
        "leaf_font_size": 7,
    }
    if truncate:
        dendrogram_kwargs.update(
            {
                "truncate_mode": "lastp",
                "p": 30,
                "show_leaf_counts": True,
            }
        )
    else:
        dendrogram_kwargs["labels"] = df["machine_label"].to_numpy()

    dendrogram(linkage_matrix, **dendrogram_kwargs)
    plt.title(title)
    plt.xlabel("Machines" if not truncate else "Machine clusters")
    plt.ylabel("Linkage distance")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_cluster_csvs(df, linkage_matrix, feature_columns, output_dir, prefix):
    assignments = df.loc[:, LABEL_COLUMNS + ["machine_label"] + ALL_NUMERIC_COLUMNS].copy()
    assignments["cluster_features"] = ", ".join(feature_columns)

    for cluster_count in CLUSTER_COUNTS:
        cluster_column = f"cluster_k{cluster_count}"
        assignments[cluster_column] = fcluster(
            linkage_matrix,
            t=cluster_count,
            criterion="maxclust",
        )

        summary = (
            assignments.groupby(cluster_column)
            .agg(
                machines=("machine_label", "count"),
                MYCT_mean=("MYCT", "mean"),
                MMIN_mean=("MMIN", "mean"),
                MMAX_mean=("MMAX", "mean"),
                CACH_mean=("CACH", "mean"),
                CHMIN_mean=("CHMIN", "mean"),
                CHMAX_mean=("CHMAX", "mean"),
                PRP_mean=("PRP", "mean"),
                ERP_mean=("ERP", "mean"),
            )
            .round(2)
            .reset_index()
        )
        summary.to_csv(
            output_dir / f"{prefix}_cluster_summary_k{cluster_count}.csv",
            index=False,
        )

    assignments.to_csv(output_dir / f"{prefix}_cluster_assignments.csv", index=False)


def save_all_numeric_linkage_outputs(df, output_dir=OUTPUT_DIR):
    all_numeric_dir = output_dir / "all_numeric_linkage_methods"

    for method in LINKAGE_METHODS:
        method_dir = all_numeric_dir / method
        method_prefix = f"all_numeric_{method}"
        linkage_matrix = make_linkage_matrix(df, ALL_NUMERIC_COLUMNS, method)

        plot_machine_dendrogram(
            df,
            linkage_matrix,
            f"All Numeric Features ({method.title()} Linkage)",
            method_dir / f"{method_prefix}_dendrogram_truncated.png",
            truncate=True,
        )
        plot_machine_dendrogram(
            df,
            linkage_matrix,
            f"All Numeric Features ({method.title()} Linkage, Labeled)",
            method_dir / f"{method_prefix}_dendrogram_labeled.png",
            truncate=False,
        )
        save_cluster_csvs(
            df,
            linkage_matrix,
            ALL_NUMERIC_COLUMNS,
            method_dir,
            method_prefix,
        )


def save_memory_io_linkage_outputs(df, output_dir=OUTPUT_DIR):
    memory_io_dir = output_dir / "memory_io_linkage_methods"

    for method in LINKAGE_METHODS:
        method_dir = memory_io_dir / method
        method_prefix = f"memory_io_{method}"
        linkage_matrix = make_linkage_matrix(df, MEMORY_IO_COLUMNS, method)

        plot_machine_dendrogram(
            df,
            linkage_matrix,
            f"Memory and I/O Features ({method.title()} Linkage)",
            method_dir / f"{method_prefix}_dendrogram_truncated.png",
            truncate=True,
        )
        plot_machine_dendrogram(
            df,
            linkage_matrix,
            f"Memory and I/O Features ({method.title()} Linkage, Labeled)",
            method_dir / f"{method_prefix}_dendrogram_labeled.png",
            truncate=False,
        )
        save_cluster_csvs(
            df,
            linkage_matrix,
            MEMORY_IO_COLUMNS,
            method_dir,
            method_prefix,
        )


def save_cycle_time_linkage_outputs(df, output_dir=OUTPUT_DIR):
    cycle_time_dir = output_dir / "cycle_time_linkage_methods"

    for method in LINKAGE_METHODS:
        method_dir = cycle_time_dir / method
        method_prefix = f"cycle_time_{method}"
        linkage_matrix = make_linkage_matrix(df, CYCLE_TIME_COLUMNS, method)

        plot_machine_dendrogram(
            df,
            linkage_matrix,
            f"Cycle Time Only ({method.title()} Linkage)",
            method_dir / f"{method_prefix}_dendrogram_truncated.png",
            truncate=True,
        )
        plot_machine_dendrogram(
            df,
            linkage_matrix,
            f"Cycle Time Only ({method.title()} Linkage, Labeled)",
            method_dir / f"{method_prefix}_dendrogram_labeled.png",
            truncate=False,
        )
        save_cluster_csvs(
            df,
            linkage_matrix,
            CYCLE_TIME_COLUMNS,
            method_dir,
            method_prefix,
        )


def plot_column_dendrogram(columns, linkage_matrix, title, output_path):
    plt.figure(figsize=(10, 6))
    dendrogram(
        linkage_matrix,
        labels=columns,
        leaf_rotation=0,
        leaf_font_size=10,
    )
    plt.title(title)
    plt.xlabel("Numeric columns")
    plt.ylabel("Linkage distance")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_column_cluster_csvs(columns, linkage_matrix, output_dir, prefix):
    linkage_df = pd.DataFrame(
        linkage_matrix,
        columns=["left_child", "right_child", "distance", "leaf_count"],
    )
    linkage_df.to_csv(output_dir / f"{prefix}_linkage_matrix.csv", index=False)

    assignments = pd.DataFrame({"column": columns})
    for cluster_count in CLUSTER_COUNTS:
        assignments[f"column_cluster_k{cluster_count}"] = fcluster(
            linkage_matrix,
            t=cluster_count,
            criterion="maxclust",
        )

    assignments.to_csv(output_dir / f"{prefix}_column_clusters.csv", index=False)


def save_column_clustering_outputs(df, output_dir=OUTPUT_DIR):
    column_dir = output_dir / "column_clustering"
    scaled = scale_numeric_features(df, ALL_NUMERIC_COLUMNS)

    for method in LINKAGE_METHODS:
        method_dir = column_dir / method
        method_prefix = f"columns_{method}"
        linkage_matrix = linkage(scaled.T, method=method)

        plot_column_dendrogram(
            ALL_NUMERIC_COLUMNS,
            linkage_matrix,
            f"Numeric Column Clustering ({method.title()} Linkage)",
            method_dir / f"{method_prefix}_dendrogram.png",
        )
        save_column_cluster_csvs(
            ALL_NUMERIC_COLUMNS,
            linkage_matrix,
            method_dir,
            method_prefix,
        )


def main():
    df = load_hardware_data()
    save_basic_summaries(df)
    save_all_numeric_linkage_outputs(df)
    save_memory_io_linkage_outputs(df)
    save_cycle_time_linkage_outputs(df)
    save_column_clustering_outputs(df)
    print(f"Wrote computer hardware dendrogram analysis to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
