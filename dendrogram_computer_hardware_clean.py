from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/computer-hardware/machine.csv")
OUTPUT_DIR = Path("outputs/computer_hardware_clean")

LABEL_COLUMNS = ["vendor_name", "Model"]
HARDWARE_COLUMNS = ["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX"]
PERFORMANCE_COLUMNS = ["PRP", "ERP"]
ALL_NUMERIC_COLUMNS = HARDWARE_COLUMNS + PERFORMANCE_COLUMNS


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


def main():
    df = load_hardware_data()
    save_basic_summaries(df)
    print(f"Wrote basic computer hardware summaries to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
