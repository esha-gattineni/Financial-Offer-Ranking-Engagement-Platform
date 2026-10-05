from pathlib import Path

import pandas as pd
import tensorflow_data_validation as tfdv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_1000000.parquet"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "tfdv"
)

STATS_FILE = ARTIFACT_DIR / "training_stats.pb"
SCHEMA_FILE = ARTIFACT_DIR / "schema.pbtxt"


def load_dataset():
    print("\nLoading dataset...")

    dataset = pd.read_parquet(DATA_FILE)

    print(
        "Rows:",
        f"{len(dataset):,}",
    )

    print(
        "Columns:",
        len(dataset.columns),
    )

    return dataset


def generate_statistics(dataset):
    print(
        "\nGenerating TFDV statistics..."
    )

    stats = (
        tfdv.generate_statistics_from_dataframe(
            dataset
        )
    )

    return stats


def infer_schema(stats):
    print(
        "\nInferring schema..."
    )

    schema = tfdv.infer_schema(
        statistics=stats
    )

    return schema


def validate_statistics(
    stats,
    schema,
):
    print(
        "\nChecking dataset for anomalies..."
    )

    anomalies = tfdv.validate_statistics(
        statistics=stats,
        schema=schema,
    )

    return anomalies


if __name__ == "__main__":

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_dataset()

    stats = generate_statistics(
        dataset
    )

    schema = infer_schema(
        stats
    )

    anomalies = validate_statistics(
        stats,
        schema,
    )

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    tfdv.write_stats_text(
        stats,
        str(STATS_FILE),
    )

    # --------------------------------------------------------
    # Save schema
    # --------------------------------------------------------

    tfdv.write_schema_text(
        schema,
        str(SCHEMA_FILE),
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(
        "\n================================"
    )

    print(
        "TFDV VALIDATION RESULTS"
    )

    print(
        "================================"
    )

    if anomalies.anomaly_info:

        print(
            "\nAnomalies detected:"
        )

        for feature_name, anomaly in (
            anomalies.anomaly_info.items()
        ):

            print(
                f"\nFeature: {feature_name}"
            )

            print(
                anomaly.description
            )

    else:

        print(
            "\nNo schema anomalies detected."
        )

    print(
        "\nStatistics saved to:"
    )

    print(
        STATS_FILE
    )

    print(
        "\nSchema saved to:"
    )

    print(
        SCHEMA_FILE
    )

    print(
        "\nStep 14 completed."
    )