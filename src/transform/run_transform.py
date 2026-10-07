from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf

import apache_beam as beam
import tensorflow_transform.beam as tft_beam

from tensorflow_transform import coders

from tensorflow_transform.tf_metadata import (
    dataset_metadata,
    schema_utils,
)

from src.models.train_historical import (
    load_dataset,
    chronological_split,
    add_historical_features,
)

from src.transform.preprocessing import (
    LABEL_KEY,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    preprocessing_fn,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

TFT_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tft_input"
)

TFT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tft_output"
)

TFT_ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "tft_transform"
)

TEMP_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "tft_temp"
)


TRAIN_PARQUET = (
    TFT_INPUT_DIR
    / "train.parquet"
)

VALIDATION_PARQUET = (
    TFT_INPUT_DIR
    / "validation.parquet"
)


# ============================================================
# CLEAN RAW MODEL FEATURES
# ============================================================

def clean_dataframe(
    dataframe,
):
    dataframe = dataframe.copy()

    required_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [LABEL_KEY]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    dataframe = dataframe[
        required_columns
    ].copy()

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    for column in NUMERIC_FEATURES:

        dataframe[column] = (
            pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )
        )

        dataframe[column] = (
            dataframe[column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
            .astype(np.float32)
        )

    # --------------------------------------------------------
    # Categorical
    # --------------------------------------------------------

    for column in CATEGORICAL_FEATURES:

        dataframe[column] = (
            dataframe[column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    dataframe[LABEL_KEY] = (
        pd.to_numeric(
            dataframe[LABEL_KEY],
            errors="raise",
        )
        .astype(np.int64)
    )

    return dataframe


# ============================================================
# BUILD LEAKAGE-SAFE DATASETS
# ============================================================

def prepare_datasets():

    print(
        "\nLoading 1M interaction dataset..."
    )

    dataset = load_dataset()

    print(
        "Rows:",
        f"{len(dataset):,}",
    )

    print(
        "\nCreating chronological split..."
    )

    train, validation = (
        chronological_split(
            dataset
        )
    )

    print(
        "\nGenerating leakage-safe "
        "historical features..."
    )

    train, validation = (
        add_historical_features(
            train,
            validation,
        )
    )

    train = clean_dataframe(
        train
    )

    validation = clean_dataframe(
        validation
    )

    print(
        "\nTrain rows:",
        f"{len(train):,}",
    )

    print(
        "Validation rows:",
        f"{len(validation):,}",
    )

    TFT_INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_parquet(
        TRAIN_PARQUET,
        index=False,
    )

    validation.to_parquet(
        VALIDATION_PARQUET,
        index=False,
    )

    print(
        "\nPrepared TFT input files:"
    )

    print(
        TRAIN_PARQUET
    )

    print(
        VALIDATION_PARQUET
    )


# ============================================================
# RAW FEATURE SCHEMA
# ============================================================

def create_raw_metadata():

    feature_spec = {}

    for feature_name in NUMERIC_FEATURES:

        feature_spec[
            feature_name
        ] = tf.io.FixedLenFeature(
            [],
            tf.float32,
        )

    for feature_name in CATEGORICAL_FEATURES:

        feature_spec[
            feature_name
        ] = tf.io.FixedLenFeature(
            [],
            tf.string,
        )

    feature_spec[LABEL_KEY] = (
        tf.io.FixedLenFeature(
            [],
            tf.int64,
        )
    )

    schema = (
        schema_utils
        .schema_from_feature_spec(
            feature_spec
        )
    )

    return (
        dataset_metadata
        .DatasetMetadata(
            schema
        )
    )


# ============================================================
# BEAM + TFT
# ============================================================

def run_beam_pipeline():

    print(
        "\nStarting Apache Beam "
        "+ TensorFlow Transform..."
    )

    raw_metadata = (
        create_raw_metadata()
    )

    # Clean outputs from previous run
    shutil.rmtree(
        TFT_OUTPUT_DIR,
        ignore_errors=True,
    )

    shutil.rmtree(
        TFT_ARTIFACT_DIR,
        ignore_errors=True,
    )

    shutil.rmtree(
        TEMP_DIR,
        ignore_errors=True,
    )

    TFT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TFT_ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pipeline_options = (
        beam.options.pipeline_options
        .PipelineOptions(
            runner="DirectRunner",
        )
    )

    with beam.Pipeline(
        options=pipeline_options
    ) as pipeline:

        # ====================================================
        # READ TRAIN
        # ====================================================

        train_data = (
            pipeline
            | "ReadTrainParquet"
            >> beam.io.parquetio.ReadFromParquet(
                str(TRAIN_PARQUET)
            )
        )

        # ====================================================
        # READ VALIDATION
        # ====================================================

        validation_data = (
            pipeline
            | "ReadValidationParquet"
            >> beam.io.parquetio.ReadFromParquet(
                str(
                    VALIDATION_PARQUET
                )
            )
        )

        with tft_beam.Context(
            temp_dir=str(TEMP_DIR)
        ):

            # ================================================
            # ANALYZE TRAIN + TRANSFORM TRAIN
            # ================================================

            (
                transformed_train_dataset,
                transform_fn,
            ) = (
                (
                    train_data,
                    raw_metadata,
                )
                | "AnalyzeAndTransformTrain"
                >> tft_beam
                .AnalyzeAndTransformDataset(
                    preprocessing_fn
                )
            )

            (
                transformed_train_data,
                transformed_metadata,
            ) = transformed_train_dataset

            # ================================================
            # TRANSFORM VALIDATION USING TRAIN TRANSFORM_FN
            # ================================================

            transformed_validation_dataset = (
                (
                    (
                        validation_data,
                        raw_metadata,
                    ),
                    transform_fn,
                )
                | "TransformValidation"
                >> tft_beam
                .TransformDataset()
            )

            (
                transformed_validation_data,
                _,
            ) = (
                transformed_validation_dataset
            )

            # ================================================
            # SAVE TFT GRAPH
            # ================================================

            _ = (
                transform_fn
                | "WriteTransformFn"
                >> tft_beam.WriteTransformFn(
                    str(
                        TFT_ARTIFACT_DIR
                    )
                )
            )

        # ====================================================
        # ENCODE AS TF.EXAMPLE
        # ====================================================

        coder = (
            coders.ExampleProtoCoder(
                transformed_metadata.schema
            )
        )

        # ====================================================
        # WRITE TRAIN TFRECORD
        # ====================================================

        _ = (
            transformed_train_data

            | "EncodeTrainExamples"
            >> beam.Map(
                coder.encode
            )

            | "WriteTrainTFRecords"
            >> beam.io.WriteToTFRecord(
                str(
                    TFT_OUTPUT_DIR
                    / "train"
                ),
                file_name_suffix=(
                    ".tfrecord"
                ),
            )
        )

        # ====================================================
        # WRITE VALIDATION TFRECORD
        # ====================================================

        _ = (
            transformed_validation_data

            | "EncodeValidationExamples"
            >> beam.Map(
                coder.encode
            )

            | "WriteValidationTFRecords"
            >> beam.io.WriteToTFRecord(
                str(
                    TFT_OUTPUT_DIR
                    / "validation"
                ),
                file_name_suffix=(
                    ".tfrecord"
                ),
            )
        )

    print(
        "\nApache Beam pipeline finished."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    prepare_datasets()

    run_beam_pipeline()

    print(
        "\n================================"
    )

    print(
        "STEP 15 COMPLETE"
    )

    print(
        "================================"
    )

    print(
        "\nTFT Transform graph:"
    )

    print(
        TFT_ARTIFACT_DIR
    )

    print(
        "\nTransformed TFRecords:"
    )

    print(
        TFT_OUTPUT_DIR
    )