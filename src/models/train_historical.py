from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_1000000.parquet"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_FILE = (
    MODEL_DIR
    / "historical_model.joblib"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():
    print("\nLoading dataset...")

    dataset = pd.read_parquet(DATA_FILE)

    dataset["timestamp"] = pd.to_numeric(
        dataset["timestamp"],
        errors="coerce",
    )

    dataset = dataset.dropna(
        subset=[
            "timestamp",
            "display_id",
            "clicked",
        ]
    ).copy()

    print("Dataset shape:", dataset.shape)

    return dataset


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(dataset, train_fraction=0.80):
    """
    Split interactions chronologically while keeping all displays
    with the same timestamp on the same side of the boundary.

    This prevents ambiguous temporal boundaries when historical
    engagement features are generated from prior outcomes.
    """

    display_times = (
        dataset[["display_id", "timestamp"]]
        .drop_duplicates(subset=["display_id"])
        .sort_values(["timestamp", "display_id"])
        .reset_index(drop=True)
    )

    unique_timestamps = (
        display_times["timestamp"]
        .dropna()
        .sort_values()
        .unique()
    )

    if len(unique_timestamps) < 2:
        raise ValueError(
            "At least two unique timestamps are required "
            "for chronological splitting."
        )

    cutoff_index = int(
        len(unique_timestamps) * train_fraction
    )

    cutoff_index = max(
        1,
        min(cutoff_index, len(unique_timestamps) - 1),
    )

    validation_start_timestamp = (
        unique_timestamps[cutoff_index]
    )

    train_display_ids = set(
        display_times.loc[
            display_times["timestamp"]
            < validation_start_timestamp,
            "display_id",
        ]
    )

    validation_display_ids = set(
        display_times.loc[
            display_times["timestamp"]
            >= validation_start_timestamp,
            "display_id",
        ]
    )

    train = dataset[
        dataset["display_id"].isin(train_display_ids)
    ].copy()

    validation = dataset[
        dataset["display_id"].isin(validation_display_ids)
    ].copy()

    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)

    print("\nChronological split:")
    print(f"Train rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")

    print(
        "Max train timestamp:",
        train["timestamp"].max(),
    )

    print(
        "Min validation timestamp:",
        validation["timestamp"].min(),
    )

    assert (
        train["timestamp"].max()
        < validation["timestamp"].min()
    ), "Temporal leakage detected at split boundary."

    return train, validation# ============================================================
# TRAINING HISTORY FOR AN ENTITY
#
# Example entities:
# ad_id
# campaign_id
# advertiser_id
# promoted_document_id
# ============================================================

def add_training_entity_history(
    train,
    entity_column,
    prefix,
):
    """
    For each training row, calculate history using
    only strictly earlier timestamps.

    Rows at the same timestamp do NOT see each other's
    clicked labels.
    """

    train = train.copy()

    stats = (
        train[
            [
                entity_column,
                "timestamp",
                "clicked",
            ]
        ]
        .dropna(
            subset=[entity_column]
        )
        .groupby(
            [
                entity_column,
                "timestamp",
            ],
            as_index=False,
        )
        .agg(
            current_impressions=(
                "clicked",
                "size",
            ),
            current_clicks=(
                "clicked",
                "sum",
            ),
        )
        .sort_values(
            [
                entity_column,
                "timestamp",
            ]
        )
    )

    stats["prior_impressions"] = (
        stats.groupby(
            entity_column
        )["current_impressions"]
        .cumsum()
        - stats["current_impressions"]
    )

    stats["prior_clicks"] = (
        stats.groupby(
            entity_column
        )["current_clicks"]
        .cumsum()
        - stats["current_clicks"]
    )

    stats[
        f"{prefix}_prior_impressions"
    ] = stats["prior_impressions"]

    stats[
        f"{prefix}_prior_clicks"
    ] = stats["prior_clicks"]

    stats[
        f"{prefix}_historical_ctr"
    ] = np.where(
        stats["prior_impressions"] > 0,
        stats["prior_clicks"]
        / stats["prior_impressions"],
        0.0,
    )

    stats[
        f"{prefix}_has_history"
    ] = (
        stats["prior_impressions"] > 0
    ).astype("int8")

    keep_columns = [
        entity_column,
        "timestamp",
        f"{prefix}_prior_impressions",
        f"{prefix}_prior_clicks",
        f"{prefix}_historical_ctr",
        f"{prefix}_has_history",
    ]

    train = train.merge(
        stats[keep_columns],
        on=[
            entity_column,
            "timestamp",
        ],
        how="left",
        validate="many_to_one",
    )

    new_columns = keep_columns[2:]

    train[new_columns] = (
        train[new_columns]
        .fillna(0)
    )

    return train


# ============================================================
# VALIDATION ENTITY HISTORY
# ============================================================

def add_validation_entity_history(
    train,
    validation,
    entity_column,
    prefix,
):
    """
    Validation rows use ONLY training-period history.

    No validation clicked labels are used to create
    validation features.
    """

    validation = validation.copy()

    history = (
        train[
            [
                entity_column,
                "clicked",
            ]
        ]
        .dropna(
            subset=[entity_column]
        )
        .groupby(
            entity_column,
            as_index=False,
        )
        .agg(
            prior_impressions=(
                "clicked",
                "size",
            ),
            prior_clicks=(
                "clicked",
                "sum",
            ),
        )
    )

    history[
        f"{prefix}_prior_impressions"
    ] = history["prior_impressions"]

    history[
        f"{prefix}_prior_clicks"
    ] = history["prior_clicks"]

    history[
        f"{prefix}_historical_ctr"
    ] = np.where(
        history["prior_impressions"] > 0,
        history["prior_clicks"]
        / history["prior_impressions"],
        0.0,
    )

    history[
        f"{prefix}_has_history"
    ] = 1

    history = history[
        [
            entity_column,
            f"{prefix}_prior_impressions",
            f"{prefix}_prior_clicks",
            f"{prefix}_historical_ctr",
            f"{prefix}_has_history",
        ]
    ]

    validation = validation.merge(
        history,
        on=entity_column,
        how="left",
        validate="many_to_one",
    )

    new_columns = [
        f"{prefix}_prior_impressions",
        f"{prefix}_prior_clicks",
        f"{prefix}_historical_ctr",
        f"{prefix}_has_history",
    ]

    validation[new_columns] = (
        validation[new_columns]
        .fillna(0)
    )

    return validation


# ============================================================
# USER ACTIVITY HISTORY
# ============================================================

def add_user_history(
    train,
    validation,
):
    """
    Count previous displays for each user.

    Does not use validation labels.
    """

    train = train.copy()
    validation = validation.copy()

    # -----------------------------------------
    # Build one row per display
    # -----------------------------------------

    display_level = (
        train[
            [
                "display_id",
                "uuid",
                "timestamp",
            ]
        ]
        .drop_duplicates(
            subset=["display_id"]
        )
        .sort_values(
            [
                "uuid",
                "timestamp",
                "display_id",
            ]
        )
    )

    # -----------------------------------------
    # Training feature
    # -----------------------------------------

    display_level[
        "user_prior_displays"
    ] = (
        display_level.groupby(
            "uuid"
        )
        .cumcount()
    )

    train = train.merge(
        display_level[
            [
                "display_id",
                "user_prior_displays",
            ]
        ],
        on="display_id",
        how="left",
        validate="many_to_one",
    )

    # -----------------------------------------
    # Validation feature
    # -----------------------------------------

    user_counts = (
        display_level.groupby(
            "uuid"
        )["display_id"]
        .nunique()
        .rename(
            "user_prior_displays"
        )
        .reset_index()
    )

    validation = validation.merge(
        user_counts,
        on="uuid",
        how="left",
        validate="many_to_one",
    )

    train[
        "user_prior_displays"
    ] = (
        train["user_prior_displays"]
        .fillna(0)
    )

    validation[
        "user_prior_displays"
    ] = (
        validation["user_prior_displays"]
        .fillna(0)
    )

    return train, validation


# ============================================================
# CREATE HISTORICAL FEATURES
# ============================================================

def add_historical_features(
    train,
    validation,
):
    print(
        "\nCreating historical engagement features..."
    )

    entities = [
        (
            "ad_id",
            "ad",
        ),
        (
            "campaign_id",
            "campaign",
        ),
        (
            "advertiser_id",
            "advertiser",
        ),
        (
            "promoted_document_id",
            "promoted_document",
        ),
    ]

    for entity_column, prefix in entities:
        print(
            f"  Building {prefix} history..."
        )

        train = (
            add_training_entity_history(
                train,
                entity_column,
                prefix,
            )
        )

        validation = (
            add_validation_entity_history(
                train,
                validation,
                entity_column,
                prefix,
            )
        )

    train, validation = add_user_history(
        train,
        validation,
    )

    print(
        "Historical feature engineering complete."
    )

    return train, validation


# ============================================================
# BUILD MODEL PIPELINE
# ============================================================

def build_model(
    numeric_features,
    categorical_features,
):
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    model = LogisticRegression(
        max_iter=500,
        solver="liblinear",
        random_state=42,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# PREPARE X / Y
# ============================================================

def prepare_features(
    dataset,
    numeric_features,
    categorical_features,
):
    feature_columns = (
        numeric_features
        + categorical_features
    )

    X = dataset[
        feature_columns
    ].copy()

    y = dataset[
        "clicked"
    ].astype("int8")

    for column in numeric_features:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    for column in categorical_features:
        X[column] = (
            X[column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    return X, y


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    name,
    pipeline,
    train,
    validation,
    numeric_features,
    categorical_features,
):
    X_train, y_train = prepare_features(
        train,
        numeric_features,
        categorical_features,
    )

    X_validation, y_validation = (
        prepare_features(
            validation,
            numeric_features,
            categorical_features,
        )
    )

    print(
        f"\nTraining {name}..."
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    probabilities = (
        pipeline.predict_proba(
            X_validation
        )[:, 1]
    )

    predictions = pipeline.predict(
        X_validation
    )

    auc = roc_auc_score(
        y_validation,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_validation,
        probabilities,
    )

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    print(
        "\n================================"
    )
    print(name.upper())
    print(
        "================================"
    )

    print(
        f"ROC-AUC:  {auc:.4f}"
    )

    print(
        f"PR-AUC:   {pr_auc:.4f}"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    return pipeline, auc, pr_auc


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    dataset = load_dataset()

    train, validation = (
        chronological_split(dataset)
    )

    # ========================================================
    # ADD HISTORICAL FEATURES
    # ========================================================

    train, validation = (
        add_historical_features(
            train,
            validation,
        )
    )

    # ========================================================
    # BASE FEATURES
    # ========================================================

    base_numeric_features = [
        "hour_of_day",
        "day_index",

        "same_category",
        "same_topic",

        "category_match_strength",
        "topic_match_strength",

        "event_category_confidence",
        "promoted_category_confidence",

        "event_topic_confidence",
        "promoted_topic_confidence",

        "event_category_missing",
        "promoted_category_missing",

        "event_topic_missing",
        "promoted_topic_missing",

        # Sparse page-view features
        "user_prior_pageviews",
        "hours_since_last_pageview",
        "user_prior_views_event_document",
        "user_prior_views_promoted_document",
    ]

    categorical_features = [
        "platform_category",
        "country",
        "region",

        "event_category_id",
        "promoted_category_id",

        "event_topic_id",
        "promoted_topic_id",
    ]

    # ========================================================
    # HISTORICAL FEATURES
    # ========================================================

    historical_features = [
        "user_prior_displays",

        "ad_prior_impressions",
        "ad_prior_clicks",
        "ad_historical_ctr",
        "ad_has_history",

        "campaign_prior_impressions",
        "campaign_prior_clicks",
        "campaign_historical_ctr",
        "campaign_has_history",

        "advertiser_prior_impressions",
        "advertiser_prior_clicks",
        "advertiser_historical_ctr",
        "advertiser_has_history",

        "promoted_document_prior_impressions",
        "promoted_document_prior_clicks",
        "promoted_document_historical_ctr",
        "promoted_document_has_history",
    ]

    # ========================================================
    # MODEL 1: CHRONOLOGICAL BASELINE
    # ========================================================

    baseline_pipeline = build_model(
        base_numeric_features,
        categorical_features,
    )

    (
        baseline_pipeline,
        baseline_auc,
        baseline_pr_auc,
    ) = evaluate_model(
        "Chronological baseline",
        baseline_pipeline,
        train,
        validation,
        base_numeric_features,
        categorical_features,
    )

    # ========================================================
    # MODEL 2: + HISTORICAL FEATURES
    # ========================================================

    enhanced_numeric_features = (
        base_numeric_features
        + historical_features
    )

    historical_pipeline = build_model(
        enhanced_numeric_features,
        categorical_features,
    )

    (
        historical_pipeline,
        historical_auc,
        historical_pr_auc,
    ) = evaluate_model(
        "Historical engagement model",
        historical_pipeline,
        train,
        validation,
        enhanced_numeric_features,
        categorical_features,
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    auc_change = (
        historical_auc
        - baseline_auc
    )

    pr_auc_change = (
        historical_pr_auc
        - baseline_pr_auc
    )

    print(
        "\n================================"
    )
    print(
        "FINAL COMPARISON"
    )
    print(
        "================================"
    )

    print(
        f"Chronological baseline AUC: "
        f"{baseline_auc:.4f}"
    )

    print(
        f"Historical model AUC:       "
        f"{historical_auc:.4f}"
    )

    print(
        f"AUC improvement:            "
        f"{auc_change:+.4f}"
    )

    print()

    print(
        f"Chronological baseline PR-AUC: "
        f"{baseline_pr_auc:.4f}"
    )

    print(
        f"Historical model PR-AUC:       "
        f"{historical_pr_auc:.4f}"
    )

    print(
        f"PR-AUC improvement:            "
        f"{pr_auc_change:+.4f}"
    )

    # ========================================================
    # HISTORY COVERAGE
    # ========================================================

    print(
        "\n================================"
    )
    print(
        "VALIDATION HISTORY COVERAGE"
    )
    print(
        "================================"
    )

    coverage_columns = [
        "ad_has_history",
        "campaign_has_history",
        "advertiser_has_history",
        "promoted_document_has_history",
    ]

    for column in coverage_columns:
        coverage = (
            validation[column].mean()
            * 100
        )

        print(
            f"{column}: "
            f"{coverage:.2f}%"
        )

    # ========================================================
    # SAVE ENHANCED MODEL
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        historical_pipeline,
        MODEL_FILE,
    )

    print("\nSaved historical model:")
    print(MODEL_FILE)

    print(
        "\nStep 11 completed successfully."
    )
