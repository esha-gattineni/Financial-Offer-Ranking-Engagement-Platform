from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)

from src.models.train_historical import (
    load_dataset,
    chronological_split,
    add_historical_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_1000000.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "tensorflow_engagement.keras"
)

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "evaluation"

LABEL_COLUMN = "clicked"

# Validation thresholds
MIN_ROC_AUC = 0.68
MIN_PR_AUC = 0.34


NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "platform_category",
    "country",
    "region",
    "event_category_id",
    "promoted_category_id",
    "event_topic_id",
    "promoted_topic_id",
]


def load_validation_data():
    print("Loading full dataset...")

    dataset = load_dataset()

    print("Creating chronological split...")
    train_df, validation_df = chronological_split(dataset)

    print("Generating historical features...")
    train_df, validation_df = add_historical_features(
        train_df,
        validation_df,
    )

    print("Normalizing numeric features...")
    train_df, validation_df = normalize_numeric_features(
        train_df,
        validation_df,
    )

    return validation_df

def normalize_numeric_features(train_df, validation_df):
    train_df = train_df.copy()
    validation_df = validation_df.copy()

    for column in NUMERIC_FEATURES:
        if column not in train_df.columns:
            continue

        train_df[column] = pd.to_numeric(
            train_df[column],
            errors="coerce",
        ).fillna(0.0)

        validation_df[column] = pd.to_numeric(
            validation_df[column],
            errors="coerce",
        ).fillna(0.0)

        mean = train_df[column].mean()
        std = train_df[column].std()

        if pd.isna(std) or std == 0:
            std = 1.0

        train_df[column] = (
            train_df[column] - mean
        ) / std

        validation_df[column] = (
            validation_df[column] - mean
        ) / std

    return train_df, validation_df


def prepare_features(df):
    features = {}

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            values = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0)
                .astype("float32")
                .to_numpy()
            )

            features[col] = values

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            values = (
                df[col]
                .fillna("__MISSING__")
                .astype(str)
                .to_numpy()
            )

            features[col] = values

    return features


def main():
    print("Loading validation data...")

    validation_df = load_validation_data()

    print(
        f"Validation rows: {len(validation_df):,}"
    )

    y_true = validation_df[LABEL_COLUMN].to_numpy()

    print("Loading TensorFlow model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Preparing features...")

    features = prepare_features(validation_df)

    print("Running predictions...")

    predictions = model.predict(
        features,
        batch_size=4096,
        verbose=1,
    ).reshape(-1)

    roc_auc = roc_auc_score(
        y_true,
        predictions,
    )

    pr_auc = average_precision_score(
        y_true,
        predictions,
    )

    predicted_classes = (
        predictions >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predicted_classes,
    )

    passed = bool(
        (roc_auc >= MIN_ROC_AUC)
        and (pr_auc >= MIN_PR_AUC)
    )

    results = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "accuracy": float(accuracy),
        "thresholds": {
            "min_roc_auc": float(MIN_ROC_AUC),
            "min_pr_auc": float(MIN_PR_AUC),
        },
        "validation_passed": bool(passed),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "evaluation_results.json"
    )

    with open(output_path, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print("\nEvaluation Results")
    print("------------------")
    print(
        f"ROC-AUC:  {roc_auc:.4f}"
    )
    print(
        f"PR-AUC:   {pr_auc:.4f}"
    )
    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print("\nValidation Gate")
    print("---------------")
    print(
        f"ROC-AUC >= {MIN_ROC_AUC}: "
        f"{roc_auc >= MIN_ROC_AUC}"
    )
    print(
        f"PR-AUC >= {MIN_PR_AUC}: "
        f"{pr_auc >= MIN_PR_AUC}"
    )

    if passed:
        print("\nMODEL STATUS: PASS")
    else:
        print("\nMODEL STATUS: FAIL")

    print(
        f"\nSaved evaluation results to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":
    main()