from pathlib import Path
import json

import pandas as pd

from src.models.train_historical import (
    load_dataset,
    chronological_split,
    add_historical_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "serving"
    / "test_request.json"
)


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


def normalize(train, validation):
    train = train.copy()
    validation = validation.copy()

    for column in NUMERIC_FEATURES:
        train[column] = pd.to_numeric(
            train[column],
            errors="coerce",
        ).fillna(0.0)

        validation[column] = pd.to_numeric(
            validation[column],
            errors="coerce",
        ).fillna(0.0)

        mean = train[column].mean()
        std = train[column].std()

        if pd.isna(std) or std == 0:
            std = 1.0

        validation[column] = (
            validation[column] - mean
        ) / std

    return validation


def main():
    print("Loading dataset...")

    dataset = load_dataset()

    train, validation = chronological_split(dataset)

    print("Creating historical features...")

    train, validation = add_historical_features(
        train,
        validation,
    )

    validation = normalize(
        train,
        validation,
    )

    row = validation.iloc[0]

    inputs = {}

    for column in NUMERIC_FEATURES:
        inputs[column] = [
            [float(row[column])]
        ]

    for column in CATEGORICAL_FEATURES:
        value = row[column]

        if pd.isna(value):
            value = "__MISSING__"

        inputs[column] = [
            [str(value)]
        ]

    request = {
        "inputs": inputs
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            request,
            f,
            indent=2,
        )

    print(f"\nCreated request:\n{OUTPUT_PATH}")
    print(f"Actual clicked label: {int(row['clicked'])}")


if __name__ == "__main__":
    main()