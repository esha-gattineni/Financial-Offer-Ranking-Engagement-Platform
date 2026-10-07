from pathlib import Path

import argparse
import numpy as np
import pandas as pd

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

CLICKS_FILE = RAW_DATA_DIR / "clicks_train.csv.zip"
EVENTS_FILE = RAW_DATA_DIR / "events.csv.zip"
PROMOTED_FILE = RAW_DATA_DIR / "promoted_content.csv.zip"
CATEGORIES_FILE = RAW_DATA_DIR / "documents_categories.csv.zip"
TOPICS_FILE = RAW_DATA_DIR / "documents_topics.csv.zip"
PAGE_VIEWS_FILE = RAW_DATA_DIR / "page_views_sample.csv.zip"


# ============================================================
# STEP 1: LOAD CLICK DATA
# ============================================================

def load_clicks(num_rows=1_000_000):
    print(f"\nLoading {num_rows:,} rows from clicks_train...")

    clicks = pd.read_csv(
        CLICKS_FILE,
        compression="zip",
        nrows=num_rows,
        usecols=[
            "display_id",
            "ad_id",
            "clicked",
        ],
    )

    print("Click rows loaded:", len(clicks))

    return clicks


# ============================================================
# STEP 2: LOAD EVENT DATA
# ============================================================

def load_events(display_ids):
    print("\nLoading matching events...")

    matching_chunks = []

    for chunk in pd.read_csv(
        EVENTS_FILE,
        compression="zip",
        usecols=[
            "display_id",
            "uuid",
            "document_id",
            "timestamp",
            "platform",
            "geo_location",
        ],
        chunksize=500_000,
    ):
        filtered_chunk = chunk[
            chunk["display_id"].isin(display_ids)
        ]

        if not filtered_chunk.empty:
            matching_chunks.append(filtered_chunk)

    if not matching_chunks:
        raise ValueError("No matching events found.")

    events = pd.concat(
        matching_chunks,
        ignore_index=True,
    )

    events = events.rename(
        columns={
            "document_id": "event_document_id",
        }
    )

    print("Matching events loaded:", len(events))

    return events


# ============================================================
# STEP 3: LOAD PROMOTED CONTENT
# ============================================================

def load_promoted_content(ad_ids):
    print("\nLoading promoted content...")

    promoted = pd.read_csv(
        PROMOTED_FILE,
        compression="zip",
        usecols=[
            "ad_id",
            "document_id",
            "campaign_id",
            "advertiser_id",
        ],
    )

    promoted = promoted[
        promoted["ad_id"].isin(ad_ids)
    ].copy()

    promoted = promoted.rename(
        columns={
            "document_id": "promoted_document_id",
        }
    )

    print("Matching promoted rows loaded:", len(promoted))

    return promoted


# ============================================================
# STEP 4: LOAD DOCUMENT CATEGORIES
# ============================================================

def load_document_categories(document_ids):
    print("\nLoading document categories...")

    matching_chunks = []

    for chunk in pd.read_csv(
        CATEGORIES_FILE,
        compression="zip",
        usecols=[
            "document_id",
            "category_id",
            "confidence_level",
        ],
        chunksize=500_000,
    ):
        filtered_chunk = chunk[
            chunk["document_id"].isin(document_ids)
        ]

        if not filtered_chunk.empty:
            matching_chunks.append(filtered_chunk)

    if not matching_chunks:
        raise ValueError(
            "No matching document categories found."
        )

    categories = pd.concat(
        matching_chunks,
        ignore_index=True,
    )

    print(
        "Category rows before filtering:",
        len(categories),
    )

    # Highest-confidence category first
    categories = categories.sort_values(
        by="confidence_level",
        ascending=False,
    )

    # Keep one category per document
    categories = categories.drop_duplicates(
        subset=["document_id"],
        keep="first",
    )

    print(
        "Unique documents with category:",
        len(categories),
    )

    return categories


# ============================================================
# STEP 5: LOAD DOCUMENT TOPICS
# ============================================================

def load_document_topics(document_ids):
    print("\nLoading document topics...")

    matching_chunks = []

    for chunk in pd.read_csv(
        TOPICS_FILE,
        compression="zip",
        usecols=[
            "document_id",
            "topic_id",
            "confidence_level",
        ],
        chunksize=500_000,
    ):
        filtered_chunk = chunk[
            chunk["document_id"].isin(document_ids)
        ]

        if not filtered_chunk.empty:
            matching_chunks.append(filtered_chunk)

    if not matching_chunks:
        raise ValueError(
            "No matching document topics found."
        )

    topics = pd.concat(
        matching_chunks,
        ignore_index=True,
    )

    print(
        "Topic rows before filtering:",
        len(topics),
    )

    # Highest-confidence topic first
    topics = topics.sort_values(
        by="confidence_level",
        ascending=False,
    )

    # Keep one topic per document
    topics = topics.drop_duplicates(
        subset=["document_id"],
        keep="first",
    )

    print(
        "Unique documents with topic:",
        len(topics),
    )

    return topics


# ============================================================
# STEP 6: CONTEXT / CONTENT FEATURE ENGINEERING
# ============================================================

def engineer_features(dataset):
    print("\nEngineering context/content features...")

    dataset = dataset.copy()

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    milliseconds_per_hour = 60 * 60 * 1000
    milliseconds_per_day = 24 * milliseconds_per_hour

    dataset["hour_of_day"] = (
        (
            dataset["timestamp"]
            // milliseconds_per_hour
        )
        % 24
    ).astype("int8")

    dataset["day_index"] = (
        dataset["timestamp"]
        // milliseconds_per_day
    ).astype("int16")

    # --------------------------------------------------------
    # Geography
    #
    # Example:
    # US>CA>807
    # --------------------------------------------------------

    geo_parts = (
        dataset["geo_location"]
        .astype("string")
        .str.extract(
            r"^(?P<country>[^>]+)"
            r"(?:>(?P<region>[^>]+))?"
            r"(?:>(?P<dma>[^>]+))?"
        )
    )

    dataset["country"] = geo_parts["country"]
    dataset["region"] = geo_parts["region"]
    dataset["dma"] = geo_parts["dma"]

    # --------------------------------------------------------
    # Platform
    # --------------------------------------------------------

    dataset["platform_category"] = (
        dataset["platform"]
        .astype("string")
        .fillna("unknown")
    )

    # --------------------------------------------------------
    # Missing-value indicators
    # --------------------------------------------------------

    dataset["event_category_missing"] = (
        dataset["event_category_id"].isna()
    ).astype("int8")

    dataset["promoted_category_missing"] = (
        dataset["promoted_category_id"].isna()
    ).astype("int8")

    dataset["event_topic_missing"] = (
        dataset["event_topic_id"].isna()
    ).astype("int8")

    dataset["promoted_topic_missing"] = (
        dataset["promoted_topic_id"].isna()
    ).astype("int8")

    # --------------------------------------------------------
    # Category match
    # --------------------------------------------------------

    valid_category_match = (
        dataset["event_category_id"].notna()
        & dataset["promoted_category_id"].notna()
        & (
            dataset["event_category_id"]
            == dataset["promoted_category_id"]
        )
    )

    dataset["same_category"] = (
        valid_category_match.astype("int8")
    )

    # --------------------------------------------------------
    # Topic match
    # --------------------------------------------------------

    valid_topic_match = (
        dataset["event_topic_id"].notna()
        & dataset["promoted_topic_id"].notna()
        & (
            dataset["event_topic_id"]
            == dataset["promoted_topic_id"]
        )
    )

    dataset["same_topic"] = (
        valid_topic_match.astype("int8")
    )

    # --------------------------------------------------------
    # Category match strength
    # --------------------------------------------------------

    dataset["category_match_strength"] = 0.0

    category_mask = (
        dataset["same_category"] == 1
    )

    dataset.loc[
        category_mask,
        "category_match_strength",
    ] = (
        dataset.loc[
            category_mask,
            "event_category_confidence",
        ]
        * dataset.loc[
            category_mask,
            "promoted_category_confidence",
        ]
    )

    # --------------------------------------------------------
    # Topic match strength
    # --------------------------------------------------------

    dataset["topic_match_strength"] = 0.0

    topic_mask = (
        dataset["same_topic"] == 1
    )

    dataset.loc[
        topic_mask,
        "topic_match_strength",
    ] = (
        dataset.loc[
            topic_mask,
            "event_topic_confidence",
        ]
        * dataset.loc[
            topic_mask,
            "promoted_topic_confidence",
        ]
    )

    print(
        "Context/content feature engineering complete."
    )

    return dataset


# ============================================================
# STEP 8: LOAD PAGE-VIEW HISTORY
# ============================================================

def load_page_views(user_ids):
    print("\nLoading relevant page-view history...")

    matching_chunks = []

    for chunk in pd.read_csv(
        PAGE_VIEWS_FILE,
        compression="zip",
        usecols=[
            "uuid",
            "document_id",
            "timestamp",
        ],
        chunksize=500_000,
    ):
        filtered_chunk = chunk[
            chunk["uuid"].isin(user_ids)
        ]

        if not filtered_chunk.empty:
            matching_chunks.append(
                filtered_chunk
            )

    if not matching_chunks:
        print(
            "No matching page-view history found."
        )

        return pd.DataFrame(
            columns=[
                "uuid",
                "document_id",
                "timestamp",
            ]
        )

    page_views = pd.concat(
        matching_chunks,
        ignore_index=True,
    )

    page_views = page_views.dropna(
        subset=[
            "uuid",
            "document_id",
            "timestamp",
        ]
    )

    print(
        "Relevant page-view rows:",
        len(page_views),
    )

    return page_views


# ============================================================
# STEP 8: BEHAVIORAL FEATURE ENGINEERING
# ============================================================

def add_behavioral_features(
    dataset,
    page_views,
):
    print(
        "\nCreating behavioral features..."
    )

    dataset = dataset.copy()
    page_views = page_views.copy()

    # --------------------------------------------------------
    # Clean numeric fields
    # --------------------------------------------------------

    dataset["timestamp"] = pd.to_numeric(
        dataset["timestamp"],
        errors="coerce",
    )

    dataset["event_document_id"] = pd.to_numeric(
        dataset["event_document_id"],
        errors="coerce",
    )

    dataset["promoted_document_id"] = pd.to_numeric(
        dataset["promoted_document_id"],
        errors="coerce",
    )

    page_views["timestamp"] = pd.to_numeric(
        page_views["timestamp"],
        errors="coerce",
    )

    page_views["document_id"] = pd.to_numeric(
        page_views["document_id"],
        errors="coerce",
    )

    page_views = page_views.dropna(
        subset=[
            "uuid",
            "timestamp",
            "document_id",
        ]
    )

    # --------------------------------------------------------
    # User-level page-view history
    # --------------------------------------------------------

    user_history = {}

    for user_id, group in page_views.groupby(
        "uuid",
        sort=False,
    ):
        user_history[user_id] = np.sort(
            group["timestamp"].to_numpy()
        )

    # --------------------------------------------------------
    # User + document history
    # --------------------------------------------------------

    user_document_history = {}

    for (
        user_id,
        document_id,
    ), group in page_views.groupby(
        [
            "uuid",
            "document_id",
        ],
        sort=False,
    ):
        user_document_history[
            (user_id, document_id)
        ] = np.sort(
            group["timestamp"].to_numpy()
        )

    # --------------------------------------------------------
    # Feature 1:
    # Number of page views before current impression
    # --------------------------------------------------------

    def get_prior_pageviews(row):
        timestamps = user_history.get(
            row["uuid"]
        )

        if timestamps is None:
            return 0

        return int(
            np.searchsorted(
                timestamps,
                row["timestamp"],
                side="left",
            )
        )

    dataset["user_prior_pageviews"] = (
        dataset.apply(
            get_prior_pageviews,
            axis=1,
        )
    )

    # --------------------------------------------------------
    # Feature 2:
    # Hours since previous page view
    # --------------------------------------------------------

    def get_recency_hours(row):
        timestamps = user_history.get(
            row["uuid"]
        )

        if timestamps is None:
            return -1.0

        position = np.searchsorted(
            timestamps,
            row["timestamp"],
            side="left",
        )

        if position == 0:
            return -1.0

        previous_timestamp = (
            timestamps[position - 1]
        )

        milliseconds_difference = (
            row["timestamp"]
            - previous_timestamp
        )

        return (
            milliseconds_difference
            / 3_600_000
        )

    dataset["hours_since_last_pageview"] = (
        dataset.apply(
            get_recency_hours,
            axis=1,
        )
    )

    # --------------------------------------------------------
    # Feature 3:
    # Previous views of current event document
    # --------------------------------------------------------

    def get_prior_event_document_views(row):
        if pd.isna(
            row["event_document_id"]
        ):
            return 0

        key = (
            row["uuid"],
            row["event_document_id"],
        )

        timestamps = (
            user_document_history.get(key)
        )

        if timestamps is None:
            return 0

        return int(
            np.searchsorted(
                timestamps,
                row["timestamp"],
                side="left",
            )
        )

    dataset[
        "user_prior_views_event_document"
    ] = dataset.apply(
        get_prior_event_document_views,
        axis=1,
    )

    # --------------------------------------------------------
    # Feature 4:
    # Previous views of promoted document
    # --------------------------------------------------------

    def get_prior_promoted_document_views(row):
        if pd.isna(
            row["promoted_document_id"]
        ):
            return 0

        key = (
            row["uuid"],
            row["promoted_document_id"],
        )

        timestamps = (
            user_document_history.get(key)
        )

        if timestamps is None:
            return 0

        return int(
            np.searchsorted(
                timestamps,
                row["timestamp"],
                side="left",
            )
        )

    dataset[
        "user_prior_views_promoted_document"
    ] = dataset.apply(
        get_prior_promoted_document_views,
        axis=1,
    )

    print(
        "Behavioral feature engineering complete."
    )

    return dataset

def add_empty_pageview_features(dataset):
    """
    Used when page_views_sample is skipped.

    These columns are still created so downstream
    training code does not break.
    """

    dataset = dataset.copy()

    dataset["user_prior_pageviews"] = 0
    dataset["hours_since_last_pageview"] = -1.0
    dataset["user_prior_views_event_document"] = 0
    dataset["user_prior_views_promoted_document"] = 0

    return dataset

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rows",
        type=int,
        default=100_000,
        help="Number of click rows to process",
    )

    parser.add_argument(
        "--include-pageviews",
        action="store_true",
        help="Include sparse page_views_sample behavioral features",
    )

    args = parser.parse_args()

    NUM_ROWS = args.rows

    clicks = load_clicks(
        num_rows=NUM_ROWS
    )

    print("\nClicks shape:")
    print(clicks.shape)

    print("\nClicks columns:")
    print(clicks.columns.tolist())

    print("\nFirst 5 click rows:")
    print(clicks.head())

    # ========================================================
    # STEP 2: LOAD + JOIN EVENTS
    # ========================================================

    display_ids = set(
        clicks["display_id"].unique()
    )

    print("\nUnique display IDs:")
    print(len(display_ids))

    events = load_events(
        display_ids
    )

    print("\nEvents shape:")
    print(events.shape)

    dataset = clicks.merge(
        events,
        on="display_id",
        how="left",
        validate="many_to_one",
    )

    print("\nDataset after events join:")
    print(dataset.shape)

    print("\nMissing event rows:")
    print(
        dataset["uuid"]
        .isna()
        .sum()
    )

    assert len(dataset) == len(clicks), (
        "ERROR: Events join changed row count."
    )

    # ========================================================
    # STEP 3: LOAD + JOIN PROMOTED CONTENT
    # ========================================================

    ad_ids = set(
        clicks["ad_id"].unique()
    )

    print("\nUnique ad IDs:")
    print(len(ad_ids))

    promoted = load_promoted_content(
        ad_ids
    )

    dataset = dataset.merge(
        promoted,
        on="ad_id",
        how="left",
        validate="many_to_one",
    )

    print(
        "\nDataset after promoted-content join:"
    )
    print(dataset.shape)

    print("\nMissing promoted documents:")
    print(
        dataset["promoted_document_id"]
        .isna()
        .sum()
    )

    assert len(dataset) == len(clicks), (
        "ERROR: Promoted-content join "
        "changed row count."
    )

    # ========================================================
    # STEP 4: DOCUMENT CATEGORIES
    # ========================================================

    event_document_ids = set(
        dataset["event_document_id"]
        .dropna()
        .unique()
    )

    promoted_document_ids = set(
        dataset["promoted_document_id"]
        .dropna()
        .unique()
    )

    document_ids = (
        event_document_ids
        | promoted_document_ids
    )

    print("\nUnique documents needed:")
    print(len(document_ids))

    categories = load_document_categories(
        document_ids
    )

    # Event categories
    event_categories = categories.rename(
        columns={
            "document_id":
                "event_document_id",
            "category_id":
                "event_category_id",
            "confidence_level":
                "event_category_confidence",
        }
    )

    dataset = dataset.merge(
        event_categories,
        on="event_document_id",
        how="left",
        validate="many_to_one",
    )

    # Promoted categories
    promoted_categories = categories.rename(
        columns={
            "document_id":
                "promoted_document_id",
            "category_id":
                "promoted_category_id",
            "confidence_level":
                "promoted_category_confidence",
        }
    )

    dataset = dataset.merge(
        promoted_categories,
        on="promoted_document_id",
        how="left",
        validate="many_to_one",
    )

    print(
        "\nDataset after category joins:"
    )
    print(dataset.shape)

    assert len(dataset) == len(clicks), (
        "ERROR: Category join changed row count."
    )

    # ========================================================
    # STEP 5: DOCUMENT TOPICS
    # ========================================================

    topics = load_document_topics(
        document_ids
    )

    # Event topics
    event_topics = topics.rename(
        columns={
            "document_id":
                "event_document_id",
            "topic_id":
                "event_topic_id",
            "confidence_level":
                "event_topic_confidence",
        }
    )

    dataset = dataset.merge(
        event_topics,
        on="event_document_id",
        how="left",
        validate="many_to_one",
    )

    # Promoted topics
    promoted_topics = topics.rename(
        columns={
            "document_id":
                "promoted_document_id",
            "topic_id":
                "promoted_topic_id",
            "confidence_level":
                "promoted_topic_confidence",
        }
    )

    dataset = dataset.merge(
        promoted_topics,
        on="promoted_document_id",
        how="left",
        validate="many_to_one",
    )

    print(
        "\nDataset after topic joins:"
    )
    print(dataset.shape)

    assert len(dataset) == len(clicks), (
        "ERROR: Topic join changed row count."
    )

    print(
        "\nSteps 1-5 completed successfully."
    )

    # ========================================================
    # STEP 6: CONTEXT + CONTENT FEATURES
    # ========================================================

    dataset = engineer_features(
        dataset
    )

    print(
        "\nDataset after feature engineering:"
    )
    print(dataset.shape)

    print("\nEngineered feature sample:")
    print(
        dataset[
            [
                "timestamp",
                "hour_of_day",
                "day_index",
                "geo_location",
                "country",
                "region",
                "dma",
                "platform_category",
                "same_category",
                "category_match_strength",
                "same_topic",
                "topic_match_strength",
            ]
        ].head(10)
    )

    assert len(dataset) == len(clicks), (
        "ERROR: Feature engineering "
        "changed row count."
    )

    print(
        "\nStep 6 completed successfully."
    )

    # ========================================================
    # STEP 8: OPTIONAL PAGE-VIEW FEATURES
    # ========================================================

    if args.include_pageviews:

        user_ids = set(
            dataset["uuid"]
            .dropna()
            .unique()
        )

        print("\nUnique users:")
        print(len(user_ids))

        page_views = load_page_views(
            user_ids
        )

        dataset = add_behavioral_features(
            dataset,
            page_views,
        )

        print(
            "\nPage-view behavioral features added."
        )

    else:

        print(
            "\nSkipping page_views_sample "
            "(low coverage in previous experiment)."
        )

        dataset = add_empty_pageview_features(
            dataset
        )

    assert len(dataset) == len(clicks)
    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print("\n================================")
    print("FINAL DATASET")
    print("================================")

    print("\nShape:")
    print(dataset.shape)

    print("\nClick rate:")
    print(dataset["clicked"].mean())

    print("\nTotal columns:")
    print(len(dataset.columns))

    print("\nColumns:")
    print(dataset.columns.tolist())

    # ========================================================
    # SAVE DATASET
    # ========================================================

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
    PROCESSED_DATA_DIR
    / f"training_{NUM_ROWS}.parquet"
    )

    dataset.to_parquet(
        output_file,
        index=False,
    )

    print("\nSaved processed dataset to:")
    print(output_file)

    print(
        "\nDataset build completed successfully."
    )