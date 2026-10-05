import tensorflow as tf
import tensorflow_transform as tft


LABEL_KEY = "clicked"

NUMERIC_FEATURES = [
    # Context
    "hour_of_day",
    "day_index",

    # Content similarity
    "same_category",
    "same_topic",
    "category_match_strength",
    "topic_match_strength",

    # Confidence
    "event_category_confidence",
    "promoted_category_confidence",
    "event_topic_confidence",
    "promoted_topic_confidence",

    # Missing indicators
    "event_category_missing",
    "promoted_category_missing",
    "event_topic_missing",
    "promoted_topic_missing",

    # User behavioral history
    "user_prior_displays",

    # Ad history
    "ad_prior_impressions",
    "ad_prior_clicks",
    "ad_historical_ctr",
    "ad_has_history",

    # Campaign history
    "campaign_prior_impressions",
    "campaign_prior_clicks",
    "campaign_historical_ctr",
    "campaign_has_history",

    # Advertiser history
    "advertiser_prior_impressions",
    "advertiser_prior_clicks",
    "advertiser_historical_ctr",
    "advertiser_has_history",

    # Promoted document history
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


def transformed_name(key):
    return f"{key}_xf"


def preprocessing_fn(inputs):
    """
    TensorFlow Transform preprocessing function.

    Numeric features:
        z-score normalization using statistics learned
        ONLY from training data.

    Categorical features:
        training-data vocabulary -> integer ID.

    Label:
        passed through unchanged.
    """

    outputs = {}

    # ========================================================
    # NUMERIC FEATURES
    # ========================================================

    for feature_name in NUMERIC_FEATURES:

        feature = tf.cast(
            inputs[feature_name],
            tf.float32,
        )

        outputs[
            transformed_name(feature_name)
        ] = tft.scale_to_z_score(
            feature
        )

    # ========================================================
    # CATEGORICAL FEATURES
    # ========================================================

    for feature_name in CATEGORICAL_FEATURES:

        outputs[
            transformed_name(feature_name)
        ] = tft.compute_and_apply_vocabulary(
            inputs[feature_name],
            vocab_filename=(
                f"{feature_name}_vocab"
            ),
            num_oov_buckets=1,
        )

    # ========================================================
    # LABEL
    # ========================================================

    outputs[LABEL_KEY] = tf.cast(
        inputs[LABEL_KEY],
        tf.int64,
    )

    return outputs