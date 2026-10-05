from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from train_historical import (
    add_historical_features,
    chronological_split,
    load_dataset,
)


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_FILE = (
    MODEL_DIR
    / "tensorflow_engagement.keras"
)

LOGISTIC_HISTORICAL_AUC = 0.6832
LOGISTIC_HISTORICAL_PR_AUC = 0.3471

BATCH_SIZE = 4096
EPOCHS = 10


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [
    # Context
    "hour_of_day",
    "day_index",

    # Content relevance
    "same_category",
    "same_topic",
    "category_match_strength",
    "topic_match_strength",

    # Content confidence
    "event_category_confidence",
    "promoted_category_confidence",
    "event_topic_confidence",
    "promoted_topic_confidence",

    # Missing indicators
    "event_category_missing",
    "promoted_category_missing",
    "event_topic_missing",
    "promoted_topic_missing",

    # User history
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


# ============================================================
# CLEAN / NORMALIZE DATA
# ============================================================

def prepare_dataframes(train, validation):
    print("\nPreparing TensorFlow features...")

    train = train.copy()
    validation = validation.copy()

    # --------------------------------------------------------
    # Numeric preprocessing
    # --------------------------------------------------------

    for column in NUMERIC_FEATURES:

        train[column] = pd.to_numeric(
            train[column],
            errors="coerce",
        )

        validation[column] = pd.to_numeric(
            validation[column],
            errors="coerce",
        )

        # Learn median ONLY from training data
        median = train[column].median()

        if pd.isna(median):
            median = 0.0

        train[column] = (
            train[column]
            .fillna(median)
        )

        validation[column] = (
            validation[column]
            .fillna(median)
        )

        # Learn normalization only from training data
        mean = train[column].mean()
        std = train[column].std()

        if pd.isna(std) or std == 0:
            std = 1.0

        train[column] = (
            (train[column] - mean)
            / std
        ).astype("float32")

        validation[column] = (
            (validation[column] - mean)
            / std
        ).astype("float32")

    # --------------------------------------------------------
    # Categorical preprocessing
    # --------------------------------------------------------

    for column in CATEGORICAL_FEATURES:

        train[column] = (
            train[column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

        validation[column] = (
            validation[column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    return train, validation


# ============================================================
# TENSORFLOW DATASET
# ============================================================

def create_tf_dataset(
    dataframe,
    shuffle=False,
):
    features = {}

    for column in NUMERIC_FEATURES:
        features[column] = (
            dataframe[column]
            .to_numpy(dtype=np.float32)
        )

    for column in CATEGORICAL_FEATURES:
        features[column] = (
            dataframe[column]
            .astype(str)
            .to_numpy()
        )

    labels = (
        dataframe["clicked"]
        .to_numpy(dtype=np.float32)
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            features,
            labels,
        )
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=100_000,
            seed=42,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# BUILD KERAS MODEL
# ============================================================

def build_model(train):
    print("\nBuilding TensorFlow model...")

    inputs = {}
    encoded_features = []

    # --------------------------------------------------------
    # Numeric inputs
    # --------------------------------------------------------

    for feature_name in NUMERIC_FEATURES:

        input_layer = tf.keras.Input(
            shape=(1,),
            name=feature_name,
            dtype=tf.float32,
        )

        inputs[feature_name] = input_layer

        encoded_features.append(
            input_layer
        )

    # --------------------------------------------------------
    # Categorical inputs
    # --------------------------------------------------------

    for feature_name in CATEGORICAL_FEATURES:

        input_layer = tf.keras.Input(
            shape=(1,),
            name=feature_name,
            dtype=tf.string,
        )

        inputs[feature_name] = input_layer

        vocabulary = (
            train[feature_name]
            .astype(str)
            .unique()
            .tolist()
        )

        lookup = tf.keras.layers.StringLookup(
            vocabulary=vocabulary,
            output_mode="one_hot",
            num_oov_indices=1,
            name=f"{feature_name}_lookup",
        )

        encoded = lookup(
            input_layer
        )

        encoded_features.append(
            encoded
        )

    # --------------------------------------------------------
    # Combine features
    # --------------------------------------------------------

    x = tf.keras.layers.Concatenate(
        name="combined_features"
    )(
        encoded_features
    )

    # --------------------------------------------------------
    # Neural network
    # --------------------------------------------------------

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        name="dense_128",
    )(x)

    x = tf.keras.layers.BatchNormalization()(
        x
    )

    x = tf.keras.layers.Dropout(
        0.25
    )(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        name="dense_64",
    )(x)

    x = tf.keras.layers.Dropout(
        0.15
    )(x)

    x = tf.keras.layers.Dense(
        32,
        activation="relu",
        name="dense_32",
    )(x)

    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="engagement_probability",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=output,
        name="engagement_prediction_model",
    )

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(
                name="auc",
                curve="ROC",
            ),
            tf.keras.metrics.AUC(
                name="pr_auc",
                curve="PR",
            ),
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy",
            ),
        ],
    )

    return model


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\nTensorFlow version:",
        tf.__version__,
    )

    print(
        "Available devices:",
        tf.config.list_physical_devices(),
    )

    # ========================================================
    # LOAD 1M DATASET
    # ========================================================

    dataset = load_dataset()

    print(
        "\nLoaded rows:",
        f"{len(dataset):,}",
    )

    # ========================================================
    # CHRONOLOGICAL SPLIT
    # ========================================================

    train, validation = chronological_split(
        dataset
    )

    # ========================================================
    # HISTORICAL FEATURES
    # ========================================================

    train, validation = add_historical_features(
        train,
        validation,
    )

    print(
        "\nTraining shape:",
        train.shape,
    )

    print(
        "Validation shape:",
        validation.shape,
    )

    # ========================================================
    # PREPARE DATA
    # ========================================================

    train, validation = prepare_dataframes(
        train,
        validation,
    )

    train_dataset = create_tf_dataset(
        train,
        shuffle=True,
    )

    validation_dataset = create_tf_dataset(
        validation,
        shuffle=False,
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = build_model(
        train
    )

    print("\nModel summary:")

    model.summary()

    # ========================================================
    # CALLBACKS
    # ========================================================

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        )
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\nTraining TensorFlow model..."
    )

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[
            early_stopping,
        ],
        verbose=1,
    )

    # ========================================================
    # EVALUATE WITH SKLEARN
    # ========================================================

    print(
        "\nGenerating validation predictions..."
    )

    probabilities = model.predict(
        validation_dataset,
        verbose=1,
    ).reshape(-1)

    labels = (
        validation["clicked"]
        .to_numpy(dtype=np.int8)
    )

    auc = roc_auc_score(
        labels,
        probabilities,
    )

    pr_auc = average_precision_score(
        labels,
        probabilities,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\n================================"
    )

    print(
        "TENSORFLOW MODEL RESULTS"
    )

    print(
        "================================"
    )

    print(
        f"ROC-AUC: {auc:.4f}"
    )

    print(
        f"PR-AUC:  {pr_auc:.4f}"
    )

    print(
        "\n================================"
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        "================================"
    )

    print(
        f"Logistic historical AUC: "
        f"{LOGISTIC_HISTORICAL_AUC:.4f}"
    )

    print(
        f"TensorFlow AUC:          "
        f"{auc:.4f}"
    )

    print(
        f"AUC change:              "
        f"{auc - LOGISTIC_HISTORICAL_AUC:+.4f}"
    )

    print()

    print(
        f"Logistic historical PR-AUC: "
        f"{LOGISTIC_HISTORICAL_PR_AUC:.4f}"
    )

    print(
        f"TensorFlow PR-AUC:          "
        f"{pr_auc:.4f}"
    )

    print(
        f"PR-AUC change:              "
        f"{pr_auc - LOGISTIC_HISTORICAL_PR_AUC:+.4f}"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(
        MODEL_FILE
    )

    print(
        "\nTensorFlow model saved to:"
    )

    print(
        MODEL_FILE
    )

    print(
        "\nStep 13 completed successfully."
    )