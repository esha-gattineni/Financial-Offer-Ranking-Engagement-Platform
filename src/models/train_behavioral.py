from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_100000.parquet"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_FILE = (
    MODEL_DIR
    / "behavioral_model.joblib"
)


# ============================================================
# BASELINE RESULT
# ============================================================

BASELINE_AUC = 0.6608


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    print(
        "\nLoading processed training dataset..."
    )

    dataset = pd.read_parquet(
        DATA_FILE
    )

    print(
        "Dataset shape:",
        dataset.shape
    )

    return dataset


# ============================================================
# TRAIN BEHAVIORAL MODEL
# ============================================================

def train_behavioral_model(dataset):

    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    numeric_features = [
        # Context
        "hour_of_day",
        "day_index",

        # Content relevance
        "same_category",
        "same_topic",
        "category_match_strength",
        "topic_match_strength",

        # Confidence
        "event_category_confidence",
        "promoted_category_confidence",
        "event_topic_confidence",
        "promoted_topic_confidence",

        # Missing-value indicators
        "event_category_missing",
        "promoted_category_missing",
        "event_topic_missing",
        "promoted_topic_missing",

        # Behavioral features
        "user_prior_pageviews",
        "hours_since_last_pageview",
        "user_prior_views_event_document",
        "user_prior_views_promoted_document",
    ]

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    categorical_features = [
        "platform_category",
        "country",
        "region",

        "event_category_id",
        "promoted_category_id",

        "event_topic_id",
        "promoted_topic_id",
    ]

    feature_columns = (
        numeric_features
        + categorical_features
    )

    # --------------------------------------------------------
    # X / y / groups
    # --------------------------------------------------------

    X = dataset[
        feature_columns
    ].copy()

    y = dataset[
        "clicked"
    ].astype("int8")

    groups = dataset[
        "display_id"
    ]

    # --------------------------------------------------------
    # Clean numeric data
    # --------------------------------------------------------

    for column in numeric_features:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Clean categorical data
    # --------------------------------------------------------

    for column in categorical_features:
        X[column] = (
            X[column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    # ========================================================
    # GROUP-AWARE TRAIN / VALIDATION SPLIT
    # ========================================================

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_index, validation_index = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    X_train = X.iloc[
        train_index
    ]

    X_validation = X.iloc[
        validation_index
    ]

    y_train = y.iloc[
        train_index
    ]

    y_validation = y.iloc[
        validation_index
    ]

    print("\nTraining rows:")
    print(len(X_train))

    print("\nValidation rows:")
    print(len(X_validation))

    print("\nTraining click rate:")
    print(y_train.mean())

    print("\nValidation click rate:")
    print(y_validation.mean())

    # ========================================================
    # NUMERIC PREPROCESSING
    # ========================================================

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

    # ========================================================
    # CATEGORICAL PREPROCESSING
    # ========================================================

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

    # ========================================================
    # COMBINE PREPROCESSING
    # ========================================================

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

    # ========================================================
    # MODEL
    # ========================================================

    model = LogisticRegression(
        max_iter=500,
        solver="liblinear",
        random_state=42,
    )

    pipeline = Pipeline(
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

    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\nTraining behavioral model..."
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    # ========================================================
    # PREDICT
    # ========================================================

    validation_probabilities = (
        pipeline.predict_proba(
            X_validation
        )[:, 1]
    )

    validation_predictions = (
        pipeline.predict(
            X_validation
        )
    )

    # ========================================================
    # EVALUATE
    # ========================================================

    auc = roc_auc_score(
        y_validation,
        validation_probabilities,
    )

    pr_auc = average_precision_score(
        y_validation,
        validation_probabilities,
    )

    accuracy = accuracy_score(
        y_validation,
        validation_predictions,
    )

    improvement = (
        auc - BASELINE_AUC
    )

    improvement_percent = (
        improvement
        / BASELINE_AUC
        * 100
    )

    print(
        "\n================================"
    )
    print(
        "BEHAVIORAL MODEL RESULTS"
    )
    print(
        "================================"
    )

    print(
        f"\nValidation ROC-AUC: {auc:.4f}"
    )

    print(
        f"Validation PR-AUC:  {pr_auc:.4f}"
    )

    print(
        f"Validation Accuracy: {accuracy:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_validation,
            validation_predictions,
            digits=4,
        )
    )

    # ========================================================
    # COMPARE TO BASELINE
    # ========================================================

    print(
        "\n================================"
    )
    print(
        "BASELINE COMPARISON"
    )
    print(
        "================================"
    )

    print(
        f"Baseline ROC-AUC:   "
        f"{BASELINE_AUC:.4f}"
    )

    print(
        f"Behavioral ROC-AUC: "
        f"{auc:.4f}"
    )

    print(
        f"AUC change:          "
        f"{improvement:+.4f}"
    )

    print(
        f"Relative change:     "
        f"{improvement_percent:+.2f}%"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        MODEL_FILE,
    )

    print("\nModel saved to:")
    print(MODEL_FILE)

    return pipeline, auc


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    dataset = load_dataset()

    model, auc = train_behavioral_model(
        dataset
    )

    print(
        "\nStep 9 completed successfully."
    )