from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_100000.parquet"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_FILE = MODEL_DIR / "baseline_model.joblib"


# --------------------------------------------------
# Load processed data
# --------------------------------------------------

def load_dataset():
    print("Loading processed training dataset...")

    dataset = pd.read_parquet(DATA_FILE)

    print("Dataset shape:", dataset.shape)

    return dataset


# --------------------------------------------------
# Train baseline
# --------------------------------------------------

def train_baseline(dataset):

    # ==================================================
    # Baseline features
    # ==================================================

    numeric_features = [
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

    feature_columns = (
        numeric_features
        + categorical_features
    )

    X = dataset[feature_columns].copy()

    y = dataset["clicked"].astype("int8")

    groups = dataset["display_id"]


    # --------------------------------------------------
    # Clean numeric columns
    # --------------------------------------------------

    for column in numeric_features:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )


    # --------------------------------------------------
    # Clean categorical columns
    # --------------------------------------------------

    for column in categorical_features:
        X[column] = (
            X[column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    # ==================================================
    # Group-aware train / validation split
    # ==================================================

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

    X_train = X.iloc[train_index]
    X_validation = X.iloc[validation_index]

    y_train = y.iloc[train_index]
    y_validation = y.iloc[validation_index]


    print("\nTraining rows:")
    print(len(X_train))

    print("\nValidation rows:")
    print(len(X_validation))

    print("\nTraining click rate:")
    print(y_train.mean())

    print("\nValidation click rate:")
    print(y_validation.mean())


    # ==================================================
    # Numeric preprocessing
    # ==================================================

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )


    # ==================================================
    # Categorical preprocessing
    # ==================================================
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
    
    # ==================================================
    # Combine preprocessing
    # ==================================================

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


    # ==================================================
    # Baseline model
    # ==================================================

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


    # ==================================================
    # Train
    # ==================================================

    print("\nTraining baseline model...")

    pipeline.fit(
        X_train,
        y_train,
    )

    print("\nMissing values after preprocessing cleanup:")
    print(X_train.isna().sum())

    print("\nFeature dtypes:")
    print(X_train.dtypes)


    # ==================================================
    # Predictions
    # ==================================================

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


    # ==================================================
    # Evaluation
    # ==================================================

    auc = roc_auc_score(
        y_validation,
        validation_probabilities,
    )

    accuracy = accuracy_score(
        y_validation,
        validation_predictions,
    )


    print("\n==============================")
    print("BASELINE RESULTS")
    print("==============================")

    print(
        f"Validation ROC-AUC: {auc:.4f}"
    )

    print(
        f"Validation Accuracy: {accuracy:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_validation,
            validation_predictions,
            digits=4,
        )
    )


    # ==================================================
    # Save model
    # ==================================================

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


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    dataset = load_dataset()

    model, auc = train_baseline(
        dataset
    )

    print("\nStep 7 completed successfully.")

    print(
        f"Baseline AUC recorded: {auc:.4f}"
    )