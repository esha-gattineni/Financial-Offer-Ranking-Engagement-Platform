from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PARQUET = PROJECT_ROOT / "data" / "processed" / "tft_input" / "train.parquet"
EVAL_PARQUET = PROJECT_ROOT / "data" / "processed" / "tft_input" / "validation.parquet"

OUTPUT_DIR = PROJECT_ROOT / "data" / "tfx_input"


CATEGORICAL_FEATURES = [
    "platform_category",
    "country",
    "region",
    "event_category_id",
    "promoted_category_id",
    "event_topic_id",
    "promoted_topic_id",
]


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Prefix categorical values so CSV inference treats them as strings.
    for column in CATEGORICAL_FEATURES:
        if column in df.columns:
            df[column] = (
                df[column]
                .fillna("__MISSING__")
                .astype(str)
                .map(lambda value: f"{column}={value}")
            )

    return df


def main():
    print("Loading TFT train data...")
    train_df = pd.read_parquet(TRAIN_PARQUET)

    print("Loading TFT validation data...")
    eval_df = pd.read_parquet(EVAL_PARQUET)

    train_df = prepare_dataframe(train_df)
    eval_df = prepare_dataframe(eval_df)

    train_dir = OUTPUT_DIR / "train"
    eval_dir = OUTPUT_DIR / "eval"

    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    train_path = train_dir / "data.csv"
    eval_path = eval_dir / "data.csv"

    print("Writing TFX training CSV...")
    train_df.to_csv(train_path, index=False)

    print("Writing TFX evaluation CSV...")
    eval_df.to_csv(eval_path, index=False)

    print("\nTFX input prepared successfully.")
    print(f"Train: {train_path}")
    print(f"Eval:  {eval_path}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Eval rows:  {len(eval_df):,}")


if __name__ == "__main__":
    main()