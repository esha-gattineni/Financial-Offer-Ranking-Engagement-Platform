from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_100000.parquet"
)


def main():
    df = pd.read_parquet(DATA_FILE)

    behavioral_features = [
        "user_prior_pageviews",
        "hours_since_last_pageview",
        "user_prior_views_event_document",
        "user_prior_views_promoted_document",
    ]

    print("\nDataset shape:")
    print(df.shape)

    print("\nBehavioral feature statistics:")
    print(
        df[behavioral_features]
        .describe()
        .T
    )

    print("\nRows with prior page views:")
    prior_pageview_rows = (
        df["user_prior_pageviews"] > 0
    )

    print(
        f"{prior_pageview_rows.sum():,} / {len(df):,}"
    )

    print(
        f"Coverage: "
        f"{prior_pageview_rows.mean() * 100:.2f}%"
    )

    print("\nRows with a previous event-document view:")
    event_view_rows = (
        df["user_prior_views_event_document"] > 0
    )

    print(
        f"{event_view_rows.mean() * 100:.2f}%"
    )

    print("\nRows with a previous promoted-document view:")
    promoted_view_rows = (
        df["user_prior_views_promoted_document"] > 0
    )

    print(
        f"{promoted_view_rows.mean() * 100:.2f}%"
    )

    print("\nUsers with page-view history:")

    user_history = (
        df.groupby("uuid")["user_prior_pageviews"]
        .max()
    )

    print(
        f"{(user_history > 0).mean() * 100:.2f}%"
    )

    print("\nClick rate: no prior pageviews")
    print(
        df.loc[
            df["user_prior_pageviews"] == 0,
            "clicked"
        ].mean()
    )

    print("\nClick rate: has prior pageviews")
    print(
        df.loc[
            df["user_prior_pageviews"] > 0,
            "clicked"
        ].mean()
    )

    # Behavioral activity buckets
    df["activity_bucket"] = pd.cut(
        df["user_prior_pageviews"],
        bins=[
            -1,
            0,
            2,
            5,
            10,
            50,
            float("inf"),
        ],
        labels=[
            "0",
            "1-2",
            "3-5",
            "6-10",
            "11-50",
            "50+",
        ],
    )

    print("\nClick rate by prior-pageview bucket:")

    print(
        df.groupby(
            "activity_bucket",
            observed=True,
        )["clicked"]
        .agg(
            [
                "count",
                "mean",
            ]
        )
    )


if __name__ == "__main__":
    main()