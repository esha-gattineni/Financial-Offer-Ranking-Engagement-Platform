import pandas as pd

from src.models.train_historical import chronological_split


def test_chronological_split_has_strict_time_boundary():
    dataset = pd.DataFrame(
        {
            "display_id": [
                1, 1,
                2, 2,
                3, 3,
                4, 4,
                5, 5,
            ],
            "timestamp": [
                100, 100,
                200, 200,
                300, 300,
                400, 400,
                500, 500,
            ],
            "clicked": [
                0, 1,
                0, 0,
                1, 0,
                0, 1,
                1, 0,
            ],
        }
    )

    train, validation = chronological_split(
        dataset,
        train_fraction=0.80,
    )

    assert (
        train["timestamp"].max()
        < validation["timestamp"].min()
    )


def test_same_timestamp_never_crosses_split():
    dataset = pd.DataFrame(
        {
            "display_id": [
                1,
                2,
                3,
                4,
                5,
                6,
            ],
            "timestamp": [
                100,
                200,
                300,
                400,
                400,
                500,
            ],
            "clicked": [
                0,
                1,
                0,
                1,
                0,
                1,
            ],
        }
    )

    train, validation = chronological_split(
        dataset,
        train_fraction=0.70,
    )

    train_timestamps = set(
        train["timestamp"]
    )

    validation_timestamps = set(
        validation["timestamp"]
    )

    assert train_timestamps.isdisjoint(
        validation_timestamps
    )
