"""Time-based data splitting to prevent lookahead and time-dependent leakage."""

import polars as pl


def time_based_split(
    df: pl.DataFrame,
    timestamp_col: str = "timestamp",
    train_ratio: float = 0.667,  # ~20 days of 30
    val_ratio: float = 0.167,  # ~5 days of 30
    # Remaining (~5 days) is test
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split time-series event data strictly by timestamp cutoff.
    Guarantees:
    - Train events < Val events < Test events
    - Disjoint sets with zero temporal overlap
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in dataframe")

    sorted_df = df.sort(timestamp_col)

    min_ts = sorted_df[timestamp_col].min()
    max_ts = sorted_df[timestamp_col].max()

    total_duration = max_ts - min_ts

    train_cutoff = min_ts + (total_duration * train_ratio)
    val_cutoff = min_ts + (total_duration * (train_ratio + val_ratio))

    train_df = sorted_df.filter(pl.col(timestamp_col) <= train_cutoff)
    val_df = sorted_df.filter(
        (pl.col(timestamp_col) > train_cutoff) & (pl.col(timestamp_col) <= val_cutoff)
    )
    test_df = sorted_df.filter(pl.col(timestamp_col) > val_cutoff)

    return train_df, val_df, test_df
