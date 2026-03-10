from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass
class FeatureEngineer:
    target_col: str = "demand"
    sku_col: str = "sku_id"
    date_col: str = "date"

    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.date_col] = pd.to_datetime(out[self.date_col])

        out["day_of_week"] = out[self.date_col].dt.dayofweek
        out["day_of_month"] = out[self.date_col].dt.day
        out["week_of_year"] = out[self.date_col].dt.isocalendar().week.astype(int)
        out["month"] = out[self.date_col].dt.month
        out["quarter"] = out[self.date_col].dt.quarter
        out["year"] = out[self.date_col].dt.year
        out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
        return out

    def add_lag_and_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.sort_values([self.sku_col, self.date_col]).copy()
        grouped = out.groupby(self.sku_col)[self.target_col]

        out["lag_7"] = grouped.shift(7)
        out["lag_14"] = grouped.shift(14)
        out["lag_28"] = grouped.shift(28)
        out["rolling_mean_7"] = grouped.shift(1).rolling(window=7).mean().reset_index(level=0, drop=True)
        out["rolling_std_14"] = (
            grouped.shift(1).rolling(window=14).std().reset_index(level=0, drop=True)
        )

        return out

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "sell_price" in out.columns:
            out["sell_price"] = out["sell_price"].astype(float)
            out["price_change_7"] = (
                out.groupby(self.sku_col)["sell_price"].pct_change(7).replace([np.inf, -np.inf], np.nan)
            )
        return out

    def transform(self, df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
        out = self.add_calendar_features(df)
        out = self.add_lag_and_rolling_features(out)
        out = self.add_price_features(out)

        if dropna:
            out = out.dropna().reset_index(drop=True)

        return out


FEATURE_COLUMNS: List[str] = [
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_14",
    "sell_price",
    "price_change_7",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_weekend",
]


CATEGORICAL_COLUMNS: List[str] = ["item_id", "store_id", "dept_id", "cat_id", "state_id"]


def get_training_columns(df: pd.DataFrame) -> Iterable[str]:
    candidates = FEATURE_COLUMNS + CATEGORICAL_COLUMNS
    return [col for col in candidates if col in df.columns]
