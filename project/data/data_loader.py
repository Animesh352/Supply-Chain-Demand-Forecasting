from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class M5DataLoader:
    """Loads and prepares the M5 dataset for SKU-level demand forecasting."""

    calendar_path: Path
    sell_prices_path: Path
    sales_train_validation_path: Path

    def load_calendar(self) -> pd.DataFrame:
        calendar = pd.read_csv(self.calendar_path)
        calendar["date"] = pd.to_datetime(calendar["date"])
        return calendar

    def load_sell_prices(self) -> pd.DataFrame:
        return pd.read_csv(self.sell_prices_path)

    def load_sales(self) -> pd.DataFrame:
        return pd.read_csv(self.sales_train_validation_path)

    @staticmethod
    def _day_columns(df: pd.DataFrame) -> List[str]:
        return [col for col in df.columns if col.startswith("d_")]

    def build_long_demand_frame(self) -> pd.DataFrame:
        sales = self.load_sales()
        day_cols = self._day_columns(sales)

        sales["sku_id"] = sales["item_id"].astype(str) + "_" + sales["store_id"].astype(str)

        long_sales = sales.melt(
            id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "sku_id"],
            value_vars=day_cols,
            var_name="d",
            value_name="demand",
        )

        # SKU-level daily demand (aggregated in case of duplicates).
        long_sales = (
            long_sales.groupby(
                ["sku_id", "item_id", "store_id", "dept_id", "cat_id", "state_id", "d"],
                as_index=False,
            )["demand"]
            .sum()
        )
        return long_sales

    def build_modeling_frame(self) -> pd.DataFrame:
        calendar = self.load_calendar()
        sell_prices = self.load_sell_prices()
        long_sales = self.build_long_demand_frame()

        df = long_sales.merge(calendar, on="d", how="left")
        df = df.merge(
            sell_prices,
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left",
        )

        # Fill sparse prices for weeks where price is not observed.
        df["sell_price"] = (
            df.groupby("sku_id")["sell_price"]
            .transform(lambda s: s.ffill().bfill())
            .fillna(df["sell_price"].median())
        )

        # Stable time ordering for feature generation and model training.
        df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)

        return df


if __name__ == "__main__":
    base = Path("data/raw")
    loader = M5DataLoader(
        calendar_path=base / "calendar.csv",
        sell_prices_path=base / "sell_prices.csv",
        sales_train_validation_path=base / "sales_train_validation.csv",
    )
    modeling_df = loader.build_modeling_frame()
    print(modeling_df.head())
    print(f"Rows: {len(modeling_df):,}")
