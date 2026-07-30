from __future__ import annotations

"""Train on a single store subset for fast pipeline verification."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import M5DataLoader
from models.forecasting_model import DemandForecastingModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train on a store subset of the M5 dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-path", type=Path, default=Path("models/artifacts/forecasting_model.pkl"))
    parser.add_argument("--store-id", type=str, default="CA_1", help="Store ID to filter (e.g. CA_1)")
    parser.add_argument("--n-skus", type=int, default=0, help="Limit to N random SKUs (0 = all)")
    parser.add_argument("--n-folds", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading data for store {args.store_id}...")
    loader = M5DataLoader(
        calendar_path=args.data_dir / "calendar.csv",
        sell_prices_path=args.data_dir / "sell_prices.csv",
        sales_train_validation_path=args.data_dir / "sales_train_validation.csv",
        store_ids=[args.store_id],
    )
    modeling_df = loader.build_modeling_frame()
    if args.n_skus > 0:
        skus = sorted(modeling_df["sku_id"].unique())[:args.n_skus]
        modeling_df = modeling_df[modeling_df["sku_id"].isin(skus)].reset_index(drop=True)
    print(f"Rows after filter: {len(modeling_df):,}  SKUs: {modeling_df['sku_id'].nunique():,}")

    model = DemandForecastingModel(model_path=args.model_path)
    artifacts = model.train(modeling_df=modeling_df, n_folds=args.n_folds)
    model.save()

    print("\nTraining complete.")
    print(f"Saved: {args.model_path}")
    print(f"MAE:   {artifacts.metrics['mae']:.4f}")
    print(f"RMSE:  {artifacts.metrics['rmse']:.4f}")
    print(f"RMSSE: {artifacts.metrics['rmsse']:.4f}")


if __name__ == "__main__":
    main()
