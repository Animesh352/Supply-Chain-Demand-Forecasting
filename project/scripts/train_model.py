from __future__ import annotations

import argparse
from pathlib import Path

from models.forecasting_model import DemandForecastingModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and persist the M5 forecasting model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Directory containing M5 CSV files.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/artifacts/forecasting_model.pkl"),
        help="Output artifact path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    calendar_path = args.data_dir / "calendar.csv"
    sell_prices_path = args.data_dir / "sell_prices.csv"
    sales_path = args.data_dir / "sales_train_validation.csv"

    missing = [str(p) for p in [calendar_path, sell_prices_path, sales_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))

    model = DemandForecastingModel(model_path=args.model_path)
    artifacts = model.train_from_raw_files(
        calendar_path=calendar_path,
        sell_prices_path=sell_prices_path,
        sales_train_validation_path=sales_path,
    )

    print("Training complete.")
    print(f"Saved model artifact: {args.model_path}")
    print(f"MAE: {artifacts.metrics.get('mae', 0.0):.4f}")
    print(f"RMSE: {artifacts.metrics.get('rmse', 0.0):.4f}")
    print(f"RMSSE: {artifacts.metrics.get('rmsse', 0.0):.4f}")


if __name__ == "__main__":
    main()
