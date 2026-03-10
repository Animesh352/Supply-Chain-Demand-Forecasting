from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from data.data_loader import M5DataLoader
from evaluation.backtesting import compute_metrics, rolling_time_series_folds, run_basic_drift_detection
from features.feature_engineering import FeatureEngineer, get_training_columns


@dataclass
class ForecastOutput:
    sku_id: str
    forecast: List[float]
    forecast_std: List[float]


@dataclass
class ModelArtifacts:
    model: XGBRegressor
    feature_columns: List[str]
    train_encoded_columns: List[str]
    metrics: Dict[str, float]
    residual_std_by_sku: Dict[str, float]
    global_residual_std: float
    history_frame: pd.DataFrame
    drift_summary: Dict[str, float | bool | str]


class DemandForecastingModel:
    def __init__(
        self,
        model_path: Path,
        random_state: int = 42,
    ) -> None:
        self.model_path = model_path
        self.random_state = random_state
        self.feature_engineer = FeatureEngineer()
        self.artifacts: Optional[ModelArtifacts] = None

    @staticmethod
    def _encode_features(x: pd.DataFrame, train_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
        encoded = pd.get_dummies(x, drop_first=False)

        if train_columns is None:
            return encoded, list(encoded.columns)

        encoded = encoded.reindex(columns=train_columns, fill_value=0)
        return encoded, train_columns

    def _build_model(self) -> XGBRegressor:
        return XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=-1,
        )

    def train(self, modeling_df: pd.DataFrame, n_folds: int = 3) -> ModelArtifacts:
        df = self.feature_engineer.transform(modeling_df, dropna=True)
        feature_cols = list(get_training_columns(df))

        fold_metrics: List[Dict[str, float]] = []
        val_predictions = []

        for fold_id, (train_start, train_end, val_start, val_end) in enumerate(
            rolling_time_series_folds(df, date_col="date", n_folds=n_folds), start=1
        ):
            train_mask = (df["date"] >= train_start) & (df["date"] <= train_end)
            val_mask = (df["date"] >= val_start) & (df["date"] <= val_end)

            train_df = df.loc[train_mask].copy()
            val_df = df.loc[val_mask].copy()

            x_train = train_df[feature_cols]
            y_train = train_df["demand"].astype(float).values
            x_val = val_df[feature_cols]
            y_val = val_df["demand"].astype(float).values

            x_train_enc, train_columns = self._encode_features(x_train)
            x_val_enc, _ = self._encode_features(x_val, train_columns=train_columns)

            model = self._build_model()
            model.fit(x_train_enc, y_train)

            y_pred = model.predict(x_val_enc)
            fold_metric = compute_metrics(y_true=y_val, y_pred=y_pred, y_train_history=y_train)
            fold_metrics.append(fold_metric)

            val_df = val_df[["sku_id", "date", "demand"]].copy()
            val_df["prediction"] = y_pred
            val_predictions.append(val_df)

            print(
                f"Fold {fold_id}: MAE={fold_metric['mae']:.4f}, RMSE={fold_metric['rmse']:.4f}, "
                f"RMSSE={fold_metric['rmsse']:.4f}"
            )

        metrics = {
            "mae": float(np.mean([m["mae"] for m in fold_metrics])),
            "rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
            "rmsse": float(np.mean([m["rmsse"] for m in fold_metrics])),
        }

        all_val = pd.concat(val_predictions, ignore_index=True)
        all_val["residual"] = all_val["demand"] - all_val["prediction"]
        residual_std_by_sku = (
            all_val.groupby("sku_id")["residual"].std().fillna(all_val["residual"].std()).to_dict()
        )
        global_residual_std = float(all_val["residual"].std()) if len(all_val) > 1 else 1.0

        # Fit final model on full training data.
        x_full = df[feature_cols]
        y_full = df["demand"].astype(float).values
        x_full_enc, train_encoded_cols = self._encode_features(x_full)

        final_model = self._build_model()
        final_model.fit(x_full_enc, y_full)

        split_date = df["date"].quantile(0.8)
        reference_df = df[df["date"] <= split_date]
        current_df = df[df["date"] > split_date]
        drift_summary = run_basic_drift_detection(
            reference_df=reference_df,
            current_df=current_df,
            columns=[col for col in feature_cols if col in df.columns and df[col].dtype != "O"],
        )

        self.artifacts = ModelArtifacts(
            model=final_model,
            feature_columns=feature_cols,
            train_encoded_columns=train_encoded_cols,
            metrics=metrics,
            residual_std_by_sku={k: float(v) for k, v in residual_std_by_sku.items()},
            global_residual_std=global_residual_std,
            history_frame=df,
            drift_summary=drift_summary,
        )
        return self.artifacts

    def save(self) -> None:
        if self.artifacts is None:
            raise RuntimeError("Model is not trained. Call train() before save().")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.artifacts, f)

    def load(self) -> ModelArtifacts:
        with open(self.model_path, "rb") as f:
            artifacts: ModelArtifacts = pickle.load(f)
        self.artifacts = artifacts
        return artifacts

    def train_from_raw_files(
        self,
        calendar_path: Path,
        sell_prices_path: Path,
        sales_train_validation_path: Path,
    ) -> ModelArtifacts:
        loader = M5DataLoader(
            calendar_path=calendar_path,
            sell_prices_path=sell_prices_path,
            sales_train_validation_path=sales_train_validation_path,
        )
        modeling_df = loader.build_modeling_frame()
        artifacts = self.train(modeling_df=modeling_df, n_folds=3)
        self.save()
        return artifacts

    def _require_artifacts(self) -> ModelArtifacts:
        if self.artifacts is None:
            raise RuntimeError("Artifacts are not loaded. Call load() or train().")
        return self.artifacts

    def forecast_next_30_days(self, sku_id: str) -> ForecastOutput:
        artifacts = self._require_artifacts()
        history = artifacts.history_frame

        sku_hist = history[history["sku_id"] == sku_id].sort_values("date").copy()
        if sku_hist.empty:
            raise ValueError(f"SKU '{sku_id}' was not found in training history.")

        latest = sku_hist.iloc[-1].copy()
        last_date = pd.Timestamp(latest["date"])

        demand_history = list(sku_hist["demand"].astype(float).values)
        recent_prices = list(sku_hist["sell_price"].astype(float).tail(8).values)

        predictions: List[float] = []
        std_series: List[float] = []

        sku_sigma = artifacts.residual_std_by_sku.get(sku_id, artifacts.global_residual_std)

        for step in range(1, 31):
            forecast_date = last_date + pd.Timedelta(days=step)
            lag_7 = demand_history[-7] if len(demand_history) >= 7 else demand_history[-1]
            lag_14 = demand_history[-14] if len(demand_history) >= 14 else demand_history[-1]
            lag_28 = demand_history[-28] if len(demand_history) >= 28 else demand_history[-1]

            rolling_mean_7 = float(np.mean(demand_history[-7:]))
            rolling_std_14 = float(np.std(demand_history[-14:], ddof=1)) if len(demand_history) >= 2 else 0.0

            current_price = recent_prices[-1]
            price_change_7 = 0.0
            if len(recent_prices) >= 8 and recent_prices[-8] != 0:
                price_change_7 = (recent_prices[-1] - recent_prices[-8]) / recent_prices[-8]

            row = {
                "lag_7": lag_7,
                "lag_14": lag_14,
                "lag_28": lag_28,
                "rolling_mean_7": rolling_mean_7,
                "rolling_std_14": rolling_std_14,
                "sell_price": current_price,
                "price_change_7": price_change_7,
                "day_of_week": forecast_date.dayofweek,
                "day_of_month": forecast_date.day,
                "week_of_year": int(forecast_date.isocalendar().week),
                "month": forecast_date.month,
                "quarter": int((forecast_date.month - 1) / 3) + 1,
                "year": forecast_date.year,
                "is_weekend": int(forecast_date.dayofweek in [5, 6]),
                "item_id": latest["item_id"],
                "store_id": latest["store_id"],
                "dept_id": latest["dept_id"],
                "cat_id": latest["cat_id"],
                "state_id": latest["state_id"],
            }

            x = pd.DataFrame([row])[artifacts.feature_columns]
            x_enc, _ = self._encode_features(x, train_columns=artifacts.train_encoded_columns)
            pred = float(artifacts.model.predict(x_enc)[0])
            pred = max(pred, 0.0)

            predictions.append(pred)
            std_series.append(float(max(sku_sigma, 1e-6)))
            demand_history.append(pred)
            recent_prices.append(current_price)

        return ForecastOutput(sku_id=sku_id, forecast=predictions, forecast_std=std_series)


def load_or_train_model(
    model_path: Path,
    calendar_path: Path,
    sell_prices_path: Path,
    sales_train_validation_path: Path,
) -> DemandForecastingModel:
    forecaster = DemandForecastingModel(model_path=model_path)
    if model_path.exists():
        forecaster.load()
    else:
        forecaster.train_from_raw_files(
            calendar_path=calendar_path,
            sell_prices_path=sell_prices_path,
            sales_train_validation_path=sales_train_validation_path,
        )
    return forecaster
