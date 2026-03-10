from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from models.forecasting_model import DemandForecastingModel, load_or_train_model
from optimization.inventory_policy import compute_inventory_policy
from optimization.monte_carlo_simulation import run_monte_carlo_comparison


class OptimizeResponse(BaseModel):
    forecast_next_30_days: List[float]
    mae: float
    rmse: float
    safety_stock: float
    reorder_point: float
    recommended_order_quantity: float
    cost_reduction_percent: float
    stockout_probability: float


app = FastAPI(
    title="Supply Chain Forecasting & Inventory Optimization System",
    version="1.0.0",
)


def _resolve_paths() -> Dict[str, Path]:
    base = Path(os.getenv("M5_DATA_DIR", "data/raw"))
    return {
        "calendar": base / "calendar.csv",
        "sell_prices": base / "sell_prices.csv",
        "sales_train_validation": base / "sales_train_validation.csv",
        "model": Path(os.getenv("MODEL_PATH", "models/artifacts/forecasting_model.pkl")),
    }


def _load_forecaster() -> DemandForecastingModel:
    paths = _resolve_paths()
    required = [paths["calendar"], paths["sell_prices"], paths["sales_train_validation"]]
    auto_train = os.getenv("AUTO_TRAIN_ON_STARTUP", "0") == "1"

    if paths["model"].exists():
        forecaster = DemandForecastingModel(model_path=paths["model"])
        forecaster.load()
        return forecaster

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Model artifact is missing and required raw files were not found: " + ", ".join(missing)
        )

    if not auto_train:
        raise FileNotFoundError(
            "Model artifact not found at "
            f"'{paths['model']}'. Train first with scripts/train_model.py or set AUTO_TRAIN_ON_STARTUP=1."
        )

    return load_or_train_model(
        model_path=paths["model"],
        calendar_path=paths["calendar"],
        sell_prices_path=paths["sell_prices"],
        sales_train_validation_path=paths["sales_train_validation"],
    )


@app.on_event("startup")
def startup_event() -> None:
    app.state.forecaster = _load_forecaster()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/optimize", response_model=OptimizeResponse)
def optimize(
    sku_id: str = Query(..., description="SKU identifier"),
    lead_time: int = Query(7, ge=1, le=90),
    service_level: float = Query(0.95, ge=0.5, le=0.999),
    holding_cost: float = Query(0.2, gt=0),
    stockout_cost: float = Query(5.0, gt=0),
    order_cost: float = Query(50.0, gt=0),
) -> OptimizeResponse:
    forecaster: DemandForecastingModel = app.state.forecaster

    try:
        forecast_output = forecaster.forecast_next_30_days(sku_id=sku_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    forecast_arr = np.array(forecast_output.forecast, dtype=float)
    std_arr = np.array(forecast_output.forecast_std, dtype=float)

    mean_demand = float(np.mean(forecast_arr))
    demand_std = float(np.mean(std_arr))
    annual_demand = mean_demand * 365.0

    policy = compute_inventory_policy(
        mean_demand=mean_demand,
        demand_std=demand_std,
        lead_time=lead_time,
        service_level=service_level,
        annual_demand=annual_demand,
        holding_cost=holding_cost,
        order_cost=order_cost,
    )

    simulation = run_monte_carlo_comparison(
        forecast_mean=forecast_arr,
        forecast_std=std_arr,
        lead_time=lead_time,
        holding_cost=holding_cost,
        stockout_cost=stockout_cost,
        policy=policy,
        n_paths=1000,
        horizon_days=90,
    )

    metrics = forecaster.artifacts.metrics if forecaster.artifacts else {"mae": 0.0, "rmse": 0.0}

    return OptimizeResponse(
        forecast_next_30_days=[float(x) for x in forecast_arr],
        mae=float(metrics.get("mae", 0.0)),
        rmse=float(metrics.get("rmse", 0.0)),
        safety_stock=policy.safety_stock,
        reorder_point=policy.reorder_point,
        recommended_order_quantity=policy.eoq,
        cost_reduction_percent=simulation.cost_reduction_percent,
        stockout_probability=simulation.stockout_probability,
    )
