from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure local packages (models/, optimization/, etc.) are importable
# regardless of the directory from which Streamlit is launched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.forecasting_model import DemandForecastingModel, load_or_train_model
from optimization.inventory_policy import compute_inventory_policy
from optimization.monte_carlo_simulation import run_monte_carlo_comparison

st.set_page_config(page_title="Supply Chain Forecasting & Inventory Optimization", layout="wide")


@st.cache_resource(show_spinner=True)
def get_forecaster() -> DemandForecastingModel:
    base = Path(os.getenv("M5_DATA_DIR", "data/raw"))
    model_path = Path(os.getenv("MODEL_PATH", "models/artifacts/forecasting_model.pkl"))
    auto_train = os.getenv("AUTO_TRAIN_ON_STARTUP", "0") == "1"

    if model_path.exists():
        forecaster = DemandForecastingModel(model_path=model_path)
        forecaster.load()
        return forecaster

    calendar_path = base / "calendar.csv"
    sell_prices_path = base / "sell_prices.csv"
    sales_path = base / "sales_train_validation.csv"

    missing = [str(p) for p in [calendar_path, sell_prices_path, sales_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required raw files: " + ", ".join(missing))

    if not auto_train:
        raise FileNotFoundError(
            "Model artifact not found. Train first with scripts/train_model.py "
            "or launch with AUTO_TRAIN_ON_STARTUP=1."
        )

    return load_or_train_model(
        model_path=model_path,
        calendar_path=calendar_path,
        sell_prices_path=sell_prices_path,
        sales_train_validation_path=sales_path,
    )


forecaster = get_forecaster()
artifacts = forecaster.artifacts

if artifacts is None:
    st.error("No model artifacts available. Ensure training completed successfully.")
    st.stop()

history = artifacts.history_frame.copy()
sku_options = sorted(history["sku_id"].unique())

st.title("Supply Chain Forecasting & Inventory Optimization System")
st.caption("Decision-support dashboard for demand forecasting, policy optimization, and risk simulation")

section1, section2 = st.columns([2, 1])
with section1:
    st.subheader("Section 1: SKU Selection")
    selected_sku = st.selectbox("Select SKU", sku_options, index=0)
with section2:
    st.subheader("Section 3: Model Metrics")
    st.metric("MAE", f"{artifacts.metrics.get('mae', 0.0):.3f}")
    st.metric("RMSE", f"{artifacts.metrics.get('rmse', 0.0):.3f}")

sku_hist = history[history["sku_id"] == selected_sku].sort_values("date")
forecast_output = forecaster.forecast_next_30_days(selected_sku)

forecast_values = np.array(forecast_output.forecast)
forecast_std = np.array(forecast_output.forecast_std)

future_dates = pd.date_range(start=sku_hist["date"].max() + pd.Timedelta(days=1), periods=30, freq="D")
lower_band = np.maximum(forecast_values - 1.96 * forecast_std, 0)
upper_band = forecast_values + 1.96 * forecast_std

st.subheader("Section 2: Forecast Visualization")
forecast_fig = go.Figure()
forecast_fig.add_trace(
    go.Scatter(
        x=sku_hist["date"],
        y=sku_hist["demand"],
        mode="lines",
        name="Historical Demand",
        line=dict(color="#1f77b4", width=2),
    )
)
forecast_fig.add_trace(
    go.Scatter(
        x=future_dates,
        y=forecast_values,
        mode="lines+markers",
        name="Forecast (30 days)",
        line=dict(color="#ff7f0e", width=2),
    )
)
forecast_fig.add_trace(
    go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(upper_band) + list(lower_band[::-1]),
        fill="toself",
        fillcolor="rgba(255,127,14,0.20)",
        line=dict(color="rgba(255,127,14,0)"),
        hoverinfo="skip",
        name="95% Confidence Band",
    )
)
forecast_fig.update_layout(height=420, template="plotly_white", legend=dict(orientation="h"))
st.plotly_chart(forecast_fig, use_container_width=True)

st.subheader("Section 4: Inventory Inputs")
inp1, inp2, inp3, inp4, inp5 = st.columns(5)
with inp1:
    lead_time = st.slider("Lead Time (days)", min_value=1, max_value=60, value=7)
with inp2:
    service_level = st.slider("Service Level", min_value=0.50, max_value=0.999, value=0.95)
with inp3:
    holding_cost = st.slider("Holding Cost", min_value=0.01, max_value=5.0, value=0.2)
with inp4:
    stockout_cost = st.slider("Stockout Cost", min_value=0.1, max_value=50.0, value=5.0)
with inp5:
    order_cost = st.slider("Order Cost", min_value=1.0, max_value=500.0, value=50.0)

mean_demand = float(np.mean(forecast_values))
demand_std = float(np.mean(forecast_std))
annual_demand = mean_demand * 365

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
    forecast_mean=forecast_values,
    forecast_std=forecast_std,
    lead_time=lead_time,
    holding_cost=holding_cost,
    stockout_cost=stockout_cost,
    policy=policy,
    n_paths=1000,
    horizon_days=90,
)

st.subheader("Section 5: Inventory Outputs")
out1, out2, out3, out4 = st.columns(4)
out1.metric("Safety Stock", f"{policy.safety_stock:.2f}")
out2.metric("Reorder Point", f"{policy.reorder_point:.2f}")
out3.metric("EOQ", f"{policy.eoq:.2f}")
out4.metric("Service Level Achieved", f"{simulation.service_level_achieved:.2%}")

st.subheader("Section 6: Cost Comparison Chart")
cost_fig = go.Figure(
    data=[
        go.Bar(name="Baseline Cost", x=["Policy Cost"], y=[simulation.baseline_cost_mean], marker_color="#d62728"),
        go.Bar(name="Optimized Cost", x=["Policy Cost"], y=[simulation.optimized_cost_mean], marker_color="#2ca02c"),
    ]
)
cost_fig.update_layout(template="plotly_white", barmode="group", height=350)
st.plotly_chart(cost_fig, use_container_width=True)

st.subheader("Section 7: Monte Carlo Risk Chart")
risk_col1, risk_col2 = st.columns(2)
with risk_col1:
    dist_fig = go.Figure()
    dist_fig.add_trace(
        go.Histogram(
            x=simulation.baseline_cost_distribution,
            nbinsx=40,
            name="Baseline",
            opacity=0.55,
            marker_color="#d62728",
        )
    )
    dist_fig.add_trace(
        go.Histogram(
            x=simulation.optimized_cost_distribution,
            nbinsx=40,
            name="Optimized",
            opacity=0.55,
            marker_color="#2ca02c",
        )
    )
    dist_fig.update_layout(template="plotly_white", barmode="overlay", height=360)
    st.plotly_chart(dist_fig, use_container_width=True)

with risk_col2:
    st.metric("Cost Reduction", f"{simulation.cost_reduction_percent:.2f}%")
    st.metric("Stockout Probability", f"{simulation.stockout_probability:.2%}")
    st.info(
        f"Drift Check: {artifacts.drift_summary.get('status', 'unknown')} | "
        f"Share drifted columns: {artifacts.drift_summary.get('drift_share', 0.0):.2%}"
    )
