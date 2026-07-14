# Supply-Chain-Demand-Forecasting

A production-style Python project for time-series demand forecasting and inventory decision optimization. Built on the M5 Walmart dataset with a FastAPI endpoint and Streamlit dashboard for delivery.

## What this does

Trains an XGBoost regressor on the M5 demand dataset, evaluates it with rolling-window backtesting, generates 30-day per-SKU forecasts, and computes Safety Stock/Reorder Point/EOQ inventory policies from those forecasts. Monte Carlo simulation lets you compare a baseline policy against an optimized one by sampling demand uncertainty.

## Architecture

```text
M5 CSV Files
(calendar, sell_prices, sales_train_validation)
        |
        v
Data Loader + Feature Engineering
(lags, rolling stats, calendar/price features)
        |
        v
XGBoost Forecasting Model
(3-fold rolling validation: MAE, RMSE, RMSSE)
        |
        +--------------------------+
        |                          |
        v                          v
Inventory Optimization        Drift Check (Evidently)
(SS, ROP, EOQ)               (basic dataset drift)
        |
        v
Monte Carlo Simulation
(cost, service level, stockout risk)
        |
        +--------------------------+
        |                          |
        v                          v
FastAPI (/optimize)          Streamlit Dashboard
```

## Project structure

```text
project/
├── api/main.py
├── dashboard/app.py
├── data/data_loader.py
├── evaluation/backtesting.py
├── features/feature_engineering.py
├── models/forecasting_model.py
├── optimization/inventory_policy.py
├── optimization/monte_carlo_simulation.py
├── scripts/train_model.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Model details

| Component | Choice |
|-----------|--------|
| Model | `XGBRegressor` |
| Hyperparameters | `n_estimators=400`, `lr=0.05`, `max_depth=8`, `subsample=0.9`, `colsample_bytree=0.9` |
| Lag features | `lag_7`, `lag_14`, `lag_28` |
| Rolling features | `rolling_mean_7`, `rolling_std_14` |
| Price features | `sell_price`, `price_change_7` |
| Calendar features | day of week, day of month, week of year, month, quarter, year, is_weekend |
| SKU metadata | `item_id`, `store_id`, `dept_id`, `cat_id`, `state_id` (one-hot encoded) |
| Validation | 3-fold expanding-window time-series CV (min 365 days train, 28-day val windows) |
| Metrics | MAE, RMSE, RMSSE (Root Mean Squared Scaled Error) |
| Forecasting | Recursive 30-day forecast per SKU using iterated one-step prediction |

Metrics are printed per fold during training. No precomputed results are committed to the repo -- run `scripts/train_model.py` to reproduce them.

TODO(animesh): save and commit fold-level MAE/RMSSE table after running on the full M5 dataset.

## Inventory optimization

Safety stock and reorder point use the normal-approximation formula with a configurable service-level z-score. EOQ uses the Wilson formula.

- Safety Stock = z * sigma_demand * sqrt(lead_time)
- Reorder Point = mean_demand * lead_time + safety_stock
- EOQ = sqrt(2 * annual_demand * order_cost / holding_cost)

These are computed from the model's per-SKU residual standard deviation as the demand uncertainty estimate.

## Setup

```bash
cd project
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place raw files in `project/data/raw/`:
- `calendar.csv`
- `sell_prices.csv`
- `sales_train_validation.csv`

## Train model

```bash
cd project
source .venv/bin/activate
python scripts/train_model.py --data-dir data/raw --model-path models/artifacts/forecasting_model.pkl
```

## Run API

```bash
cd project
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API docs: `http://localhost:8000/docs`

## Run dashboard

```bash
cd project
source .venv/bin/activate
streamlit run dashboard/app.py --server.port 8501
```

Dashboard: `http://localhost:8501`

## Important runtime note

By default, the API and dashboard do **not** auto-train at startup. They expect a pre-trained model artifact at the configured path.

To allow auto-training at startup:

```bash
export AUTO_TRAIN_ON_STARTUP=1
```

## Docker

Build:

```bash
cd project
docker build -t supply-chain-forecasting:latest .
```

Run API:

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_PATH=models/artifacts/forecasting_model.pkl \
  -v $(pwd)/data/raw:/app/data/raw \
  -v $(pwd)/models/artifacts:/app/models/artifacts \
  supply-chain-forecasting:latest \
  uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Run dashboard:

```bash
docker run --rm -p 8501:8501 \
  -e MODEL_PATH=models/artifacts/forecasting_model.pkl \
  -v $(pwd)/data/raw:/app/data/raw \
  -v $(pwd)/models/artifacts:/app/models/artifacts \
  supply-chain-forecasting:latest \
  streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

## Sample API call

```bash
curl -G "http://localhost:8000/optimize" \
  --data-urlencode "sku_id=FOODS_1_001_CA_1" \
  --data-urlencode "lead_time=7" \
  --data-urlencode "service_level=0.95" \
  --data-urlencode "holding_cost=0.2" \
  --data-urlencode "stockout_cost=5.0" \
  --data-urlencode "order_cost=50"
```
