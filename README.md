# Supply Chain Forecasting & Inventory Optimization System

Production-style Python project for demand forecasting, inventory decision optimization, and risk simulation, built on the M5 dataset with API and dashboard delivery.

## Why This Project Matters
This project demonstrates the same end-to-end skills used in compliance and logistics analytics roles:
- ETL-style data pipeline construction from multiple source tables
- Predictive modeling with measurable validation metrics
- Decision automation logic (policy optimization)
- Monitoring hooks (basic drift checks)
- Deployment-ready interfaces (FastAPI + Streamlit + Docker)

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
(rolling validation: MAE, RMSE, RMSSE)
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

## Project Structure

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

## Core Features
- Aggregates M5 demand to SKU-day level
- Generates lag and rolling demand features (`lag_7`, `lag_14`, `lag_28`, `rolling_mean_7`, `rolling_std_14`)
- Trains XGBoost forecaster with rolling-window validation
- Computes `MAE`, `RMSE`, and manual `RMSSE`
- Forecasts next 30 days per SKU
- Computes Safety Stock, Reorder Point, EOQ
- Runs Monte Carlo policy comparison (baseline vs optimized)
- Serves decisions via API and visualizes via dashboard

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

## Train Model Artifact

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

## Run Dashboard

```bash
cd project
source .venv/bin/activate
streamlit run dashboard/app.py --server.port 8501
```

Dashboard: `http://localhost:8501`

## Important Runtime Note
By default, API/dashboard do **not** auto-train at startup. They expect a pre-trained model artifact.

To allow auto-training at startup (heavier):

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

## Sample API Call

```bash
curl -G "http://localhost:8000/optimize" \
  --data-urlencode "sku_id=FOODS_1_001_CA_1" \
  --data-urlencode "lead_time=7" \
  --data-urlencode "service_level=0.95" \
  --data-urlencode "holding_cost=0.2" \
  --data-urlencode "stockout_cost=5.0" \
  --data-urlencode "order_cost=50"
```

## GitHub Upload Checklist
1. Keep raw M5 files out of Git (already ignored in root `.gitignore`).
2. Keep `models/artifacts/` out of Git.
3. Commit source code + README only.
4. Add screenshots of API docs and dashboard for recruiter-friendly repo presentation.

## Suggested Resume Bullets (Role-Aligned)
- Built an end-to-end Python analytics system that ingests multi-table supply-chain data, engineers time-series features, and trains an XGBoost forecasting model with rolling-window validation (`MAE`, `RMSE`, `RMSSE`).
- Designed an inventory decision engine that automates safety stock, reorder point, and EOQ recommendations based on forecast uncertainty and service-level targets.
- Implemented Monte Carlo simulation to quantify stockout risk and compare baseline vs optimized policy cost outcomes for operational decision support.
- Delivered production-style interfaces using FastAPI and Streamlit, with containerized deployment via Docker and basic drift monitoring integration using Evidently.
- Applied modular software engineering practices (separated data, features, modeling, evaluation, optimization, API, and dashboard layers) to support maintainability and enterprise integration.

## Trade Compliance Positioning (for interviews)
Even though this project is supply-chain focused, the architecture maps directly to trade compliance analytics:
- Replace SKU demand forecasting with compliance-risk forecasting.
- Replace inventory policy logic with risk scoring/decision thresholds.
- Reuse ETL, validation, API, and dashboard framework for entry audits, document checks, and KPI monitoring.
