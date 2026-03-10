from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from optimization.inventory_policy import InventoryPolicyResult


@dataclass
class SimulationResult:
    baseline_cost_mean: float
    optimized_cost_mean: float
    service_level_achieved: float
    cost_reduction_percent: float
    stockout_probability: float
    baseline_cost_distribution: List[float]
    optimized_cost_distribution: List[float]


def _simulate_policy(
    demand_path: np.ndarray,
    reorder_point: float,
    order_qty: float,
    lead_time: int,
    holding_cost: float,
    stockout_cost: float,
) -> Dict[str, float]:
    inventory = reorder_point + order_qty
    outstanding_orders: List[tuple[int, float]] = []

    total_holding_cost = 0.0
    total_stockout_cost = 0.0
    stockout_days = 0

    for day_idx, demand in enumerate(demand_path):
        arrivals = [qty for (arrival_day, qty) in outstanding_orders if arrival_day == day_idx]
        if arrivals:
            inventory += float(np.sum(arrivals))
            outstanding_orders = [o for o in outstanding_orders if o[0] != day_idx]

        sold = min(inventory, demand)
        unmet = max(demand - inventory, 0.0)
        inventory = max(inventory - demand, 0.0)

        if unmet > 0:
            stockout_days += 1

        total_holding_cost += inventory * holding_cost
        total_stockout_cost += unmet * stockout_cost

        if inventory <= reorder_point:
            arrival_day = day_idx + lead_time
            outstanding_orders.append((arrival_day, order_qty))

    total_cost = total_holding_cost + total_stockout_cost
    service_level = 1.0 - (stockout_days / len(demand_path))

    return {
        "total_holding_cost": float(total_holding_cost),
        "total_stockout_cost": float(total_stockout_cost),
        "total_cost": float(total_cost),
        "service_level": float(service_level),
        "stockout": float(stockout_days > 0),
    }


def run_monte_carlo_comparison(
    forecast_mean: np.ndarray,
    forecast_std: np.ndarray,
    lead_time: int,
    holding_cost: float,
    stockout_cost: float,
    policy: InventoryPolicyResult,
    n_paths: int = 1000,
    horizon_days: int = 90,
    random_state: int = 42,
) -> SimulationResult:
    rng = np.random.default_rng(random_state)

    repeated_mean = np.resize(forecast_mean, horizon_days)
    repeated_std = np.resize(forecast_std, horizon_days)

    baseline_costs = []
    optimized_costs = []
    optimized_service_levels = []
    optimized_stockout_flags = []

    naive_reorder_point = float(np.mean(repeated_mean) * lead_time)
    naive_order_qty = float(max(np.mean(repeated_mean) * 7, 1.0))

    for _ in range(n_paths):
        sampled_demand = rng.normal(loc=repeated_mean, scale=np.maximum(repeated_std, 1e-6))
        sampled_demand = np.maximum(sampled_demand, 0.0)

        baseline = _simulate_policy(
            demand_path=sampled_demand,
            reorder_point=naive_reorder_point,
            order_qty=naive_order_qty,
            lead_time=lead_time,
            holding_cost=holding_cost,
            stockout_cost=stockout_cost,
        )
        optimized = _simulate_policy(
            demand_path=sampled_demand,
            reorder_point=policy.reorder_point,
            order_qty=policy.eoq,
            lead_time=lead_time,
            holding_cost=holding_cost,
            stockout_cost=stockout_cost,
        )

        baseline_costs.append(baseline["total_cost"])
        optimized_costs.append(optimized["total_cost"])
        optimized_service_levels.append(optimized["service_level"])
        optimized_stockout_flags.append(optimized["stockout"])

    baseline_cost_mean = float(np.mean(baseline_costs))
    optimized_cost_mean = float(np.mean(optimized_costs))
    cost_reduction_percent = (
        ((baseline_cost_mean - optimized_cost_mean) / baseline_cost_mean) * 100.0
        if baseline_cost_mean > 0
        else 0.0
    )

    return SimulationResult(
        baseline_cost_mean=baseline_cost_mean,
        optimized_cost_mean=optimized_cost_mean,
        service_level_achieved=float(np.mean(optimized_service_levels)),
        cost_reduction_percent=float(cost_reduction_percent),
        stockout_probability=float(np.mean(optimized_stockout_flags)),
        baseline_cost_distribution=[float(x) for x in baseline_costs],
        optimized_cost_distribution=[float(x) for x in optimized_costs],
    )
