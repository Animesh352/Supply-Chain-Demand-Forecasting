from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist


@dataclass
class InventoryPolicyResult:
    safety_stock: float
    reorder_point: float
    eoq: float


def _z_value(service_level: float) -> float:
    level = min(max(service_level, 0.5), 0.9999)
    return NormalDist().inv_cdf(level)


def compute_inventory_policy(
    mean_demand: float,
    demand_std: float,
    lead_time: int,
    service_level: float,
    annual_demand: float,
    holding_cost: float,
    order_cost: float,
) -> InventoryPolicyResult:
    z = _z_value(service_level)

    safety_stock = z * demand_std * math.sqrt(max(lead_time, 1))
    reorder_point = (mean_demand * lead_time) + safety_stock

    safe_holding_cost = max(holding_cost, 1e-6)
    safe_demand = max(annual_demand, 1e-6)
    safe_order_cost = max(order_cost, 1e-6)

    eoq = math.sqrt((2.0 * safe_demand * safe_order_cost) / safe_holding_cost)

    return InventoryPolicyResult(
        safety_stock=float(max(safety_stock, 0.0)),
        reorder_point=float(max(reorder_point, 0.0)),
        eoq=float(max(eoq, 1.0)),
    )
