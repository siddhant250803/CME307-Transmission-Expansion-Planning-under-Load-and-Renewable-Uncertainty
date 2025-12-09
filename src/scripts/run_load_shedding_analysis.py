"""
Load Shedding Stress Test Analysis
==================================

This script performs a supply shortage stress test by:
1. Derating all generators to 20% of nameplate capacity (simulating outages)
2. Increasing loads by 40% (stress multiplier)
3. Solving DC-OPF with load shedding enabled (penalty: $50k/MWh VOLL)
4. Quantifying the amount and cost of unserved energy

The analysis runs across 12 representative periods to understand how load
shedding varies with demand patterns. This provides a quantitative benchmark
for evaluating resilience investments.

Output:
- results/load_shedding_periods.csv: Period-by-period load shedding results

Usage:
    python src/scripts/run_load_shedding_analysis.py

Author: CME307 Team (Edouard Rabasse, Siddhant Sukhani)
Date: December 2025
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.data_loader import RTSDataLoader
from src.core.timeseries_loader import TimeseriesLoader
from src.core.tep_with_shedding import TEPWithLoadShedding


@dataclass
class ScenarioData:
    """Wrap RTSDataLoader so we can override loads/generation."""

    base_loader: RTSDataLoader
    custom_loads: Dict[int, float]
    custom_generators: pd.DataFrame

    def __getattr__(self, name):
        return getattr(self.base_loader, name)

    def get_load_by_bus(self):
        return self.custom_loads

    def get_generator_data(self):
        return self.custom_generators

    def get_generators_by_bus(self):
        grouped: Dict[int, List[pd.Series]] = {}
        for _, row in self.custom_generators.iterrows():
            grouped.setdefault(int(row["Bus ID"]), []).append(row)
        return grouped


def solve_with_load_shedding(
    period: int,
    base_loader: RTSDataLoader,
    timeseries: TimeseriesLoader,
    participation: Dict[int, float],
    availability_factor: float,
    stress_multiplier: float,
    shedding_cost: float,
) -> Tuple[Dict[str, float], bool]:
    buses = base_loader.get_bus_data()
    nodal_loads = timeseries.get_nodal_loads(period, buses, participation)
    stressed_loads = {bus: load * stress_multiplier for bus, load in nodal_loads.items()}

    generators = base_loader.get_generator_data().copy()
    generators["PMax MW"] *= availability_factor
    generators["PMin MW"] = np.minimum(generators["PMin MW"], generators["PMax MW"])

    scenario_loader = ScenarioData(base_loader, stressed_loads, generators)

    model = TEPWithLoadShedding(
        scenario_loader,
        candidate_lines=[],
        line_cost_per_mw=0.0,
        custom_loads=stressed_loads,
        shedding_cost_per_mw=shedding_cost,
    )
    model.build_model()

    def try_solve(solver_name: str) -> bool:
        try:
            return model.solve(solver=solver_name)
        except Exception:
            return False

    solved = try_solve("gurobi") or try_solve("glpk")
    if not solved:
        return {"period": period, "solved": False}, False

    results = model.get_results()
    generation_values = results.get("generation", {})
    computed_generation = float(sum(generation_values.values())) if generation_values else 0.0
    total_load = sum(stressed_loads.values())
    load_shed = results.get("total_load_shed", 0.0)

    record = {
        "period": period,
        "solved": True,
        "total_load_mw": total_load,
        "total_generation_mw": computed_generation,
        "objective_cost": results.get("objective_value", 0.0),
        "operating_cost": results.get("operating_cost", 0.0),
        "load_shed_mw": load_shed,
        "load_shed_pct": (load_shed / total_load) * 100 if total_load > 0 else 0.0,
        "load_shed_cost": results.get("load_shedding_cost", 0.0),
        "availability_factor": availability_factor,
        "stress_multiplier": stress_multiplier,
    }
    return record, True


def main():
    periods = list(range(1, 13))  # All 12 periods
    availability_factor = 0.2
    stress_multiplier = 1.40
    shedding_cost = 50_000.0

    base_loader = RTSDataLoader(data_dir="data/RTS_Data/SourceData")
    timeseries = TimeseriesLoader(data_dir="data/RTS_Data")
    timeseries.load_regional_load()
    timeseries.load_wind_generation()
    timeseries.load_pv_generation()
    timeseries.load_hydro_generation()

    buses = base_loader.get_bus_data()
    participation = timeseries.get_bus_load_participation_factors(buses)

    records: List[Dict[str, float]] = []
    for period in periods:
        record, solved = solve_with_load_shedding(
            period,
            base_loader,
            timeseries,
            participation,
            availability_factor,
            stress_multiplier,
            shedding_cost,
        )
        records.append(record)
        status = "solved" if solved else "failed"
        load_shed = record.get("load_shed_mw", 0.0)
        print(
            f"Period {period:2d}: {status.upper()} | "
            f"Load={record.get('total_load_mw', 0.0):7.1f} MW | "
            f"Load shed={load_shed:7.2f} MW"
        )

    df = pd.DataFrame(records)
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", "load_shedding_periods.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved detailed results to {output_path}")


if __name__ == "__main__":
    main()
