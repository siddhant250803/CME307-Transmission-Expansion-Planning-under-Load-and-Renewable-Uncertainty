"""
Analyze supply-shortage scenarios using the DC OPF model with load shedding.

The script:
1. Loads RTS-GMLC network and time-series data.
2. Builds nodal loads for selected time periods.
3. Derates generator capacities to emulate large outages.
4. Solves a DC OPF with load shedding (using TEPWithLoadShedding with zero candidates).
5. Records the amount and cost of unserved energy and saves results to CSV.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import RTSDataLoader
from src.timeseries_loader import TimeseriesLoader
from src.tep_with_shedding import TEPWithLoadShedding


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
    periods = [1, 2, 3, 4, 5, 6]
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
