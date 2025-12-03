"""
Run scenario-based robust TEP.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.data_loader import RTSDataLoader
from src.core.scenario_robust_tep import Scenario, ScenarioRobustTEP


def main():
    print("=" * 70)
    print("Scenario-Based Robust Transmission Expansion Planning")
    print("=" * 70)

    data_loader = RTSDataLoader(data_dir="data/RTS_Data/SourceData")

    scenarios = [
        Scenario(
            name="base",
            load_scale=1.0,
            renewable_scale=1.0,
            thermal_scale=1.0,
            weight=0.4,
            description="Nominal demand and renewable availability",
        ),
        Scenario(
            name="high_load_low_renew",
            load_scale=1.20,
            renewable_scale=0.70,
            thermal_scale=0.98,
            branch_scale=0.85,
            branch_overrides={"A27": 0.4, "CA-1": 0.4, "CB-1": 0.4},
            weight=0.35,
            description="Stressed load growth with renewable drought",
        ),
        Scenario(
            name="low_load_high_renew",
            load_scale=0.90,
            renewable_scale=1.15,
            thermal_scale=1.0,
            branch_scale=1.05,
            weight=0.25,
            description="Mild demand with surplus renewables",
        ),
    ]

    robust_tep = ScenarioRobustTEP(
        data_loader,
        scenarios,
        line_cost_per_mw=120_000,  # encourage proactive builds under stress
    )
    robust_tep.generate_candidate_lines(max_candidates=20)
    robust_tep.build_model()

    success = robust_tep.solve(solver="gurobi", time_limit=1800)
    if not success:
        print("Robust TEP did not solve successfully.")
        return

    results = robust_tep.get_results()
    scenario_results = results["scenario_results"]

    print("\nScenario summaries:")
    for scenario in scenarios:
        sr = scenario_results[scenario.name]
        print(
            f"  - {scenario.name}: load={sr['total_load']:.1f} MW, "
            f"generation={sr['total_generation']:.1f} MW, "
            f"operating cost=${sr['operating_cost']:,.2f}"
        )

    print(f"\nLines built: {len(results['lines_built'])}")
    for line in results["lines_built"]:
        print(
            f"  • Bus {int(line['from_bus'])} -> Bus {int(line['to_bus'])}: "
            f"{line['capacity']:.0f} MW, cost=${line['cost']:,.0f}"
        )

    os.makedirs("results", exist_ok=True)
    output_csv = os.path.join("results", "robust_tep_summary.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["scenario", "description", "total_load_mw", "total_generation_mw", "operating_cost"]
        )
        for scenario in scenarios:
            sr = scenario_results[scenario.name]
            writer.writerow(
                [
                    scenario.name,
                    scenario.description,
                    f"{sr['total_load']:.2f}",
                    f"{sr['total_generation']:.2f}",
                    f"{sr['operating_cost']:.2f}",
                ]
            )

    print(f"\nScenario details saved to {output_csv}")
    print(f"Objective value: ${results['objective_value']:,.2f}")
    print(f"Investment cost: ${results['investment_cost']:,.2f}")


if __name__ == "__main__":
    main()
