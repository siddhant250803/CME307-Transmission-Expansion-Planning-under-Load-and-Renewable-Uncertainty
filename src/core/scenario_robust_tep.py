"""
Scenario-based robust Transmission Expansion Planning model.

This module extends the base TEP formulation so that a single investment plan
must withstand multiple load / generation scenarios simultaneously.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pyomo.environ import (
    Binary,
    Constraint,
    ConcreteModel,
    NonNegativeReals,
    Objective,
    Param,
    Reals,
    Set,
    Var,
    minimize,
    value,
)
from pyomo.opt import SolverFactory, TerminationCondition

from .data_loader import RTSDataLoader
from .tep import TEP


RENEWABLE_TYPES = {"WIND", "PV", "CSP", "HYDRO", "ROR"}


@dataclass
class Scenario:
    name: str
    load_scale: float = 1.0
    renewable_scale: float = 1.0
    thermal_scale: float = 1.0
    branch_scale: float = 1.0
    branch_overrides: Optional[Dict[str, float]] = None
    weight: float = 1.0
    description: str = ""


class ScenarioRobustTEP(TEP):
    """Multi-scenario TEP model."""

    def __init__(
        self,
        data_loader: RTSDataLoader,
        scenarios: List[Scenario],
        candidate_lines: Optional[List[dict]] = None,
        line_cost_per_mw: float = 1_000_000.0,
    ):
        super().__init__(data_loader, candidate_lines, line_cost_per_mw)
        self.scenarios = scenarios
        self.scenario_map = {s.name: s for s in scenarios}

    def build_model(self):
        """Build Pyomo model with scenario-indexed operational variables."""
        if not self.scenarios:
            raise ValueError("Scenario list is empty.")

        self.model = ConcreteModel()

        # Ensure candidates exist
        if self.candidate_lines is None:
            self.generate_candidate_lines(max_candidates=25)

        buses = self.data.get_bus_data()
        base_load = self.data.get_load_by_bus()
        branches = self.data.get_branch_data()
        generators = self.data.get_generator_data()

        bus_ids = list(buses.index)
        gen_ids = list(generators["GEN UID"])
        branch_ids = list(branches["UID"])
        candidate_ids = list(range(len(self.candidate_lines)))
        scenario_ids = [s.name for s in self.scenarios]

        self.model.buses = Set(initialize=bus_ids)
        self.model.generators = Set(initialize=gen_ids)
        self.model.branches = Set(initialize=branch_ids)
        self.model.candidates = Set(initialize=candidate_ids)
        self.model.scenarios = Set(initialize=scenario_ids)

        # Scenario weights
        total_weight = sum(s.weight for s in self.scenarios)
        scenario_weights = {s.name: s.weight / total_weight for s in self.scenarios}
        self.model.scenario_weight = Param(
            self.model.scenarios, initialize=scenario_weights
        )

        # Generator parameters
        gen_to_bus = {}
        gen_pmax = {}
        gen_pmin = {}
        gen_cost = {}
        gen_unit_type = {}
        renewable_flags = {}

        for _, row in generators.iterrows():
            gid = row["GEN UID"]
            gen_to_bus[gid] = row["Bus ID"]
            gen_pmax[gid] = row["PMax MW"]
            gen_pmin[gid] = row["PMin MW"]
            avg_hr = row.get("HR_avg_0", 10000) if not pd.isna(row.get("HR_avg_0")) else 10000
            fuel_price = row.get("Fuel Price $/MMBTU", 0)
            gen_cost[gid] = fuel_price * avg_hr / 1000.0
            unit_type = str(row.get("Unit Type", "")).upper()
            gen_unit_type[gid] = unit_type
            renewable_flags[gid] = 1 if any(rt in unit_type for rt in RENEWABLE_TYPES) else 0

        self.model.gen_to_bus = Param(self.model.generators, initialize=gen_to_bus)
        self.model.gen_pmax = Param(self.model.generators, initialize=gen_pmax)
        self.model.gen_pmin = Param(self.model.generators, initialize=gen_pmin)
        self.model.gen_cost = Param(self.model.generators, initialize=gen_cost)
        self.model.gen_is_renewable = Param(
            self.model.generators, initialize=renewable_flags
        )

        # Scenario-specific generator caps
        scenario_gen_pmax = {}
        scenario_gen_pmin = {}
        for scenario in self.scenarios:
            for gid in gen_ids:
                is_renew = renewable_flags[gid]
                base_pmax = gen_pmax[gid]
                base_pmin = gen_pmin[gid]
                scale = (
                    scenario.renewable_scale if is_renew else scenario.thermal_scale
                )
                pmax = base_pmax * scale
                pmin = min(base_pmin * scale, pmax)
                scenario_gen_pmax[(gid, scenario.name)] = pmax
                scenario_gen_pmin[(gid, scenario.name)] = pmin

        self.model.scenario_gen_pmax = Param(
            self.model.generators, self.model.scenarios, initialize=scenario_gen_pmax
        )
        self.model.scenario_gen_pmin = Param(
            self.model.generators, self.model.scenarios, initialize=scenario_gen_pmin
        )

        # Load parameters
        scenario_loads = {}
        for scenario in self.scenarios:
            for bid in bus_ids:
                scenario_loads[(bid, scenario.name)] = base_load.get(bid, 0.0) * scenario.load_scale

        self.model.bus_load = Param(
            self.model.buses, self.model.scenarios, initialize=scenario_loads, default=0.0
        )

        # Branch parameters
        branch_params = self.data.get_branch_parameters()
        branch_susceptance = {}
        branch_rating = {}
        branch_from = {}
        branch_to = {}

        for branch_id in branch_ids:
            params = branch_params[branch_id]
            branch_susceptance[branch_id] = params["susceptance"]
            branch_rating[branch_id] = params["rating"]
            branch_from[branch_id] = params["from_bus"]
            branch_to[branch_id] = params["to_bus"]

        self.model.branch_susceptance = Param(self.model.branches, initialize=branch_susceptance)
        self.model.branch_rating = Param(self.model.branches, initialize=branch_rating)
        self.model.branch_from = Param(self.model.branches, initialize=branch_from)
        self.model.branch_to = Param(self.model.branches, initialize=branch_to)

        scenario_branch_rating = {}
        for scenario in self.scenarios:
            overrides = scenario.branch_overrides or {}
            for br in branch_ids:
                scale = overrides.get(br, scenario.branch_scale)
                scenario_branch_rating[(br, scenario.name)] = branch_rating[br] * scale

        self.model.scenario_branch_rating = Param(
            self.model.branches, self.model.scenarios, initialize=scenario_branch_rating
        )

        # Candidate parameters
        candidate_from = {}
        candidate_to = {}
        candidate_capacity = {}
        candidate_cost = {}
        candidate_susceptance = {}

        for idx, cand in enumerate(self.candidate_lines):
            candidate_from[idx] = cand["from_bus"]
            candidate_to[idx] = cand["to_bus"]
            candidate_capacity[idx] = cand["capacity"]
            candidate_cost[idx] = cand["cost"]
            candidate_susceptance[idx] = cand.get("susceptance", 10.0)

        self.model.candidate_from = Param(self.model.candidates, initialize=candidate_from)
        self.model.candidate_to = Param(self.model.candidates, initialize=candidate_to)
        self.model.candidate_capacity = Param(self.model.candidates, initialize=candidate_capacity)
        self.model.candidate_cost = Param(self.model.candidates, initialize=candidate_cost)
        self.model.candidate_susceptance = Param(self.model.candidates, initialize=candidate_susceptance)

        # Decision variables
        self.model.build_line = Var(self.model.candidates, domain=Binary)
        self.model.p_gen = Var(self.model.generators, self.model.scenarios, domain=NonNegativeReals)
        self.model.theta = Var(self.model.buses, self.model.scenarios, domain=Reals)
        self.model.p_flow = Var(self.model.branches, self.model.scenarios, domain=Reals)
        self.model.p_flow_candidate = Var(self.model.candidates, self.model.scenarios, domain=Reals)

        # Objective: investment + weighted operating cost
        def objective_rule(m):
            investment = sum(m.candidate_cost[c] * m.build_line[c] for c in m.candidates)
            operating = sum(
                m.scenario_weight[s] * sum(m.gen_cost[g] * m.p_gen[g, s] for g in m.generators)
                for s in m.scenarios
            )
            return investment + operating

        self.model.obj = Objective(rule=objective_rule, sense=minimize)

        # Power balance
        def power_balance_rule(m, b, s):
            gen_at_bus = [g for g in m.generators if m.gen_to_bus[g] == b]
            existing_in = sum(m.p_flow[br, s] for br in m.branches if m.branch_to[br] == b)
            existing_out = sum(m.p_flow[br, s] for br in m.branches if m.branch_from[br] == b)
            cand_in = sum(m.p_flow_candidate[c, s] for c in m.candidates if m.candidate_to[c] == b)
            cand_out = sum(m.p_flow_candidate[c, s] for c in m.candidates if m.candidate_from[c] == b)
            return (
                sum(m.p_gen[g, s] for g in gen_at_bus)
                + existing_in
                - existing_out
                + cand_in
                - cand_out
                == m.bus_load[b, s]
            )

        self.model.power_balance = Constraint(self.model.buses, self.model.scenarios, rule=power_balance_rule)

        # DC flow for existing lines
        def dc_flow_rule(m, br, s):
            return m.p_flow[br, s] == m.branch_susceptance[br] * (
                m.theta[m.branch_from[br], s] - m.theta[m.branch_to[br], s]
            )

        self.model.dc_flow = Constraint(self.model.branches, self.model.scenarios, rule=dc_flow_rule)

        # DC flow for candidate lines with Big-M
        M = 10000.0

        def cand_flow_upper(m, c, s):
            delta = m.theta[m.candidate_from[c], s] - m.theta[m.candidate_to[c], s]
            return m.p_flow_candidate[c, s] - m.candidate_susceptance[c] * delta <= M * (1 - m.build_line[c])

        def cand_flow_lower(m, c, s):
            delta = m.theta[m.candidate_from[c], s] - m.theta[m.candidate_to[c], s]
            return m.p_flow_candidate[c, s] - m.candidate_susceptance[c] * delta >= -M * (1 - m.build_line[c])

        def cand_zero_upper(m, c, s):
            return m.p_flow_candidate[c, s] <= M * m.build_line[c]

        def cand_zero_lower(m, c, s):
            return m.p_flow_candidate[c, s] >= -M * m.build_line[c]

        self.model.cand_flow_upper = Constraint(self.model.candidates, self.model.scenarios, rule=cand_flow_upper)
        self.model.cand_flow_lower = Constraint(self.model.candidates, self.model.scenarios, rule=cand_flow_lower)
        self.model.cand_zero_upper = Constraint(self.model.candidates, self.model.scenarios, rule=cand_zero_upper)
        self.model.cand_zero_lower = Constraint(self.model.candidates, self.model.scenarios, rule=cand_zero_lower)

        # Limits
        def branch_max(m, br, s):
            return m.p_flow[br, s] <= m.scenario_branch_rating[br, s]

        def branch_min(m, br, s):
            return m.p_flow[br, s] >= -m.scenario_branch_rating[br, s]

        self.model.branch_max = Constraint(self.model.branches, self.model.scenarios, rule=branch_max)
        self.model.branch_min = Constraint(self.model.branches, self.model.scenarios, rule=branch_min)

        def candidate_flow_max(m, c, s):
            return m.p_flow_candidate[c, s] <= m.candidate_capacity[c] * m.build_line[c]

        def candidate_flow_min(m, c, s):
            return m.p_flow_candidate[c, s] >= -m.candidate_capacity[c] * m.build_line[c]

        self.model.candidate_flow_max = Constraint(self.model.candidates, self.model.scenarios, rule=candidate_flow_max)
        self.model.candidate_flow_min = Constraint(self.model.candidates, self.model.scenarios, rule=candidate_flow_min)

        # Generator bounds
        def gen_max_rule(m, g, s):
            return m.p_gen[g, s] <= m.scenario_gen_pmax[g, s]

        def gen_min_rule(m, g, s):
            return m.p_gen[g, s] >= m.scenario_gen_pmin[g, s]

        self.model.gen_max = Constraint(self.model.generators, self.model.scenarios, rule=gen_max_rule)
        self.model.gen_min = Constraint(self.model.generators, self.model.scenarios, rule=gen_min_rule)

        # Reference angle for each scenario (use min bus id)
        ref_bus = min(bus_ids)

        def ref_rule(m, s):
            return m.theta[ref_bus, s] == 0

        self.model.ref_bus = Constraint(self.model.scenarios, rule=ref_rule)

    def solve(self, solver: str = "gurobi", time_limit: int = 3600):
        if self.model is None:
            self.build_model()

        opt = SolverFactory(solver)
        if solver == "gurobi":
            opt.options["TimeLimit"] = time_limit
            opt.options["MIPGap"] = 0.02

        print(f"Solving scenario-robust TEP with {len(self.scenarios)} scenarios...")
        results = opt.solve(self.model, tee=True)

        term = results.solver.termination_condition
        if term in (TerminationCondition.optimal, TerminationCondition.feasible):
            self.results = results
            return True
        else:
            print(f"Solver terminated with condition: {term}")
            return False

    def get_results(self):
        if self.model is None or self.results is None:
            return None

        scenario_results = {}
        for s in self.model.scenarios:
            total_load = sum(value(self.model.bus_load[b, s]) for b in self.model.buses)
            total_gen = sum(value(self.model.p_gen[g, s]) for g in self.model.generators)
            operating_cost = sum(
                value(self.model.gen_cost[g] * self.model.p_gen[g, s])
                for g in self.model.generators
            )
            scenario_results[s] = {
                "total_load": total_load,
                "total_generation": total_gen,
                "operating_cost": operating_cost,
            }

        lines_built = []
        for c in self.model.candidates:
            if value(self.model.build_line[c]) > 0.5:
                lines_built.append(
                    {
                        "candidate_id": c,
                        "from_bus": value(self.model.candidate_from[c]),
                        "to_bus": value(self.model.candidate_to[c]),
                        "capacity": value(self.model.candidate_capacity[c]),
                        "cost": value(self.model.candidate_cost[c]),
                    }
                )

        investment_cost = sum(
            value(self.model.candidate_cost[c] * self.model.build_line[c])
            for c in self.model.candidates
        )
        operating_cost = sum(
            value(self.model.scenario_weight[s])
            * sum(value(self.model.gen_cost[g] * self.model.p_gen[g, s]) for g in self.model.generators)
            for s in self.model.scenarios
        )

        return {
            "objective_value": value(self.model.obj),
            "investment_cost": investment_cost,
            "scenario_results": scenario_results,
            "lines_built": lines_built,
        }
