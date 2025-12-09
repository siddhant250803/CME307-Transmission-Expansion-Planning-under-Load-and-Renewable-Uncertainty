"""DC Optimal Power Flow model for power system dispatch optimization."""

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint,
    NonNegativeReals, Reals, minimize, value
)
from pyomo.opt import SolverFactory, TerminationCondition
import pandas as pd
import numpy as np
from .data_loader import RTSDataLoader


class DCOPF:
    """DC Optimal Power Flow model."""
    
    def __init__(self, data_loader: RTSDataLoader):
        """Initialize DC-OPF model."""
        self.data = data_loader
        self.model = None
        self.results = None
        
    def build_model(self) -> None:
        """Build the DC-OPF optimization model."""
        self.model = ConcreteModel(name="DC-OPF")
        
        buses = self.data.get_bus_data()
        branches = self.data.get_branch_data()
        generators = self.data.get_generator_data()
        load = self.data.get_load_by_bus()
        gen_by_bus = self.data.get_generators_by_bus()
        
        bus_ids = list(buses.index)
        gen_ids = list(generators['GEN UID'])
        branch_ids = list(branches['UID'])
        
        self.model.buses = Set(initialize=bus_ids, doc="Set of system buses")
        self.model.generators = Set(initialize=gen_ids, doc="Set of generators")
        self.model.branches = Set(initialize=branch_ids, doc="Set of transmission lines")
        
        gen_to_bus = {}
        gen_pmax = {}
        gen_pmin = {}
        gen_cost = {}
        
        for _, gen in generators.iterrows():
            gen_id = gen['GEN UID']
            bus_id = gen['Bus ID']
            
            gen_to_bus[gen_id] = bus_id
            gen_pmax[gen_id] = gen['PMax MW']
            gen_pmin[gen_id] = gen['PMin MW']
            
            fuel_price = gen['Fuel Price $/MMBTU']
            avg_hr = gen.get('HR_avg_0', 10000)
            if pd.isna(avg_hr):
                avg_hr = 10000
            gen_cost[gen_id] = fuel_price * avg_hr / 1000.0
        
        self.model.gen_to_bus = Param(self.model.generators, initialize=gen_to_bus)
        self.model.gen_pmax = Param(self.model.generators, initialize=gen_pmax)
        self.model.gen_pmin = Param(self.model.generators, initialize=gen_pmin)
        self.model.gen_cost = Param(self.model.generators, initialize=gen_cost)
        
        branch_params = self.data.get_branch_parameters()
        branch_susceptance = {}
        branch_rating = {}
        branch_from = {}
        branch_to = {}
        
        for branch_id in branch_ids:
            params = branch_params[branch_id]
            branch_susceptance[branch_id] = params['susceptance']
            branch_rating[branch_id] = params['rating']
            branch_from[branch_id] = params['from_bus']
            branch_to[branch_id] = params['to_bus']
        
        self.model.branch_susceptance = Param(self.model.branches, initialize=branch_susceptance)
        self.model.branch_rating = Param(self.model.branches, initialize=branch_rating)
        self.model.branch_from = Param(self.model.branches, initialize=branch_from)
        self.model.branch_to = Param(self.model.branches, initialize=branch_to)
        
        self.model.bus_load = Param(self.model.buses, initialize=load, default=0.0)
        
        self.model.p_gen = Var(self.model.generators, domain=NonNegativeReals)
        self.model.theta = Var(self.model.buses, domain=Reals)
        self.model.p_flow = Var(self.model.branches, domain=Reals)
        
        self.model.obj = Objective(
            expr=sum(self.model.gen_cost[g] * self.model.p_gen[g] for g in self.model.generators),
            sense=minimize
        )
        
        def power_balance_rule(m, b):
            gen_at_bus = [g for g in m.generators if m.gen_to_bus[g] == b]
            flow_in = sum(m.p_flow[br] for br in m.branches if m.branch_to[br] == b)
            flow_out = sum(m.p_flow[br] for br in m.branches if m.branch_from[br] == b)
            return sum(m.p_gen[g] for g in gen_at_bus) + flow_in - flow_out == m.bus_load[b]
        
        self.model.power_balance = Constraint(self.model.buses, rule=power_balance_rule)
        
        def dc_flow_rule(m, br):
            return m.p_flow[br] == m.branch_susceptance[br] * (
                m.theta[m.branch_from[br]] - m.theta[m.branch_to[br]]
            )
        
        self.model.dc_flow = Constraint(self.model.branches, rule=dc_flow_rule)
        
        self.model.gen_max = Constraint(
            self.model.generators,
            rule=lambda m, g: m.p_gen[g] <= m.gen_pmax[g]
        )
        
        self.model.gen_min = Constraint(
            self.model.generators,
            rule=lambda m, g: m.p_gen[g] >= m.gen_pmin[g]
        )
        
        self.model.branch_max = Constraint(
            self.model.branches,
            rule=lambda m, br: m.p_flow[br] <= m.branch_rating[br]
        )
        
        self.model.branch_min = Constraint(
            self.model.branches,
            rule=lambda m, br: m.p_flow[br] >= -m.branch_rating[br]
        )
        
        ref_bus = min(bus_ids)
        self.model.ref_bus = Constraint(expr=self.model.theta[ref_bus] == 0)
        
    def solve(self, solver: str = 'gurobi') -> bool:
        """Solve the DC-OPF optimization problem."""
        if self.model is None:
            self.build_model()
        
        if solver == 'gurobi':
            opt = SolverFactory('gurobi')
        else:
            opt = SolverFactory('glpk')
        
        print(f"Solving DC-OPF with {solver}...")
        results = opt.solve(self.model, tee=False)
        
        if results.solver.termination_condition == TerminationCondition.optimal:
            print("✓ Optimal solution found!")
            self.results = results
            return True
        else:
            print(f"✗ Solver termination: {results.solver.termination_condition}")
            return False
    
    def get_results(self) -> dict:
        """Extract results from the solved model."""
        if self.model is None or self.results is None:
            return None
        
        results_dict = {
            'objective_value': value(self.model.obj),
            'generation': {},
            'flows': {},
            'angles': {},
            'total_generation': 0,
            'total_load': sum(value(self.model.bus_load[b]) for b in self.model.buses),
            'congested_branches': []
        }
        
        for g in self.model.generators:
            p_val = value(self.model.p_gen[g])
            results_dict['generation'][g] = p_val
            results_dict['total_generation'] += p_val
        
        for br in self.model.branches:
            flow_val = value(self.model.p_flow[br])
            results_dict['flows'][br] = flow_val
            rating = value(self.model.branch_rating[br])
            if abs(flow_val) >= 0.99 * rating:
                results_dict['congested_branches'].append({
                    'branch': br,
                    'flow': flow_val,
                    'rating': rating,
                    'utilization': abs(flow_val) / rating
                })
        
        for b in self.model.buses:
            results_dict['angles'][b] = value(self.model.theta[b])
        
        return results_dict
    
    def print_summary(self) -> None:
        """Print a formatted summary of the DC-OPF solution."""
        results = self.get_results()
        if results is None:
            print("No results available - solve the model first")
            return
        
        print("\n" + "="*60)
        print("DC-OPF RESULTS SUMMARY")
        print("="*60)
        print(f"Total Operating Cost:     ${results['objective_value']:,.2f}")
        print(f"Total Generation:         {results['total_generation']:.2f} MW")
        print(f"Total Load:               {results['total_load']:.2f} MW")
        print(f"Power Balance Check:      {results['total_generation'] - results['total_load']:.4f} MW")
        print(f"Congested Branches:       {len(results['congested_branches'])}")
        
        if results['congested_branches']:
            print("\nCongested Lines (≥99% utilized):")
            for cb in results['congested_branches'][:10]:  # Show top 10
                print(f"  {cb['branch']:12s}: {cb['flow']:8.2f} MW / "
                      f"{cb['rating']:8.2f} MW ({cb['utilization']*100:.1f}%)")
        
        print("="*60)
