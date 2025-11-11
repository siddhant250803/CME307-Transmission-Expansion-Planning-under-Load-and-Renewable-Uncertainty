"""
DC Optimal Power Flow (OPF) baseline model
"""
from pyomo.environ import *
import pandas as pd
import numpy as np
from data_loader import RTSDataLoader

class DCOPF:
    """DC Optimal Power Flow model"""
    
    def __init__(self, data_loader):
        self.data = data_loader
        self.model = None
        self.results = None
        
    def build_model(self):
        """Build DC OPF model"""
        self.model = ConcreteModel()
        
        # Get data
        buses = self.data.get_bus_data()
        branches = self.data.get_branch_data()
        generators = self.data.get_generator_data()
        load = self.data.get_load_by_bus()
        gen_by_bus = self.data.get_generators_by_bus()
        
        # Sets
        bus_ids = list(buses.index)
        gen_ids = list(generators['GEN UID'])
        branch_ids = list(branches['UID'])
        
        self.model.buses = Set(initialize=bus_ids)
        self.model.generators = Set(initialize=gen_ids)
        self.model.branches = Set(initialize=branch_ids)
        
        # Create generator-to-bus mapping
        gen_to_bus = {}
        gen_pmax = {}
        gen_pmin = {}
        gen_cost = {}  # Simplified: use fuel price * heat rate
        
        for _, gen in generators.iterrows():
            gen_id = gen['GEN UID']
            bus_id = gen['Bus ID']
            gen_to_bus[gen_id] = bus_id
            gen_pmax[gen_id] = gen['PMax MW']
            gen_pmin[gen_id] = gen['PMin MW']
            
            # Simple cost: fuel price * average heat rate (BTU/kWh) / 1000
            # Convert to $/MWh
            fuel_price = gen['Fuel Price $/MMBTU']
            avg_hr = gen.get('HR_avg_0', 10000)  # Default if missing
            if pd.isna(avg_hr):
                avg_hr = 10000
            # Cost in $/MWh = (fuel_price $/MMBTU) * (avg_hr BTU/kWh) / 1000
            gen_cost[gen_id] = fuel_price * avg_hr / 1000.0
        
        self.model.gen_to_bus = Param(self.model.generators, initialize=gen_to_bus)
        self.model.gen_pmax = Param(self.model.generators, initialize=gen_pmax)
        self.model.gen_pmin = Param(self.model.generators, initialize=gen_pmin)
        self.model.gen_cost = Param(self.model.generators, initialize=gen_cost)
        
        # Branch parameters
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
        
        # Load (renamed from 'load' to avoid Pyomo reserved attribute)
        self.model.bus_load = Param(self.model.buses, initialize=load, default=0.0)
        
        # Variables
        self.model.p_gen = Var(self.model.generators, domain=NonNegativeReals)
        self.model.theta = Var(self.model.buses, domain=Reals)  # Voltage angles
        self.model.p_flow = Var(self.model.branches, domain=Reals)  # Branch flows
        
        # Objective: minimize generation cost
        self.model.obj = Objective(
            expr=sum(self.model.gen_cost[g] * self.model.p_gen[g] for g in self.model.generators),
            sense=minimize
        )
        
        # Power balance at each bus
        def power_balance_rule(m, b):
            gen_at_bus = [g for g in m.generators if m.gen_to_bus[g] == b]
            flow_in = sum(m.p_flow[br] for br in m.branches if m.branch_to[br] == b)
            flow_out = sum(m.p_flow[br] for br in m.branches if m.branch_from[br] == b)
            return sum(m.p_gen[g] for g in gen_at_bus) + flow_in - flow_out == m.bus_load[b]
        
        self.model.power_balance = Constraint(self.model.buses, rule=power_balance_rule)
        
        # DC power flow: p_flow = susceptance * (theta_from - theta_to)
        def dc_flow_rule(m, br):
            return m.p_flow[br] == m.branch_susceptance[br] * (m.theta[m.branch_from[br]] - m.theta[m.branch_to[br]])
        
        self.model.dc_flow = Constraint(self.model.branches, rule=dc_flow_rule)
        
        # Generator limits
        self.model.gen_max = Constraint(
            self.model.generators,
            rule=lambda m, g: m.p_gen[g] <= m.gen_pmax[g]
        )
        self.model.gen_min = Constraint(
            self.model.generators,
            rule=lambda m, g: m.p_gen[g] >= m.gen_pmin[g]
        )
        
        # Branch flow limits
        self.model.branch_max = Constraint(
            self.model.branches,
            rule=lambda m, br: m.p_flow[br] <= m.branch_rating[br]
        )
        self.model.branch_min = Constraint(
            self.model.branches,
            rule=lambda m, br: m.p_flow[br] >= -m.branch_rating[br]
        )
        
        # Reference bus (set angle to 0)
        ref_bus = min(bus_ids)  # Use minimum bus ID as reference
        self.model.ref_bus = Constraint(expr=self.model.theta[ref_bus] == 0)
        
    def solve(self, solver='gurobi'):
        """Solve the DC OPF model"""
        if self.model is None:
            self.build_model()
        
        if solver == 'gurobi':
            opt = SolverFactory('gurobi')
        else:
            opt = SolverFactory('glpk')
        
        print(f"Solving DC OPF with {solver}...")
        results = opt.solve(self.model, tee=False)
        
        if results.solver.termination_condition == TerminationCondition.optimal:
            print("Solution found!")
            self.results = results
            return True
        else:
            print(f"Solver termination: {results.solver.termination_condition}")
            return False
    
    def get_results(self):
        """Extract results from solved model"""
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
        
        # Generation
        for g in self.model.generators:
            p_val = value(self.model.p_gen[g])
            results_dict['generation'][g] = p_val
            results_dict['total_generation'] += p_val
        
        # Branch flows
        for br in self.model.branches:
            flow_val = value(self.model.p_flow[br])
            results_dict['flows'][br] = flow_val
            rating = value(self.model.branch_rating[br])
            if abs(flow_val) >= 0.99 * rating:  # Near capacity
                results_dict['congested_branches'].append({
                    'branch': br,
                    'flow': flow_val,
                    'rating': rating,
                    'utilization': abs(flow_val) / rating
                })
        
        # Voltage angles
        for b in self.model.buses:
            results_dict['angles'][b] = value(self.model.theta[b])
        
        return results_dict
    
    def print_summary(self):
        """Print summary of results"""
        results = self.get_results()
        if results is None:
            print("No results available")
            return
        
        print("\n" + "="*60)
        print("DC OPF RESULTS SUMMARY")
        print("="*60)
        print(f"Total Operating Cost: ${results['objective_value']:,.2f}")
        print(f"Total Generation: {results['total_generation']:.2f} MW")
        print(f"Total Load: {results['total_load']:.2f} MW")
        print(f"Number of Congested Branches: {len(results['congested_branches'])}")
        
        if results['congested_branches']:
            print("\nCongested Branches:")
            for cb in results['congested_branches'][:10]:  # Show top 10
                print(f"  {cb['branch']}: {cb['flow']:.2f} MW / {cb['rating']:.2f} MW ({cb['utilization']*100:.1f}%)")
        
        print("="*60)

