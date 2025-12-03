"""
Transmission Expansion Planning (TEP) MILP model
"""
from pyomo.environ import *
import pandas as pd
import numpy as np
from .data_loader import RTSDataLoader

class TEP:
    """Transmission Expansion Planning model"""
    
    def __init__(self, data_loader, candidate_lines=None, line_cost_per_mw=1000000, 
                 cost_model='capacity_distance'):
        """
        Initialize TEP model
        
        Parameters:
        -----------
        data_loader : RTSDataLoader
            Data loader instance
        candidate_lines : list of dict, optional
            List of candidate lines to build. Each dict should have:
            {'from_bus': int, 'to_bus': int, 'capacity': float, 'cost': float}
            If None, generates candidates based on existing network
        line_cost_per_mw : float
            Cost per MW of new line capacity (default: $1M/MW)
            Used for 'capacity_distance' model: cost = line_cost_per_mw * capacity * (dist/100)
        cost_model : str
            Cost calculation model:
            - 'capacity_distance': cost = line_cost_per_mw * capacity * (dist/100) [current]
            - 'distance_based': cost = cost_per_mile * distance_miles (more realistic)
            - 'capacity_only': cost = line_cost_per_mw * capacity (ignores distance)
        """
        self.data = data_loader
        self.line_cost_per_mw = line_cost_per_mw
        self.cost_model = cost_model
        self.candidate_lines = candidate_lines
        self.model = None
        self.results = None
        
    def generate_candidate_lines(self, max_candidates=50):
        """Generate candidate transmission lines"""
        if self.candidate_lines is not None:
            return self.candidate_lines
        
        buses = self.data.get_bus_data()
        existing_branches = self.data.get_branch_data()
        
        # Get existing connections
        existing_connections = set()
        for _, branch in existing_branches.iterrows():
            from_bus = int(branch['From Bus'])
            to_bus = int(branch['To Bus'])
            existing_connections.add((min(from_bus, to_bus), max(from_bus, to_bus)))
        
        # Generate candidates: connect buses in same area or adjacent areas
        candidates = []
        bus_ids = list(buses.index)
        
        for i, bus1 in enumerate(bus_ids):
            for bus2 in bus_ids[i+1:]:
                # Skip if already connected
                key = (min(bus1, bus2), max(bus1, bus2))
                if key in existing_connections:
                    continue
                
                # Only consider buses in same or adjacent areas
                area1 = buses.loc[bus1, 'Area']
                area2 = buses.loc[bus2, 'Area']
                if area1 == area2 or abs(area1 - area2) == 1:
                    # Estimate capacity based on voltage level
                    base_kv = buses.loc[bus1, 'BaseKV']
                    if base_kv >= 230:
                        capacity = 500  # MW
                    else:
                        capacity = 200  # MW
                    
                    # Calculate distance
                    if 'lat' in buses.columns and 'lng' in buses.columns:
                        lat1, lng1 = buses.loc[bus1, 'lat'], buses.loc[bus1, 'lng']
                        lat2, lng2 = buses.loc[bus2, 'lat'], buses.loc[bus2, 'lng']
                        # Rough distance estimate (degrees to km approximation)
                        dist_km = np.sqrt((lat1-lat2)**2 + (lng1-lng2)**2) * 111  # km
                        dist_miles = dist_km * 0.621371  # Convert to miles
                    else:
                        dist_km = 50  # Default distance
                        dist_miles = dist_km * 0.621371
                    
                    # Calculate cost based on selected model
                    cost = self._calculate_line_cost(capacity, dist_miles, base_kv)
                    
                    candidates.append({
                        'from_bus': bus1,
                        'to_bus': bus2,
                        'capacity': capacity,
                        'cost': cost,
                        'susceptance': 1.0 / 0.1  # Default reactance of 0.1 p.u.
                    })
                    
                    if len(candidates) >= max_candidates:
                        break
            
            if len(candidates) >= max_candidates:
                break
        
        self.candidate_lines = candidates
        print(f"Generated {len(candidates)} candidate lines")
        return candidates
    
    def _calculate_line_cost(self, capacity_mw, distance_miles, voltage_kv):
        """
        Calculate transmission line cost based on selected model
        
        Parameters:
        -----------
        capacity_mw : float
            Line capacity in MW
        distance_miles : float
            Line distance in miles
        voltage_kv : float
            Line voltage in kV
        
        Returns:
        --------
        cost : float
            Total line cost in dollars
        """
        if self.cost_model == 'capacity_distance':
            # Original model: cost = line_cost_per_mw * capacity * (dist/100)
            # Note: This scales distance by 100km, so dist_miles converted to km
            dist_km = distance_miles / 0.621371
            cost = self.line_cost_per_mw * capacity_mw * (dist_km / 100.0)
        
        elif self.cost_model == 'distance_based':
            # More realistic: cost per mile varies by voltage
            # Based on MISO/FERC data: 230kV ~$1.5M/mile, 345kV ~$2.5M/mile, 500kV ~$4M/mile
            cost_per_mile = {
                138: 1.0e6,   # $1M/mile for 138kV
                230: 1.5e6,   # $1.5M/mile for 230kV
                345: 2.5e6,   # $2.5M/mile for 345kV
                500: 4.0e6,   # $4M/mile for 500kV
                765: 6.0e6    # $6M/mile for 765kV
            }
            # Find closest voltage level
            voltage_levels = sorted(cost_per_mile.keys())
            closest_voltage = min(voltage_levels, key=lambda v: abs(v - voltage_kv))
            cost = cost_per_mile[closest_voltage] * distance_miles
        
        elif self.cost_model == 'capacity_only':
            # Simple: cost = line_cost_per_mw * capacity (ignores distance)
            cost = self.line_cost_per_mw * capacity_mw
        
        else:
            # Default to capacity_distance
            dist_km = distance_miles / 0.621371
            cost = self.line_cost_per_mw * capacity_mw * (dist_km / 100.0)
        
        return cost
    
    def build_model(self):
        """Build TEP MILP model"""
        self.model = ConcreteModel()
        
        # Generate candidates if not provided
        if self.candidate_lines is None:
            self.generate_candidate_lines()
        
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
        candidate_ids = list(range(len(self.candidate_lines)))
        
        self.model.buses = Set(initialize=bus_ids)
        self.model.generators = Set(initialize=gen_ids)
        self.model.branches = Set(initialize=branch_ids)
        self.model.candidates = Set(initialize=candidate_ids)
        
        # Existing branch parameters
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
        
        # Candidate line parameters
        candidate_from = {}
        candidate_to = {}
        candidate_capacity = {}
        candidate_cost = {}
        candidate_susceptance = {}
        
        for idx, cand in enumerate(self.candidate_lines):
            candidate_from[idx] = cand['from_bus']
            candidate_to[idx] = cand['to_bus']
            candidate_capacity[idx] = cand['capacity']
            candidate_cost[idx] = cand['cost']
            candidate_susceptance[idx] = cand.get('susceptance', 10.0)
        
        self.model.candidate_from = Param(self.model.candidates, initialize=candidate_from)
        self.model.candidate_to = Param(self.model.candidates, initialize=candidate_to)
        self.model.candidate_capacity = Param(self.model.candidates, initialize=candidate_capacity)
        self.model.candidate_cost = Param(self.model.candidates, initialize=candidate_cost)
        self.model.candidate_susceptance = Param(self.model.candidates, initialize=candidate_susceptance)
        
        # Generator parameters
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
        
        # Load (renamed from 'load' to avoid Pyomo reserved attribute)
        self.model.bus_load = Param(self.model.buses, initialize=load, default=0.0)
        
        # Variables
        self.model.p_gen = Var(self.model.generators, domain=NonNegativeReals)
        self.model.theta = Var(self.model.buses, domain=Reals)
        self.model.p_flow = Var(self.model.branches, domain=Reals)
        self.model.p_flow_candidate = Var(self.model.candidates, domain=Reals)
        self.model.build_line = Var(self.model.candidates, domain=Binary)  # 1 if line is built
        
        # Objective: minimize investment cost + operating cost
        investment_cost = sum(self.model.candidate_cost[c] * self.model.build_line[c] 
                              for c in self.model.candidates)
        operating_cost = sum(self.model.gen_cost[g] * self.model.p_gen[g] 
                            for g in self.model.generators)
        
        self.model.obj = Objective(expr=investment_cost + operating_cost, sense=minimize)
        
        # Power balance
        def power_balance_rule(m, b):
            gen_at_bus = [g for g in m.generators if m.gen_to_bus[g] == b]
            flow_in_existing = sum(m.p_flow[br] for br in m.branches if m.branch_to[br] == b)
            flow_out_existing = sum(m.p_flow[br] for br in m.branches if m.branch_from[br] == b)
            flow_in_candidate = sum(m.p_flow_candidate[c] for c in m.candidates if m.candidate_to[c] == b)
            flow_out_candidate = sum(m.p_flow_candidate[c] for c in m.candidates if m.candidate_from[c] == b)
            
            return (sum(m.p_gen[g] for g in gen_at_bus) + 
                   flow_in_existing - flow_out_existing +
                   flow_in_candidate - flow_out_candidate == m.bus_load[b])
        
        self.model.power_balance = Constraint(self.model.buses, rule=power_balance_rule)
        
        # DC flow for existing branches
        def dc_flow_rule(m, br):
            return m.p_flow[br] == m.branch_susceptance[br] * (m.theta[m.branch_from[br]] - m.theta[m.branch_to[br]])
        
        self.model.dc_flow = Constraint(self.model.branches, rule=dc_flow_rule)
        
        # DC flow for candidate lines (only if built)
        # Big-M formulation: p_flow = susceptance * (theta_from - theta_to) if built, else 0
        M = 10000  # Large number for Big-M
        
        def dc_flow_candidate_upper(m, c):
            from_bus = m.candidate_from[c]
            to_bus = m.candidate_to[c]
            susceptance = m.candidate_susceptance[c]
            delta_theta = m.theta[from_bus] - m.theta[to_bus]
            # If built: p_flow = susceptance * delta_theta
            # If not built: relax constraint
            return m.p_flow_candidate[c] - susceptance * delta_theta <= M * (1 - m.build_line[c])
        
        def dc_flow_candidate_lower(m, c):
            from_bus = m.candidate_from[c]
            to_bus = m.candidate_to[c]
            susceptance = m.candidate_susceptance[c]
            delta_theta = m.theta[from_bus] - m.theta[to_bus]
            # If built: p_flow = susceptance * delta_theta
            # If not built: relax constraint
            return m.p_flow_candidate[c] - susceptance * delta_theta >= -M * (1 - m.build_line[c])
        
        def dc_flow_candidate_zero(m, c):
            # If not built, flow must be zero
            return m.p_flow_candidate[c] <= M * m.build_line[c]
        
        def dc_flow_candidate_zero_lower(m, c):
            # If not built, flow must be zero (lower bound)
            return m.p_flow_candidate[c] >= -M * m.build_line[c]
        
        self.model.dc_flow_candidate_upper = Constraint(self.model.candidates, rule=dc_flow_candidate_upper)
        self.model.dc_flow_candidate_lower = Constraint(self.model.candidates, rule=dc_flow_candidate_lower)
        self.model.dc_flow_candidate_zero = Constraint(self.model.candidates, rule=dc_flow_candidate_zero)
        self.model.dc_flow_candidate_zero_lower = Constraint(self.model.candidates, rule=dc_flow_candidate_zero_lower)
        
        # Generator limits
        self.model.gen_max = Constraint(
            self.model.generators,
            rule=lambda m, g: m.p_gen[g] <= m.gen_pmax[g]
        )
        self.model.gen_min = Constraint(
            self.model.generators,
            rule=lambda m, g: m.p_gen[g] >= m.gen_pmin[g]
        )
        
        # Existing branch flow limits
        self.model.branch_max = Constraint(
            self.model.branches,
            rule=lambda m, br: m.p_flow[br] <= m.branch_rating[br]
        )
        self.model.branch_min = Constraint(
            self.model.branches,
            rule=lambda m, br: m.p_flow[br] >= -m.branch_rating[br]
        )
        
        # Candidate line flow limits (only if built)
        def candidate_flow_max_rule(m, c):
            return m.p_flow_candidate[c] <= m.candidate_capacity[c] * m.build_line[c]
        
        def candidate_flow_min_rule(m, c):
            return m.p_flow_candidate[c] >= -m.candidate_capacity[c] * m.build_line[c]
        
        self.model.candidate_flow_max = Constraint(self.model.candidates, rule=candidate_flow_max_rule)
        self.model.candidate_flow_min = Constraint(self.model.candidates, rule=candidate_flow_min_rule)
        
        # Reference bus
        ref_bus = min(bus_ids)
        self.model.ref_bus = Constraint(expr=self.model.theta[ref_bus] == 0)
    
    def solve(self, solver='gurobi', time_limit=3600):
        """Solve the TEP model"""
        if self.model is None:
            self.build_model()
        
        if solver == 'gurobi':
            opt = SolverFactory('gurobi')
            opt.options['TimeLimit'] = time_limit
            opt.options['MIPGap'] = 0.01  # 1% optimality gap
        else:
            opt = SolverFactory('glpk')
        
        print(f"Solving TEP MILP with {solver}...")
        results = opt.solve(self.model, tee=True)
        
        if results.solver.termination_condition == TerminationCondition.optimal:
            print("Optimal solution found!")
            self.results = results
            return True
        elif results.solver.termination_condition == TerminationCondition.feasible:
            print("Feasible solution found (may not be optimal)")
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
            'investment_cost': sum(value(self.model.candidate_cost[c] * self.model.build_line[c]) 
                                   for c in self.model.candidates),
            'operating_cost': sum(value(self.model.gen_cost[g] * self.model.p_gen[g]) 
                                 for g in self.model.generators),
            'lines_built': [],
            'generation': {},
            'flows': {},
            'congested_branches': []
        }
        
        # Lines built
        for c in self.model.candidates:
            if value(self.model.build_line[c]) > 0.5:
                results_dict['lines_built'].append({
                    'candidate_id': c,
                    'from_bus': value(self.model.candidate_from[c]),
                    'to_bus': value(self.model.candidate_to[c]),
                    'capacity': value(self.model.candidate_capacity[c]),
                    'cost': value(self.model.candidate_cost[c]),
                    'flow': value(self.model.p_flow_candidate[c])
                })
        
        # Generation
        for g in self.model.generators:
            results_dict['generation'][g] = value(self.model.p_gen[g])
        
        # Branch flows
        for br in self.model.branches:
            flow_val = value(self.model.p_flow[br])
            results_dict['flows'][br] = flow_val
            rating = value(self.model.branch_rating[br])
            if abs(flow_val) >= 0.99 * rating:
                results_dict['congested_branches'].append({
                    'branch': br,
                    'flow': flow_val,
                    'rating': rating
                })
        
        return results_dict
    
    def print_summary(self):
        """Print summary of results"""
        results = self.get_results()
        if results is None:
            print("No results available")
            return
        
        print("\n" + "="*60)
        print("TEP RESULTS SUMMARY")
        print("="*60)
        print(f"Total System Cost: ${results['objective_value']:,.2f}")
        print(f"  Investment Cost: ${results['investment_cost']:,.2f}")
        print(f"  Operating Cost: ${results['operating_cost']:,.2f}")
        print(f"\nLines Built: {len(results['lines_built'])}")
        for line in results['lines_built']:
            print(f"  Bus {line['from_bus']} -> Bus {line['to_bus']}: "
                  f"{line['capacity']:.0f} MW, Cost: ${line['cost']:,.0f}, "
                  f"Flow: {line['flow']:.2f} MW")
        print(f"\nCongested Branches: {len(results['congested_branches'])}")
        print("="*60)

