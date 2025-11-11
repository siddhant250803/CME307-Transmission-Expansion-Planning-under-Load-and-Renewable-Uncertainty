"""
Multi-period Transmission Expansion Planning model with time series data
"""
from pyomo.environ import *
import pandas as pd
import numpy as np
from data_loader import RTSDataLoader
from timeseries_loader import TimeseriesLoader
from tep import TEP

class MultiPeriodTEP(TEP):
    """Multi-period TEP model using time series data"""
    
    def __init__(self, data_loader, timeseries_loader, candidate_lines=None, 
                 line_cost_per_mw=1000000, periods=None, representative_method='peak_avg_low'):
        """
        Initialize multi-period TEP model
        
        Parameters:
        -----------
        data_loader : RTSDataLoader
        timeseries_loader : TimeseriesLoader
        candidate_lines : list, optional
        line_cost_per_mw : float
        periods : list of int, optional - specific periods to use
        representative_method : str - method to select periods if not provided
        """
        super().__init__(data_loader, candidate_lines, line_cost_per_mw)
        self.timeseries = timeseries_loader
        self.periods = periods
        self.representative_method = representative_method
        self.participation_factors = None
        
    def prepare_time_series(self, n_periods=24):
        """Prepare time series data"""
        # Load time series
        self.timeseries.load_regional_load()
        self.timeseries.load_wind_generation()
        self.timeseries.load_pv_generation()
        self.timeseries.load_hydro_generation()
        
        # Select representative periods
        if self.periods is None:
            self.periods = self.timeseries.select_representative_periods(
                n_periods=n_periods, 
                method=self.representative_method
            )
        
        # Calculate load participation factors
        buses = self.data.get_bus_data()
        self.participation_factors = self.timeseries.get_bus_load_participation_factors(buses)
        
        print(f"Selected {len(self.periods)} representative periods: {self.periods[:5]}...")
    
    def build_model(self):
        """Build multi-period TEP MILP model"""
        if self.periods is None:
            self.prepare_time_series()
        
        self.model = ConcreteModel()
        
        # Get data
        buses = self.data.get_bus_data()
        branches = self.data.get_branch_data()
        generators = self.data.get_generator_data()
        gen_by_bus = self.data.get_generators_by_bus()
        
        # Sets
        bus_ids = list(buses.index)
        gen_ids = list(generators['GEN UID'])
        branch_ids = list(branches['UID'])
        period_ids = self.periods
        
        # Generate candidates if not provided
        if self.candidate_lines is None:
            self.generate_candidate_lines()
        candidate_ids = list(range(len(self.candidate_lines)))
        
        self.model.buses = Set(initialize=bus_ids)
        self.model.generators = Set(initialize=gen_ids)
        self.model.branches = Set(initialize=branch_ids)
        self.model.candidates = Set(initialize=candidate_ids)
        self.model.periods = Set(initialize=period_ids)
        
        # Generator parameters
        gen_to_bus = {}
        gen_pmax = {}
        gen_pmin = {}
        gen_cost = {}
        gen_type = {}  # For renewable generators
        
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
            
            # Identify renewable generators
            unit_type = gen.get('Unit Type', '')
            gen_type[gen_id] = unit_type
        
        self.model.gen_to_bus = Param(self.model.generators, initialize=gen_to_bus)
        self.model.gen_pmax = Param(self.model.generators, initialize=gen_pmax)
        self.model.gen_pmin = Param(self.model.generators, initialize=gen_pmin)
        self.model.gen_cost = Param(self.model.generators, initialize=gen_cost)
        self.model.gen_type = Param(self.model.generators, initialize=gen_type)
        
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
        
        # Time-varying parameters
        # Load by bus and period
        bus_load_param = {}
        for period in period_ids:
            nodal_loads = self.timeseries.get_nodal_loads(
                period, buses, self.participation_factors
            )
            for bus_id in bus_ids:
                bus_load_param[(bus_id, period)] = nodal_loads.get(bus_id, 0.0)
        
        self.model.bus_load = Param(self.model.buses, self.model.periods, initialize=bus_load_param, default=0.0)
        
        # Renewable generation by period
        # Match generators to time series data
        renewable_max = {}
        
        # Initialize all generators to their PMax (default, can be overridden by renewables)
        for gen_id in gen_ids:
            for period in period_ids:
                renewable_max[(gen_id, period)] = None  # None means use PMax
        
        # Match wind generators
        if self.timeseries.wind_data:
            for period in period_ids:
                if period in self.timeseries.wind_data:
                    for wind_gen_name, wind_power in self.timeseries.wind_data[period].items():
                        # Find matching generator (wind generators have names like "309_WIND_1")
                        matching_gens = [g for g in gen_ids if wind_gen_name in g or g in wind_gen_name]
                        for gen_id in matching_gens:
                            renewable_max[(gen_id, period)] = min(wind_power, gen_pmax[gen_id])
        
        # Match PV generators
        if self.timeseries.pv_data:
            for period in period_ids:
                if period in self.timeseries.pv_data:
                    for pv_gen_name, pv_power in self.timeseries.pv_data[period].items():
                        matching_gens = [g for g in gen_ids if pv_gen_name in g or g in pv_gen_name]
                        for gen_id in matching_gens:
                            renewable_max[(gen_id, period)] = min(pv_power, gen_pmax[gen_id])
        
        # Match hydro generators
        if self.timeseries.hydro_data:
            for period in period_ids:
                if period in self.timeseries.hydro_data:
                    for hydro_gen_name, hydro_power in self.timeseries.hydro_data[period].items():
                        matching_gens = [g for g in gen_ids if hydro_gen_name in g or g in hydro_gen_name]
                        for gen_id in matching_gens:
                            renewable_max[(gen_id, period)] = min(hydro_power, gen_pmax[gen_id])
        
        # Convert None to PMax for non-renewable generators
        renewable_max_final = {}
        for gen_id in gen_ids:
            for period in period_ids:
                if (gen_id, period) in renewable_max and renewable_max[(gen_id, period)] is not None:
                    renewable_max_final[(gen_id, period)] = renewable_max[(gen_id, period)]
                else:
                    # Use PMax for non-renewable or unmatched generators
                    renewable_max_final[(gen_id, period)] = gen_pmax[gen_id]
        
        self.model.renewable_max = Param(self.model.generators, self.model.periods, 
                                        initialize=renewable_max_final)
        
        # Variables
        # Investment decisions (same for all periods)
        self.model.build_line = Var(self.model.candidates, domain=Binary)
        
        # Operational variables (indexed by period)
        self.model.p_gen = Var(self.model.generators, self.model.periods, domain=NonNegativeReals)
        self.model.theta = Var(self.model.buses, self.model.periods, domain=Reals)
        self.model.p_flow = Var(self.model.branches, self.model.periods, domain=Reals)
        self.model.p_flow_candidate = Var(self.model.candidates, self.model.periods, domain=Reals)
        
        # Objective: investment cost (one-time) + sum of operating costs over all periods
        investment_cost = sum(self.model.candidate_cost[c] * self.model.build_line[c] 
                             for c in self.model.candidates)
        
        # Operating cost weighted by period duration (assume 1 hour per period)
        operating_cost = sum(
            sum(self.model.gen_cost[g] * self.model.p_gen[g, t] for g in self.model.generators)
            for t in self.model.periods
        )
        
        self.model.obj = Objective(expr=investment_cost + operating_cost, sense=minimize)
        
        # Power balance for each bus and period
        def power_balance_rule(m, b, t):
            gen_at_bus = [g for g in m.generators if m.gen_to_bus[g] == b]
            flow_in_existing = sum(m.p_flow[br, t] for br in m.branches if m.branch_to[br] == b)
            flow_out_existing = sum(m.p_flow[br, t] for br in m.branches if m.branch_from[br] == b)
            flow_in_candidate = sum(m.p_flow_candidate[c, t] for c in m.candidates if m.candidate_to[c] == b)
            flow_out_candidate = sum(m.p_flow_candidate[c, t] for c in m.candidates if m.candidate_from[c] == b)
            
            return (sum(m.p_gen[g, t] for g in gen_at_bus) + 
                   flow_in_existing - flow_out_existing +
                   flow_in_candidate - flow_out_candidate == m.bus_load[b, t])
        
        self.model.power_balance = Constraint(self.model.buses, self.model.periods, rule=power_balance_rule)
        
        # DC flow for existing branches (each period)
        def dc_flow_rule(m, br, t):
            return m.p_flow[br, t] == m.branch_susceptance[br] * (m.theta[m.branch_from[br], t] - m.theta[m.branch_to[br], t])
        
        self.model.dc_flow = Constraint(self.model.branches, self.model.periods, rule=dc_flow_rule)
        
        # DC flow for candidate lines (Big-M formulation, each period)
        M = 10000
        
        def dc_flow_candidate_upper(m, c, t):
            from_bus = m.candidate_from[c]
            to_bus = m.candidate_to[c]
            susceptance = m.candidate_susceptance[c]
            delta_theta = m.theta[from_bus, t] - m.theta[to_bus, t]
            return m.p_flow_candidate[c, t] - susceptance * delta_theta <= M * (1 - m.build_line[c])
        
        def dc_flow_candidate_lower(m, c, t):
            from_bus = m.candidate_from[c]
            to_bus = m.candidate_to[c]
            susceptance = m.candidate_susceptance[c]
            delta_theta = m.theta[from_bus, t] - m.theta[to_bus, t]
            return m.p_flow_candidate[c, t] - susceptance * delta_theta >= -M * (1 - m.build_line[c])
        
        def dc_flow_candidate_zero(m, c, t):
            return m.p_flow_candidate[c, t] <= M * m.build_line[c]
        
        def dc_flow_candidate_zero_lower(m, c, t):
            return m.p_flow_candidate[c, t] >= -M * m.build_line[c]
        
        self.model.dc_flow_candidate_upper = Constraint(self.model.candidates, self.model.periods, rule=dc_flow_candidate_upper)
        self.model.dc_flow_candidate_lower = Constraint(self.model.candidates, self.model.periods, rule=dc_flow_candidate_lower)
        self.model.dc_flow_candidate_zero = Constraint(self.model.candidates, self.model.periods, rule=dc_flow_candidate_zero)
        self.model.dc_flow_candidate_zero_lower = Constraint(self.model.candidates, self.model.periods, rule=dc_flow_candidate_zero_lower)
        
        # Generator limits (each period)
        def gen_max_rule(m, g, t):
            # Limit by PMax and renewable availability
            return m.p_gen[g, t] <= min(m.gen_pmax[g], m.renewable_max[g, t])
        
        self.model.gen_max = Constraint(self.model.generators, self.model.periods, rule=gen_max_rule)
        self.model.gen_min = Constraint(
            self.model.generators, self.model.periods,
            rule=lambda m, g, t: m.p_gen[g, t] >= m.gen_pmin[g]
        )
        
        # Branch flow limits (each period)
        self.model.branch_max = Constraint(
            self.model.branches, self.model.periods,
            rule=lambda m, br, t: m.p_flow[br, t] <= m.branch_rating[br]
        )
        self.model.branch_min = Constraint(
            self.model.branches, self.model.periods,
            rule=lambda m, br, t: m.p_flow[br, t] >= -m.branch_rating[br]
        )
        
        # Candidate line flow limits (each period)
        def candidate_flow_max_rule(m, c, t):
            return m.p_flow_candidate[c, t] <= m.candidate_capacity[c] * m.build_line[c]
        
        def candidate_flow_min_rule(m, c, t):
            return m.p_flow_candidate[c, t] >= -m.candidate_capacity[c] * m.build_line[c]
        
        self.model.candidate_flow_max = Constraint(self.model.candidates, self.model.periods, rule=candidate_flow_max_rule)
        self.model.candidate_flow_min = Constraint(self.model.candidates, self.model.periods, rule=candidate_flow_min_rule)
        
        # Reference bus (each period)
        ref_bus = min(bus_ids)
        self.model.ref_bus = Constraint(
            self.model.periods,
            rule=lambda m, t: m.theta[ref_bus, t] == 0
        )
    
    def solve(self, solver='gurobi', time_limit=1800):
        """Solve the multi-period TEP model"""
        if self.model is None:
            self.build_model()
        
        if solver == 'gurobi':
            opt = SolverFactory('gurobi')
            opt.options['TimeLimit'] = time_limit
            opt.options['MIPGap'] = 0.02  # 2% optimality gap for larger problem
        else:
            opt = SolverFactory('glpk')
        
        print(f"Solving Multi-Period TEP MILP with {solver}...")
        print(f"  Periods: {len(self.periods)}, Candidates: {len(self.candidate_lines)}")
        results = opt.solve(self.model, tee=True)
        
        if results.solver.termination_condition == TerminationCondition.optimal:
            print("Optimal solution found!")
            self.results = results
            return True
        elif results.solver.termination_condition == TerminationCondition.feasible:
            print("Feasible solution found (may not be optimal)")
            self.results = results
            return True
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            print("Model is infeasible!")
            print("Possible causes:")
            print("  - Selected periods have incompatible load/generation patterns")
            print("  - System cannot meet demand with available generation")
            print("  - Consider using run_simplified_tep.py with load shedding option")
            return False
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
            'operating_cost': sum(
                sum(value(self.model.gen_cost[g] * self.model.p_gen[g, t]) 
                    for g in self.model.generators)
                for t in self.model.periods
            ),
            'lines_built': [],
            'generation': {},
            'flows': {},
            'congested_branches': [],
            'period_results': {}
        }
        
        # Lines built
        for c in self.model.candidates:
            if value(self.model.build_line[c]) > 0.5:
                # Calculate average flow across periods
                avg_flow = sum(value(self.model.p_flow_candidate[c, t]) 
                              for t in self.model.periods) / len(self.model.periods)
                results_dict['lines_built'].append({
                    'candidate_id': c,
                    'from_bus': value(self.model.candidate_from[c]),
                    'to_bus': value(self.model.candidate_to[c]),
                    'capacity': value(self.model.candidate_capacity[c]),
                    'cost': value(self.model.candidate_cost[c]),
                    'avg_flow': avg_flow
                })
        
        # Generation by period
        for t in self.model.periods:
            period_gen = {}
            for g in self.model.generators:
                period_gen[g] = value(self.model.p_gen[g, t])
            results_dict['generation'][t] = period_gen
        
        # Branch flows and congestion by period
        for t in self.model.periods:
            period_flows = {}
            period_congested = []
            for br in self.model.branches:
                flow_val = value(self.model.p_flow[br, t])
                period_flows[br] = flow_val
                rating = value(self.model.branch_rating[br])
                if abs(flow_val) >= 0.99 * rating:
                    period_congested.append({
                        'branch': br,
                        'flow': flow_val,
                        'rating': rating,
                        'period': t
                    })
            
            results_dict['flows'][t] = period_flows
            results_dict['congested_branches'].extend(period_congested)
            
            # Period summary
            total_gen = sum(period_gen.values())
            total_load = sum(value(self.model.bus_load[b, t]) for b in self.model.buses)
            period_cost = sum(value(self.model.gen_cost[g] * self.model.p_gen[g, t]) 
                            for g in self.model.generators)
            
            results_dict['period_results'][t] = {
                'total_generation': total_gen,
                'total_load': total_load,
                'operating_cost': period_cost,
                'congested_branches': len(period_congested)
            }
        
        return results_dict
    
    def print_summary(self):
        """Print summary of results"""
        results = self.get_results()
        if results is None:
            print("No results available")
            return
        
        print("\n" + "="*60)
        print("MULTI-PERIOD TEP RESULTS SUMMARY")
        print("="*60)
        print(f"Total System Cost: ${results['objective_value']:,.2f}")
        print(f"  Investment Cost: ${results['investment_cost']:,.2f}")
        print(f"  Operating Cost (all periods): ${results['operating_cost']:,.2f}")
        print(f"  Average Operating Cost per Period: ${results['operating_cost']/len(self.periods):,.2f}")
        print(f"\nLines Built: {len(results['lines_built'])}")
        for line in results['lines_built']:
            print(f"  Bus {line['from_bus']} -> Bus {line['to_bus']}: "
                  f"{line['capacity']:.0f} MW, Cost: ${line['cost']:,.0f}, "
                  f"Avg Flow: {line['avg_flow']:.2f} MW")
        
        print(f"\nPeriod-by-Period Summary:")
        for period in sorted(results['period_results'].keys()):
            pr = results['period_results'][period]
            print(f"  Period {period}: Load={pr['total_load']:.1f} MW, "
                  f"Gen={pr['total_generation']:.1f} MW, "
                  f"Cost=${pr['operating_cost']:,.2f}, "
                  f"Congested={pr['congested_branches']} branches")
        
        print(f"\nTotal Congested Branches (across all periods): {len(results['congested_branches'])}")
        print("="*60)

