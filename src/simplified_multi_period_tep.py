"""
Simplified Multi-Period TEP using two-stage approach
Stage 1: Identify congestion periods
Stage 2: Solve TEP for critical periods only
"""
from pyomo.environ import *
import pandas as pd
import numpy as np
from data_loader import RTSDataLoader
from timeseries_loader import TimeseriesLoader
from tep import TEP
from dc_opf import DCOPF

class DCOPFWithLoads(DCOPF):
    """DC OPF that accepts custom loads"""
    def __init__(self, data_loader, custom_loads):
        super().__init__(data_loader)
        self.custom_loads = custom_loads
    
    def build_model(self):
        """Build model with custom loads"""
        # Temporarily replace load dict
        original_load = self.data.get_load_by_bus()
        # Create modified data loader
        class ModifiedLoader:
            def __init__(self, original, custom):
                self.original = original
                self.custom = custom
            def get_load_by_bus(self):
                return self.custom
        
        mod_loader = ModifiedLoader(self.data, self.custom_loads)
        self.data.get_load_by_bus = mod_loader.get_load_by_bus
        super().build_model()
        # Restore original
        self.data.get_load_by_bus = lambda: original_load

class TEPWithLoads(TEP):
    """TEP that accepts custom loads"""
    def __init__(self, data_loader, candidate_lines, line_cost_per_mw, custom_loads):
        super().__init__(data_loader, candidate_lines, line_cost_per_mw)
        self.custom_loads = custom_loads
    
    def build_model(self):
        """Build model with custom loads"""
        original_load = self.data.get_load_by_bus()
        class ModifiedLoader:
            def __init__(self, original, custom):
                self.original = original
                self.custom = custom
            def get_load_by_bus(self):
                return self.custom
        
        mod_loader = ModifiedLoader(self.data, self.custom_loads)
        self.data.get_load_by_bus = mod_loader.get_load_by_bus
        super().build_model()
        self.data.get_load_by_bus = lambda: original_load

class TEPWithLoadsAndOutages(TEP):
    """TEP with custom loads and simulated line outages"""
    def __init__(self, data_loader, candidate_lines, line_cost_per_mw, custom_loads, outage_fraction=0.15):
        super().__init__(data_loader, candidate_lines, line_cost_per_mw)
        self.custom_loads = custom_loads
        self.outage_fraction = outage_fraction
    
    def build_model(self):
        """Build model with custom loads and reduced line capacities"""
        original_load = self.data.get_load_by_bus()
        original_get_branch_params = self.data.get_branch_parameters
        
        # Modify branch parameters to simulate outages
        import random
        random.seed(42)
        branch_params = self.data.get_branch_parameters()
        branches_list = list(branch_params.keys())
        branches_to_derate = random.sample(branches_list, 
                                          int(len(branches_list) * self.outage_fraction))
        
        for br_id in branches_to_derate:
            branch_params[br_id]['rating'] *= 0.5  # Reduce by 50%
        
        print(f"Derated {len(branches_to_derate)} lines to simulate outages")
        
        # Temporarily modify get_branch_parameters
        def modified_get_branch_params():
            return branch_params
        
        self.data.get_branch_parameters = modified_get_branch_params
        
        class ModifiedLoader:
            def __init__(self, original, custom):
                self.original = original
                self.custom = custom
            def get_load_by_bus(self):
                return self.custom
        
        mod_loader = ModifiedLoader(self.data, self.custom_loads)
        self.data.get_load_by_bus = mod_loader.get_load_by_bus
        
        super().build_model()
        
        # Restore originals
        self.data.get_load_by_bus = lambda: original_load
        self.data.get_branch_parameters = original_get_branch_params

class TEPWithLoadsAndTargetedOutages(TEP):
    """TEP with custom loads and targeted line outages on congested lines"""
    def __init__(self, data_loader, candidate_lines, line_cost_per_mw, custom_loads, target_branches=None):
        super().__init__(data_loader, candidate_lines, line_cost_per_mw)
        self.custom_loads = custom_loads
        self.target_branches = target_branches or []
    
    def build_model(self):
        """Build model with custom loads and reduced capacities on target branches"""
        original_load = self.data.get_load_by_bus()
        original_get_branch_params = self.data.get_branch_parameters
        
        # Modify branch parameters - reduce target branches significantly
        branch_params = self.data.get_branch_parameters()
        
        for br_id in self.target_branches:
            if br_id in branch_params:
                branch_params[br_id]['rating'] *= 0.3  # Reduce to 30% capacity
                print(f"  Reduced {br_id} capacity to 30%")
        
        # Also reduce some inter-regional lines
        branches = self.data.get_branch_data()
        inter_regional = []
        for _, branch in branches.iterrows():
            from_bus = int(branch['From Bus'])
            to_bus = int(branch['To Bus'])
            area_from = self.data.get_bus_data().loc[from_bus, 'Area']
            area_to = self.data.get_bus_data().loc[to_bus, 'Area']
            if area_from != area_to and branch['UID'] not in self.target_branches:
                inter_regional.append(branch['UID'])
        
        # Reduce 5 additional inter-regional lines
        import random
        random.seed(42)
        additional = random.sample(inter_regional, min(5, len(inter_regional)))
        for br_id in additional:
            branch_params[br_id]['rating'] *= 0.5
            print(f"  Reduced {br_id} capacity to 50%")
        
        print(f"Total lines derated: {len(self.target_branches) + len(additional)}")
        
        # Temporarily modify get_branch_parameters
        def modified_get_branch_params():
            return branch_params
        
        self.data.get_branch_parameters = modified_get_branch_params
        
        class ModifiedLoader:
            def __init__(self, original, custom):
                self.original = original
                self.custom = custom
            def get_load_by_bus(self):
                return self.custom
        
        mod_loader = ModifiedLoader(self.data, self.custom_loads)
        self.data.get_load_by_bus = mod_loader.get_load_by_bus
        
        super().build_model()
        
        # Restore originals
        self.data.get_load_by_bus = lambda: original_load
        self.data.get_branch_parameters = original_get_branch_params

class SimplifiedMultiPeriodTEP:
    """Simplified approach: solve TEP for peak periods only"""
    
    def __init__(self, data_loader, timeseries_loader, line_cost_per_mw=500000):
        self.data = data_loader
        self.timeseries = timeseries_loader
        self.line_cost_per_mw = line_cost_per_mw
        self.candidate_lines = None
        self.peak_periods = None
        self.results = None
        
    def identify_congestion_periods(self, n_periods=3):
        """Identify periods with highest congestion"""
        print("Analyzing time series to identify peak congestion periods...")
        
        # Load time series
        self.timeseries.load_regional_load()
        buses = self.data.get_bus_data()
        participation = self.timeseries.get_bus_load_participation_factors(buses)
        
        # Run DC OPF for multiple periods to find congestion
        from dc_opf import DCOPF
        
        period_congestion = {}
        
        # Sample periods throughout the year
        sample_periods = list(range(1, min(25, 25)))  # First 24 hours
        
        for period in sample_periods[:12]:  # Check first 12 hours
            try:
                nodal_loads = self.timeseries.get_nodal_loads(period, buses, participation)
                
                # Create DC OPF with this period's load
                # Need to create a custom DC OPF that accepts loads
                dcopf = DCOPFWithLoads(self.data, nodal_loads)
                dcopf.build_model()
                
                if dcopf.solve(solver='gurobi'):
                    results = dcopf.get_results()
                    congestion_count = len(results['congested_branches'])
                    total_load = results['total_load']
                    period_congestion[period] = {
                        'congestion': congestion_count,
                        'load': total_load,
                        'cost': results['objective_value']
                    }
                    print(f"  Period {period}: Load={total_load:.1f} MW, Congested={congestion_count} branches")
            except Exception as e:
                print(f"  Period {period}: Error - {e}")
                continue
        
        # Select top N periods by congestion
        sorted_periods = sorted(period_congestion.items(), 
                               key=lambda x: (x[1]['congestion'], x[1]['load']), 
                               reverse=True)
        self.peak_periods = [p[0] for p in sorted_periods[:n_periods]]
        
        print(f"\nSelected peak periods: {self.peak_periods}")
        return self.peak_periods
    
    def solve_aggregated_tep(self, max_candidates=8):
        """Solve TEP for aggregated peak periods"""
        if self.peak_periods is None:
            self.identify_congestion_periods(n_periods=2)  # Use 2 peak periods
        
        print(f"\nSolving TEP for {len(self.peak_periods)} peak periods...")
        
        # Generate candidates
        if self.candidate_lines is None:
            self.generate_candidate_lines(max_candidates=max_candidates)
        
        # Create TEP model for peak load scenario (aggregate of peak periods)
        buses = self.data.get_bus_data()
        participation = self.timeseries.get_bus_load_participation_factors(buses)
        
        # Use maximum load across peak periods, with stress test multiplier
        # Scale up loads by 50% to create expansion need
        stress_multiplier = 1.50
        max_loads = {}
        for bus_id in buses.index:
            max_load = 0
            for period in self.peak_periods:
                nodal_loads = self.timeseries.get_nodal_loads(period, buses, participation)
                max_load = max(max_load, nodal_loads.get(bus_id, 0))
            max_loads[bus_id] = max_load * stress_multiplier
        
        # Add new large load center (simulating new industrial area)
        # Add 500 MW load at a remote bus (e.g., bus 122 in region 1)
        new_load_bus = 122
        if new_load_bus in buses.index:
            max_loads[new_load_bus] = max_loads.get(new_load_bus, 0) + 500
            print(f"Added 500 MW new load at bus {new_load_bus} (simulating new industrial area)")
        
        print(f"Using peak load scenario with {stress_multiplier*100:.0f}% stress multiplier")
        print(f"Total peak load: {sum(max_loads.values()):.1f} MW")
        
        # Create TEP with load shedding option - this will make expansion attractive
        # when load shedding cost > expansion cost
        from tep_with_shedding import TEPWithLoadShedding
        
        # Use targeted outages to create congestion
        branch_params = self.data.get_branch_parameters()
        target_branches = ['A27', 'CA-1', 'CB-1']
        for br_id in target_branches:
            if br_id in branch_params:
                branch_params[br_id]['rating'] *= 0.3
        
        # Temporarily modify branch params
        original_get_branch_params = self.data.get_branch_parameters
        def modified_get_branch_params():
            return branch_params
        self.data.get_branch_parameters = modified_get_branch_params
        
        # Create TEP with load shedding
        tep = TEPWithLoadShedding(
            self.data, 
            self.candidate_lines, 
            self.line_cost_per_mw, 
            max_loads,
            shedding_cost_per_mw=50000  # $50k/MW - very high penalty
        )
        tep.build_model()
        
        # Restore
        self.data.get_branch_parameters = original_get_branch_params
        
        print(f"Solving TEP with peak load scenario and load shedding option...")
        success = tep.solve(solver='gurobi', time_limit=600)
        
        if success:
            self.results = tep.get_results()
            self.results['peak_periods'] = self.peak_periods
            self.results['peak_loads'] = max_loads
            # Use TEP's print method
            tep.print_summary()
            return True
        
        return False
    
    def generate_candidate_lines(self, max_candidates=8):
        """Generate candidate lines (same as parent class)"""
        buses = self.data.get_bus_data()
        existing_branches = self.data.get_branch_data()
        
        existing_connections = set()
        for _, branch in existing_branches.iterrows():
            from_bus = int(branch['From Bus'])
            to_bus = int(branch['To Bus'])
            existing_connections.add((min(from_bus, to_bus), max(from_bus, to_bus)))
        
        candidates = []
        bus_ids = list(buses.index)
        
        for i, bus1 in enumerate(bus_ids):
            for bus2 in bus_ids[i+1:]:
                key = (min(bus1, bus2), max(bus1, bus2))
                if key in existing_connections:
                    continue
                
                area1 = buses.loc[bus1, 'Area']
                area2 = buses.loc[bus2, 'Area']
                if area1 == area2 or abs(area1 - area2) == 1:
                    base_kv = buses.loc[bus1, 'BaseKV']
                    if base_kv >= 230:
                        capacity = 500
                    else:
                        capacity = 200
                    
                    if 'lat' in buses.columns and 'lng' in buses.columns:
                        lat1, lng1 = buses.loc[bus1, 'lat'], buses.loc[bus1, 'lng']
                        lat2, lng2 = buses.loc[bus2, 'lat'], buses.loc[bus2, 'lng']
                        dist = np.sqrt((lat1-lat2)**2 + (lng1-lng2)**2) * 111
                    else:
                        dist = 50
                    
                    cost = self.line_cost_per_mw * capacity * (dist / 100.0)
                    
                    candidates.append({
                        'from_bus': bus1,
                        'to_bus': bus2,
                        'capacity': capacity,
                        'cost': cost,
                        'susceptance': 10.0
                    })
                    
                    if len(candidates) >= max_candidates:
                        break
            
            if len(candidates) >= max_candidates:
                break
        
        self.candidate_lines = candidates
        print(f"Generated {len(candidates)} candidate lines")
        return candidates
    
    def print_summary(self):
        """Print summary"""
        if self.results is None:
            print("No results available")
            return
        
        print("\n" + "="*60)
        print("SIMPLIFIED MULTI-PERIOD TEP RESULTS")
        print("="*60)
        print(f"Peak Periods Analyzed: {self.results['peak_periods']}")
        print(f"Total System Cost: ${self.results['objective_value']:,.2f}")
        print(f"  Investment Cost: ${self.results['investment_cost']:,.2f}")
        print(f"  Operating Cost: ${self.results['operating_cost']:,.2f}")
        if 'total_load_shed' in self.results:
            print(f"  Load Shedding Cost: ${self.results.get('load_shedding_cost', 0):,.2f}")
            print(f"  Total Load Shed: {self.results.get('total_load_shed', 0):.2f} MW")
        print(f"\nLines Built: {len(self.results['lines_built'])}")
        for line in self.results['lines_built']:
            print(f"  Bus {line['from_bus']} -> Bus {line['to_bus']}: "
                  f"{line['capacity']:.0f} MW, Cost: ${line['cost']:,.0f}")
        print("="*60)

