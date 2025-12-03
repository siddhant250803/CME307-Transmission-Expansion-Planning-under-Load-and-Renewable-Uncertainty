"""
TEP model with load shedding (unserved energy) option
This makes expansion more attractive by allowing comparison of expansion cost vs. load shedding cost
"""
from pyomo.environ import *
import pandas as pd
import numpy as np
from .data_loader import RTSDataLoader
from .tep import TEP

class TEPWithLoadShedding(TEP):
    """TEP model that allows load shedding with high penalty cost"""
    
    def __init__(self, data_loader, candidate_lines, line_cost_per_mw, custom_loads, 
                 shedding_cost_per_mw=10000):
        """
        Parameters:
        -----------
        shedding_cost_per_mw : float
            Cost per MW of unserved energy (very high to discourage, but allows feasibility)
        """
        super().__init__(data_loader, candidate_lines, line_cost_per_mw)
        self.custom_loads = custom_loads
        self.shedding_cost = shedding_cost_per_mw
    
    def build_model(self):
        """Build TEP model with load shedding option"""
        # Temporarily replace load
        original_load = self.data.get_load_by_bus()
        class ModifiedLoader:
            def __init__(self, original, custom):
                self.original = original
                self.custom = custom
            def get_load_by_bus(self):
                return self.custom
        
        mod_loader = ModifiedLoader(self.data, self.custom_loads)
        self.data.get_load_by_bus = mod_loader.get_load_by_bus
        
        # Build base model
        super().build_model()
        
        # Restore original
        self.data.get_load_by_bus = lambda: original_load
        
        # Add load shedding variables
        self.model.load_shed = Var(self.model.buses, domain=NonNegativeReals)
        
        # Modify power balance to allow load shedding
        def power_balance_with_shedding(m, b):
            gen_at_bus = [g for g in m.generators if m.gen_to_bus[g] == b]
            flow_in_existing = sum(m.p_flow[br] for br in m.branches if m.branch_to[br] == b)
            flow_out_existing = sum(m.p_flow[br] for br in m.branches if m.branch_from[br] == b)
            flow_in_candidate = sum(m.p_flow_candidate[c] for c in m.candidates if m.candidate_to[c] == b)
            flow_out_candidate = sum(m.p_flow_candidate[c] for c in m.candidates if m.candidate_from[c] == b)
            
            # Load can be partially or fully shed
            return (sum(m.p_gen[g] for g in gen_at_bus) + 
                   flow_in_existing - flow_out_existing +
                   flow_in_candidate - flow_out_candidate + m.load_shed[b] == m.bus_load[b])
        
        # Remove old power balance and add new one
        self.model.del_component(self.model.power_balance)
        self.model.power_balance = Constraint(self.model.buses, rule=power_balance_with_shedding)
        
        # Update objective to include load shedding penalty
        investment_cost = sum(self.model.candidate_cost[c] * self.model.build_line[c] 
                              for c in self.model.candidates)
        operating_cost = sum(self.model.gen_cost[g] * self.model.p_gen[g] 
                            for g in self.model.generators)
        shedding_penalty = sum(self.shedding_cost * self.model.load_shed[b] 
                               for b in self.model.buses)
        
        self.model.del_component(self.model.obj)
        self.model.obj = Objective(expr=investment_cost + operating_cost + shedding_penalty, 
                                   sense=minimize)
    
    def get_results(self):
        """Extract results including load shedding"""
        results = super().get_results()
        if results is None:
            return None
        
        # Add load shedding info
        total_shed = sum(value(self.model.load_shed[b]) for b in self.model.buses)
        results['total_load_shed'] = total_shed
        results['load_shedding_cost'] = total_shed * self.shedding_cost
        
        return results
    
    def print_summary(self):
        """Print summary including load shedding"""
        results = self.get_results()
        if results is None:
            print("No results available")
            return
        
        print("\n" + "="*60)
        print("TEP WITH LOAD SHEDDING RESULTS")
        print("="*60)
        print(f"Total System Cost: ${results['objective_value']:,.2f}")
        print(f"  Investment Cost: ${results['investment_cost']:,.2f}")
        print(f"  Operating Cost: ${results['operating_cost']:,.2f}")
        print(f"  Load Shedding Cost: ${results['load_shedding_cost']:,.2f}")
        print(f"  Total Load Shed: {results['total_load_shed']:.2f} MW")
        print(f"\nLines Built: {len(results['lines_built'])}")
        for line in results['lines_built']:
            print(f"  Bus {line['from_bus']} -> Bus {line['to_bus']}: "
                  f"{line['capacity']:.0f} MW, Cost: ${line['cost']:,.0f}, "
                  f"Flow: {line['flow']:.2f} MW")
        print("="*60)

