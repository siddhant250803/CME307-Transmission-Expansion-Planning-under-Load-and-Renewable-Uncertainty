"""
Transmission Expansion Planning with Load Shedding
==================================================

This module extends the base TEP model to allow load shedding (unserved energy)
as an explicit decision variable with penalty costs. This enables the model to:

1. Handle infeasible scenarios by allowing demand curtailment
2. Quantify the cost of supply shortages (value of lost load)
3. Compare expansion costs against load shedding penalties

Mathematical Formulation
------------------------
The model extends base TEP by adding load shedding variables s_b ≥ 0:

    min  Σ_c F_c × x_c + Σ_g c_g × p_g + γ × Σ_b s_b     (Investment + Operating + Shedding)
    
    s.t. [Power balance with load shedding]
         Σ_g∈G_b p_g + Σ_l f_l^in - Σ_l f_l^out 
         + Σ_c f_c^in - Σ_c f_c^out + s_b = d_b          ∀b    (Balance)
         
         [All other TEP constraints]
         s_b ≥ 0                                          ∀b    (Non-negative shedding)

Where:
    - s_b: Load shed at bus b (MW)
    - γ: Penalty cost per MW of unserved energy ($/MWh), typically $50,000/MWh (VOLL)

Usage Example
-------------
>>> from src.core.data_loader import RTSDataLoader
>>> from src.core.tep_with_shedding import TEPWithLoadShedding
>>>
>>> data = RTSDataLoader('data/RTS_Data/SourceData')
>>> custom_loads = {101: 500, 102: 300, ...}  # Stressed loads
>>> 
>>> # Create model with load shedding enabled
>>> tep = TEPWithLoadShedding(data, candidate_lines=[], 
>>>                           line_cost_per_mw=0,  # Disable building
>>>                           custom_loads=custom_loads,
>>>                           shedding_cost_per_mw=50000)  # $50k/MWh VOLL
>>> tep.build_model()
>>> tep.solve(solver='gurobi')
>>>
>>> results = tep.get_results()
>>> print(f"Total load shed: {results['total_load_shed']} MW")
>>> print(f"Shedding cost: ${results['load_shedding_cost']:,.0f}")

Author: CME307 Team (Edouard Rabasse, Siddhant Sukhani)
Date: December 2025
"""
from pyomo.environ import *
import pandas as pd
import numpy as np
from .data_loader import RTSDataLoader
from .tep import TEP

class TEPWithLoadShedding(TEP):
    """
    TEP model that allows load shedding with penalty cost.
    
    This class extends the base TEP model to include load shedding variables
    that allow the model to curtail demand when supply is insufficient. The
    penalty cost (typically set to value of lost load, VOLL) makes load
    shedding expensive but allows the model to remain feasible under extreme
    stress conditions.
    
    Attributes
    ----------
    custom_loads : dict
        Dictionary mapping bus_id -> load (MW) for custom load scenario
    shedding_cost : float
        Penalty cost per MW of unserved energy ($/MWh)
    """
    
    def __init__(self, data_loader, candidate_lines, line_cost_per_mw, custom_loads, 
                 shedding_cost_per_mw=10000):
        """
        Initialize TEP model with load shedding capability.
        
        Parameters
        ----------
        data_loader : RTSDataLoader
            Data loader instance
        candidate_lines : list
            List of candidate line dictionaries (can be empty to disable building)
        line_cost_per_mw : float
            Capital cost per MW for candidate lines ($/MW)
        custom_loads : dict
            Dictionary mapping bus_id -> load (MW) for stress scenario
        shedding_cost_per_mw : float, optional
            Penalty cost per MW of unserved energy ($/MWh), default is 10,000.
            Typical VOLL (value of lost load) is $50,000/MWh.
        """
        super().__init__(data_loader, candidate_lines, line_cost_per_mw)
        self.custom_loads = custom_loads
        self.shedding_cost = shedding_cost_per_mw
    
    def build_model(self):
        """
        Build TEP model with load shedding variables.
        
        This method:
        1. Temporarily replaces load data with custom_loads
        2. Builds base TEP model
        3. Adds load_shed variables for each bus
        4. Modifies power balance constraint to include load shedding
        5. Updates objective function to include shedding penalty
        """
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

