"""
DC Optimal Power Flow (DC-OPF) Model
====================================

This module implements a linearized DC Optimal Power Flow model for power system
dispatch optimization. The DC-OPF is a fundamental building block for more complex
models like Transmission Expansion Planning (TEP).

Mathematical Formulation
------------------------
The DC-OPF minimizes total generation cost subject to:

    min  Σ_g c_g × p_g                               (Objective)
    
    s.t. Σ_g∈G_b p_g + Σ_l f_l^in - Σ_l f_l^out = d_b   ∀b    (Power Balance)
         f_l = B_l × (θ_i - θ_j)                     ∀l    (DC Power Flow)
         -f̄_l ≤ f_l ≤ f̄_l                           ∀l    (Thermal Limits)
         p_g^min ≤ p_g ≤ p_g^max                     ∀g    (Generator Limits)
         θ_ref = 0                                         (Reference Bus)

Where:
    - p_g: Generation output of unit g (MW)
    - f_l: Power flow on line l (MW)
    - θ_b: Voltage angle at bus b (radians)
    - c_g: Marginal cost of generator g ($/MWh)
    - d_b: Net load at bus b (MW)
    - B_l: Line susceptance (1/reactance)
    - f̄_l: Thermal rating of line l (MW)

Key Assumptions (DC Approximation)
----------------------------------
1. Voltage magnitudes are 1.0 p.u. at all buses
2. Angle differences are small (sin(θ) ≈ θ)
3. Line resistance is negligible (R << X)
4. Reactive power flows are ignored

These assumptions yield a linear program that can be solved efficiently.

Usage Example
-------------
>>> from src.core.data_loader import RTSDataLoader
>>> from src.core.dc_opf import DCOPF
>>> 
>>> # Load system data
>>> data = RTSDataLoader('data/RTS_Data/SourceData')
>>> 
>>> # Create and solve DC-OPF
>>> opf = DCOPF(data)
>>> opf.build_model()
>>> opf.solve(solver='gurobi')
>>> 
>>> # Get results
>>> results = opf.get_results()
>>> print(f"Total cost: ${results['objective_value']:,.2f}")

References
----------
- Stott, B., Jardim, J., & Alsac, O. (2009). DC power flow revisited.
  IEEE Transactions on Power Systems, 24(3), 1290-1300.
- Wood, A. J., & Wollenberg, B. F. (2012). Power generation, operation,
  and control (3rd ed.). Wiley.

Author: CME307 Team (Edouard Rabasse, Siddhant Sukhani)
Date: December 2025
"""

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint,
    NonNegativeReals, Reals, minimize, value
)
from pyomo.opt import SolverFactory, TerminationCondition
import pandas as pd
import numpy as np
from .data_loader import RTSDataLoader


class DCOPF:
    """
    DC Optimal Power Flow Model.
    
    This class implements a standard DC-OPF formulation using Pyomo for
    algebraic modeling and Gurobi/GLPK for optimization.
    
    Attributes
    ----------
    data : RTSDataLoader
        Data loader containing bus, branch, and generator information
    model : pyomo.ConcreteModel
        The Pyomo optimization model (None until build_model() is called)
    results : pyomo.SolverResults
        Solver results (None until solve() is called)
    
    Methods
    -------
    build_model()
        Construct the Pyomo DC-OPF model
    solve(solver='gurobi')
        Solve the optimization problem
    get_results()
        Extract and return solution values
    print_summary()
        Print a formatted summary of results
    """
    
    def __init__(self, data_loader: RTSDataLoader):
        """
        Initialize the DC-OPF model.
        
        Parameters
        ----------
        data_loader : RTSDataLoader
            Instance of the RTS-GMLC data loader with system data
        """
        self.data = data_loader
        self.model = None
        self.results = None
        
    def build_model(self) -> None:
        """
        Build the Pyomo DC-OPF optimization model.
        
        This method constructs the complete DC-OPF formulation including:
        - Index sets (buses, generators, branches)
        - Parameters (costs, limits, susceptances)
        - Decision variables (generation, angles, flows)
        - Objective function (minimize generation cost)
        - Constraints (power balance, DC flow, limits)
        
        The model is stored in self.model and can be solved using solve().
        """
        self.model = ConcreteModel(name="DC-OPF")
        
        # =====================================================================
        # LOAD DATA
        # =====================================================================
        buses = self.data.get_bus_data()
        branches = self.data.get_branch_data()
        generators = self.data.get_generator_data()
        load = self.data.get_load_by_bus()
        gen_by_bus = self.data.get_generators_by_bus()
        
        # =====================================================================
        # INDEX SETS
        # Define the fundamental sets that index our decision variables
        # =====================================================================
        bus_ids = list(buses.index)
        gen_ids = list(generators['GEN UID'])
        branch_ids = list(branches['UID'])
        
        self.model.buses = Set(initialize=bus_ids, doc="Set of system buses")
        self.model.generators = Set(initialize=gen_ids, doc="Set of generators")
        self.model.branches = Set(initialize=branch_ids, doc="Set of transmission lines")
        
        # =====================================================================
        # GENERATOR PARAMETERS
        # Build mappings from generator ID to bus, capacity, and cost
        # =====================================================================
        gen_to_bus = {}      # Maps generator -> bus location
        gen_pmax = {}        # Maximum generation capacity (MW)
        gen_pmin = {}        # Minimum generation (MW) - for baseload units
        gen_cost = {}        # Marginal cost ($/MWh)
        
        for _, gen in generators.iterrows():
            gen_id = gen['GEN UID']
            bus_id = gen['Bus ID']
            
            gen_to_bus[gen_id] = bus_id
            gen_pmax[gen_id] = gen['PMax MW']
            gen_pmin[gen_id] = gen['PMin MW']
            
            # Calculate marginal cost from fuel price and heat rate
            # Cost ($/MWh) = Fuel Price ($/MMBTU) × Heat Rate (BTU/kWh) / 1000
            fuel_price = gen['Fuel Price $/MMBTU']
            avg_hr = gen.get('HR_avg_0', 10000)  # Default heat rate if missing
            if pd.isna(avg_hr):
                avg_hr = 10000  # 10,000 BTU/kWh is typical for thermal plants
            gen_cost[gen_id] = fuel_price * avg_hr / 1000.0
        
        # Register generator parameters with the model
        self.model.gen_to_bus = Param(
            self.model.generators, initialize=gen_to_bus,
            doc="Bus location for each generator"
        )
        self.model.gen_pmax = Param(
            self.model.generators, initialize=gen_pmax,
            doc="Maximum generation capacity (MW)"
        )
        self.model.gen_pmin = Param(
            self.model.generators, initialize=gen_pmin,
            doc="Minimum generation output (MW)"
        )
        self.model.gen_cost = Param(
            self.model.generators, initialize=gen_cost,
            doc="Marginal generation cost ($/MWh)"
        )
        
        # =====================================================================
        # BRANCH PARAMETERS
        # Network topology and electrical characteristics
        # =====================================================================
        branch_params = self.data.get_branch_parameters()
        branch_susceptance = {}  # B = 1/X (p.u.)
        branch_rating = {}       # Thermal limit (MW)
        branch_from = {}         # From bus
        branch_to = {}           # To bus
        
        for branch_id in branch_ids:
            params = branch_params[branch_id]
            branch_susceptance[branch_id] = params['susceptance']
            branch_rating[branch_id] = params['rating']
            branch_from[branch_id] = params['from_bus']
            branch_to[branch_id] = params['to_bus']
        
        self.model.branch_susceptance = Param(
            self.model.branches, initialize=branch_susceptance,
            doc="Line susceptance B = 1/X (p.u.)"
        )
        self.model.branch_rating = Param(
            self.model.branches, initialize=branch_rating,
            doc="Line thermal rating (MW)"
        )
        self.model.branch_from = Param(
            self.model.branches, initialize=branch_from,
            doc="From bus for each branch"
        )
        self.model.branch_to = Param(
            self.model.branches, initialize=branch_to,
            doc="To bus for each branch"
        )
        
        # =====================================================================
        # LOAD PARAMETER
        # Note: 'bus_load' used instead of 'load' to avoid Pyomo reserved name
        # =====================================================================
        self.model.bus_load = Param(
            self.model.buses, initialize=load, default=0.0,
            doc="Active power load at each bus (MW)"
        )
        
        # =====================================================================
        # DECISION VARIABLES
        # =====================================================================
        
        # Generator output (MW) - non-negative, bounded by gen_pmin/pmax
        self.model.p_gen = Var(
            self.model.generators, domain=NonNegativeReals,
            doc="Generator active power output (MW)"
        )
        
        # Voltage angles (radians) - unbounded real numbers
        self.model.theta = Var(
            self.model.buses, domain=Reals,
            doc="Bus voltage angle (radians)"
        )
        
        # Line power flows (MW) - can be positive or negative
        self.model.p_flow = Var(
            self.model.branches, domain=Reals,
            doc="Active power flow on transmission line (MW)"
        )
        
        # =====================================================================
        # OBJECTIVE FUNCTION
        # Minimize total generation cost
        # =====================================================================
        self.model.obj = Objective(
            expr=sum(
                self.model.gen_cost[g] * self.model.p_gen[g] 
                for g in self.model.generators
            ),
            sense=minimize,
            doc="Minimize total generation cost"
        )
        
        # =====================================================================
        # CONSTRAINTS
        # =====================================================================
        
        # -----------------------------------------------------------------
        # Power Balance at Each Bus (Kirchhoff's Current Law)
        # Generation + Flow_in - Flow_out = Load
        # -----------------------------------------------------------------
        def power_balance_rule(m, b):
            """
            Enforce power balance at bus b.
            
            Sum of generation at the bus plus net power injection from
            transmission lines must equal the load at the bus.
            """
            # Generators located at this bus
            gen_at_bus = [g for g in m.generators if m.gen_to_bus[g] == b]
            
            # Power flowing INTO this bus (lines where this bus is the 'to' end)
            flow_in = sum(m.p_flow[br] for br in m.branches if m.branch_to[br] == b)
            
            # Power flowing OUT of this bus (lines where this bus is the 'from' end)
            flow_out = sum(m.p_flow[br] for br in m.branches if m.branch_from[br] == b)
            
            return sum(m.p_gen[g] for g in gen_at_bus) + flow_in - flow_out == m.bus_load[b]
        
        self.model.power_balance = Constraint(
            self.model.buses, rule=power_balance_rule,
            doc="Power balance at each bus"
        )
        
        # -----------------------------------------------------------------
        # DC Power Flow Equation
        # f_l = B_l × (θ_from - θ_to)
        # -----------------------------------------------------------------
        def dc_flow_rule(m, br):
            """
            DC power flow approximation.
            
            Line flow is proportional to the angle difference across the line,
            scaled by the line susceptance.
            """
            return m.p_flow[br] == m.branch_susceptance[br] * (
                m.theta[m.branch_from[br]] - m.theta[m.branch_to[br]]
            )
        
        self.model.dc_flow = Constraint(
            self.model.branches, rule=dc_flow_rule,
            doc="DC power flow equation"
        )
        
        # -----------------------------------------------------------------
        # Generator Output Limits
        # p_min ≤ p_gen ≤ p_max
        # -----------------------------------------------------------------
        self.model.gen_max = Constraint(
            self.model.generators,
            rule=lambda m, g: m.p_gen[g] <= m.gen_pmax[g],
            doc="Generator maximum output constraint"
        )
        
        self.model.gen_min = Constraint(
            self.model.generators,
            rule=lambda m, g: m.p_gen[g] >= m.gen_pmin[g],
            doc="Generator minimum output constraint"
        )
        
        # -----------------------------------------------------------------
        # Transmission Line Thermal Limits
        # -f_max ≤ f ≤ f_max (bidirectional flow limits)
        # -----------------------------------------------------------------
        self.model.branch_max = Constraint(
            self.model.branches,
            rule=lambda m, br: m.p_flow[br] <= m.branch_rating[br],
            doc="Branch maximum flow limit"
        )
        
        self.model.branch_min = Constraint(
            self.model.branches,
            rule=lambda m, br: m.p_flow[br] >= -m.branch_rating[br],
            doc="Branch minimum flow limit (reverse direction)"
        )
        
        # -----------------------------------------------------------------
        # Reference Bus Angle
        # Fix one bus angle to 0 to remove rotational degree of freedom
        # -----------------------------------------------------------------
        ref_bus = min(bus_ids)  # Use lowest-numbered bus as reference
        self.model.ref_bus = Constraint(
            expr=self.model.theta[ref_bus] == 0,
            doc="Reference bus angle constraint"
        )
        
    def solve(self, solver: str = 'gurobi') -> bool:
        """
        Solve the DC-OPF optimization problem.
        
        Parameters
        ----------
        solver : str, optional
            Solver to use ('gurobi' or 'glpk'). Default is 'gurobi'.
        
        Returns
        -------
        bool
            True if optimal solution found, False otherwise
        
        Notes
        -----
        Gurobi is preferred for its speed and robustness. GLPK can be used
        as a free alternative but may be slower for larger problems.
        """
        if self.model is None:
            self.build_model()
        
        # Select and configure solver
        if solver == 'gurobi':
            opt = SolverFactory('gurobi')
        else:
            opt = SolverFactory('glpk')
        
        print(f"Solving DC-OPF with {solver}...")
        results = opt.solve(self.model, tee=False)
        
        # Check solution status
        if results.solver.termination_condition == TerminationCondition.optimal:
            print("✓ Optimal solution found!")
            self.results = results
            return True
        else:
            print(f"✗ Solver termination: {results.solver.termination_condition}")
            return False
    
    def get_results(self) -> dict:
        """
        Extract results from the solved model.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'objective_value': Total generation cost ($)
            - 'generation': Dict mapping generator ID to output (MW)
            - 'flows': Dict mapping branch ID to flow (MW)
            - 'angles': Dict mapping bus ID to angle (radians)
            - 'total_generation': Sum of all generation (MW)
            - 'total_load': Sum of all load (MW)
            - 'congested_branches': List of branches near thermal limit
        
        Raises
        ------
        Returns None if model hasn't been solved
        """
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
        
        # Extract generation dispatch
        for g in self.model.generators:
            p_val = value(self.model.p_gen[g])
            results_dict['generation'][g] = p_val
            results_dict['total_generation'] += p_val
        
        # Extract branch flows and identify congestion
        for br in self.model.branches:
            flow_val = value(self.model.p_flow[br])
            results_dict['flows'][br] = flow_val
            rating = value(self.model.branch_rating[br])
            
            # Flag lines at ≥99% of thermal capacity as congested
            if abs(flow_val) >= 0.99 * rating:
                results_dict['congested_branches'].append({
                    'branch': br,
                    'flow': flow_val,
                    'rating': rating,
                    'utilization': abs(flow_val) / rating
                })
        
        # Extract voltage angles
        for b in self.model.buses:
            results_dict['angles'][b] = value(self.model.theta[b])
        
        return results_dict
    
    def print_summary(self) -> None:
        """
        Print a formatted summary of the DC-OPF solution.
        
        Displays key metrics including total cost, generation, load balance,
        and congested transmission lines.
        """
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
