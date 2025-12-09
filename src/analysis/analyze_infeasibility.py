"""
Analyze infeasibility in periods 1-6 using IIS (Irreducible Infeasible Set) analysis
and parametric analysis to find feasibility boundaries
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.data_loader import RTSDataLoader
from src.core.timeseries_loader import TimeseriesLoader
from src.core.dc_opf import DCOPF

class DCOPFWithLoads(DCOPF):
    """DC OPF that accepts custom loads"""
    def __init__(self, data_loader, custom_loads):
        super().__init__(data_loader)
        self.custom_loads = custom_loads
    
    def build_model(self):
        """Build model with custom loads"""
        # Temporarily replace load dict
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
        # Restore original
        self.data.get_load_by_bus = lambda: original_load

def analyze_iis(period, data_loader, timeseries_loader, results_dir):
    """Use Gurobi's IIS feature to identify conflicting constraints"""
    print(f"\n{'='*60}")
    print(f"IIS Analysis for Period {period}")
    print(f"{'='*60}")
    
    # Get loads for this period
    buses = data_loader.get_bus_data()
    participation = timeseries_loader.get_bus_load_participation_factors(buses)
    nodal_loads = timeseries_loader.get_nodal_loads(period, buses, participation)
    
    # Build model
    dcopf = DCOPFWithLoads(data_loader, nodal_loads)
    dcopf.build_model()
    
    # Try to solve - this should fail
    opt = SolverFactory('gurobi')
    opt.options['IISMethod'] = 1  # Enable IIS computation
    opt.options['LogFile'] = os.path.join(results_dir, f'iis_period_{period}.log')
    
    print(f"Attempting to solve period {period}...")
    results = opt.solve(dcopf.model, tee=False)
    
    if results.solver.termination_condition == TerminationCondition.infeasible:
        print(f"Model is infeasible. Computing IIS...")
        
        # Get IIS information
        # Note: Pyomo doesn't directly expose IIS, but we can check the log file
        # or use Gurobi's Python API directly
        
        # Alternative: Check constraint violations manually
        print("\nAnalyzing constraint violations...")
        
        # Check power balance
        total_load = sum(nodal_loads.values())
        generators = data_loader.get_generator_data()
        total_max_gen = generators['PMax MW'].sum()
        total_min_gen = generators['PMin MW'].sum()
        
        print(f"  Total Load: {total_load:.2f} MW")
        print(f"  Total Max Generation: {total_max_gen:.2f} MW")
        print(f"  Total Min Generation: {total_min_gen:.2f} MW")
        print(f"  Generation Margin: {total_max_gen - total_load:.2f} MW")
        print(f"  Min Gen Constraint: {total_min_gen:.2f} MW (may exceed load)")
        
        if total_min_gen > total_load:
            print(f"\n  ⚠️  PROBLEM: Minimum generation ({total_min_gen:.2f} MW) exceeds load ({total_load:.2f} MW)")
            print(f"     Required reduction: {total_min_gen - total_load:.2f} MW")
        
        # Check renewable availability
        renewable_max = {}
        if timeseries_loader.wind_data and period in timeseries_loader.wind_data:
            wind_total = sum(timeseries_loader.wind_data[period].values())
            renewable_max['wind'] = wind_total
            print(f"  Wind Generation Available: {wind_total:.2f} MW")
        else:
            renewable_max['wind'] = 0
            print(f"  Wind Generation Available: 0 MW")
        
        if timeseries_loader.pv_data and period in timeseries_loader.pv_data:
            pv_total = sum(timeseries_loader.pv_data[period].values())
            renewable_max['pv'] = pv_total
            print(f"  PV Generation Available: {pv_total:.2f} MW")
        else:
            renewable_max['pv'] = 0
            print(f"  PV Generation Available: 0 MW")
        
        if timeseries_loader.hydro_data and period in timeseries_loader.hydro_data:
            hydro_total = sum(timeseries_loader.hydro_data[period].values())
            renewable_max['hydro'] = hydro_total
            print(f"  Hydro Generation Available: {hydro_total:.2f} MW")
        else:
            renewable_max['hydro'] = 0
            print(f"  Hydro Generation Available: 0 MW")
        
        total_renewable = sum(renewable_max.values())
        print(f"  Total Renewable Available: {total_renewable:.2f} MW")
        
        # Thermal generation needed
        thermal_needed = total_load - total_renewable
        thermal_available = total_max_gen - total_renewable
        thermal_min = total_min_gen - total_renewable
        
        print(f"\n  Thermal Generation Needed: {thermal_needed:.2f} MW")
        print(f"  Thermal Generation Available: {thermal_available:.2f} MW")
        print(f"  Thermal Min Generation: {thermal_min:.2f} MW")
        
        if thermal_min > thermal_needed:
            print(f"\n  ⚠️  PROBLEM: Thermal min gen ({thermal_min:.2f} MW) > thermal needed ({thermal_needed:.2f} MW)")
        
        return {
            'period': period,
            'total_load': total_load,
            'total_max_gen': total_max_gen,
            'total_min_gen': total_min_gen,
            'renewable_available': total_renewable,
            'thermal_needed': thermal_needed,
            'thermal_min': thermal_min,
            'infeasible': True,
            'likely_cause': 'min_gen_exceeds_load' if total_min_gen > total_load else 'insufficient_generation'
        }
    else:
        print(f"  Period {period} is feasible!")
        return {'period': period, 'infeasible': False}
    
    return None

def parametric_load_analysis(period, data_loader, timeseries_loader, results_dir):
    """Sweep load reduction to find feasibility boundary"""
    print(f"\n{'='*60}")
    print(f"Parametric Load Analysis for Period {period}")
    print(f"{'='*60}")
    
    buses = data_loader.get_bus_data()
    participation = timeseries_loader.get_bus_load_participation_factors(buses)
    base_loads = timeseries_loader.get_nodal_loads(period, buses, participation)
    base_total_load = sum(base_loads.values())
    
    print(f"Base total load: {base_total_load:.2f} MW")
    print(f"Testing load reductions from 0% to 50%...")
    
    results = []
    load_factors = np.linspace(1.0, 0.5, num=21)  # 100% to 50% in 2.5% steps
    
    for factor in load_factors:
        scaled_loads = {bus: load * factor for bus, load in base_loads.items()}
        
        dcopf = DCOPFWithLoads(data_loader, scaled_loads)
        dcopf.build_model()
        
        opt = SolverFactory('gurobi')
        opt.options['LogFile'] = ''  # Suppress log
        results_solve = opt.solve(dcopf.model, tee=False)
        
        feasible = (results_solve.solver.termination_condition == TerminationCondition.optimal)
        reduction_pct = (1 - factor) * 100
        
        if feasible:
            obj_val = value(dcopf.model.obj) if feasible else None
            results.append({
                'load_factor': factor,
                'load_reduction_pct': reduction_pct,
                'total_load': sum(scaled_loads.values()),
                'feasible': True,
                'cost': obj_val
            })
            print(f"  {reduction_pct:5.1f}% reduction ({sum(scaled_loads.values()):7.1f} MW): FEASIBLE (Cost: ${obj_val:,.2f})")
        else:
            results.append({
                'load_factor': factor,
                'load_reduction_pct': reduction_pct,
                'total_load': sum(scaled_loads.values()),
                'feasible': False,
                'cost': None
            })
            print(f"  {reduction_pct:5.1f}% reduction ({sum(scaled_loads.values()):7.1f} MW): INFEASIBLE")
    
    # Find boundary
    feasible_results = [r for r in results if r['feasible']]
    if feasible_results:
        boundary = min(feasible_results, key=lambda x: x['load_reduction_pct'])
        print(f"\n  ✓ Feasibility boundary: {boundary['load_reduction_pct']:.1f}% load reduction")
        print(f"    Minimum feasible load: {boundary['total_load']:.2f} MW")
        print(f"    Load reduction needed: {base_total_load - boundary['total_load']:.2f} MW")
    else:
        print(f"\n  ✗ No feasible solution found even with 50% load reduction")
    
    return results

def parametric_generation_analysis(period, data_loader, timeseries_loader, results_dir):
    """Sweep minimum generation reduction to find feasibility boundary"""
    print(f"\n{'='*60}")
    print(f"Parametric Generation Analysis for Period {period}")
    print(f"{'='*60}")
    
    buses = data_loader.get_bus_data()
    participation = timeseries_loader.get_bus_load_participation_factors(buses)
    nodal_loads = timeseries_loader.get_nodal_loads(period, buses, participation)
    total_load = sum(nodal_loads.values())
    
    generators = data_loader.get_generator_data()
    base_pmin = generators['PMin MW'].copy()
    base_pmin_dict = dict(zip(generators['GEN UID'], base_pmin))
    
    print(f"Base total min generation: {base_pmin.sum():.2f} MW")
    print(f"Total load: {total_load:.2f} MW")
    print(f"Testing minimum generation reductions from 0% to 50%...")
    print(f"(Note: The issue is likely minimum generation > load, not maximum)")
    
    results = []
    gen_factors = np.linspace(1.0, 0.5, num=21)  # 100% to 50% in 2.5% steps
    
    for factor in gen_factors:
        # Build model with modified PMin
        dcopf = DCOPFWithLoads(data_loader, nodal_loads)
        
        # Modify PMin before building model
        original_get_gen_data = data_loader.get_generator_data
        modified_gens = generators.copy()
        modified_gens['PMin MW'] = base_pmin * factor
        
        class ModifiedDataLoader:
            def __init__(self, original, modified_gens):
                self.original = original
                self.modified_gens = modified_gens
            def get_generator_data(self):
                return self.modified_gens
            def __getattr__(self, name):
                return getattr(self.original, name)
        
        mod_loader = ModifiedDataLoader(data_loader, modified_gens)
        dcopf.data = mod_loader
        dcopf.build_model()
        
        opt = SolverFactory('gurobi')
        opt.options['LogFile'] = ''
        results_solve = opt.solve(dcopf.model, tee=False)
        
        feasible = (results_solve.solver.termination_condition == TerminationCondition.optimal)
        reduction_pct = (1 - factor) * 100
        total_min_gen = (base_pmin * factor).sum()
        
        if feasible:
            obj_val = value(dcopf.model.obj) if feasible else None
            results.append({
                'gen_factor': factor,
                'gen_reduction_pct': reduction_pct,
                'total_min_gen': total_min_gen,
                'feasible': True,
                'cost': obj_val
            })
            print(f"  {reduction_pct:5.1f}% reduction ({total_min_gen:7.1f} MW): FEASIBLE (Cost: ${obj_val:,.2f})")
        else:
            results.append({
                'gen_factor': factor,
                'gen_reduction_pct': reduction_pct,
                'total_min_gen': total_min_gen,
                'feasible': False,
                'cost': None
            })
            print(f"  {reduction_pct:5.1f}% reduction ({total_min_gen:7.1f} MW): INFEASIBLE")
    
    # Find boundary
    feasible_results = [r for r in results if r['feasible']]
    if feasible_results:
        boundary = min(feasible_results, key=lambda x: x['gen_reduction_pct'])
        print(f"\n  ✓ Feasibility boundary: {boundary['gen_reduction_pct']:.1f}% minimum generation reduction")
        print(f"    Maximum feasible min gen: {boundary['total_min_gen']:.2f} MW")
        print(f"    Min gen reduction needed: {base_pmin.sum() - boundary['total_min_gen']:.2f} MW")
    else:
        print(f"\n  ✗ No feasible solution found even with 50% minimum generation reduction")
    
    return results

def main():
    print("="*60)
    print("Infeasibility Analysis for Periods 1-6")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    data_loader = RTSDataLoader(data_dir='data/RTS_Data/SourceData')
    timeseries_loader = TimeseriesLoader(data_dir='data/RTS_Data')
    timeseries_loader.load_regional_load()
    timeseries_loader.load_wind_generation()
    timeseries_loader.load_pv_generation()
    timeseries_loader.load_hydro_generation()
    
    # Create results subdirectory
    results_dir = 'results/infeasibility_analysis'
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}/")
    
    # Analyze each infeasible period
    infeasible_periods = [1, 2, 3, 4, 5, 6]
    
    iis_results = []
    load_analysis_results = {}
    gen_analysis_results = {}
    
    for period in infeasible_periods:
        # IIS Analysis
        iis_result = analyze_iis(period, data_loader, timeseries_loader, results_dir)
        if iis_result:
            iis_results.append(iis_result)
        
        # Parametric Load Analysis
        load_results = parametric_load_analysis(period, data_loader, timeseries_loader, results_dir)
        load_analysis_results[period] = load_results
        
        # Parametric Generation Analysis
        gen_results = parametric_generation_analysis(period, data_loader, timeseries_loader, results_dir)
        gen_analysis_results[period] = gen_results
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if iis_results:
        print("\nIIS Analysis Summary:")
        df_iis = pd.DataFrame(iis_results)
        print(df_iis.to_string(index=False))
        df_iis.to_csv(os.path.join(results_dir, 'iis_analysis.csv'), index=False)
    
    # Find minimum load reduction needed
    print("\nMinimum Load Reduction Needed:")
    for period in infeasible_periods:
        if period in load_analysis_results:
            feasible = [r for r in load_analysis_results[period] if r['feasible']]
            if feasible:
                min_reduction = min(feasible, key=lambda x: x['load_reduction_pct'])
                print(f"  Period {period}: {min_reduction['load_reduction_pct']:.1f}% reduction ({min_reduction['total_load']:.1f} MW)")
    
    # Find minimum generation reduction needed
    print("\nMinimum Generation Reduction Needed:")
    for period in infeasible_periods:
        if period in gen_analysis_results:
            feasible = [r for r in gen_analysis_results[period] if r['feasible']]
            if feasible:
                min_reduction = min(feasible, key=lambda x: x['gen_reduction_pct'])
                print(f"  Period {period}: {min_reduction['gen_reduction_pct']:.1f}% reduction ({min_reduction['total_min_gen']:.1f} MW)")
    
    # Save detailed results
    for period in infeasible_periods:
        if period in load_analysis_results:
            df_load = pd.DataFrame(load_analysis_results[period])
            df_load.to_csv(os.path.join(results_dir, f'load_analysis_period_{period}.csv'), index=False)
        
        if period in gen_analysis_results:
            df_gen = pd.DataFrame(gen_analysis_results[period])
            df_gen.to_csv(os.path.join(results_dir, f'gen_analysis_period_{period}.csv'), index=False)
    
    print("\n" + "="*60)
    print(f"Results saved to {results_dir}/ directory")
    print("="*60)

if __name__ == '__main__':
    main()

