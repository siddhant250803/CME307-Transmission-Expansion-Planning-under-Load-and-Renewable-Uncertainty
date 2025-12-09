"""
Cost Sensitivity Analysis for Transmission Expansion Planning
============================================================

This script performs a cost sensitivity analysis by solving TEP models across
a range of capital cost parameters. The goal is to identify the break-even point
where transmission expansion becomes economically attractive.

Methodology:
1. Run baseline DC-OPF to establish reference operating cost
2. Sweep line capital costs from $100k/MW to $1M/MW (10 logarithmically-spaced points)
3. For each cost level, solve TEP and record:
   - Number of lines built
   - Operating cost
   - Investment cost
   - Net benefit (operating savings - investment cost)

Output:
- results/cost_sensitivity.csv: Detailed results for each cost level

Usage:
    python src/scripts/run_cost_sensitivity.py

Author: CME307 Team (Edouard Rabasse, Siddhant Sukhani)
Date: December 2025
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.data_loader import RTSDataLoader
from src.core.dc_opf import DCOPF
from src.core.tep import TEP

def run_cost_sensitivity():
    """
    Run cost sensitivity analysis.
    
    Sweeps line capital costs from $100k/MW to $1M/MW and solves TEP at each
    level to determine when expansion becomes economically justified.
    """
    print("="*60)
    print("CME307: TEP Cost Sensitivity Analysis")
    print("="*60)
    
    # Load data
    print("\nLoading RTS-GMLC data...")
    data_loader = RTSDataLoader(data_dir='data/RTS_Data/SourceData')
    
    # Run baseline
    print("\nRunning baseline DC OPF...")
    dcopf = DCOPF(data_loader)
    dcopf.build_model()
    baseline_success = dcopf.solve(solver='gurobi')
    
    if not baseline_success:
        print("Baseline failed!")
        return
    
    baseline_results = dcopf.get_results()
    baseline_cost = baseline_results['objective_value']
    print(f"Baseline Operating Cost: ${baseline_cost:,.2f}")
    
    # Define cost range to sweep
    # Realistic range: $100k-$1M per MW (based on typical transmission costs)
    cost_values = np.logspace(5, 6, num=10)  # $100k to $1M, 10 points on log scale
    # Or use linear: np.linspace(100000, 1000000, num=10)
    
    results = []
    
    print("\n" + "="*60)
    print("Running TEP for different cost parameters...")
    print("="*60)
    
    for cost_per_mw in cost_values:
        print(f"\n--- Testing line_cost_per_mw = ${cost_per_mw:,.0f}/MW ---")
        
        # Create TEP model
        tep = TEP(data_loader, line_cost_per_mw=cost_per_mw)
        tep.generate_candidate_lines(max_candidates=20)  # Limit for speed
        
        tep.build_model()
        success = tep.solve(solver='gurobi', time_limit=300)
        
        if success:
            tep_results = tep.get_results()
            investment_cost = tep_results['investment_cost']
            operating_cost = tep_results['operating_cost']
            total_cost = tep_results['objective_value']
            lines_built = len(tep_results['lines_built'])
            
            savings = baseline_cost - operating_cost
            net_benefit = savings - investment_cost
            
            results.append({
                'cost_per_mw': cost_per_mw,
                'investment_cost': investment_cost,
                'operating_cost': operating_cost,
                'total_cost': total_cost,
                'lines_built': lines_built,
                'operating_savings': savings,
                'net_benefit': net_benefit,
                'break_even': net_benefit >= 0
            })
            
            print(f"  Investment: ${investment_cost:,.2f}")
            print(f"  Operating: ${operating_cost:,.2f}")
            print(f"  Total: ${total_cost:,.2f}")
            print(f"  Lines Built: {lines_built}")
            print(f"  Net Benefit: ${net_benefit:,.2f}")
        else:
            print(f"  Failed to solve")
            results.append({
                'cost_per_mw': cost_per_mw,
                'investment_cost': None,
                'operating_cost': None,
                'total_cost': None,
                'lines_built': None,
                'operating_savings': None,
                'net_benefit': None,
                'break_even': None
            })
    
    # Summary table
    print("\n" + "="*60)
    print("COST SENSITIVITY SUMMARY")
    print("="*60)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Find break-even point
    valid_results = df[df['net_benefit'].notna()]
    if len(valid_results) > 0:
        break_even = valid_results[valid_results['net_benefit'] >= 0]
        if len(break_even) > 0:
            min_cost = break_even['cost_per_mw'].min()
            print(f"\nBreak-even point: Expansion becomes attractive at cost <= ${min_cost:,.0f}/MW")
        else:
            print("\nNo break-even point found in tested range")
    
    # Save results
    output_file = 'results/cost_sensitivity.csv'
    os.makedirs('results', exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    return df

if __name__ == '__main__':
    run_cost_sensitivity()

