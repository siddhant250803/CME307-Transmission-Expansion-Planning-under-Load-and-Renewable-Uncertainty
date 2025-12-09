#!/usr/bin/env python3
"""
CME307 Transmission Expansion Planning - Unified Analysis Script
================================================================

This script provides a unified interface to run all TEP analyses:
- Baseline DC-OPF and single-period TEP
- Cost sensitivity analysis
- Multi-period TEP (simplified two-stage approach)
- Load shedding stress test
- Scenario-based robust TEP
- Visualization generation

Usage:
    python src/scripts/main.py --all              # Run all analyses
    python src/scripts/main.py --baseline         # Run baseline + TEP
    python src/scripts/main.py --cost-sensitivity # Run cost sweep
    python src/scripts/main.py --multi-period     # Run simplified multi-period
    python src/scripts/main.py --load-shedding    # Run load shedding analysis
    python src/scripts/main.py --robust           # Run robust TEP
    python src/scripts/main.py --visualize        # Generate all plots

Author: CME307 Team (Edouard Rabasse, Siddhant Sukhani)
Date: December 2025
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_baseline_and_tep() -> dict:
    """Run baseline DC-OPF and single-period TEP."""
    print_header("BASELINE DC-OPF & SINGLE-PERIOD TEP")
    
    from src.core.data_loader import RTSDataLoader
    from src.core.dc_opf import DCOPF
    from src.core.tep import TEP
    
    # Load data
    print("\nLoading RTS-GMLC data...")
    data_loader = RTSDataLoader(data_dir='data/RTS_Data/SourceData')
    
    # Run baseline DC-OPF
    print("\n--- Step 1: Baseline DC-OPF ---")
    dcopf = DCOPF(data_loader)
    dcopf.build_model()
    baseline_success = dcopf.solve(solver='gurobi')
    
    baseline_results = None
    if baseline_success:
        baseline_results = dcopf.get_results()
        dcopf.print_summary()
    else:
        print("Baseline DC-OPF failed!")
        return {'baseline': None, 'tep': None}
    
    # Run single-period TEP
    print("\n--- Step 2: Single-Period TEP ---")
    tep = TEP(data_loader, line_cost_per_mw=1_000_000)
    tep.generate_candidate_lines(max_candidates=30)
    tep.build_model()
    tep_success = tep.solve(solver='gurobi', time_limit=600)
    
    tep_results = None
    if tep_success:
        tep_results = tep.get_results()
        tep.print_summary()
    else:
        print("TEP failed!")
    
    return {'baseline': baseline_results, 'tep': tep_results, 'data_loader': data_loader}


def run_cost_sensitivity() -> None:
    """Run cost sensitivity analysis."""
    print_header("COST SENSITIVITY ANALYSIS")
    
    import numpy as np
    import pandas as pd
    from src.core.data_loader import RTSDataLoader
    from src.core.dc_opf import DCOPF
    from src.core.tep import TEP
    
    data_loader = RTSDataLoader(data_dir='data/RTS_Data/SourceData')
    
    # Baseline
    dcopf = DCOPF(data_loader)
    dcopf.build_model()
    dcopf.solve(solver='gurobi')
    baseline_cost = dcopf.get_results()['objective_value']
    print(f"Baseline Operating Cost: ${baseline_cost:,.2f}")
    
    # Cost sweep
    cost_values = np.logspace(5, 6, num=10)
    results = []
    
    print("\nSweeping line costs from $100k to $1M per MW...")
    for cost_per_mw in cost_values:
        tep = TEP(data_loader, line_cost_per_mw=cost_per_mw)
        tep.generate_candidate_lines(max_candidates=20)
        tep.build_model()
        
        if tep.solve(solver='gurobi', time_limit=300):
            r = tep.get_results()
            results.append({
                'cost_per_mw': cost_per_mw,
                'investment_cost': r['investment_cost'],
                'operating_cost': r['operating_cost'],
                'total_cost': r['objective_value'],
                'lines_built': len(r['lines_built']),
                'operating_savings': baseline_cost - r['operating_cost'],
                'net_benefit': baseline_cost - r['operating_cost'] - r['investment_cost'],
                'break_even': baseline_cost - r['operating_cost'] >= r['investment_cost']
            })
            print(f"  ${cost_per_mw:,.0f}/MW: {len(r['lines_built'])} lines built")
    
    # Save results
    df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/cost_sensitivity.csv', index=False)
    print(f"\nResults saved to results/cost_sensitivity.csv")


def run_multi_period() -> None:
    """Run simplified multi-period TEP."""
    print_header("SIMPLIFIED MULTI-PERIOD TEP")
    
    from src.core.data_loader import RTSDataLoader
    from src.core.timeseries_loader import TimeseriesLoader
    from src.core.simplified_multi_period_tep import SimplifiedMultiPeriodTEP
    
    data_loader = RTSDataLoader(data_dir='data/RTS_Data/SourceData')
    timeseries_loader = TimeseriesLoader(data_dir='data/RTS_Data')
    
    tep = SimplifiedMultiPeriodTEP(
        data_loader,
        timeseries_loader,
        line_cost_per_mw=200_000
    )
    
    peak_periods = tep.identify_congestion_periods(n_periods=2)
    success = tep.solve_aggregated_tep(max_candidates=8)
    
    if success:
        tep.print_summary()


def run_load_shedding() -> None:
    """Run load shedding stress test."""
    print_header("LOAD SHEDDING STRESS TEST")
    
    # Import and run the existing script's main function
    from src.scripts.run_load_shedding_analysis import main as load_shed_main
    load_shed_main()


def run_robust_tep() -> None:
    """Run scenario-based robust TEP."""
    print_header("SCENARIO-BASED ROBUST TEP")
    
    # Import and run the existing script's main function
    from src.scripts.run_robust_tep import main as robust_main
    robust_main()


def run_visualizations() -> None:
    """Generate all visualizations."""
    print_header("GENERATING VISUALIZATIONS")
    
    from src.core.data_loader import RTSDataLoader
    from src.core.dc_opf import DCOPF
    from src.analysis.enhanced_visualizations import generate_all_visualizations
    
    data_loader = RTSDataLoader(data_dir='data/RTS_Data/SourceData')
    
    dcopf = DCOPF(data_loader)
    dcopf.build_model()
    dcopf.solve(solver='gurobi')
    baseline_results = dcopf.get_results()
    
    generate_all_visualizations(data_loader, baseline_results)


def run_all() -> None:
    """Run all analyses."""
    print_header("CME307: COMPLETE ANALYSIS PIPELINE")
    print("\nThis will run all analyses in sequence. This may take 10-15 minutes.")
    
    # 1. Baseline and TEP
    results = run_baseline_and_tep()
    
    # 2. Cost sensitivity
    run_cost_sensitivity()
    
    # 3. Multi-period (simplified)
    run_multi_period()
    
    # 4. Load shedding
    run_load_shedding()
    
    # 5. Robust TEP
    run_robust_tep()
    
    # 6. Visualizations
    run_visualizations()
    
    print_header("ALL ANALYSES COMPLETE")
    print("\nResults saved to:")
    print("  - results/cost_sensitivity.csv")
    print("  - results/load_shedding_periods.csv")
    print("  - results/robust_tep_summary.csv")
    print("  - results/*.png (network, generation mix, congestion)")
    print("  - results/plots/*.png (all analysis plots)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='CME307 Transmission Expansion Planning Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/scripts/main.py --all              Run all analyses
  python src/scripts/main.py --baseline         Run baseline + TEP
  python src/scripts/main.py --cost-sensitivity Run cost sweep
  python src/scripts/main.py --robust           Run robust TEP
  python src/scripts/main.py --visualize        Generate plots
        """
    )
    
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    parser.add_argument('--baseline', action='store_true', help='Run baseline DC-OPF and TEP')
    parser.add_argument('--cost-sensitivity', action='store_true', help='Run cost sensitivity analysis')
    parser.add_argument('--multi-period', action='store_true', help='Run simplified multi-period TEP')
    parser.add_argument('--load-shedding', action='store_true', help='Run load shedding stress test')
    parser.add_argument('--robust', action='store_true', help='Run scenario-based robust TEP')
    parser.add_argument('--visualize', action='store_true', help='Generate all visualizations')
    
    args = parser.parse_args()
    
    # If no args, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Run selected analyses
    if args.all:
        run_all()
    else:
        if args.baseline:
            run_baseline_and_tep()
        if args.cost_sensitivity:
            run_cost_sensitivity()
        if args.multi_period:
            run_multi_period()
        if args.load_shedding:
            run_load_shedding()
        if args.robust:
            run_robust_tep()
        if args.visualize:
            run_visualizations()


if __name__ == '__main__':
    main()

