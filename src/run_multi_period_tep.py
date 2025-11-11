"""
Run Multi-Period Transmission Expansion Planning model

NOTE: This script requires a full Gurobi license. For size-limited licenses,
use 'run_simplified_tep.py' instead, which uses a two-stage approach that
works within license limits.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import RTSDataLoader
from src.dc_opf import DCOPF
from src.timeseries_loader import TimeseriesLoader
from src.multi_period_tep import MultiPeriodTEP

def main():
    print("="*60)
    print("CME307: Multi-Period Transmission Expansion Planning")
    print("Using Time Series Data")
    print("="*60)
    
    # Load data
    print("\nLoading RTS-GMLC data...")
    data_loader = RTSDataLoader(data_dir='data/RTS_Data/SourceData')
    
    print("Loading time series data...")
    timeseries_loader = TimeseriesLoader(data_dir='data/RTS_Data')
    
    # Run baseline for first period (for comparison) - skip for now, just run TEP
    print("\n" + "="*60)
    print("STEP 1: Baseline DC OPF (Skipped - using static load)")
    print("="*60)
    print("Note: Baseline uses static load from bus.csv")
    print("Multi-period TEP will use time-varying load from time series")
    baseline_cost = None
    
    # Run Multi-Period TEP
    print("\n" + "="*60)
    print("STEP 2: Multi-Period Transmission Expansion Planning")
    print("="*60)
    
    # Create multi-period TEP model
    # Use fewer periods to fit within Gurobi license limits
    tep = MultiPeriodTEP(
        data_loader, 
        timeseries_loader,
        line_cost_per_mw=500000,  # Lower cost to make expansion more attractive
        representative_method='peak_avg_low',
        candidate_lines=None  # Will generate limited candidates
    )
    # Limit candidate generation (very small for license)
    tep.generate_candidate_lines(max_candidates=10)
    
    # Prepare time series and select periods
    print("\nPreparing time series data...")
    tep.prepare_time_series(n_periods=4)  # 4 representative periods (minimal for license)
    
    print("\nBuilding Multi-Period TEP MILP model...")
    tep.build_model()
    
    # Limit candidates to reduce problem size
    if len(tep.candidate_lines) > 20:
        print(f"Limiting candidates from {len(tep.candidate_lines)} to 20...")
        tep.candidate_lines = tep.candidate_lines[:20]
        tep.build_model()  # Rebuild with fewer candidates
    
    print("\nSolving Multi-Period TEP MILP...")
    try:
        tep_success = tep.solve(solver='gurobi', time_limit=1200)  # 20 minute limit
    except Exception as e:
        if "too large" in str(e) or "license" in str(e).lower():
            print(f"\nGurobi license limit reached. Model size: {len(tep.periods)} periods, {len(tep.candidate_lines)} candidates")
            print("Note: This script requires a full Gurobi license for multi-period models.")
            print("Recommendation: Use 'run_simplified_tep.py' instead, which uses a two-stage approach")
            print("that works within license limits.")
            print("\nAttempting to reduce model size...")
            # Reduce model size and rebuild
            tep.periods = None  # Reset periods
            tep.prepare_time_series(n_periods=2)  # Use only 2 periods
            tep.generate_candidate_lines(max_candidates=5)  # Reduce to 5 candidates
            tep.build_model()  # Rebuild with smaller size
            try:
                print(f"Trying with reduced size: {len(tep.periods)} periods, {len(tep.candidate_lines)} candidates")
                tep_success = tep.solve(solver='gurobi', time_limit=600)
            except Exception as e2:
                if "too large" in str(e2) or "license" in str(e2).lower():
                    print("\nModel still too large for Gurobi license.")
                    print("Please use 'run_simplified_tep.py' which is designed to work within license limits.")
                    tep_success = False
                else:
                    raise
        else:
            raise
    
    if tep_success:
        tep.print_summary()
        tep_results = tep.get_results()
        
        if baseline_cost is not None:
            # Compare single period baseline to multi-period average
            avg_period_cost = tep_results['operating_cost'] / len(tep.periods)
            print(f"\n" + "="*60)
            print("COMPARISON")
            print("="*60)
            print(f"Baseline (Period 1) Operating Cost: ${baseline_cost:,.2f}")
            print(f"TEP Average Operating Cost per Period: ${avg_period_cost:,.2f}")
            print(f"TEP Total Operating Cost (all periods): ${tep_results['operating_cost']:,.2f}")
            print(f"Investment Cost: ${tep_results['investment_cost']:,.2f}")
            
            # Estimate annual savings (scale single period to annual)
            annual_baseline = baseline_cost * 365  # Rough estimate
            annual_tep = tep_results['operating_cost'] * (365 / len(tep.periods))
            annual_savings = annual_baseline - annual_tep
            
            print(f"\nEstimated Annual Comparison:")
            print(f"  Baseline Annual Operating Cost: ${annual_baseline:,.2f}")
            print(f"  TEP Annual Operating Cost: ${annual_tep:,.2f}")
            print(f"  Annual Operating Cost Savings: ${annual_savings:,.2f}")
            print(f"  Investment Cost: ${tep_results['investment_cost']:,.2f}")
            print(f"  Net Annual Savings: ${annual_savings - tep_results['investment_cost']:,.2f}")
            print(f"  Payback Period: {tep_results['investment_cost'] / max(annual_savings, 1):.1f} years")
            print("="*60)
    else:
        print("Failed to solve Multi-Period TEP model")

if __name__ == '__main__':
    main()

