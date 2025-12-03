"""
Run Multi-Period Transmission Expansion Planning model

NOTE: Gurobi WLS (Web License Service) academic licenses may have size limits
on the number of constraints or nonzeros, even if variable limits are fine.
This model creates ~3240 constraints and ~5980 nonzeros which may exceed
some academic license limits.

For size-limited licenses, use 'run_simplified_tep.py' instead, which uses
a two-stage approach that works within license limits.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.data_loader import RTSDataLoader
from src.core.dc_opf import DCOPF
from src.core.timeseries_loader import TimeseriesLoader
from src.core.multi_period_tep import MultiPeriodTEP

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
    print(f"Model size: {len(tep.periods)} periods, {len(tep.candidate_lines)} candidates")
    print(f"Estimated: ~{len(tep.periods) * (len(tep.model.generators) + len(tep.model.buses) + len(tep.model.branches) + len(tep.candidate_lines))} variables, ~{len(tep.periods) * 800} constraints\n")
    
    try:
        tep_success = tep.solve(solver='gurobi', time_limit=1200)  # 20 minute limit
    except Exception as e:
        error_str = str(e).lower()
        if "too large" in error_str or ("license" in error_str and "size" in error_str):
            print(f"\n{'='*60}")
            print("Gurobi License Issue Detected")
            print(f"{'='*60}")
            print(f"Model size: {len(tep.periods)} periods, {len(tep.candidate_lines)} candidates")
            print("\nNOTE: Your Gurobi WLS academic license may have size restrictions.")
            print("Even though variable/constraint counts may be within limits, the model")
            print("complexity (MIP structure, Big-M constraints) may exceed license limits.")
            print("\nRECOMMENDATION: Use 'run_simplified_tep.py' instead.")
            print("It uses a two-stage approach that works reliably:")
            print("  1. Identifies peak congestion periods")
            print("  2. Solves TEP for aggregated peak load scenario")
            print("  3. Includes load shedding for feasibility")
            print("\nTo run the simplified version:")
            print("  python src/scripts/run_simplified_tep.py")
            print(f"{'='*60}\n")
            tep_success = False
        elif "infeasible" in error_str:
            print(f"\n{'='*60}")
            print("Model is Infeasible")
            print(f"{'='*60}")
            print("The model cannot satisfy all constraints simultaneously.")
            print("This can happen when:")
            print("  - Selected periods have incompatible load/generation patterns")
            print("  - System cannot meet demand with available generation across all periods")
            print("  - Renewable generation is too low in some periods")
            print("\nRECOMMENDATION: Use 'run_simplified_tep.py' which includes:")
            print("  - Load shedding option for feasibility")
            print("  - Better period selection (peak periods only)")
            print("  - Stress testing with targeted outages")
            print("\nTo run the simplified version:")
            print("  python src/scripts/run_simplified_tep.py")
            print(f"{'='*60}\n")
            tep_success = False
        else:
            print(f"\nUnexpected error: {e}")
            print("Try using 'run_simplified_tep.py' for a more robust approach.")
            tep_success = False
    
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

