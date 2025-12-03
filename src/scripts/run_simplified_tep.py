"""
Run Simplified Multi-Period TEP (works within Gurobi license limits)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.data_loader import RTSDataLoader
from src.core.timeseries_loader import TimeseriesLoader
from src.core.simplified_multi_period_tep import SimplifiedMultiPeriodTEP

def main():
    print("="*60)
    print("CME307: Simplified Multi-Period TEP")
    print("Using Time Series Data (Peak Period Analysis)")
    print("="*60)
    
    # Load data
    print("\nLoading RTS-GMLC data...")
    data_loader = RTSDataLoader(data_dir='data/RTS_Data/SourceData')
    
    print("Loading time series data...")
    timeseries_loader = TimeseriesLoader(data_dir='data/RTS_Data')
    
    # Create simplified TEP with very low line cost to encourage expansion
    # Also will stress test with higher loads
    tep = SimplifiedMultiPeriodTEP(
        data_loader,
        timeseries_loader,
        line_cost_per_mw=200000  # Very low cost ($200k/MW) to make expansion attractive
    )
    
    # Identify peak periods
    print("\n" + "="*60)
    print("STEP 1: Identify Peak Congestion Periods")
    print("="*60)
    peak_periods = tep.identify_congestion_periods(n_periods=2)
    
    # Solve TEP
    print("\n" + "="*60)
    print("STEP 2: Solve TEP for Peak Load Scenario")
    print("="*60)
    success = tep.solve_aggregated_tep(max_candidates=8)
    
    if success:
        tep.print_summary()
        results = tep.results
        
        print(f"\n" + "="*60)
        print("ANALYSIS")
        print("="*60)
        print(f"Peak periods identified: {results['peak_periods']}")
        print(f"Investment decision based on worst-case (peak) load scenario")
        print(f"Lines built will help relieve congestion during peak periods")
        print("="*60)
    else:
        print("Failed to solve TEP model")

if __name__ == '__main__':
    main()

