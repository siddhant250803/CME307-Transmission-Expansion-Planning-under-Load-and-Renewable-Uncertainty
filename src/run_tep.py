"""
Run Transmission Expansion Planning model
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import RTSDataLoader
from src.dc_opf import DCOPF
from src.tep import TEP

def main():
    print("="*60)
    print("CME307: Transmission Expansion Planning")
    print("TEP MILP Model")
    print("="*60)
    
    # Load data
    print("\nLoading RTS-GMLC data...")
    data_loader = RTSDataLoader(data_dir='data/RTS_Data/SourceData')
    
    # Run baseline first
    print("\n" + "="*60)
    print("STEP 1: Baseline DC OPF (No Expansion)")
    print("="*60)
    dcopf = DCOPF(data_loader)
    dcopf.build_model()
    baseline_success = dcopf.solve(solver='gurobi')
    
    if baseline_success:
        baseline_results = dcopf.get_results()
        baseline_cost = baseline_results['objective_value']
        dcopf.print_summary()
    else:
        print("Baseline failed, continuing with TEP...")
        baseline_cost = None
    
    # Run TEP
    print("\n" + "="*60)
    print("STEP 2: Transmission Expansion Planning")
    print("="*60)
    
    # Create TEP model
    tep = TEP(data_loader, line_cost_per_mw=1000000)  # $1M per MW
    tep.generate_candidate_lines(max_candidates=30)  # Limit candidates for faster solve
    
    print("\nBuilding TEP MILP model...")
    tep.build_model()
    
    print("\nSolving TEP MILP...")
    tep_success = tep.solve(solver='gurobi', time_limit=600)  # 10 minute limit
    
    if tep_success:
        tep.print_summary()
        tep_results = tep.get_results()
        
        if baseline_cost is not None:
            savings = baseline_cost - tep_results['operating_cost']
            print(f"\n" + "="*60)
            print("COMPARISON")
            print("="*60)
            print(f"Baseline Operating Cost: ${baseline_cost:,.2f}")
            print(f"TEP Operating Cost: ${tep_results['operating_cost']:,.2f}")
            print(f"Operating Cost Savings: ${savings:,.2f}")
            print(f"Investment Cost: ${tep_results['investment_cost']:,.2f}")
            print(f"Net Savings: ${savings - tep_results['investment_cost']:,.2f}")
            print("="*60)
    else:
        print("Failed to solve TEP model")

if __name__ == '__main__':
    main()

