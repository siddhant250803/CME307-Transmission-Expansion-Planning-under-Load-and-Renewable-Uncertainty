"""
Run baseline DC OPF model
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import RTSDataLoader
from src.dc_opf import DCOPF

def main():
    print("="*60)
    print("CME307: Transmission Expansion Planning")
    print("Baseline DC OPF Model")
    print("="*60)
    
    # Load data
    print("\nLoading RTS-GMLC data...")
    data_loader = RTSDataLoader(data_dir='data/RTS_Data/SourceData')
    
    # Create and solve DC OPF
    print("\nBuilding DC OPF model...")
    dcopf = DCOPF(data_loader)
    dcopf.build_model()
    
    print("\nSolving DC OPF...")
    success = dcopf.solve(solver='gurobi')
    
    if success:
        dcopf.print_summary()
        
        # Save results
        results = dcopf.get_results()
        print(f"\nResults saved. Total cost: ${results['objective_value']:,.2f}")
    else:
        print("Failed to solve DC OPF model")

if __name__ == '__main__':
    main()

