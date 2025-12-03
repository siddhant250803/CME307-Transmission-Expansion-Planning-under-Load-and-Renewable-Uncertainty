"""
Run baseline DC OPF model
"""
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.data_loader import RTSDataLoader
from src.core.dc_opf import DCOPF

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

