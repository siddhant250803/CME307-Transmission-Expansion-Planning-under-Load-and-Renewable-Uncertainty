"""RTS-GMLC data loader for power system optimization."""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Any


class RTSDataLoader:
    """Data loader for the RTS-GMLC power system dataset."""
    
    def __init__(self, data_dir: str = 'data/RTS_Data/SourceData'):
        """Initialize the data loader."""
        # Handle relative path resolution
        if not os.path.isabs(data_dir):
            # Try resolving relative to the project root
            project_root = Path(__file__).parent.parent
            data_path = project_root / data_dir
            if data_path.exists():
                self.data_dir = str(data_path)
            else:
                self.data_dir = data_dir
        else:
            self.data_dir = data_dir
        
        # Initialize data containers
        self.buses = None
        self.branches = None
        self.generators = None
        
        # Load all data on initialization
        self.load_data()
    
    def load_data(self) -> None:
        """Load all CSV data files into pandas DataFrames."""
        bus_file = os.path.join(self.data_dir, 'bus.csv')
        if not os.path.exists(bus_file):
            raise FileNotFoundError(f"Bus file not found: {bus_file}")
        
        self.buses = pd.read_csv(bus_file)
        self.buses.set_index('Bus ID', inplace=True)
        
        branch_file = os.path.join(self.data_dir, 'branch.csv')
        if not os.path.exists(branch_file):
            raise FileNotFoundError(f"Branch file not found: {branch_file}")
        
        self.branches = pd.read_csv(branch_file)
        
        gen_file = os.path.join(self.data_dir, 'gen.csv')
        if not os.path.exists(gen_file):
            raise FileNotFoundError(f"Generator file not found: {gen_file}")
        
        self.generators = pd.read_csv(gen_file)
        
        print(f"✓ Loaded {len(self.buses)} buses, {len(self.branches)} branches, "
              f"{len(self.generators)} generators")
    
    def get_bus_data(self) -> pd.DataFrame:
        """Get the bus data DataFrame."""
        return self.buses
    
    def get_branch_data(self) -> pd.DataFrame:
        """Get the branch data DataFrame."""
        return self.branches
    
    def get_generator_data(self) -> pd.DataFrame:
        """Get the generator data DataFrame."""
        return self.generators
    
    def get_load_by_bus(self) -> Dict[int, float]:
        """Get active power load at each bus."""
        return self.buses['MW Load'].to_dict()
    
    def get_generators_by_bus(self) -> Dict[int, List[pd.Series]]:
        """Group generators by their bus location."""
        gen_by_bus = {}
        for _, gen in self.generators.iterrows():
            bus_id = gen['Bus ID']
            if bus_id not in gen_by_bus:
                gen_by_bus[bus_id] = []
            gen_by_bus[bus_id].append(gen)
        return gen_by_bus
    
    def get_total_capacity_by_bus(self) -> Dict[int, float]:
        """Calculate total generation capacity at each bus."""
        gen_by_bus = self.get_generators_by_bus()
        capacity = {}
        for bus_id, gens in gen_by_bus.items():
            capacity[bus_id] = sum(gen['PMax MW'] for gen in gens)
        return capacity
    
    def get_branch_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Extract electrical parameters for each branch."""
        branch_params = {}
        for _, branch in self.branches.iterrows():
            uid = branch['UID']
            x_value = branch['X']
            
            branch_params[uid] = {
                'from_bus': int(branch['From Bus']),
                'to_bus': int(branch['To Bus']),
                'r': branch['R'],
                'x': x_value,
                'b': branch['B'],
                'rating': branch['Cont Rating'],
                'susceptance': 1.0 / x_value if x_value > 0 else 0.0
            }
        return branch_params
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get a summary of the power system."""
        return {
            'n_buses': len(self.buses),
            'n_branches': len(self.branches),
            'n_generators': len(self.generators),
            'total_load': self.buses['MW Load'].sum(),
            'total_capacity': self.generators['PMax MW'].sum(),
            'n_areas': self.buses['Area'].nunique()
        }


if __name__ == '__main__':
    # Test the data loader
    print("="*60)
    print("Testing RTS-GMLC Data Loader")
    print("="*60)
    
    loader = RTSDataLoader('data/RTS_Data/SourceData')
    
    summary = loader.get_system_summary()
    print(f"\nSystem Summary:")
    print(f"  Buses:        {summary['n_buses']}")
    print(f"  Branches:     {summary['n_branches']}")
    print(f"  Generators:   {summary['n_generators']}")
    print(f"  Total Load:   {summary['total_load']:,.0f} MW")
    print(f"  Total Cap:    {summary['total_capacity']:,.0f} MW")
    print(f"  Regions:      {summary['n_areas']}")
