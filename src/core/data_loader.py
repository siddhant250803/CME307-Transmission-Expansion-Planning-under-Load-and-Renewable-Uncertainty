"""
Data loader for RTS-GMLC dataset
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

class RTSDataLoader:
    """Load and process RTS-GMLC data files"""
    
    def __init__(self, data_dir='data/RTS_Data/SourceData'):
        # Handle relative paths
        if not os.path.isabs(data_dir):
            # Try relative to project root
            project_root = Path(__file__).parent.parent
            data_path = project_root / data_dir
            if data_path.exists():
                self.data_dir = str(data_path)
            else:
                self.data_dir = data_dir
        else:
            self.data_dir = data_dir
        
        self.buses = None
        self.branches = None
        self.generators = None
        self.load_data()
    
    def load_data(self):
        """Load all CSV data files"""
        # Load buses
        bus_file = os.path.join(self.data_dir, 'bus.csv')
        if not os.path.exists(bus_file):
            raise FileNotFoundError(f"Bus file not found: {bus_file}")
        self.buses = pd.read_csv(bus_file)
        self.buses.set_index('Bus ID', inplace=True)
        
        # Load branches
        branch_file = os.path.join(self.data_dir, 'branch.csv')
        if not os.path.exists(branch_file):
            raise FileNotFoundError(f"Branch file not found: {branch_file}")
        self.branches = pd.read_csv(branch_file)
        
        # Load generators
        gen_file = os.path.join(self.data_dir, 'gen.csv')
        if not os.path.exists(gen_file):
            raise FileNotFoundError(f"Generator file not found: {gen_file}")
        self.generators = pd.read_csv(gen_file)
        
        print(f"Loaded {len(self.buses)} buses, {len(self.branches)} branches, {len(self.generators)} generators")
    
    def get_bus_data(self):
        """Get bus data"""
        return self.buses
    
    def get_branch_data(self):
        """Get branch data"""
        return self.branches
    
    def get_generator_data(self):
        """Get generator data"""
        return self.generators
    
    def get_load_by_bus(self):
        """Get load (MW) by bus"""
        return self.buses['MW Load'].to_dict()
    
    def get_generators_by_bus(self):
        """Get generators grouped by bus"""
        gen_by_bus = {}
        for _, gen in self.generators.iterrows():
            bus_id = gen['Bus ID']
            if bus_id not in gen_by_bus:
                gen_by_bus[bus_id] = []
            gen_by_bus[bus_id].append(gen)
        return gen_by_bus
    
    def get_total_capacity_by_bus(self):
        """Get total generation capacity by bus"""
        gen_by_bus = self.get_generators_by_bus()
        capacity = {}
        for bus_id, gens in gen_by_bus.items():
            capacity[bus_id] = sum(gen['PMax MW'] for gen in gens)
        return capacity
    
    def get_branch_parameters(self):
        """Get branch parameters for DC power flow"""
        branch_params = {}
        for _, branch in self.branches.iterrows():
            uid = branch['UID']
            branch_params[uid] = {
                'from_bus': int(branch['From Bus']),
                'to_bus': int(branch['To Bus']),
                'r': branch['R'],
                'x': branch['X'],
                'b': branch['B'],
                'rating': branch['Cont Rating'],  # Continuous rating in MW
                'susceptance': 1.0 / branch['X'] if branch['X'] > 0 else 0.0
            }
        return branch_params

