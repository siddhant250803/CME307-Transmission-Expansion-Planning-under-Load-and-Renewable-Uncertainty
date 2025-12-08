"""
RTS-GMLC Data Loader
====================

This module provides utilities to load and process the IEEE RTS-GMLC 
(Reliability Test System - Grid Modernization Lab Consortium) dataset
for power system optimization studies.

The RTS-GMLC Dataset
--------------------
The RTS-GMLC is a modernized version of the classic IEEE Reliability Test 
System, featuring:
- 73 buses across 3 interconnected regions
- 120 AC transmission lines + 1 DC tie
- 158 generating units including wind, solar, and storage
- Full hourly time series for load and renewable generation

Dataset Structure
-----------------
The source data is organized as CSV files:
- bus.csv: Bus data (ID, area, voltage, load, coordinates)
- branch.csv: Transmission line data (from/to bus, impedance, ratings)
- gen.csv: Generator data (bus, capacity, costs, fuel type)

Usage Example
-------------
>>> from src.core.data_loader import RTSDataLoader
>>>
>>> # Load all system data
>>> data = RTSDataLoader('data/RTS_Data/SourceData')
>>>
>>> # Access individual data frames
>>> buses = data.get_bus_data()
>>> generators = data.get_generator_data()
>>> branches = data.get_branch_data()
>>>
>>> # Get derived quantities
>>> load_by_bus = data.get_load_by_bus()
>>> branch_params = data.get_branch_parameters()

References
----------
- Barrows, C., et al. (2019). The IEEE Reliability Test System: A proposed 
  2019 update. IEEE Transactions on Power Systems, 35(1), 119-127.
- GitHub: https://github.com/GridMod/RTS-GMLC

Author: CME307 Team (Edouard Rabasse, Siddhant Sukhani)
Date: December 2025
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Any


class RTSDataLoader:
    """
    Data loader for the RTS-GMLC power system dataset.
    
    This class handles loading and preprocessing of all static network data
    required for DC-OPF and TEP optimization models.
    
    Attributes
    ----------
    data_dir : str
        Path to the SourceData directory containing CSV files
    buses : pd.DataFrame
        Bus data indexed by Bus ID
    branches : pd.DataFrame
        Branch/transmission line data
    generators : pd.DataFrame
        Generator unit data
        
    Methods
    -------
    load_data()
        Load all CSV files into DataFrames
    get_bus_data()
        Return bus DataFrame
    get_branch_data()
        Return branch DataFrame
    get_generator_data()
        Return generator DataFrame
    get_load_by_bus()
        Return dict mapping bus ID to load (MW)
    get_generators_by_bus()
        Return dict grouping generators by bus
    get_total_capacity_by_bus()
        Return dict mapping bus ID to total generation capacity
    get_branch_parameters()
        Return dict of electrical parameters for each branch
    
    Notes
    -----
    The loader automatically handles relative paths from the project root
    and validates that required files exist.
    """
    
    def __init__(self, data_dir: str = 'data/RTS_Data/SourceData'):
        """
        Initialize the data loader.
        
        Parameters
        ----------
        data_dir : str
            Path to the RTS-GMLC SourceData directory.
            Can be relative (resolved from project root) or absolute.
            
        Raises
        ------
        FileNotFoundError
            If required CSV files are not found
        """
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
        """
        Load all CSV data files into pandas DataFrames.
        
        This method reads:
        - bus.csv: Network topology and load data
        - branch.csv: Transmission line parameters
        - gen.csv: Generator characteristics and costs
        
        Raises
        ------
        FileNotFoundError
            If any required file is missing
        """
        # =====================================================================
        # LOAD BUS DATA
        # Contains: Bus ID, Area, BaseKV, MW Load, MVAR Load, coordinates
        # =====================================================================
        bus_file = os.path.join(self.data_dir, 'bus.csv')
        if not os.path.exists(bus_file):
            raise FileNotFoundError(f"Bus file not found: {bus_file}")
        
        self.buses = pd.read_csv(bus_file)
        self.buses.set_index('Bus ID', inplace=True)  # Use Bus ID as index for easy lookup
        
        # =====================================================================
        # LOAD BRANCH DATA
        # Contains: UID, From Bus, To Bus, R, X, B, ratings
        # =====================================================================
        branch_file = os.path.join(self.data_dir, 'branch.csv')
        if not os.path.exists(branch_file):
            raise FileNotFoundError(f"Branch file not found: {branch_file}")
        
        self.branches = pd.read_csv(branch_file)
        
        # =====================================================================
        # LOAD GENERATOR DATA
        # Contains: GEN UID, Bus ID, Unit Type, Fuel, PMin, PMax, costs, etc.
        # =====================================================================
        gen_file = os.path.join(self.data_dir, 'gen.csv')
        if not os.path.exists(gen_file):
            raise FileNotFoundError(f"Generator file not found: {gen_file}")
        
        self.generators = pd.read_csv(gen_file)
        
        # Print summary
        print(f"✓ Loaded {len(self.buses)} buses, {len(self.branches)} branches, "
              f"{len(self.generators)} generators")
    
    def get_bus_data(self) -> pd.DataFrame:
        """
        Get the bus data DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Bus data indexed by Bus ID with columns:
            - Area: Region number (1, 2, or 3)
            - BaseKV: Nominal voltage (kV)
            - MW Load: Active power load (MW)
            - MVAR Load: Reactive power load (MVAR)
            - lat, lng: Geographic coordinates (if available)
        """
        return self.buses
    
    def get_branch_data(self) -> pd.DataFrame:
        """
        Get the branch data DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Branch data with columns:
            - UID: Unique branch identifier
            - From Bus, To Bus: Terminal bus IDs
            - R, X, B: Series resistance, reactance, shunt susceptance (p.u.)
            - Cont Rating: Continuous thermal rating (MW)
            - LTE Rating: Long-term emergency rating (MW)
            - STE Rating: Short-term emergency rating (MW)
        """
        return self.branches
    
    def get_generator_data(self) -> pd.DataFrame:
        """
        Get the generator data DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Generator data with columns:
            - GEN UID: Unique generator identifier
            - Bus ID: Location bus
            - Unit Type: Technology (e.g., CT, CC, WIND, PV)
            - Fuel: Primary fuel type
            - PMin MW, PMax MW: Operating limits
            - Fuel Price $/MMBTU: Fuel cost
            - HR_avg_0: Average heat rate (BTU/kWh)
        """
        return self.generators
    
    def get_load_by_bus(self) -> Dict[int, float]:
        """
        Get active power load at each bus.
        
        Returns
        -------
        dict
            Mapping of Bus ID → MW Load
            
        Example
        -------
        >>> load = data.get_load_by_bus()
        >>> print(f"Load at bus 101: {load[101]:.1f} MW")
        """
        return self.buses['MW Load'].to_dict()
    
    def get_generators_by_bus(self) -> Dict[int, List[pd.Series]]:
        """
        Group generators by their bus location.
        
        Returns
        -------
        dict
            Mapping of Bus ID → list of generator Series objects
            
        Example
        -------
        >>> gen_by_bus = data.get_generators_by_bus()
        >>> for gen in gen_by_bus.get(101, []):
        ...     print(f"  {gen['GEN UID']}: {gen['PMax MW']} MW")
        """
        gen_by_bus = {}
        for _, gen in self.generators.iterrows():
            bus_id = gen['Bus ID']
            if bus_id not in gen_by_bus:
                gen_by_bus[bus_id] = []
            gen_by_bus[bus_id].append(gen)
        return gen_by_bus
    
    def get_total_capacity_by_bus(self) -> Dict[int, float]:
        """
        Calculate total generation capacity at each bus.
        
        Returns
        -------
        dict
            Mapping of Bus ID → total PMax (MW)
        """
        gen_by_bus = self.get_generators_by_bus()
        capacity = {}
        for bus_id, gens in gen_by_bus.items():
            capacity[bus_id] = sum(gen['PMax MW'] for gen in gens)
        return capacity
    
    def get_branch_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract electrical parameters for each branch.
        
        This method computes the susceptance (B = 1/X) used in DC power flow
        calculations and organizes branch data for easy model construction.
        
        Returns
        -------
        dict
            Mapping of branch UID → parameter dict with keys:
            - 'from_bus': Source bus ID
            - 'to_bus': Destination bus ID
            - 'r': Series resistance (p.u.)
            - 'x': Series reactance (p.u.)
            - 'b': Shunt susceptance (p.u.)
            - 'rating': Continuous thermal rating (MW)
            - 'susceptance': Line susceptance B = 1/X for DC flow
            
        Notes
        -----
        The susceptance is computed as B = 1/X. For lines with X = 0
        (which shouldn't occur in practice), susceptance is set to 0.
        """
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
                'rating': branch['Cont Rating'],  # Continuous (normal) rating in MW
                'susceptance': 1.0 / x_value if x_value > 0 else 0.0
            }
        return branch_params
    
    def get_system_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the power system.
        
        Returns
        -------
        dict
            Summary statistics including:
            - n_buses: Number of buses
            - n_branches: Number of transmission lines
            - n_generators: Number of generating units
            - total_load: System-wide load (MW)
            - total_capacity: Total generation capacity (MW)
            - n_areas: Number of regions
        """
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
