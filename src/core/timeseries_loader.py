"""
Time Series Data Loader for RTS-GMLC
====================================

This module handles loading and processing of hourly time series data from the
RTS-GMLC dataset, including:
- Regional load profiles (3 regions)
- Wind generation by unit
- PV (photovoltaic) generation by unit
- Hydro generation by unit

The time series data spans a full year (8,760 hours) and is used for multi-period
optimization models that account for load and renewable variability.

Key Features
------------
- Load regional load profiles and distribute to individual buses using participation factors
- Load renewable generation time series (wind, PV, hydro)
- Select representative periods using peak/average/low or k-means clustering
- Calculate bus-level load participation factors based on base case load distribution

Usage Example
-------------
>>> from src.core.timeseries_loader import TimeseriesLoader
>>> from src.core.data_loader import RTSDataLoader
>>>
>>> # Load time series data
>>> timeseries = TimeseriesLoader(data_dir='data/RTS_Data')
>>> timeseries.load_regional_load()
>>> timeseries.load_wind_generation()
>>>
>>> # Get nodal loads for a specific period
>>> data_loader = RTSDataLoader('data/RTS_Data/SourceData')
>>> buses = data_loader.get_bus_data()
>>> participation = timeseries.get_bus_load_participation_factors(buses)
>>> nodal_loads = timeseries.get_nodal_loads(period=1, buses=buses, 
>>>                                          participation_factors=participation)
>>>
>>> # Select representative periods
>>> periods = timeseries.select_representative_periods(n_periods=12, 
>>>                                                    method='peak_avg_low')

Author: CME307 Team (Edouard Rabasse, Siddhant Sukhani)
Date: December 2025
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

class TimeseriesLoader:
    """
    Load and process time series data from RTS-GMLC dataset.
    
    This class provides methods to:
    - Load hourly regional load profiles
    - Load hourly renewable generation profiles (wind, PV, hydro)
    - Calculate bus-level load participation factors
    - Distribute regional loads to individual buses
    - Select representative periods for optimization
    
    Attributes
    ----------
    data_dir : str
        Path to the RTS_Data directory containing timeseries_data_files
    load_data : dict, optional
        Dictionary mapping period -> {region: load} after load_regional_load()
    wind_data : dict, optional
        Dictionary mapping period -> {gen_id: generation} after load_wind_generation()
    pv_data : dict, optional
        Dictionary mapping period -> {gen_id: generation} after load_pv_generation()
    hydro_data : dict, optional
        Dictionary mapping period -> {gen_id: generation} after load_hydro_generation()
    """
    
    def __init__(self, data_dir='data/RTS_Data'):
        """
        Initialize the time series loader.
        
        Parameters
        ----------
        data_dir : str, optional
            Path to the RTS_Data directory. If relative, resolved relative to
            project root. Default is 'data/RTS_Data'.
        """
        # Resolve relative paths relative to project root
        if not os.path.isabs(data_dir):
            project_root = Path(__file__).resolve().parents[2]
            data_path = project_root / data_dir
            if data_path.exists():
                self.data_dir = str(data_path)
            else:
                self.data_dir = data_dir
        else:
            self.data_dir = data_dir
        
        # Initialize data storage (loaded lazily)
        self.load_data = None      # Regional load time series
        self.wind_data = None      # Wind generation time series
        self.pv_data = None        # PV generation time series
        self.hydro_data = None     # Hydro generation time series
        
    def load_regional_load(self, simulation='DAY_AHEAD'):
        """
        Load regional load time series from CSV file.
        
        The load data is organized by period (hour) and region (1, 2, 3).
        Each period contains the total load for each of the three regions.
        
        Parameters
        ----------
        simulation : str, optional
            Simulation type, default is 'DAY_AHEAD'. Determines which CSV file
            to load from the Load directory.
            
        Returns
        -------
        dict
            Dictionary mapping period (int) -> {region (int): load (float)}
            Format: {period: {1: load_region1, 2: load_region2, 3: load_region3}}
            
        Raises
        ------
        FileNotFoundError
            If the load CSV file does not exist
        """
        load_file = os.path.join(self.data_dir, 'timeseries_data_files', 'Load', 
                                 f'{simulation}_regional_Load.csv')
        if not os.path.exists(load_file):
            raise FileNotFoundError(f"Load file not found: {load_file}")
        
        df = pd.read_csv(load_file)
        # Convert to dict: {period: {region: load}}
        load_dict = {}
        for _, row in df.iterrows():
            period = int(row['Period'])
            load_dict[period] = {
                1: row['1'],
                2: row['2'],
                3: row['3']
            }
        
        self.load_data = load_dict
        return load_dict
    
    def load_wind_generation(self, simulation='DAY_AHEAD'):
        """
        Load wind generation time series from CSV file.
        
        The wind data contains hourly generation for each wind generator unit.
        Generation is expressed as a fraction of nameplate capacity (0-1).
        
        Parameters
        ----------
        simulation : str, optional
            Simulation type, default is 'DAY_AHEAD'
            
        Returns
        -------
        dict
            Dictionary mapping period (int) -> {gen_id (str): generation (float)}
            Generation values are capacity factors (0-1)
            
        Raises
        ------
        FileNotFoundError
            If the wind CSV file does not exist
        """
        wind_file = os.path.join(self.data_dir, 'timeseries_data_files', 'WIND',
                                 f'{simulation}_wind.csv')
        if not os.path.exists(wind_file):
            raise FileNotFoundError(f"Wind file not found: {wind_file}")
        
        df = pd.read_csv(wind_file)
        # Get wind generator IDs from columns (skip Year, Month, Day, Period)
        wind_gens = [col for col in df.columns if col not in ['Year', 'Month', 'Day', 'Period']]
        
        wind_dict = {}
        for _, row in df.iterrows():
            period = int(row['Period'])
            wind_dict[period] = {gen: row[gen] for gen in wind_gens}
        
        self.wind_data = wind_dict
        return wind_dict
    
    def load_pv_generation(self, simulation='DAY_AHEAD'):
        """
        Load PV (photovoltaic) generation time series from CSV file.
        
        Parameters
        ----------
        simulation : str, optional
            Simulation type, default is 'DAY_AHEAD'
            
        Returns
        -------
        dict
            Dictionary mapping period (int) -> {gen_id (str): generation (float)}
            Generation values are capacity factors (0-1)
            
        Raises
        ------
        FileNotFoundError
            If the PV CSV file does not exist
        """
        pv_file = os.path.join(self.data_dir, 'timeseries_data_files', 'PV',
                               f'{simulation}_pv.csv')
        if not os.path.exists(pv_file):
            raise FileNotFoundError(f"PV file not found: {pv_file}")
        
        df = pd.read_csv(pv_file)
        pv_gens = [col for col in df.columns if col not in ['Year', 'Month', 'Day', 'Period']]
        
        pv_dict = {}
        for _, row in df.iterrows():
            period = int(row['Period'])
            pv_dict[period] = {gen: row[gen] for gen in pv_gens}
        
        self.pv_data = pv_dict
        return pv_dict
    
    def load_hydro_generation(self, simulation='DAY_AHEAD'):
        """
        Load hydroelectric generation time series from CSV file.
        
        Parameters
        ----------
        simulation : str, optional
            Simulation type, default is 'DAY_AHEAD'
            
        Returns
        -------
        dict
            Dictionary mapping period (int) -> {gen_id (str): generation (float)}
            Generation values are capacity factors (0-1)
            
        Raises
        ------
        FileNotFoundError
            If the hydro CSV file does not exist
        """
        hydro_file = os.path.join(self.data_dir, 'timeseries_data_files', 'Hydro',
                                  f'{simulation}_hydro.csv')
        if not os.path.exists(hydro_file):
            raise FileNotFoundError(f"Hydro file not found: {hydro_file}")
        
        df = pd.read_csv(hydro_file)
        hydro_gens = [col for col in df.columns if col not in ['Year', 'Month', 'Day', 'Period']]
        
        hydro_dict = {}
        for _, row in df.iterrows():
            period = int(row['Period'])
            hydro_dict[period] = {gen: row[gen] for gen in hydro_gens}
        
        self.hydro_data = hydro_dict
        return hydro_dict
    
    def get_bus_load_participation_factors(self, buses):
        """
        Calculate load participation factors for each bus within its region.
        
        Participation factors represent the fraction of regional load served by
        each bus. They are calculated as: bus_load / total_region_load.
        This allows regional load time series to be distributed to individual buses
        proportionally to their base case loads.
        
        Parameters
        ----------
        buses : pd.DataFrame
            Bus data frame with columns 'Area' (region) and 'MW Load' (base load)
            
        Returns
        -------
        dict
            Dictionary mapping bus_id (int) -> participation_factor (float)
            Participation factors sum to 1.0 within each region
        """
        # Group buses by region
        region_loads = {}
        for bus_id in buses.index:
            region = buses.loc[bus_id, 'Area']
            load = buses.loc[bus_id, 'MW Load']
            if region not in region_loads:
                region_loads[region] = 0.0
            region_loads[region] += load
        
        # Calculate participation factors
        participation = {}
        for bus_id in buses.index:
            region = buses.loc[bus_id, 'Area']
            bus_load = buses.loc[bus_id, 'MW Load']
            if region_loads[region] > 0:
                participation[bus_id] = bus_load / region_loads[region]
            else:
                participation[bus_id] = 0.0
        
        return participation
    
    def get_nodal_loads(self, period, buses, participation_factors):
        """
        Calculate nodal (bus-level) loads for a specific time period.
        
        This method takes regional load for the period and distributes it to
        individual buses using participation factors.
        
        Parameters
        ----------
        period : int
            Time period (hour) to get loads for
        buses : pd.DataFrame
            Bus data frame with 'Area' column indicating region
        participation_factors : dict
            Dictionary mapping bus_id -> participation_factor (from 
            get_bus_load_participation_factors)
            
        Returns
        -------
        dict
            Dictionary mapping bus_id (int) -> nodal_load (float) in MW
            
        Raises
        ------
        ValueError
            If period is not found in loaded data
        """
        if self.load_data is None:
            self.load_regional_load()
        
        if period not in self.load_data:
            raise ValueError(f"Period {period} not found in load data")
        
        nodal_loads = {}
        for bus_id in buses.index:
            region = buses.loc[bus_id, 'Area']
            regional_load = self.load_data[period][region]
            participation = participation_factors[bus_id]
            nodal_loads[bus_id] = regional_load * participation
        
        return nodal_loads
    
    def select_representative_periods(self, n_periods=24, method='peak_avg_low'):
        """
        Select representative time periods for multi-period optimization.
        
        Since solving optimization over all 8,760 hours is computationally
        intractable, this method selects a subset of representative periods
        that capture key system conditions.
        
        Parameters
        ----------
        n_periods : int, optional
            Number of periods to select, default is 24
        method : str, optional
            Selection method:
            - 'peak_avg_low': Select periods from top 25% (peak), middle 50% (average),
              and bottom 25% (low) of system load distribution
            - 'kmeans': Use k-means clustering to group similar load profiles and
              select one representative from each cluster
            - 'all': Return all available periods (up to 25)
            
        Returns
        -------
        list
            List of selected period numbers (integers)
        """
        if self.load_data is None:
            self.load_regional_load()
        
        if method == 'all':
            return list(range(1, min(25, len(self.load_data) + 1)))
        
        # Calculate total system load for each period
        period_loads = {}
        for period, regional_loads in self.load_data.items():
            total_load = sum(regional_loads.values())
            period_loads[period] = total_load
        
        if method == 'peak_avg_low':
            # Select peak, average, and low load periods
            sorted_periods = sorted(period_loads.items(), key=lambda x: x[1], reverse=True)
            
            # Get peak periods (top 25%)
            n_peak = max(1, n_periods // 4)
            peak_periods = [p[0] for p in sorted_periods[:n_peak]]
            
            # Get average periods (middle 50%)
            n_avg = n_periods - 2 * n_peak
            mid_start = len(sorted_periods) // 4
            mid_end = mid_start + n_avg
            avg_periods = [p[0] for p in sorted_periods[mid_start:mid_end]]
            
            # Get low periods (bottom 25%)
            low_periods = [p[0] for p in sorted_periods[-n_peak:]]
            
            selected = sorted(peak_periods + avg_periods + low_periods)
            return selected[:n_periods]
        
        elif method == 'kmeans':
            # Simple k-means clustering
            from sklearn.cluster import KMeans
            periods = list(period_loads.keys())
            loads = np.array([[period_loads[p]] for p in periods])
            
            kmeans = KMeans(n_clusters=n_periods, random_state=42, n_init=10)
            kmeans.fit(loads)
            
            # Select one period from each cluster (closest to centroid)
            selected = []
            for i in range(n_periods):
                cluster_periods = [periods[j] for j in range(len(periods)) if kmeans.labels_[j] == i]
                if cluster_periods:
                    cluster_loads = [period_loads[p] for p in cluster_periods]
                    centroid_load = kmeans.cluster_centers_[i][0]
                    closest = min(cluster_periods, key=lambda p: abs(period_loads[p] - centroid_load))
                    selected.append(closest)
            
            return sorted(selected)
        
        return list(range(1, n_periods + 1))

