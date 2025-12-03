"""
Time series data loader for RTS-GMLC
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

class TimeseriesLoader:
    """Load and process time series data"""
    
    def __init__(self, data_dir='data/RTS_Data'):
        if not os.path.isabs(data_dir):
            project_root = Path(__file__).parent.parent
            data_path = project_root / data_dir
            if data_path.exists():
                self.data_dir = str(data_path)
            else:
                self.data_dir = data_dir
        else:
            self.data_dir = data_dir
        
        self.load_data = None
        self.wind_data = None
        self.pv_data = None
        self.hydro_data = None
        
    def load_regional_load(self, simulation='DAY_AHEAD'):
        """Load regional load time series"""
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
        """Load wind generation time series"""
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
        """Load PV generation time series"""
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
        """Load hydro generation time series"""
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
        """Calculate load participation factors for each bus within its region"""
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
        """Get nodal loads for a specific period"""
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
        """Select representative periods for optimization
        
        Args:
            n_periods: Number of periods to select
            method: 'peak_avg_low' or 'kmeans' or 'all'
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

