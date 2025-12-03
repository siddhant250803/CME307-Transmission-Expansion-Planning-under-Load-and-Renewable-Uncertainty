"""
Example: Using different cost models for TEP
Demonstrates how to use distance-based, capacity-based, and hybrid cost models
"""
from .data_loader import RTSDataLoader
from .tep import TEP

def compare_cost_models():
    """Compare different cost models"""
    print("="*60)
    print("Comparing Transmission Line Cost Models")
    print("="*60)
    
    data_loader = RTSDataLoader(data_dir='data/RTS_Data/SourceData')
    
    # Example: 500MW line, 100 miles, 230kV
    capacity = 500  # MW
    distance = 100  # miles
    voltage = 230   # kV
    
    print(f"\nExample Line: {capacity}MW, {distance} miles, {voltage}kV")
    print("-"*60)
    
    # Model 1: Capacity-Distance (original)
    tep1 = TEP(data_loader, line_cost_per_mw=1000000, cost_model='capacity_distance')
    cost1 = tep1._calculate_line_cost(capacity, distance, voltage)
    print(f"1. Capacity-Distance Model:")
    print(f"   cost = $1M/MW * {capacity}MW * ({distance*1.609}km / 100km)")
    print(f"   Cost: ${cost1:,.0f}")
    
    # Model 2: Distance-Based (more realistic)
    tep2 = TEP(data_loader, cost_model='distance_based')
    cost2 = tep2._calculate_line_cost(capacity, distance, voltage)
    print(f"\n2. Distance-Based Model:")
    print(f"   cost = $1.5M/mile * {distance} miles (for 230kV)")
    print(f"   Cost: ${cost2:,.0f}")
    
    # Model 3: Capacity-Only
    tep3 = TEP(data_loader, line_cost_per_mw=200000, cost_model='capacity_only')
    cost3 = tep3._calculate_line_cost(capacity, distance, voltage)
    print(f"\n3. Capacity-Only Model:")
    print(f"   cost = $200k/MW * {capacity}MW")
    print(f"   Cost: ${cost3:,.0f}")
    
    print("\n" + "="*60)
    print("Recommendation: Use 'distance_based' for realistic costs")
    print("Or run sensitivity analysis with 'capacity_distance' model")
    print("="*60)

if __name__ == '__main__':
    compare_cost_models()

