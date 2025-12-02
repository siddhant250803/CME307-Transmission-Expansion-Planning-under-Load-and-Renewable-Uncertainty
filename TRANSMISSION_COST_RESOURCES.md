# Transmission Line Cost Estimation Resources

## Real-World Cost Data Sources

### 1. **MISO (Midcontinent Independent System Operator)**
- **Transmission Cost Estimation Guide for MTEP** (updated annually)
- Provides per-mile cost estimates by voltage level and region
- Available at: MISO website → Planning → Transmission Cost Estimation
- Typical ranges:
  - 230kV: $1-2M per mile
  - 345kV: $1.5-3M per mile
  - 500kV: $2-4M per mile

### 2. **FERC (Federal Energy Regulatory Commission)**
- **Formula Rates**: Utilities file formula rates for transmission service
- Includes: return on investment, O&M, depreciation, taxes
- Database: FERC eLibrary (search "formula rate")
- URL: https://www.ferc.gov/formula-rates-electric-transmission-proceedings

### 3. **NERC (North American Electric Reliability Corporation)**
- **Transmission Planning Documents**: Regional transmission planning studies
- Cost data embedded in planning documents
- URL: https://www.nerc.com/comm/PC/Pages/Transmission-Planning.aspx

### 4. **DOE (Department of Energy)**
- **National Transmission Needs Study** (2023)
- Provides cost ranges for different transmission types
- URL: https://www.energy.gov/gdo/national-transmission-needs-study

### 5. **Academic/Industry References**
- **LBNL (Lawrence Berkeley National Lab)**: Transmission cost databases
- **EIA (Energy Information Administration)**: Historical cost data
- **IEEE Power & Energy Society**: Technical papers with cost data

## Typical Cost Ranges (2024-2025)

### Per-Mile Costs (varies by region, terrain, voltage)
- **138kV**: $0.8-1.5M/mile
- **230kV**: $1.0-2.5M/mile  
- **345kV**: $1.5-3.0M/mile
- **500kV**: $2.0-4.5M/mile
- **765kV**: $3.0-6.0M/mile

### Per-MW Costs (total project, not per mile)
- **230kV line (500MW capacity)**: $100k-$300k per MW
- **345kV line (1000MW capacity)**: $80k-$200k per MW
- **500kV line (2000MW capacity)**: $60k-$150k per MW

**Note**: Per-MW costs decrease with higher capacity due to economies of scale.

## Cost Components

1. **Materials**: Conductors, towers, insulators, transformers (~30-40%)
2. **Labor**: Construction, installation (~20-30%)
3. **Right-of-Way**: Land acquisition, easements (~10-20%)
4. **Engineering**: Design, environmental studies (~5-10%)
5. **Contingency**: Risk buffer (~10-15%)

## Factors Affecting Cost

- **Voltage Level**: Higher voltage = higher cost per mile, but lower cost per MW
- **Distance**: Longer lines have economies of scale (fixed costs amortized)
- **Terrain**: Mountainous/urban = 1.5-2x multiplier
- **Region**: Labor costs vary significantly (CA/NY vs. TX/OK)
- **Capacity**: Higher capacity lines more cost-effective per MW
- **Technology**: HVDC vs. HVAC, overhead vs. underground

## Recommended Cost Model

For your TEP model, consider:

```python
# Option 1: Distance-based (most realistic)
cost_per_mile = {
    230: 1.5e6,  # $1.5M/mile for 230kV
    345: 2.5e6,  # $2.5M/mile for 345kV
    500: 4.0e6   # $4M/mile for 500kV
}
cost = cost_per_mile[voltage_kv] * distance_miles

# Option 2: Fixed + variable
fixed_cost = 10e6  # $10M fixed (substations, etc.)
variable_cost_per_mile = 1.5e6  # $1.5M/mile
cost = fixed_cost + variable_cost_per_mile * distance_miles

# Option 3: Capacity-based (simpler, less accurate)
cost_per_mw = 200000  # $200k/MW (typical for 230-345kV)
cost = cost_per_mw * capacity_mw
```

## How to Use in Your Model

1. **Sensitivity Analysis**: Sweep cost parameters (see `run_cost_sensitivity.py`)
2. **Realistic Calibration**: Use MISO/FERC data to calibrate your cost model
3. **Scenario Analysis**: Test different cost assumptions (low/medium/high)
4. **Break-Even Analysis**: Find cost threshold where expansion becomes attractive

