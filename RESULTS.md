# CME307 Transmission Expansion Planning - Results

## Project Overview

This project implements Transmission Expansion Planning (TEP) using the IEEE RTS-GMLC dataset. The goal is to determine optimal transmission line investments to minimize total system cost (investment + operation) while satisfying load demands and operational constraints. The model accounts for uncertainty in renewable generation and load, ensuring reliability under realistic variability.

## Models Implemented

### 1. Baseline DC Optimal Power Flow (OPF)
- **Objective**: Minimize generation cost
- **Constraints**: Power balance, DC power flow, generator limits, line thermal limits
- **Solver**: Gurobi
- **Status**: ✅ Implemented and validated

### 2. Transmission Expansion Planning (TEP) MILP
- **Objective**: Minimize investment cost + operating cost
- **Decision Variables**: Binary variables for line construction (build/don't build)
- **Constraints**: DC power flow (with Big-M formulation for candidate lines), thermal limits, power balance
- **Solver**: Gurobi
- **Status**: ✅ Implemented and validated

### 3. Multi-Period TEP with Time Series
- **Methodology**: Two-stage approach: (1) Analyze time series to identify peak congestion periods (peak/avg/low or k-means), (2) Solve TEP for aggregated peak load scenario
- **Features**: Hourly load profiles, wind/PV/hydro variability, load participation factors
- **Solver**: Gurobi (GLPK fallback if needed)
- **Status**: ✅ Implemented and validated

## Results Summary

### Baseline DC OPF Results (Static Load)
- **Total Operating Cost**: $138,966.59
- **Total Generation**: 8,550 MW
- **Total Load**: 8,550 MW
- **Congested Branches**: 3 lines at 100% utilization
  - Line A27: 500 MW / 500 MW (100%)
  - Line CA-1: 500 MW / 500 MW (100%)
  - Line CB-1: 500 MW / 500 MW (100%)

### Single-Period TEP Model Results
- **Total System Cost**: $138,966.59
- **Investment Cost**: $0.00
- **Operating Cost**: $138,966.59
- **Lines Built**: 0
- **Finding**: No expansion needed with static load scenario

### Multi-Period TEP with Time Series Data

**Methodology**: 
- Analyzed time series data (hourly load, wind, PV, hydro) for 2020
- Identified peak congestion periods using representative day analysis
- Implemented simplified two-stage approach:
  1. Stage 1: Identify peak load periods with highest congestion
  2. Stage 2: Solve TEP for aggregated peak load scenario

**Time Series Analysis**:
- Load varies significantly: ~3,800-6,100 MW across periods
- Peak periods identified: Periods 10-11 (highest load ~4,084 MW)
- Renewable generation shows high variability (wind: 0-800 MW, PV: 0-200 MW)

**Multi-Period TEP Results** (Peak Load Scenario):
- **Peak Load Analyzed**: 6,161 MW (150% of base + new load)
- **Total System Cost**: $129,078.68
- **Investment Cost**: $0.00
- **Operating Cost**: $129,078.68
- **Load Shedding**: 0 MW (system can meet all demand)
- **Lines Built**: 0

### Key Findings

1. **System Robustness**: The RTS-GMLC system demonstrates significant redundancy:
   - Can handle 150% of peak load without expansion
   - Even with targeted line outages (30% capacity on congested lines), system remains feasible
   - No load shedding required even under extreme stress scenarios

2. **Time Series Integration Success**:
   - Successfully integrated hourly load profiles from time series data
   - Load participation factors correctly distribute regional load to buses
   - Representative period selection (peak/avg/low) captures load variability
   - Renewable generation variability properly modeled

3. **Expansion Economics**:
   - With line costs of $200k-$500k per MW, expansion not justified
   - System has sufficient capacity margin for current and near-term future loads
   - Expansion would be beneficial if:
     - Load growth exceeds 50% of current peak
     - Investment costs decrease significantly
     - New large load centers are added (>500 MW)

4. **Network Characteristics**:
   - 73 buses across 3 regions
   - 120 existing AC transmission lines
   - 1 HVDC inter-regional link
   - 158 generators with total capacity ~10,500 MW
   - Significant generation reserve margin

5. **Congestion Patterns**:
   - Inter-regional lines (CA-1, CB-1) are fully utilized under base load
   - Intra-regional line A27 at capacity
   - Congestion manageable through generation redispatch
   - No transmission bottlenecks requiring immediate expansion

## Technical Implementation

### Data Structure
```
src/
├── data_loader.py              # Load RTS-GMLC CSV files
├── timeseries_loader.py        # Load time series data (load, wind, PV, hydro)
├── dc_opf.py                  # Baseline DC OPF model
├── tep.py                     # Single-period TEP MILP model
├── multi_period_tep.py        # Full multi-period TEP (large-scale)
├── simplified_multi_period_tep.py  # Simplified multi-period TEP (works within license limits)
├── tep_with_shedding.py       # TEP with load shedding option
├── visualize.py               # Plotting utilities
├── run_baseline.py            # Run baseline model
├── run_tep.py                 # Run single-period TEP
├── run_multi_period_tep.py    # Run full multi-period TEP
└── run_simplified_tep.py      # Run simplified multi-period TEP (recommended)
```

### Key Features
- **Time Series Integration**: Loads hourly load, wind, PV, and hydro data from CSV files
- **Representative Period Selection**: Peak/avg/low load periods or k-means clustering
- **Load Participation Factors**: Distributes regional load to individual buses
- **Renewable Generation Modeling**: Time-varying capacity limits for wind, PV, hydro
- **Automatic Candidate Generation**: Based on network topology and geographic proximity
- **Big-M Formulation**: Binary line construction decisions with proper DC flow constraints
- **Load Shedding Option**: Allows unserved energy with penalty cost for feasibility
- **Stress Testing**: Simulates line outages and load growth scenarios
- **DC Power Flow**: Computational efficiency for large-scale problems
- **Modular Design**: Easy extension to stochastic/robust optimization

## Implementation Highlights

### Completed Enhancements
1. **Time Series Integration** ✅
   - Hourly load profiles from regional time series
   - Wind, PV, and hydro generation variability
   - Representative period selection (peak/avg/low method)
   - Load participation factor calculation

2. **Multi-Period Analysis** ✅
   - Peak congestion period identification
   - Aggregated peak load scenario
   - Period-by-period cost analysis

3. **Advanced Modeling** ✅
   - Load shedding with penalty costs
   - Simulated line outages for stress testing
   - Targeted congestion relief analysis

4. **Visualization** ✅
   - Network topology plots with congestion highlighting
   - Congestion bar charts
   - Generation mix pie charts

### Future Extensions
1. **Full Stochastic Programming**
   - Multiple renewable generation scenarios
   - Probabilistic load forecasts
   - Two-stage stochastic TEP

2. **Extended Time Horizon**
   - Full year analysis (8,760 hours) with clustering
   - Seasonal representative days
   - Multi-year expansion planning

3. **Advanced Features**
   - N-1 security constraints
   - Multi-stage expansion (phased construction)
   - Storage optimization integration
   - HVDC candidate lines

## Installation & Usage

### Setup
```bash
# Create virtual environment
python3 -m venv cme307
source cme307/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Models
```bash
# Run baseline DC OPF (static load)
python src/run_baseline.py

# Run single-period TEP
python src/run_tep.py

# Run multi-period TEP with time series data (RECOMMENDED)
python src/run_simplified_tep.py
```

**Recommended**: Use `run_simplified_tep.py` which:
- Analyzes time series data to identify peak periods
- Uses representative periods to reduce computational complexity
- Incorporates renewable generation variability
- Provides meaningful expansion recommendations

### Requirements
- Python 3.13+
- Pyomo 6.9+
- Gurobi 13.0+ (or GLPK as alternative)
- pandas, numpy, matplotlib, networkx, scipy, scikit-learn

## References

1. Garver, L. L. (1970). "Transmission network estimation using linear programming"
2. Conejo et al. (2006). "Decomposition techniques in mathematical programming"
3. Ruiz & Conejo (2015). "Robust transmission expansion planning"
4. RTS-GMLC Dataset: https://github.com/GridMod/RTS-GMLC

## Team
- Edouard Rabasse
- Siddhant Sukhani

## Course
CME307: Optimization - Stanford University
November 2025

