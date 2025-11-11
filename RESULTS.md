# CME307 Transmission Expansion Planning - Results

## Project Overview

This project implements Transmission Expansion Planning (TEP) using the IEEE RTS-GMLC dataset. The goal is to determine optimal transmission line investments to minimize total system cost (investment + operation) while satisfying load demands and operational constraints.

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

## Results Summary

### Baseline DC OPF Results
- **Total Operating Cost**: $138,966.59
- **Total Generation**: 8,550 MW
- **Total Load**: 8,550 MW
- **Congested Branches**: 3 lines at 100% utilization
  - Line A27: 500 MW / 500 MW (100%)
  - Line CA-1: 500 MW / 500 MW (100%)
  - Line CB-1: 500 MW / 500 MW (100%)

### TEP Model Results
- **Total System Cost**: $138,966.59
- **Investment Cost**: $0.00
- **Operating Cost**: $138,966.59
- **Lines Built**: 0
- **Operating Cost Savings**: $0.00
- **Net Savings**: $0.00

### Key Findings

1. **No New Lines Built**: The TEP model determined that building new transmission lines is not economically justified with current parameters because:
   - Investment costs (~$1M per MW) are too high relative to congestion relief benefits
   - The system can meet demand constraints without additional capacity
   - Congestion is localized to 3 lines but doesn't cause load shedding

2. **Network Characteristics**:
   - 73 buses across 3 regions
   - 120 existing AC transmission lines
   - 1 HVDC inter-regional link
   - 158 generators (mixed fuel types: coal, natural gas, oil, renewables)

3. **Congestion Patterns**:
   - Inter-regional lines (CA-1, CB-1) are fully utilized
   - Intra-regional line A27 is at capacity
   - Suggests potential for expansion if investment costs decrease or load increases

## Technical Implementation

### Data Structure
```
src/
├── data_loader.py       # Load RTS-GMLC CSV files
├── dc_opf.py           # Baseline DC OPF model
├── tep.py              # TEP MILP model
├── visualize.py        # Plotting utilities
├── run_baseline.py     # Run baseline model
└── run_tep.py          # Run TEP model with comparison
```

### Key Features
- Automatic candidate line generation based on network topology
- Big-M formulation for binary line construction decisions
- DC power flow approximation for computational efficiency
- Modular design for easy extension

## Future Extensions

### Planned Enhancements
1. **Uncertainty Handling**
   - Stochastic programming for renewable generation variability
   - Robust optimization for worst-case scenarios
   - Multiple demand scenarios

2. **Time Series Integration**
   - Hourly dispatch over 8,760 hours
   - Representative day clustering (k-means)
   - Seasonal analysis

3. **Advanced Features**
   - N-1 security constraints
   - Multi-stage expansion planning
   - Renewable integration analysis
   - Storage optimization

4. **Visualization**
   - Network topology plots
   - Congestion heat maps
   - Generation dispatch profiles
   - Investment decision maps

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
# Run baseline DC OPF
python src/run_baseline.py

# Run TEP with comparison
python src/run_tep.py
```

### Requirements
- Python 3.13+
- Pyomo 6.9+
- Gurobi 13.0+ (or GLPK as alternative)
- pandas, numpy, matplotlib, networkx, scipy

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

