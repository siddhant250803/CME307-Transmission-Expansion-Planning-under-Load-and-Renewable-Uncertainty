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

### Cost Sensitivity Analysis

**Methodology**:
- Swept line cost parameter from $100k to $1M per MW (10 points on logarithmic scale)
- Tested each cost level with TEP model using static load scenario
- Baseline operating cost: $138,966.59
- Analyzed break-even point where expansion becomes attractive

**Results**:
- **All cost levels tested**: No expansion built (0 lines)
- **Cost range tested**: $100,000 - $1,000,000 per MW
- **Operating cost**: Constant at $138,966.59 (no savings from expansion)
- **Net benefit**: $0 at all cost levels
- **Key Insight**: System has no congestion that expansion can economically relieve under static load

**Cost Model Comparison** (for 500MW, 100-mile, 230kV line):
- **Capacity-Distance Model** (original): $804.7M
  - Formula: `cost = $1M/MW × 500MW × (160.9km / 100km)`
  - Note: This model double-counts capacity and distance
- **Distance-Based Model** (realistic): $150M
  - Formula: `cost = $1.5M/mile × 100 miles` (for 230kV)
  - Based on MISO/FERC industry data
- **Capacity-Only Model**: $100M
  - Formula: `cost = $200k/MW × 500MW`
  - Ignores distance, simpler but less accurate

**Recommendation**: Use distance-based model for realistic cost estimation, or run sensitivity analysis to find cost thresholds.

### Multi-Period TEP with Time Series Data

**Methodology**: 
- Analyzed time series data (hourly load, wind, PV, hydro) for 2020
- Identified peak congestion periods using representative day analysis
- Implemented simplified two-stage approach:
  1. Stage 1: Identify peak load periods with highest congestion
  2. Stage 2: Solve TEP for aggregated peak load scenario

**Time Series Analysis**:
- Analyzed periods 1-12 to identify peak congestion
- Load varies significantly: ~3,800-4,100 MW across analyzed periods
- **Period Feasibility**:
  - Periods 1-6: Infeasible due to **minimum generation constraints exceeding load**
    - Root cause: Total minimum generation (3,775 MW) > Total load (3,400-3,700 MW)
    - Excess minimum generation: ~371 MW
    - Even with 50% load reduction, periods remain infeasible
    - Solution: Allow load shedding or reduce minimum generation constraints
    - See `results/infeasibility_analysis_summary.md` for detailed analysis
  - Periods 7-12: Feasible with successful DC OPF solutions
  - Period 7: Load = 3,857.1 MW, 0 congested branches
  - Period 8: Load = 3,911.3 MW, 0 congested branches
  - Period 9: Load = 3,991.3 MW, 0 congested branches
  - Period 10: Load = 4,052.0 MW, 0 congested branches
  - Period 11: Load = 4,084.1 MW, 0 congested branches
  - Period 12: Load = 4,047.2 MW, 0 congested branches
- **Peak Periods Identified**: Periods 10-11 (highest loads: 4,052 MW and 4,084 MW)
- Renewable generation shows high variability (wind: 0-800 MW, PV: 0-200 MW)

**Multi-Period TEP Results** (Simplified Two-Stage Approach):
- **Method**: Peak period identification + aggregated peak load scenario
- **Peak Periods Identified**: Periods 10-11 (loads: 4,052 MW and 4,084 MW)
- **Stress Test Scenario**: 
  - Base peak load: ~4,084 MW (Period 11)
  - Applied 150% stress multiplier
  - Added 500 MW new load at bus 122 (simulating new industrial area)
  - **Total Peak Load**: 6,661.1 MW
- **Model Performance**:
  - Model size: 798 constraints, 440 variables (432 continuous, 8 binary), 1,532 nonzeros
  - Solver: Gurobi with 1% optimality gap
  - Solution time: <0.01 seconds (optimal solution found immediately)
  - Presolve: Removed 640 rows and 231 columns (80% reduction)
- **Results**:
- **Total System Cost**: $129,078.68
- **Investment Cost**: $0.00
- **Operating Cost**: $129,078.68
- **Load Shedding**: 0 MW (system can meet all demand even under extreme stress)
- **Lines Built**: 0
- **Key Finding**: Even with 150% load stress + 500 MW new load, no expansion needed

### Supply Shortage Stress Test (Load Shedding Model)

Reviewer question (a) asked how we handle periods where load exceeds available supply.  
We implemented a DC-OPF variant with explicit load shedding using `TEPWithLoadShedding`
and created the runnable script `src/run_load_shedding_analysis.py`.  
Each run temporarily (i) derates every generator to 20% of its nameplate capacity, (ii) scales nodal loads by 40%,
and (iii) enforces a \$50k/MWh penalty on unserved energy. This combination creates an acute shortage
that forces the model to use the `load_shed` slack variables. The script prints the period-by-period summary
and writes the results to `results/load_shedding_periods.csv`.

| Period | Total Load (MW) | Generation (MW) | Load Shed (MW) | Load Shed (%) | Load Shed Cost (\$M) |
| --- | --- | --- | --- | --- | --- |
| 1 | 4765.4 | 2910.0 | 1855.5 | 38.9% | 92.77 |
| 2 | 4669.1 | 2910.0 | 1759.1 | 37.7% | 87.96 |
| 3 | 4642.9 | 2910.0 | 1732.9 | 37.3% | 86.65 |
| 4 | 4658.6 | 2910.0 | 1748.7 | 37.5% | 87.43 |
| 5 | 4830.0 | 2910.0 | 1920.0 | 39.8% | 96.00 |
 | 6 | 5108.1 | 2910.0 | 2198.2 | 43.0% | 109.91 |

Interpretation:
- The solution keeps every remaining generator at the new 2910 MW ceiling and sheds the balance of load,
demonstrating that shortages are captured as explicit decision variables in the optimization problem.
- The \$50k/MWh penalty translates the shortage into economic terms, which can be contrasted with
candidate line investments or alternative mitigation actions.
- Because the shortage data are exported to CSV, we can easily plug the numbers into the final report
and discuss how different penalty levels or derating assumptions affect the amount of unserved load.

### Scenario-Based Robust TEP

To answer reviewer question (b) about robustness, we prototyped a scenario-based formulation
(`src/scenario_robust_tep.py`) and a driver script `src/run_robust_tep.py`. We defined three scenarios:

| Scenario | Load scale | Renewable scale | Branch stress | Weight | Description |
| --- | --- | --- | --- | --- | --- |
| base | 1.00 | 1.00 | 1.00 on all lines | 0.40 | Nominal demand and availability |
| high\_load\_low\_renew | 1.20 | 0.70 | 0.85 on all lines, 0.40 on A27/CA-1/CB-1 | 0.35 | Load surge with renewable drought and corridor outages |
| low\_load\_high\_renew | 0.90 | 1.15 | 1.05 on all lines | 0.25 | Mild demand with surplus renewables |

Each scenario shares the same binary build variables, so the MILP selects one investment plan that satisfies
all three simultaneously. We lowered the candidate line cost to \$120k/MW to reflect the more urgent
reinforcement need under the stressed case. Results (saved in `results/robust_tep_summary.csv`) are:

| Scenario | Total load (MW) | Total generation (MW) | Operating cost (\$) |
| --- | --- | --- | --- |
| base | 8,550 | 8,550 | 138,925.58 |
| high\_load\_low\_renew | 10,260 | 10,260 | 225,753.27 |
| low\_load\_high\_renew | 7,695 | 7,695 | 129,078.68 |

The robust solve built two lines (Bus 101 → 106 and Bus 101 → 117, both 200 MW) at a combined
\$61.3 M investment. Those reinforcements relieve the derated interties in the high-load scenario and
ensure feasibility without resorting to load shedding. The weighted objective is \$61.46 M, dominated by
the capital expenditures—highlighting how the robustness requirement drives proactive expansion, unlike the
single-scenario runs where no new lines were needed.

### Key Findings

1. **System Robustness**: The RTS-GMLC system demonstrates significant redundancy:
   - Can handle 150% of peak load (6,661 MW) + 500 MW new load without expansion
   - Even with targeted line outages (30% capacity on congested lines), system remains feasible
   - No load shedding required even under extreme stress scenarios (150% load + new industrial area)
   - System successfully meets demand in peak periods (10-11) with loads ~4,050-4,085 MW

2. **Time Series Integration Success**:
   - Successfully integrated hourly load profiles from time series data
   - Load participation factors correctly distribute regional load to buses
   - Peak period identification successfully finds periods with highest load (Periods 10-11)
   - Two-stage approach effectively handles time series complexity:
     * Stage 1: Analyzes multiple periods to identify peak congestion
     * Stage 2: Solves TEP for aggregated peak scenario
   - Note: Some early periods (1-6) are infeasible, likely due to low renewable generation during those hours

3. **Expansion Economics**:
   - **Cost Sensitivity Analysis**: Tested line costs from $100k-$1M per MW (10 points on log scale)
   - **Result**: No expansion built at any cost level tested
   - **Finding**: Even at very low costs ($100k/MW), expansion not economically justified
   - System has sufficient capacity margin for current and near-term future loads
   - Operating cost savings from expansion are negligible (baseline = $138,966.59)
   - Expansion would be beneficial if:
     - Load growth exceeds 50% of current peak
     - Investment costs decrease significantly (<$100k/MW)
     - New large load centers are added (>500 MW)
     - Significant congestion exists that expansion can relieve

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
├── tep.py                     # Single-period TEP MILP model (with multiple cost models)
├── multi_period_tep.py        # Full multi-period TEP (large-scale)
├── simplified_multi_period_tep.py  # Simplified multi-period TEP (works within license limits)
├── tep_with_shedding.py       # TEP with load shedding option
├── visualize.py               # Plotting utilities
├── run_baseline.py            # Run baseline model
├── run_tep.py                 # Run single-period TEP
├── run_multi_period_tep.py    # Run full multi-period TEP
├── run_simplified_tep.py      # Run simplified multi-period TEP (recommended)
├── run_cost_sensitivity.py    # Cost sensitivity analysis (sweep cost parameters)
└── example_cost_models.py     # Compare different cost calculation models
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
- **Multiple Cost Models**: 
  - Capacity-distance model (original)
  - Distance-based model (realistic, uses MISO/FERC data)
  - Capacity-only model (simplified)
- **Cost Sensitivity Analysis**: Sweep cost parameters to find break-even points
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

4. **Cost Analysis** ✅
   - Multiple cost models (capacity-distance, distance-based, capacity-only)
   - Cost sensitivity analysis with parameter sweeping
   - Break-even point identification
   - Realistic cost calibration using MISO/FERC data

5. **Visualization** ✅
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

