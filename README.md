# CME307: Transmission Expansion Planning

Transmission Expansion Planning (TEP) under load and renewable uncertainty using the IEEE RTS-GMLC dataset.

## Project Structure

```
.
├── data/                           # RTS-GMLC dataset
│   └── RTS_Data/
│       ├── SourceData/            # Network data (buses, branches, generators)
│       └── timeseries_data_files/ # Time series (load, wind, PV, hydro)
├── src/                            # Source code
│   ├── data_loader.py             # Load RTS-GMLC CSV files
│   ├── timeseries_loader.py       # Load time series data
│   ├── dc_opf.py                  # Baseline DC OPF model
│   ├── tep.py                     # Single-period TEP MILP
│   ├── multi_period_tep.py        # Full multi-period TEP
│   ├── simplified_multi_period_tep.py  # Simplified multi-period TEP
│   ├── tep_with_shedding.py       # TEP with load shedding
│   ├── visualize.py               # Plotting utilities
│   ├── run_baseline.py            # Run baseline DC OPF
│   ├── run_tep.py                 # Run single-period TEP
│   ├── run_multi_period_tep.py    # Run full multi-period TEP
│   └── run_simplified_tep.py      # Run simplified multi-period TEP (recommended)
├── results/                        # Output visualizations
├── requirements.txt                # Python dependencies
├── proposal.tex                   # Project proposal
├── RESULTS.md                     # Detailed results report
└── README.md
```

## Installation

```bash
# Create virtual environment
python3 -m venv cme307
source cme307/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup Gurobi license (if using Gurobi)
export GRB_LICENSE_FILE="$(pwd)/gurobi.lic"
# Or run: source setup_gurobi.sh
```

**Requirements:**
- Python 3.13+
- Pyomo 6.9+
- Gurobi 13.0+ (or GLPK as alternative)
- pandas, numpy, matplotlib, networkx, scipy, scikit-learn

## Usage

### Baseline DC OPF (Static Load)
```bash
python src/run_baseline.py
```
Deterministic DC power flow minimizing generation cost. No transmission expansion.

### Single-Period TEP
```bash
python src/run_tep.py
```
MILP with binary investment variables. Minimizes investment + operating cost.

### Multi-Period TEP with Time Series (Recommended)
```bash
python src/run_simplified_tep.py
```
Two-stage approach: (1) Analyze time series to identify peak congestion periods, (2) Solve TEP for aggregated peak load scenario. Integrates hourly load profiles, wind/PV/hydro variability, and load participation factors.

## Models

### Baseline DC OPF
- **Objective**: Minimize generation cost
- **Constraints**: Power balance, DC flow, generator limits, line thermal limits
- **Solver**: Gurobi

### TEP MILP
- **Objective**: Minimize investment cost + operating cost
- **Decision Variables**: Binary variables for line construction
- **Constraints**: DC power flow (Big-M formulation for candidate lines), thermal limits
- **Solver**: Gurobi

### Multi-Period TEP
- **Methodology**: Two-stage approach with representative period selection (peak/avg/low or k-means)
- **Features**: Time-varying load, renewable generation variability, load participation factors
- **Solver**: Gurobi (GLPK fallback if needed)

## Key Features

- **Time Series Integration**: Hourly load, wind, PV, and hydro data
- **Representative Period Selection**: Peak/avg/low method or k-means clustering
- **Load Participation Factors**: Regional-to-bus load distribution
- **Renewable Variability**: Time-varying capacity limits
- **Automatic Candidate Generation**: Based on network topology
- **Load Shedding Option**: Unserved energy with penalty costs
- **Stress Testing**: Line outages and load growth scenarios

## Results

See `RESULTS.md` for detailed analysis. Models output:
- Total system cost (investment + operation)
- Investment costs and payback period
- Operating costs vs baseline
- Lines to build
- Congestion frequency and severity
- Period-by-period analysis
- Robustness under uncertainty

## References

1. Garver, L. L. (1970). "Transmission network estimation using linear programming"
2. Conejo et al. (2006). "Decomposition techniques in mathematical programming"
3. Ruiz & Conejo (2015). "Robust transmission expansion planning"
4. RTS-GMLC Dataset: https://github.com/GridMod/RTS-GMLC

