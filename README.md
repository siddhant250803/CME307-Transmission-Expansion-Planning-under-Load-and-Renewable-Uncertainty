# CME307: Transmission Expansion Planning

Transmission Expansion Planning (TEP) under load and renewable uncertainty using the IEEE RTS-GMLC dataset.

## Project Structure

```
.
|-- data/
|   `-- RTS_Data/
|       |-- SourceData/               # Network data (buses, branches, generators)
|       `-- timeseries_data_files/    # Time series (load, wind, PV, hydro)
|-- src/
|   |-- core/                        # Models and data utilities
|   |   |-- data_loader.py
|   |   |-- timeseries_loader.py
|   |   |-- dc_opf.py
|   |   |-- tep.py
|   |   |-- multi_period_tep.py
|   |   |-- simplified_multi_period_tep.py
|   |   |-- scenario_robust_tep.py
|   |   |-- tep_with_shedding.py
|   |   `-- example_cost_models.py
|   |-- analysis/                    # Diagnostics and plotting
|   |   |-- analyze_infeasibility.py
|   |   `-- visualize.py
|   `-- scripts/                     # Entry points
|       |-- run_baseline.py
|       |-- run_tep.py
|       |-- run_multi_period_tep.py
|       |-- run_simplified_tep.py
|       |-- run_load_shedding_analysis.py
|       |-- run_cost_sensitivity.py
|       `-- run_robust_tep.py
|-- results/                         # Output visualizations and tables
|-- requirements.txt
|-- RESULTS.md
`-- README.md
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
python src/scripts/run_baseline.py
```
Deterministic DC power flow minimizing generation cost. No transmission expansion.

### Single-Period TEP
```bash
python src/scripts/run_tep.py
```
MILP with binary investment variables. Minimizes investment + operating cost.

### Multi-Period TEP with Time Series (Simplified, Recommended)
```bash
python src/scripts/run_simplified_tep.py
```
Two-stage approach: (1) Analyze time series to identify peak congestion periods, (2) Solve TEP for aggregated peak load scenario. Integrates hourly load profiles, wind/PV/hydro variability, and load participation factors.

### Full Multi-Period TEP (License Permitting)
```bash
python src/scripts/run_multi_period_tep.py
```
Direct MILP across representative periods; may exceed academic WLS limits.

### Load Shedding Stress Test
```bash
python src/scripts/run_load_shedding_analysis.py
```
Derates generators, scales load, and reports unserved energy and its cost.

### Cost Sensitivity Sweep
```bash
python src/scripts/run_cost_sensitivity.py
```
Sweeps line-cost parameters to see when expansion becomes attractive.

### Scenario-Based Robust TEP
```bash
python src/scripts/run_robust_tep.py
```
Single build plan that must satisfy multiple load/renewable/outage scenarios; saves summary to `results/robust_tep_summary.csv`.

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

### Scenario-Based Robust TEP
- **Objective**: Minimize weighted operating cost across scenarios with shared build decisions
- **Features**: Load/renewable scaling per scenario, branch derating, scenario weights
- **Solver**: Gurobi

## Key Features

- **Time Series Integration**: Hourly load, wind, PV, and hydro data
- **Representative Period Selection**: Peak/avg/low method or k-means clustering
- **Load Participation Factors**: Regional-to-bus load distribution
- **Renewable Variability**: Time-varying capacity limits
- **Automatic Candidate Generation**: Based on network topology
- **Load Shedding Option**: Unserved energy with penalty costs
- **Stress Testing**: Line outages and load growth scenarios
- **Robust Optimization**: Scenario-based formulation with shared investment plan

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

