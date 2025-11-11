# CME307: Transmission Expansion Planning

Transmission Expansion Planning (TEP) under load and renewable uncertainty using the IEEE RTS-GMLC dataset.

## Project Structure

```
.
├── data/                    # RTS-GMLC dataset
│   └── RTS_Data/
├── src/                     # Source code
│   ├── data_loader.py      # Data loading utilities
│   ├── dc_opf.py           # Baseline DC OPF model
│   ├── tep.py              # Transmission Expansion Planning MILP
│   ├── run_baseline.py     # Run baseline DC OPF
│   └── run_tep.py          # Run TEP model
├── requirements.txt         # Python dependencies
└── README.md
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure Gurobi is installed and licensed (or use GLPK as alternative)

## Usage

### Baseline DC OPF

Run the baseline model (no expansion):
```bash
cd src
python run_baseline.py
```

### Transmission Expansion Planning

Run the TEP MILP model:
```bash
cd src
python run_tep.py
```

This will:
1. Run baseline DC OPF
2. Generate candidate transmission lines
3. Solve TEP MILP to determine optimal expansion
4. Compare results

## Models

### Baseline DC OPF
- Deterministic DC power flow
- Minimizes generation cost
- No transmission expansion

### TEP MILP
- Binary variables for line construction decisions
- Minimizes investment + operating cost
- DC power flow constraints
- Thermal limits

## Results

The models output:
- Total system cost
- Investment costs
- Operating costs
- Lines to build
- Congestion levels
- Generation dispatch

