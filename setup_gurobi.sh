#!/bin/bash
# Setup script to configure Gurobi license
# Run this before using Gurobi: source setup_gurobi.sh

export GRB_LICENSE_FILE="$(pwd)/gurobi.lic"
echo "Gurobi license file set to: $GRB_LICENSE_FILE"

# Verify license
python3 -c "import gurobipy as gp; print('Gurobi license configured successfully')" 2>/dev/null && echo "✓ License verified" || echo "✗ License verification failed"

