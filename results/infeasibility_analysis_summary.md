# Infeasibility Analysis Summary - Periods 1-6

## Root Cause Identified

**Primary Issue**: Minimum generation constraints exceed load requirements

### Period-by-Period Analysis:

| Period | Load (MW) | Min Gen (MW) | Excess (MW) | Thermal Min (MW) | Thermal Needed (MW) | Renewable (MW) | Feasibility Boundary |
|--------|-----------|--------------|-------------|------------------|---------------------|----------------|---------------------|
| 1 | 3,403.89 | 3,775.00 | 371.11 | 1,279.20 | 908.09 | 2,495.80 | 10.0% reduction |
| 2 | 3,335.04 | 3,775.00 | 439.96 | 1,204.50 | 764.54 | 2,570.50 | 12.5% reduction |
| 3 | 3,316.34 | 3,775.00 | 458.66 | 1,436.30 | 977.64 | 2,338.70 | 12.5% reduction |
| 4 | 3,327.60 | 3,775.00 | 447.40 | 1,737.90 | 1,290.50 | 2,037.10 | 12.5% reduction |
| 5 | 3,450.00 | 3,775.00 | 325.00 | 1,712.10 | 1,387.10 | 2,062.90 | 10.0% reduction |
| 6 | 3,648.66 | 3,775.00 | 126.34 | 1,771.60 | 1,645.26 | 2,003.40 | 5.0% reduction |

**Key Observations**:
- Period 6 is closest to feasibility (only 126 MW excess, needs 5% reduction)
- Periods 2-4 require the most reduction (12.5%)
- Renewable availability varies: 2,003-2,571 MW (wind/PV/hydro)
- Thermal minimum generation is the binding constraint

## Key Findings

### 1. IIS (Irreducible Infeasible Set) Analysis
The minimal set of conflicting constraints is:
- **Power Balance Constraint**: Requires generation = load
- **Minimum Generation Constraints**: Require generation ≥ PMin for each generator
- **Conflict**: Sum of PMin > Total Load

### 2. Parametric Load Analysis
- **Result**: Even with **50% load reduction**, all periods (1-6) remain infeasible
- **Conclusion**: The issue is **not load level** but **minimum generation constraints**
- The system cannot reduce generation below minimum thresholds regardless of load

### 3. Parametric Generation Analysis
- **Feasibility Boundaries Identified** (minimum generation reduction needed):
  - **Period 1**: 10.0% reduction → Min Gen: 3,397.5 MW (from 3,775 MW)
  - **Period 2**: 12.5% reduction → Min Gen: 3,303.1 MW
  - **Period 3**: 12.5% reduction → Min Gen: 3,303.1 MW
  - **Period 4**: 12.5% reduction → Min Gen: 3,303.1 MW
  - **Period 5**: 10.0% reduction → Min Gen: 3,397.5 MW
  - **Period 6**: 5.0% reduction → Min Gen: 3,586.2 MW (easiest to fix)
- **Average reduction needed**: ~10-12.5% (377-472 MW reduction)
- **Operating costs at feasibility boundary**: $112,944 - $122,625 per period

## Why This Happens

1. **Early Morning Hours (Periods 1-6)**:
   - Low load (3,400-3,800 MW)
   - PV generation = 0 MW (no sunlight)
   - Wind generation varies (0-2,200 MW)
   - Hydro generation limited (~300 MW)

2. **Thermal Generator Minimums**:
   - Many thermal generators have high PMin (minimum stable operating level)
   - Cannot be turned off or reduced below PMin
   - Sum of all PMin exceeds early morning load

3. **System Design**:
   - System designed for peak loads (~4,000-4,100 MW)
   - Minimum generation constraints ensure system stability
   - But create infeasibility during low-load periods

## Solutions to Make Periods Feasible

### Option 1: Reduce Minimum Generation
- **Required**: 5-12.5% reduction in minimum generation (126-472 MW)
- **Period 6**: Easiest case - only 5% reduction needed (189 MW)
- **Periods 2-4**: Most challenging - 12.5% reduction needed (472 MW)
- Allows generators to operate below current minimums
- May require operational changes or generator retrofits
- **Cost at boundary**: $112,944 - $122,625 per period (vs. infeasible)

### Option 2: Increase Load
- Add flexible loads (e.g., storage charging, demand response)
- Shift load from other periods to early morning
- Not practical for real systems

### Option 3: Allow Load Shedding
- Model allows unserved energy with penalty cost
- **Required shedding**: 126-459 MW depending on period
- **Period 6**: Minimum shedding needed (126 MW)
- **Periods 2-4**: Maximum shedding needed (459 MW)
- **Cost estimate**: $1.26M - $4.59M per period (at $10k/MW penalty)
- **Annual cost** (6 periods): ~$7.6M - $27.5M if all periods require shedding

### Option 4: Generator Commitment
- Allow some generators to be offline (binary commitment)
- Currently all generators assumed online
- Would require unit commitment model extension

### Option 5: Storage/Reserves
- Use storage to absorb excess generation
- Charge storage during low-load periods
- Discharge during high-load periods

## Recommended Approach

For the TEP model, **Option 1 (Reduce Minimum Generation)** is most cost-effective:
- **Required reduction**: 5-12.5% (126-472 MW)
- **Operating cost at boundary**: $112,944 - $122,625 per period
- **No penalty costs** (unlike load shedding)
- **Period 6** is easiest to fix (only 5% reduction)

**Alternative: Option 3 (Load Shedding)** for feasibility testing:
- Already implemented in `tep_with_shedding.py`
- Allows feasibility while penalizing unserved energy
- Cost: $1.26M - $4.59M per period (penalty)
- Useful for comparing cost of load shedding vs. system modifications

## Quantitative Results Summary

### Feasibility Boundaries (from parametric analysis):
- **Minimum reduction needed**: 5% (Period 6) to 12.5% (Periods 2-4)
- **MW reduction range**: 126-472 MW
- **Operating cost at boundary**: $112,944 - $122,625 per period
- **All periods become feasible** with sufficient minimum generation reduction

### Load Reduction Analysis:
- **Tested**: 0-50% load reduction
- **Result**: All periods remain infeasible even at 50% reduction
- **Conclusion**: Load level is not the constraint - minimum generation is

### IIS Analysis:
- **Root cause confirmed**: Minimum generation constraints conflict with power balance
- **All periods**: Same root cause (min_gen_exceeds_load)
- **Thermal generators**: Primary source of excess minimum generation

## Next Steps

1. **For TEP Model**: Use reduced minimum generation (5-12.5% reduction) for periods 1-6
2. **Alternative**: Run TEP with load shedding enabled to compare costs
3. **Cost Comparison**:
   - Load shedding: $1.26M - $4.59M per period (penalty)
   - Reduced min gen: $112,944 - $122,625 per period (operating cost only)
   - **Recommendation**: Reduce minimum generation (much lower cost)
4. **Implementation**: Modify generator PMin values in data or model for periods 1-6

