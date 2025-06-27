# ToySystemQuad.jl

A modular capacity expansion and operations modeling framework for power system analysis with thermal generation outages and parameterized forecast modeling.

## Overview

ToySystemQuad.jl is a comprehensive Julia package for studying capacity expansion equilibrium and operations optimization in power systems. It implements three complementary optimization models with simulated actuals and stochastic forecast scenarios. 

1. **Capacity Expansion Model (CEM)** - Joint investment and operations optimization
2. **Perfect Foresight Operations (DLAC-p)** - One-shot operations with perfect information
3. **DLAC-i Operations** - Rolling horizon operations with imperfect information 


## System Architecture

```
ToySystemEquilibrium/
├── src/
│   ├── SystemConfig.jl          # Technology parameters, system setup, and profile coordination
│   ├── ProfileGeneration.jl     # Simulation of generation, demand, wind forecasts, scenarios
│   ├── OptimizationModels.jl    # Three core optimization models
│   ├── PlottingModule.jl        # Plotting and analysis
│   ├── TestRunner.jl            # Main test system runner with configurable parameters
│   └── ToySystemQuad.jl         # Main module with all exports
├── run_complete_test.jl         # Configurable execution script
├── Project.toml                 # Package dependencies
└── Manifest.toml               # Package versions
```

## Quick Start

### Prerequisites

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Required packages: `JuMP`, `Gurobi`, `CSV`, `DataFrames`, `Plots`, `Statistics`, `Random`

### Running the Complete Test System

#### **Basic Run (Default Configuration)**
```bash
julia --project=. run_complete_test.jl
```

#### **Custom Configuration**
Edit the parameters in `run_complete_test.jl`:

```julia
# Configure system parameters
params = SystemParameters(
    720,     # hours (30 days)
    30,      # days  
    5,       # N (number of generators per technology fleet)
    42,      # random_seed
    10000.0, # load_shed_penalty ($/MWh)
    0.001    # load_shed_quad
)
```

### Programmatic Usage

```julia
include("src/ToySystemQuad.jl")
using .ToySystemQuad

# Create custom parameters
params = SystemParameters(720, 30, 3, 42, 10000.0, 0.001)

# Run complete test system
results = run_complete_test_system(params=params, output_dir="results")

# Access individual model results
cem_result = results["cem"]
pf_result = results["perfect_foresight"] 
dlac_result = results["dlac_i"]
```

## Models

### 1. Capacity Expansion Model (CEM)

Joint optimization of capacity investments and operations with fleet-based availability:

```
min: Σ(c^inv_n * y_n) + Σ(c^fix_n * y_n) + Σ_t Σ_n (c^op_n * p_n,t)
```
- Perfect foresight for planning horizon
- Determines optimal capacity mix

### 2. Perfect Foresight Operations (DLAC-p)

Operations optimization with fixed capacities and perfect information:

```
min: Σ_t Σ_n (c^op_n * p_n,t)
```

- Uses optimal capacities from CEM
- Perfect demand/wind/outage forecasts
- Provides optimal operations benchmark

### 3. DLAC-i Operations

Rolling horizon optimization with imperfect information:

- **N stochastic scenarios** generated independently of actuals
- **Simulated wind forecasts** with error patterns
- Actual values for current period, forecast means for lookahead
- 24-hour lookahead window by default

## Output Files

### CSV Results
- **Capacity Expansion**: `capacity_expansion_results.csv`, `capacity_expansion_operations.csv`, `capacity_expansion_summary.csv`, `capacity_expansion_profits.csv`
- **Perfect Foresight**: `perfect_foresight_operations.csv`, `perfect_foresight_summary.csv`, `perfect_foresight_profits.csv`  
- **DLAC-i**: `dlac_i_operations.csv`, `dlac_i_summary.csv`, `dlac_i_profits.csv`
- **Comparisons**: `three_model_comprehensive_comparison.csv`, `pf_vs_dlac_i_comprehensive_comparison.csv`
- **System Data**: `demand_wind_profiles.csv` (actuals), `demand_wind_outage_profiles.csv` (5 scenarios)
- **Analysis**: `comprehensive_forecast_quality_analysis.csv`, `price_statistics_summary.csv`

### Plots (in `results/plots/`)
- **Price Analysis**: Individual time series, duration curves, comprehensive comparisons
- **Generation**: Dispatch stacks by technology for each model
- **Battery Storage**: 
  - `battery_operations.png` - Charge/discharge decisions by model
  - `battery_soc_comparison.png` - State of charge comparison across models
- **System Profiles**: Demand, wind, and availability patterns
- **Capacity**: Optimal investment comparison

## Profit Analysis

Implements proper Profit-to-Market-Rate (PMR) calculation:

```
π_n(y) = Σ_t (λ_t * p_n,t - c^op_n * p_n,t) - (c^inv_n + c^fix_n) * y_n
PMR_n = π_n(y) / (c^inv_n + c^fix_n) * 100%
```
## Dependencies

- **JuMP.jl**: Optimization modeling framework
- **Gurobi.jl**: Commercial solver (free academic license available)
- **CSV.jl, DataFrames.jl**: Data handling and export
- **Plots.jl**: Comprehensive visualization
- **Statistics.jl, Random.jl**: Mathematical operations and stochastic modeling