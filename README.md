# ToySystemQuad.jl

A modular capacity expansion and operations modeling framework for power system analysis.

## Overview

ToySystemQuad.jl is a comprehensive Julia package for studying capacity expansion equilibrium and operations optimization in power systems. It implements three complementary optimization models:

1. **Capacity Expansion Model (CEM)** - Joint investment and operations optimization
2. **Perfect Foresight Operations (DLAC-p)** - One-shot operations with perfect information
3. **DLAC-i Operations** - Rolling horizon operations with imperfect information

## Features

- **4-Technology System**: Nuclear (baseload), Wind (renewable), Gas (peaker), Battery (storage)
- **Full Modular Architecture**: Separate modules for configuration, profiles, optimization, diagnostics, and visualization
- **Comprehensive Analysis**: PMR calculations, convergence diagnostics, model comparisons
- **Rich Visualizations**: Price analysis, generation stacks, duration curves, system profiles
- **CSV Export**: Detailed results for all models including operations, profits, and comparisons

## System Architecture

```
TestSys/
├── SystemConfig.jl          # Technology parameters and system setup
├── ProfileGeneration.jl     # Demand, wind, and outage profile generation
├── OptimizationModels.jl    # Three core optimization models
├── ConvergenceDiagnostics.jl # PMR calculation and convergence analysis
├── VisualizationTools.jl    # Legacy plotting tools
├── PlottingModule.jl        # Comprehensive plotting and analysis
├── TestRunner.jl            # Main test system runner
└── run_complete_test.jl     # Execution script
```

## Quick Start

### Prerequisites

```julia
using Pkg
Pkg.add(["JuMP", "Gurobi", "CSV", "DataFrames", "Plots", "Statistics"])
```

### Running the Complete Test System

```bash
julia run_complete_test.jl
```

This will:
- Run all three optimization models with 720-hour (30-day) horizon
- Save detailed CSV results to `results/` directory
- Generate comprehensive plots in `results/plots/`
- Perform model comparisons and analysis

### Basic Usage

```julia
include("TestRunner.jl")
using .TestRunner

# Run complete test system
results = run_complete_test_system(output_dir="results")

# Results contain all three model outputs
cem_result = results["cem"]
pf_result = results["perfect_foresight"] 
dlac_result = results["dlac_i"]
```

## Models

### 1. Capacity Expansion Model (CEM)

Joint optimization of capacity investments and operations:

```
min: Σ(c^inv_n * y_n) + Σ(c^fix_n * y_n) + Σ_t Σ_n (c^op_n * p_n,t)
```

Subject to power balance, generation limits, and storage constraints.

### 2. Perfect Foresight Operations (PF)

Operations optimization with fixed capacities and perfect information:

```
min: Σ_t Σ_n (c^op_n * p_n,t)
```

Uses optimal capacities from CEM with perfect demand/wind/outage forecasts.

### 3. DLAC-i Operations

Rolling horizon optimization with imperfect information:
- Actual values for current period
- Mean forecasts from scenarios for lookahead horizon
- 24-hour lookahead window by default

## Output Files

### CSV Results
- `capacity_expansion_*`: Capacity results, operations, summary, profits
- `perfect_foresight_*`: Operations, summary, profits
- `dlac_i_*`: Operations, summary, profits
- `*_comparison.csv`: Model comparisons and analysis
- `*_profiles.csv`: System demand, wind, and availability data

### Plots
- Individual price time series for each model
- Combined price duration curves
- Comprehensive price analysis with differences
- Generation dispatch stacks
- System profiles and capacity comparison

## Key Technologies

- **Nuclear**: 1200 MW max, $120k/MW/yr investment, $12/MWh fuel
- **Wind**: 1500 MW max, $85k/MW/yr investment, $0/MWh fuel, variable output
- **Gas**: 1000 MW max, $55k/MW/yr investment, $90/MWh fuel, flexible peaker
- **Battery**: 800 MW/3200 MWh max, $95k/MW/yr + $100/MWh/yr investment, 4-hour duration

## Profit Analysis

Implements proper Profit-to-Market-Rate (PMR) calculation:

```
π_n(y) = Σ_t (λ_t * p_n,t - c^op_n * p_n,t) - (c^inv_n + c^fix_n) * y_n
PMR_n = π_n(y) / (c^inv_n + c^fix_n) * 100%
```

Zero PMR indicates capacity market equilibrium.

## Convergence Diagnostics

- Oscillation detection in capacity evolution
- Convergence rate estimation
- Step size adaptation analysis
- Equilibrium stability assessment

## Dependencies

- **JuMP.jl**: Optimization modeling
- **Gurobi.jl**: Commercial solver (free academic license available)
- **CSV.jl, DataFrames.jl**: Data handling
- **Plots.jl**: Visualization
- **Statistics.jl, LinearAlgebra.jl**: Mathematical operations

## License

MIT License

## Contributing

This is a research tool for power systems analysis. Contributions welcome for:
- Additional generator types
- Enhanced convergence algorithms
- Alternative solver interfaces
- Extended visualization capabilities

## Citation

If you use ToySystemQuad.jl in your research, please cite:

```
ToySystemQuad.jl: A Modular Framework for Power System Capacity Expansion Analysis
```