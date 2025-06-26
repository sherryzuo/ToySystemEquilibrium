# ToySystemQuad.jl - Documentation and Analysis Guide

## Overview

ToySystemQuad.jl implements a 4-technology power system model designed to test and debug Stochastic Programming Capacity Market (SPCM) equilibrium algorithms and fixed point iteration methods. The system includes Nuclear (baseload), Wind (renewable), Gas (peaker), and Battery storage technologies.

## System Architecture

### Technologies
1. **Nuclear**: Low fuel cost, high investment, baseload operation (90% min generation)
2. **Wind**: Zero fuel cost, moderate investment, variable output (capacity factor driven)
3. **Gas**: High fuel cost, low investment, flexible peaker (20% min generation)
4. **Battery**: Medium investment, provides flexibility and arbitrage (4-hour duration)

### Key Parameters
- **Time Horizon**: 720 hours (30 days)
- **Demand Range**: 450-1400 MW with strong daily/seasonal patterns
- **Wind Profile**: Anticorrelated with demand (high at night, low during peak)
- **Thermal Outages**: Nuclear (99.5% availability), Gas (99.0% availability)

## Mathematical Formulation

### 1. Capacity Expansion Model
Optimizes investment decisions for the actual deterministic profiles:

```
min: Σ(inv_cost_g * cap_g) + Σ(fixed_om_g * cap_g) + Σ_t Σ_g [(fuel_g + vom_g) * gen_g,t] + penalty * load_shed
```

Subject to:
- Power balance: `Σ_g gen_g,t + bat_discharge_t - bat_charge_t + shed_t = demand_t`
- Generation limits with availability factors
- Battery energy/power constraints
- Technology-specific operational constraints

### 2. Operations Models

#### Perfect Foresight Operations
Uses fixed capacities from capacity expansion to solve operations with perfect information about demand, wind, and outages.

#### DLAC-i (Deterministic Look-Ahead with Imperfect Information)
Deterministic (imperfect) lookahead model that:
- Uses scenarios generated around actual profiles, taking the mean of forecasted scenarios in the lookahead horizon at each iteration 
- Generates sequential prices
### 3. Fixed Point Iteration for Equilibrium

The equilibrium condition should either yield zero economic profits or zero capacity:
```
if g >0: /
PMR_g = (EnergyRevenue_g - OpCost_g - FixedCost_g - InvestCost_g) / (InvestCost_g + FixedCost_g) = 0
```

Current iteration scheme:
```julia
adjustment = current_capacities + step_size * (pmr / 100)
next_capacities = softplus(adjustment, smoothing_param)
```

## Current Issues and Debugging Points

### 1. Convergence Problems
- **Oscillation**: Capacities may oscillate around equilibrium
- **Slow convergence**: Step size too small or profit gradients too flat
- **Divergence**: Step size too large or system instabilities

### 2. Mathematical Inconsistencies
- **Price formation differences** differences between optimal prices in the capacity expansion and perfect forecast operations models

### 3. Numerical Issues
- **Anderson acceleration**: Memory and regularization parameters
- **Step size adaptation**: Current heuristic may be suboptimal

## File Structure and Components

### Current Structure (Monolithic)
```
ToySystemQuad.jl:
├── System Data Definition (structs, parameters)
├── Profile Generation (demand, wind, outages, scenarios)
├── Capacity Expansion Model
├── Perfect Foresight Operations
├── DLAC-i Operations  
├── Profit Calculations
└── Complete Solve Function
```

### Current Modular Structure ✅ IMPLEMENTED
```
ToySystemQuad.jl:
├── System Data Definition (structs: Generator, Battery, SystemParameters)
├── System Configuration Module 
│   ├── get_default_system_parameters()
│   ├── create_nuclear_generator(), create_wind_generator(), create_gas_generator()
│   ├── create_battery_storage(), create_toy_system()
│   └── validate_system_configuration()
├── Profile Generation Module
│   ├── get_base_demand_profile(), get_base_wind_profile()
│   ├── generate_demand_profile(), generate_wind_profile()
│   ├── generate_nuclear_availability(), generate_gas_availability()
│   ├── generate_scenarios(), create_actual_and_scenarios()
│   └── validate_profiles()
├── Optimization Models Module
│   ├── create_base_optimization_model()
│   ├── add_power_balance_constraints!(), add_generation_constraints!()
│   ├── add_battery_constraints!()
│   ├── compute_operational_costs(), compute_investment_costs()
│   ├── create_optimization_result(), save_capacity_results()
│   └── solve_capacity_expansion() [REFACTORED]
├── Equilibrium Analysis and Convergence Diagnostics Module
│   ├── compute_pmr() - Profit-to-Market-Rate calculations
│   ├── analyze_convergence_properties() - Oscillation and rate analysis
│   ├── diagnose_convergence_issues() - Issue identification and suggestions
│   ├── create_convergence_summary() - Comprehensive reporting
│   └── compute_equilibrium_jacobian() - Stability analysis [PLACEHOLDER]
└── Original Models (Perfect Foresight, DLAC-i, etc.) [PENDING REFACTOR]
```

## Usage Examples

### Basic System Setup and Validation
```julia
# Create system with validation
generators, battery = create_toy_system()
params = get_default_system_parameters()
validate_system_configuration(generators, battery, params)

# Generate and validate profiles
actual_demand, actual_wind, nuclear_availability, gas_availability,
demand_scenarios, wind_scenarios, nuclear_avail_scenarios, gas_avail_scenarios = create_actual_and_scenarios(params)
validate_profiles(actual_demand, actual_wind, nuclear_availability, gas_availability, params)
```

### Modular Capacity Expansion
```julia
# Run capacity expansion with new modular approach
generators, battery = create_toy_system()
result = solve_capacity_expansion(generators, battery; output_dir="results/")

# Analyze results
println("Investment costs: $(result["investment_cost"])")
println("Operational costs: $(result["operational_cost"])")
println("Final capacities: $(result["capacity"])")
```

### Convergence Analysis and Debugging
```julia
# After running equilibrium iterations (placeholder - requires integration)
capacity_history = [cap1, cap2, cap3, ...]  # From equilibrium iteration
pmr_history = [pmr1, pmr2, pmr3, ...]       # From equilibrium iteration
step_size_history = [step1, step2, step3, ...]

# Analyze convergence
convergence_summary = create_convergence_summary(capacity_history, pmr_history, step_size_history, generators, battery)
println(convergence_summary)

# Diagnose issues
diagnostics = diagnose_convergence_issues(capacity_history, pmr_history, step_size_history)
for issue in diagnostics["issues"]
    println("⚠️  $issue")
end
for suggestion in diagnostics["suggestions"]
    println("💡 $suggestion")
end
```

### Equilibrium Analysis (Existing Framework)
```julia
# Run with existing equilibrium framework
using .CapacityEquilibrium
context = initialize_toy_context()
result = compute_equilibrium(context, "dlac-i"; max_iterations=100)
```

## Key Metrics for Debugging

### Enhanced Convergence Indicators (NEW)
- **Max PMR**: `maximum(abs.(pmr))` - should approach 0
- **Capacity Changes**: `norm(new_caps - old_caps)` - should decrease
- **Oscillation Metric**: Detects alternating capacity changes
- **Convergence Rate**: Exponential decay rate of PMR
- **Step Size Adaptation**: Tracks dynamic step size adjustments

### Automated Diagnostics (NEW)
The new convergence diagnostics module provides:
- **Issue Detection**: Automatically identifies slow convergence, oscillation, stagnation
- **Remedy Suggestions**: Specific recommendations for fixing convergence problems
- **Convergence Rate Analysis**: Mathematical analysis of convergence properties
- **Comprehensive Reporting**: Formatted summaries with actionable insights

### Profit Analysis (NEW)
- **Modular PMR Calculation**: Separate function for computing Profit-to-Market-Rate
- **Revenue Breakdown**: Detailed analysis of energy revenue, costs, and profits
- **Battery Arbitrage**: Specific calculations for battery storage economics
- **Technology Comparison**: Side-by-side profit analysis across technologies