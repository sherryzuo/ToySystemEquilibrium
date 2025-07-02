"""
ToySystemQuad.jl

A modular capacity expansion and operations modeling framework for power system analysis
with thermal outage generation and parametric forecasts .

# Development tip:
# For live code reloading, use Revise.jl and load this module with:
#   using Revise; Revise.includet("src/ToySystemQuad.jl")
# This will automatically track changes in this file and all included submodules.

This package implements three complementary optimization models:
1. Capacity Expansion Model (CEM) - Joint investment and operations optimization
2. Perfect Foresight Operations (DLAC-p) - One-shot operations with perfect information  
3. DLAC-i Operations - Rolling horizon operations with imperfect information using 5 scenarios

The system models a 4-technology power system: Nuclear (baseload fleet), Wind (renewable with forecasts), 
Gas (peaker fleet), and Battery (storage).
"""
module ToySystemQuad

# Include all submodules (order matters for dependencies)
include("ProfileGeneration.jl")
include("SystemConfig.jl") 
include("OptimizationModels.jl")
include("PlottingModule.jl")
include("TestRunner.jl")
include("EquilibriumModule.jl")

# Re-export key functions from submodules
using .SystemConfig
using .ProfileGeneration
using .OptimizationModels
using .PlottingModule
using .TestRunner
using .EquilibriumModule

# Export main functions for external use
export run_complete_test_system

# Core system creation and configuration
export create_complete_toy_system, get_default_system_parameters
export SystemProfiles, SystemParameters, Generator, Battery

# Optimization models  
export solve_capacity_expansion_model, solve_perfect_foresight_operations, solve_dlac_i_operations

# Profile generation functions
export generate_system_profiles, generate_wind_forecast
export generate_fleet_availability, generate_single_nuclear_availability, generate_single_gas_availability

# Analysis and plotting
export calculate_profits_and_save, compute_pmr
export plot_price_time_series, plot_price_duration_curves, plot_combined_price_analysis
export plot_generation_stacks, plot_system_profiles, plot_capacity_comparison, generate_all_plots

# Equilibrium solver
export EquilibriumParameters, solve_equilibrium, run_policy_equilibrium
export save_equilibrium_results, analyze_equilibrium_convergence, resume_from_log
export PolicyFunction, PerfectForesight, DLAC_i

# Validation
export validate_system_configuration, validate_profiles

end # module ToySystemQuad