"""
ToySystemQuad.jl

A modular capacity expansion and operations modeling framework for power system analysis.

This package implements three complementary optimization models:
1. Capacity Expansion Model (CEM) - Joint investment and operations optimization
2. Perfect Foresight Operations (DLAC-p) - One-shot operations with perfect information  
3. DLAC-i Operations - Rolling horizon operations with imperfect information

The system models a 4-technology power system: Nuclear (baseload), Wind (renewable), 
Gas (peaker), and Battery (storage).
"""
module ToySystemQuad

# Include all submodules
include("SystemConfig.jl")
include("ProfileGeneration.jl") 
include("OptimizationModels.jl")
include("PlottingModule.jl")
include("TestRunner.jl")

# Re-export key functions from submodules
using .SystemConfig
using .ProfileGeneration
using .OptimizationModels
using .PlottingModule
using .TestRunner

# Export main functions for external use
export run_complete_test_system
export solve_capacity_expansion_model, solve_perfect_foresight_operations, solve_dlac_i_operations
export create_toy_system, get_default_system_parameters
export plot_price_duration_curve, plot_generation_stack, generate_all_plots

end # module ToySystemQuad