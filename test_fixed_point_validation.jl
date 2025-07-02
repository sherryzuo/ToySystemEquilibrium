#!/usr/bin/env julia

"""
test_fixed_point_validation.jl

Validation test for the fixed-point equilibrium solver.
Tests if starting from zero capacities, the Perfect Foresight equilibrium
converges to the same capacity mix as the Capacity Expansion Model (CEM).

This is a fundamental validation - if the fixed-point iteration is correct,
then PF equilibrium should match CEM since both optimize the same objective
(minimize total system cost with perfect foresight).
"""

using Revise
using ToySystemQuad
function test_fixed_point_validation()
    println("="^100)
    println("FIXED-POINT VALIDATION TEST")
    println("Testing if PF equilibrium from zero capacities matches CEM solution")
    println("="^100)
    
    # System configuration
    params = SystemParameters(
        720,     # hours (30 days)
        30,      # days  
        5,       # N (number of generators per technology fleet)
        42,      # random_seed
        10000.0, # load_shed_penalty ($/MWh)
        0.001,   # load_shed_quad
        100.0    # flex_demand_mw
    )
    
    generators, battery, profiles = create_complete_toy_system(params)
    
    println("System configured:")
    println("  - $(params.hours)-hour horizon ($(params.days) days)")
    println("  - $(length(generators)) generator types")
    
    # =============================================================================
    # STEP 1: Solve CEM (benchmark solution)
    # =============================================================================
    println("\n" * "="^80)
    println("STEP 1: Solving Capacity Expansion Model (CEM)")
    println("="^80)
    
    cem_result = solve_capacity_expansion_model(generators, battery, profiles)
    
    if cem_result["status"] != "optimal"
        println("ERROR: CEM failed to solve optimally")
        return nothing
    end
    
    cem_capacities = cem_result["capacity"]
    cem_battery_power = cem_result["battery_power_cap"]
    cem_battery_energy = cem_result["battery_energy_cap"]
    cem_total_cost = cem_result["total_cost"]
    
    println("CEM Solution:")
    for (i, gen) in enumerate(generators)
        println("  $(gen.name): $(round(cem_capacities[i], digits=1)) MW")
    end
    println("  Battery Power: $(round(cem_battery_power, digits=1)) MW")
    println("  Battery Energy: $(round(cem_battery_energy, digits=1)) MWh")
    println("  Total Cost: \$$(round(cem_total_cost, digits=0))")
    
    # =============================================================================
    # STEP 2: Run PF equilibrium from zero capacities
    # =============================================================================
    println("\n" * "="^80)
    println("STEP 2: Running Perfect Foresight Equilibrium from Zero Capacities")
    println("="^80)
    
    # Start with zero capacities
    zero_capacities = [100,100,1500]
    zero_battery_power = 400
    zero_battery_energy = 400*4
    
    println("Starting capacities (all zero):")
    for (i, gen) in enumerate(generators)
        println("  $(gen.name): $(zero_capacities[i]) MW")
    end
    println("  Battery Power: $(zero_battery_power) MW")
    println("  Battery Energy: $(zero_battery_energy) MWh")
    
    # Equilibrium parameters - use adaptive step size for better convergence
    equilibrium_params = EquilibriumParameters(
        max_iterations = 10000,    
        tolerance = 1e-2,       
        step_size = 0.7,          # Fixed step size for validation test
        smoothing_beta = 5.0,   
        min_capacity_threshold = 1e-6,
        anderson_acceleration = true,  # Enable AAopt1_T Anderson acceleration for faster convergence
        anderson_depth = 3,           # Use fewer iterates for validation test
        anderson_beta_default = 1.0,  # Default relaxation parameter
        anderson_beta_max = 2.0,      # Conservative maximum for validation test
        anderson_T = 3,              # Recompute optimal β every 3 iterations
        # Selective updating examples (uncomment to test):
        # update_generators = [true, true, false],  # Only update Nuclear and Wind, freeze Gas
        # update_generators = [false, false, true], # Only update Gas, freeze Nuclear and Wind
        # update_generators = [true, false, true],  # Update Nuclear and Gas, freeze Wind
        update_battery = true                    # Update battery
    )
    
    println("\nEquilibrium parameters:")
    println("  Max iterations: $(equilibrium_params.max_iterations)")
    println("  Tolerance: $(equilibrium_params.tolerance)")
    println("  Step size: $(equilibrium_params.step_size)")
    if equilibrium_params.anderson_acceleration
        println("  AAopt1_T acceleration: depth=$(equilibrium_params.anderson_depth), β_default=$(equilibrium_params.anderson_beta_default), β_max=$(equilibrium_params.anderson_beta_max), T=$(equilibrium_params.anderson_T)")
    end
    println("  Smoothing beta: $(equilibrium_params.smoothing_beta)")
    
    # Run equilibrium with Perfect Foresight
    equilibrium_result = run_policy_equilibrium(
        generators, battery, zero_capacities, zero_battery_power, zero_battery_energy, 
        profiles, PerfectForesight;
        equilibrium_params = equilibrium_params,
        base_output_dir = "results/validation"
    )

end
# Run the validation test
println("Starting Fixed-Point Validation Test...")
validation_result = test_fixed_point_validation()