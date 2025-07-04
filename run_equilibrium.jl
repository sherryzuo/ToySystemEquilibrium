#!/usr/bin/env julia

"""
run_equilibrium.jl

Run script for the equilibrium solver with configurable policy functions.
Demonstrates the fixed-point iteration equilibrium system adapted from legacy code.

Usage:
    julia run_equilibrium.jl [policy] [options]
    
    policy: "pf" (Perfect Foresight) or "dlac" (DLAC-i) or "both"
    
Examples:
    julia run_equilibrium.jl pf        # Run equilibrium with Perfect Foresight policy
    julia run_equilibrium.jl dlac      # Run equilibrium with DLAC-i policy  
    julia run_equilibrium.jl both      # Run equilibrium with both policies
"""

using Revise
using ToySystemQuad

"""
    parse_command_line_args()

Parse command line arguments for policy selection.
"""
function parse_command_line_args()
    if length(ARGS) == 0
        return "both"  # Default to running both policies
    end
    
    policy_arg = lowercase(ARGS[1])
    if policy_arg in ["pf", "perfectforesight", "perfect_foresight"]
        return "pf"
    elseif policy_arg in ["dlac", "dlac_i", "dlac-i"]
        return "dlac"
    elseif policy_arg in ["both", "all"]
        return "both"
    else
        println("Invalid policy argument: $(ARGS[1])")
        println("Valid options: pf, dlac, both")
        exit(1)
    end
end

"""
    run_equilibrium_with_policy(policy_name, policy_func, generators, battery, profiles, equilibrium_params)

Run equilibrium for a single policy function with configurable parameters.
"""
function run_equilibrium_with_policy(policy_name, policy_func, generators, battery, profiles, equilibrium_params)
    println("\n" * "="^100)
    println("RUNNING EQUILIBRIUM: $policy_name Policy")
    println("="^100)
    
    # Get initial capacities from capacity expansion model (as starting point)
    println("Computing initial capacities from Capacity Expansion Model...")
    cem_result = solve_capacity_expansion_model(generators, battery, profiles)
    
    if cem_result["status"] != "optimal"
        println("ERROR: Failed to solve capacity expansion model for initial capacities")
        return nothing
    end
    
    initial_capacities = cem_result["capacity"]
    initial_battery_power_cap = cem_result["battery_power_cap"]
    initial_battery_energy_cap = cem_result["battery_energy_cap"]
    
    println("Initial capacities from CEM:")
    for (i, gen) in enumerate(generators)
        println("  $(gen.name): $(round(initial_capacities[i], digits=1)) MW")
    end
    println("  Battery Power: $(round(initial_battery_power_cap, digits=1)) MW")
    println("  Battery Energy: $(round(initial_battery_energy_cap, digits=1)) MWh")
    
    # Run equilibrium
    equilibrium_result = run_policy_equilibrium(
        generators, battery, initial_capacities, initial_battery_power_cap, 
        initial_battery_energy_cap, profiles, policy_func;
        equilibrium_params = equilibrium_params,
        base_output_dir = "results"
    )
    
    # Print summary
    println("\n" * "="^80)
    println("EQUILIBRIUM SUMMARY: $policy_name")
    println("="^80)
    println("Converged: $(equilibrium_result["converged"])")
    println("Iterations: $(equilibrium_result["iterations"])")
    println("Final max |PMR|: $(round(maximum(abs.(equilibrium_result["final_pmr"])), digits=3))%")
    
    println("\nFinal capacities:")
    for (i, gen) in enumerate(generators)
        println("  $(gen.name): $(round(equilibrium_result["final_capacities"][i], digits=1)) MW")
    end
    println("  Battery Power: $(round(equilibrium_result["final_battery_power_cap"], digits=1)) MW")
    println("  Battery Energy: $(round(equilibrium_result["final_battery_energy_cap"], digits=1)) MWh")
    
    println("\nFinal PMR values:")
    for (i, gen) in enumerate(generators)
        println("  $(gen.name): $(round(equilibrium_result["final_pmr"][i], digits=2))%")
    end
    println("  Battery: $(round(equilibrium_result["final_pmr"][end], digits=2))%")
    
    println("\nLog file: $(equilibrium_result["log_file"])")
    
    return equilibrium_result
end

"""
    main(policy_choice="dlac")

Main execution function with direct policy specification.
"""
function main(policy_choice="dlac")  # Default to DLAC-i policy
    println("="^100)
    println("EQUILIBRIUM SOLVER - ToySystemEquilibrium")
    println("Adapted from Legacy/ToySystemQuad.jl fixed-point iteration")
    println("="^100)
    
    println("Policy selection: $policy_choice")
    
    # ========================================
    # EQUILIBRIUM PARAMETERS - CONFIGURE HERE
    # ========================================
    equilibrium_params = EquilibriumParameters(
        max_iterations = 10000,        # Maximum number of iterations
        tolerance = 5e-3,           # Convergence tolerance (PMR threshold) - PMR < 0.5%
        step_size = 0.7,           # Fixed step size for capacity updates
        smoothing_beta = 10.0,      # Softplus smoothing parameter
        min_capacity_threshold = 1e-6,  # Minimum capacity threshold
        anderson_acceleration = true,   # Enable AAopt1_T Anderson acceleration
        anderson_depth = 2,            # Use fewer previous iterates for smoother updates
        anderson_beta_default = 0.3,   # More conservative default relaxation parameter
        anderson_beta_max = 0.8,       # More conservative maximum relaxation parameter  
        anderson_T = 10                # Recompute optimal β less frequently for stability
    )
    
    println("Equilibrium parameters:")
    println("  Max iterations: $(equilibrium_params.max_iterations)")
    println("  Tolerance: $(equilibrium_params.tolerance)")
    println("  Step size: $(equilibrium_params.step_size)")
    if equilibrium_params.anderson_acceleration
        println("  AAopt1_T acceleration: depth=$(equilibrium_params.anderson_depth), β_default=$(equilibrium_params.anderson_beta_default), β_max=$(equilibrium_params.anderson_beta_max), T=$(equilibrium_params.anderson_T)")
    end
    println("  Smoothing beta: $(equilibrium_params.smoothing_beta)")
    
    # Set up system configuration
    println("\nSetting up system configuration...")
        # Configure system parameters
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
    println("System configured with:")
    println("  - $(params.hours)-hour horizon ($(params.days) days)")
    println("  - Flexible demand: $(params.flex_demand_mw) MW with quadratic pricing")

    # Run equilibrium based on policy choice
    results = Dict()
    
    if policy_choice in ["pf", "both"]
        results["PerfectForesight"] = run_equilibrium_with_policy(
            "Perfect Foresight", PerfectForesight, generators, battery, profiles, equilibrium_params
        )
    end
    
    if policy_choice in ["dlac", "both"]
        results["DLAC_i"] = run_equilibrium_with_policy(
            "DLAC-i", DLAC_i, generators, battery, profiles, equilibrium_params
        )
    end
    
    println("\n" * "="^100)
    println("EQUILIBRIUM SOLVER COMPLETE")
    println("Results saved to: results/equilibrium/")
    println("="^100)
    
    return results
end

# Run with DLAC-i policy by default
# You can change this to:
# results = main("pf")     # for Perfect Foresight
# results = main("dlac")   # for DLAC-i  
# results = main("both")   # for both policies

results = main("dlac")