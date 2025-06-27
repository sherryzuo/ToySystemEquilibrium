"""
EquilibriumModule.jl

Implements fixed-point iteration equilibrium solver for capacity expansion with zero-profit equilibrium.
Adapts the equilibrium logic from Legacy/ToySystemQuad.jl to work with the new modular src structure.

Key Features:
- Policy-agnostic equilibrium solver (works with any operational policy)
- PMR-based capacity updates toward zero-profit equilibrium
- Comprehensive CSV logging of iterations and convergence
- Softplus smoothing for numerical stability
- Configurable convergence parameters
"""

module EquilibriumModule

using CSV, DataFrames, Statistics, LinearAlgebra
using ..SystemConfig: Generator, Battery, SystemParameters, SystemProfiles
using ..OptimizationModels: solve_perfect_foresight_operations, solve_dlac_i_operations, compute_pmr

export EquilibriumParameters, solve_equilibrium, run_policy_equilibrium
export save_equilibrium_results, analyze_equilibrium_convergence

# =============================================================================
# EQUILIBRIUM CONFIGURATION
# =============================================================================

"""
    EquilibriumParameters

Configuration parameters for the equilibrium solver.
"""
struct EquilibriumParameters
    max_iterations::Int
    tolerance::Float64
    step_size::Float64
    smoothing_beta::Float64
    min_capacity_threshold::Float64
    
    function EquilibriumParameters(;
        max_iterations::Int = 20,
        tolerance::Float64 = 1e-3,
        step_size::Float64 = 0.05,
        smoothing_beta::Float64 = 10.0,
        min_capacity_threshold::Float64 = 1e-6
    )
        new(max_iterations, tolerance, step_size, smoothing_beta, min_capacity_threshold)
    end
end

"""
    PolicyFunction

Enum for different operational policy functions.
"""
@enum PolicyFunction PerfectForesight DLAC_i

# =============================================================================
# POLICY DISPATCHER
# =============================================================================

"""
    call_policy_function(policy::PolicyFunction, generators, battery, capacities, 
                        battery_power_cap, battery_energy_cap, profiles; kwargs...)

Dispatch operational optimization to the appropriate policy function.
"""
function call_policy_function(policy::PolicyFunction, generators, battery, capacities, 
                             battery_power_cap, battery_energy_cap, profiles::SystemProfiles; 
                             output_dir="results", kwargs...)
    if policy == PerfectForesight
        return solve_perfect_foresight_operations(generators, battery, capacities, 
                                                battery_power_cap, battery_energy_cap, 
                                                profiles; output_dir=output_dir, kwargs...)
    elseif policy == DLAC_i
        return solve_dlac_i_operations(generators, battery, capacities, 
                                     battery_power_cap, battery_energy_cap, 
                                     profiles; output_dir=output_dir, kwargs...)
    else
        error("Unknown policy function: $policy")
    end
end

# =============================================================================
# CAPACITY UPDATE MECHANISM
# =============================================================================

"""
    update_capacities_with_smoothing(current_capacities, pmr_values, generators, 
                                    step_size, smoothing_beta, min_threshold)

Update capacities based on PMR values using softplus smoothing for numerical stability.
Positive PMR → capacity increases, Negative PMR → capacity decreases.
"""
function update_capacities_with_smoothing(current_capacities, pmr_values, generators, 
                                         step_size, smoothing_beta, min_threshold)
    G = length(generators)
    new_capacities = zeros(G)
    
    for g in 1:G
        if current_capacities[g] > min_threshold
            # Proportional update based on current capacity and PMR
            capacity_update = current_capacities[g] + step_size * current_capacities[g] * pmr_values[g] / 100.0
        else
            # For zero/small capacities, use absolute update if PMR is positive
            capacity_update = pmr_values[g] > 0 ? step_size * generators[g].max_capacity / 10.0 : 0.0
        end
        
        # Apply softplus smoothing: f(x) = log(1 + exp(β*x)) / β
        # This ensures capacity is always positive and smooth
        capacity_update = max(capacity_update, 0.0)  # Ensure non-negative input
        new_capacities[g] = log(1 + exp(smoothing_beta * capacity_update)) / smoothing_beta
        
        # Enforce maximum capacity constraints
        new_capacities[g] = min(new_capacities[g], generators[g].max_capacity)
    end
    
    return new_capacities
end

"""
    update_battery_capacity_with_smoothing(current_power_cap, current_energy_cap, 
                                          battery_pmr, battery, step_size, smoothing_beta, min_threshold)

Update battery capacities based on PMR using softplus smoothing.
"""
function update_battery_capacity_with_smoothing(current_power_cap, current_energy_cap, 
                                               battery_pmr, battery, step_size, smoothing_beta, min_threshold)
    if current_power_cap > min_threshold
        # Proportional update for power capacity
        power_update = current_power_cap + step_size * current_power_cap * battery_pmr / 100.0
    else
        # For zero/small capacity, use absolute update if PMR is positive
        power_update = battery_pmr > 0 ? step_size * battery.max_power_capacity / 10.0 : 0.0
    end
    
    # Apply softplus smoothing
    power_update = max(power_update, 0.0)
    new_power_cap = log(1 + exp(smoothing_beta * power_update)) / smoothing_beta
    new_power_cap = min(new_power_cap, battery.max_power_capacity)
    
    # Energy capacity follows power capacity with duration constraint
    new_energy_cap = min(new_power_cap * battery.duration, battery.max_energy_capacity)
    
    return new_power_cap, new_energy_cap
end

# =============================================================================
# CONVERGENCE ANALYSIS
# =============================================================================

"""
    check_convergence(pmr_values, tolerance)

Check if equilibrium has converged based on PMR tolerance.
"""
function check_convergence(pmr_values, tolerance)
    max_abs_pmr = maximum(abs.(pmr_values))
    return max_abs_pmr < tolerance * 100.0  # Convert tolerance to percentage
end

"""
    compute_convergence_metrics(pmr_history, capacity_history)

Compute convergence metrics for equilibrium analysis.
"""
function compute_convergence_metrics(pmr_history, capacity_history)
    n_iterations = length(pmr_history)
    
    if n_iterations < 2
        return Dict(
            "max_pmr_change" => NaN,
            "capacity_change_norm" => NaN,
            "oscillation_detected" => false,
            "convergence_rate" => NaN
        )
    end
    
    # PMR convergence
    current_pmr = pmr_history[end]
    prev_pmr = pmr_history[end-1]
    max_pmr_change = maximum(abs.(current_pmr .- prev_pmr))
    
    # Capacity convergence
    current_cap = capacity_history[end]
    prev_cap = capacity_history[end-1]
    capacity_change_norm = norm(current_cap .- prev_cap)
    
    # Oscillation detection (simple check for sign changes)
    oscillation_detected = false
    if n_iterations >= 4
        recent_pmr = pmr_history[end-3:end]
        for i in 1:length(recent_pmr[1])
            pmr_series = [iter[i] for iter in recent_pmr]
            sign_changes = sum(diff(sign.(pmr_series)) .!= 0)
            if sign_changes >= 2
                oscillation_detected = true
                break
            end
        end
    end
    
    # Convergence rate estimate (geometric)
    convergence_rate = NaN
    if n_iterations >= 3
        e_curr = norm(current_pmr)
        e_prev = norm(prev_pmr)
        e_prev2 = norm(pmr_history[end-2])
        
        if e_prev2 > 1e-10 && e_prev > 1e-10
            convergence_rate = log(e_curr / e_prev) / log(e_prev / e_prev2)
        end
    end
    
    return Dict(
        "max_pmr_change" => max_pmr_change,
        "capacity_change_norm" => capacity_change_norm,
        "oscillation_detected" => oscillation_detected,
        "convergence_rate" => convergence_rate
    )
end

# =============================================================================
# MAIN EQUILIBRIUM SOLVER
# =============================================================================

"""
    solve_equilibrium(generators, battery, initial_capacities, initial_battery_power_cap, 
                     initial_battery_energy_cap, profiles, policy; 
                     equilibrium_params=EquilibriumParameters(), output_dir="results/equilibrium")

Solve for capacity equilibrium using fixed-point iteration with the specified policy function.

Returns:
- Dictionary with final capacities, convergence info, and iteration history
"""
function solve_equilibrium(generators, battery, initial_capacities, initial_battery_power_cap, 
                          initial_battery_energy_cap, profiles::SystemProfiles, policy::PolicyFunction;
                          equilibrium_params::EquilibriumParameters = EquilibriumParameters(),
                          output_dir="results/equilibrium")
    
    println("="^80)
    println("EQUILIBRIUM SOLVER: $(policy) Policy")
    println("="^80)
    println("Parameters:")
    println("  Max iterations: $(equilibrium_params.max_iterations)")
    println("  Tolerance: $(equilibrium_params.tolerance)")
    println("  Step size: $(equilibrium_params.step_size)")
    println("  Smoothing beta: $(equilibrium_params.smoothing_beta)")
    println()
    
    # Initialize tracking variables
    G = length(generators)
    current_capacities = copy(initial_capacities)
    current_battery_power_cap = initial_battery_power_cap
    current_battery_energy_cap = initial_battery_energy_cap
    
    # History tracking
    capacity_history = Vector{Vector{Float64}}()
    battery_power_history = Float64[]
    battery_energy_history = Float64[]
    pmr_history = Vector{Vector{Float64}}()
    convergence_metrics_history = Vector{Dict}()
    
    converged = false
    iteration = 0
    
    println("Starting fixed-point iteration...")
    
    for iter in 1:equilibrium_params.max_iterations
        iteration = iter
        println("\n--- Iteration $iter ---")
        
        # Store current state
        push!(capacity_history, copy(current_capacities))
        push!(battery_power_history, current_battery_power_cap)
        push!(battery_energy_history, current_battery_energy_cap)
        
        println("Current capacities:")
        for g in 1:G
            println("  $(generators[g].name): $(round(current_capacities[g], digits=1)) MW")
        end
        println("  Battery Power: $(round(current_battery_power_cap, digits=1)) MW")
        println("  Battery Energy: $(round(current_battery_energy_cap, digits=1)) MWh")
        
        # Solve operational problem with current capacities
        operational_result = call_policy_function(policy, generators, battery, current_capacities, 
                                                 current_battery_power_cap, current_battery_energy_cap, 
                                                 profiles; output_dir=output_dir)
        
        if operational_result["status"] != "optimal"
            println("WARNING: Operational optimization failed at iteration $iter")
            println("Termination status: $(operational_result.get("termination_status", "unknown"))")
            break
        end
        
        # Compute PMR for all technologies
        pmr_values = compute_pmr(operational_result, generators, battery, current_capacities, 
                               current_battery_power_cap, current_battery_energy_cap)
        
        push!(pmr_history, copy(pmr_values))
        
        println("PMR values:")
        for g in 1:G
            println("  $(generators[g].name): $(round(pmr_values[g], digits=2))%")
        end
        println("  Battery: $(round(pmr_values[G+1], digits=2))%")
        
        # Check convergence
        if check_convergence(pmr_values, equilibrium_params.tolerance)
            println("\n✓ CONVERGENCE ACHIEVED at iteration $iter")
            println("  Max |PMR|: $(round(maximum(abs.(pmr_values)), digits=3))%")
            converged = true
            break
        end
        
        # Update capacities based on PMR
        new_capacities = update_capacities_with_smoothing(
            current_capacities, pmr_values[1:G], generators,
            equilibrium_params.step_size, equilibrium_params.smoothing_beta,
            equilibrium_params.min_capacity_threshold
        )
        
        new_battery_power_cap, new_battery_energy_cap = update_battery_capacity_with_smoothing(
            current_battery_power_cap, current_battery_energy_cap, pmr_values[G+1], battery,
            equilibrium_params.step_size, equilibrium_params.smoothing_beta,
            equilibrium_params.min_capacity_threshold
        )
        
        # Compute convergence metrics
        convergence_metrics = compute_convergence_metrics(pmr_history, capacity_history)
        push!(convergence_metrics_history, convergence_metrics)
        
        println("Convergence metrics:")
        println("  Max PMR change: $(round(convergence_metrics["max_pmr_change"], digits=3))%")
        println("  Capacity change norm: $(round(convergence_metrics["capacity_change_norm"], digits=2))")
        if convergence_metrics["oscillation_detected"]
            println("  ⚠ Oscillation detected")
        end
        
        # Update for next iteration
        current_capacities = new_capacities
        current_battery_power_cap = new_battery_power_cap
        current_battery_energy_cap = new_battery_energy_cap
    end
    
    # Final results
    final_pmr = length(pmr_history) > 0 ? pmr_history[end] : zeros(G+1)
    
    result = Dict(
        "converged" => converged,
        "iterations" => iteration,
        "policy" => string(policy),
        "final_capacities" => current_capacities,
        "final_battery_power_cap" => current_battery_power_cap,
        "final_battery_energy_cap" => current_battery_energy_cap,
        "final_pmr" => final_pmr,
        "capacity_history" => capacity_history,
        "battery_power_history" => battery_power_history,
        "battery_energy_history" => battery_energy_history,
        "pmr_history" => pmr_history,
        "convergence_metrics_history" => convergence_metrics_history,
        "equilibrium_params" => equilibrium_params
    )
    
    println("\n" * "="^80)
    if converged
        println("EQUILIBRIUM CONVERGED in $iteration iterations")
        println("Final max |PMR|: $(round(maximum(abs.(final_pmr)), digits=3))%")
    else
        println("EQUILIBRIUM DID NOT CONVERGE in $(equilibrium_params.max_iterations) iterations")
        println("Final max |PMR|: $(round(maximum(abs.(final_pmr)), digits=3))%")
    end
    println("="^80)
    
    return result
end

# =============================================================================
# RESULTS SAVING
# =============================================================================

"""
    save_equilibrium_results(equilibrium_result, generators, battery, output_dir)

Save equilibrium results to CSV files in the specified directory.
"""
function save_equilibrium_results(equilibrium_result, generators, battery, output_dir)
    mkpath(output_dir)
    G = length(generators)
    
    # 1. Equilibrium summary
    summary_df = DataFrame(
        Policy = [equilibrium_result["policy"]],
        Converged = [equilibrium_result["converged"]],
        Iterations = [equilibrium_result["iterations"]],
        Final_Max_PMR = [maximum(abs.(equilibrium_result["final_pmr"]))],
        Tolerance = [equilibrium_result["equilibrium_params"].tolerance],
        Step_Size = [equilibrium_result["equilibrium_params"].step_size]
    )
    
    # Add final capacities
    for g in 1:G
        summary_df[!, "Final_$(generators[g].name)_MW"] = [equilibrium_result["final_capacities"][g]]
    end
    summary_df[!, "Final_Battery_Power_MW"] = [equilibrium_result["final_battery_power_cap"]]
    summary_df[!, "Final_Battery_Energy_MWh"] = [equilibrium_result["final_battery_energy_cap"]]
    
    CSV.write(joinpath(output_dir, "equilibrium_summary.csv"), summary_df)
    
    # 2. Iteration-by-iteration results
    n_iterations = length(equilibrium_result["capacity_history"])
    
    if n_iterations > 0
        iterations_df = DataFrame(Iteration = 1:n_iterations)
        
        # Capacities
        for g in 1:G
            iterations_df[!, "$(generators[g].name)_MW"] = [equilibrium_result["capacity_history"][i][g] for i in 1:n_iterations]
        end
        iterations_df[!, "Battery_Power_MW"] = equilibrium_result["battery_power_history"]
        iterations_df[!, "Battery_Energy_MWh"] = equilibrium_result["battery_energy_history"]
        
        # PMR values
        for g in 1:G
            iterations_df[!, "$(generators[g].name)_PMR"] = [equilibrium_result["pmr_history"][i][g] for i in 1:length(equilibrium_result["pmr_history"])]
        end
        iterations_df[!, "Battery_PMR"] = [equilibrium_result["pmr_history"][i][G+1] for i in 1:length(equilibrium_result["pmr_history"])]
        
        CSV.write(joinpath(output_dir, "equilibrium_iterations.csv"), iterations_df)
        
        # 3. Convergence metrics
        if length(equilibrium_result["convergence_metrics_history"]) > 0
            convergence_df = DataFrame(
                Iteration = 2:n_iterations,  # Convergence metrics start from iteration 2
                Max_PMR_Change = [m["max_pmr_change"] for m in equilibrium_result["convergence_metrics_history"]],
                Capacity_Change_Norm = [m["capacity_change_norm"] for m in equilibrium_result["convergence_metrics_history"]],
                Oscillation_Detected = [m["oscillation_detected"] for m in equilibrium_result["convergence_metrics_history"]],
                Convergence_Rate = [m["convergence_rate"] for m in equilibrium_result["convergence_metrics_history"]]
            )
            
            CSV.write(joinpath(output_dir, "equilibrium_convergence.csv"), convergence_df)
        end
    end
    
    println("Equilibrium results saved to: $output_dir")
end

"""
    run_policy_equilibrium(generators, battery, initial_capacities, initial_battery_power_cap, 
                          initial_battery_energy_cap, profiles, policy;
                          equilibrium_params=EquilibriumParameters(), base_output_dir="results")

Complete equilibrium run with automatic directory creation and result saving.
"""
function run_policy_equilibrium(generators, battery, initial_capacities, initial_battery_power_cap, 
                               initial_battery_energy_cap, profiles::SystemProfiles, policy::PolicyFunction;
                               equilibrium_params::EquilibriumParameters = EquilibriumParameters(),
                               base_output_dir="results")
    
    # Create equilibrium-specific output directory
    equilibrium_dir = joinpath(base_output_dir, "equilibrium")
    policy_dir = joinpath(equilibrium_dir, lowercase(string(policy)))
    mkpath(policy_dir)
    
    # Solve equilibrium
    result = solve_equilibrium(generators, battery, initial_capacities, initial_battery_power_cap, 
                              initial_battery_energy_cap, profiles, policy;
                              equilibrium_params=equilibrium_params, output_dir=policy_dir)
    
    # Save results
    save_equilibrium_results(result, generators, battery, policy_dir)
    
    return result
end

end # module EquilibriumModule