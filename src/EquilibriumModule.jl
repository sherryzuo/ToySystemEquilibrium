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
export save_equilibrium_results, analyze_equilibrium_convergence, resume_from_log
export PolicyFunction, PerfectForesight, DLAC_i

# =============================================================================
# LOG FILE UTILITIES
# =============================================================================

"""
    resume_from_log(log_file, generators, battery)

Read the last line from an equilibrium log file and extract capacities for resuming.
Returns (last_iteration, capacities, battery_power_cap, battery_energy_cap) or nothing if file doesn't exist.
"""
function resume_from_log(log_file::String, generators, battery)
    if !isfile(log_file)
        return nothing
    end
    
    try
        # Read the CSV file
        df = CSV.read(log_file, DataFrame)
        
        if nrow(df) == 0
            return nothing
        end
        
        # Get the last row
        last_row = df[end, :]
        last_iteration = last_row.Iteration
        
        # Extract capacities
        G = length(generators)
        capacities = zeros(G)
        
        for g in 1:G
            cap_col = "$(generators[g].name)_capacity_MW"
            if hasproperty(last_row, Symbol(cap_col))
                capacities[g] = last_row[Symbol(cap_col)]
            else
                println("Warning: Could not find column $cap_col in log file")
                return nothing
            end
        end
        
        # Extract battery power capacity
        battery_power_cap = 0.0
        if hasproperty(last_row, :Battery_capacity_MW)
            battery_power_cap = last_row.Battery_capacity_MW
        else
            println("Warning: Could not find Battery_capacity_MW column in log file")
            return nothing
        end
        
        # Calculate battery energy capacity from power capacity
        battery_energy_cap = min(battery_power_cap * battery.duration, battery.max_energy_capacity)
        
        println("Resuming from log file: $log_file")
        println("  Last iteration: $last_iteration")
        println("  Resuming capacities:")
        for g in 1:G
            println("    $(generators[g].name): $(round(capacities[g], digits=1)) MW")
        end
        println("    Battery Power: $(round(battery_power_cap, digits=1)) MW")
        println("    Battery Energy: $(round(battery_energy_cap, digits=1)) MWh")
        
        return (last_iteration, capacities, battery_power_cap, battery_energy_cap)
        
    catch e
        println("Error reading log file $log_file: $e")
        return nothing
    end
end

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
    step_size::Float64               # Fixed step size
    smoothing_beta::Float64
    min_capacity_threshold::Float64
    update_generators::Vector{Bool}  # Which generators to update (true = update, false = freeze)
    update_battery::Bool             # Whether to update battery
    anderson_acceleration::Bool      # Whether to use Anderson acceleration
    anderson_depth::Int              # Number of previous iterations to use in Anderson acceleration
    anderson_beta_default::Float64   # Default relaxation parameter
    anderson_beta_max::Float64       # Maximum relaxation parameter
    anderson_T::Int                  # Interval for recomputing optimal beta
    
    function EquilibriumParameters(;
        max_iterations::Int = 20,
        tolerance::Float64 = 1e-3,
        step_size::Float64 = 0.05,
        smoothing_beta::Float64 = 10.0,
        min_capacity_threshold::Float64 = 1e-6,
        update_generators::Vector{Bool} = Bool[],  # Empty = update all
        update_battery::Bool = true,
        anderson_acceleration::Bool = false,
        anderson_depth::Int = 5,
        anderson_beta_default::Float64 = 1.0,
        anderson_beta_max::Float64 = 3.0,
        anderson_T::Int = 5
    )
        new(max_iterations, tolerance, step_size, smoothing_beta, min_capacity_threshold, 
            update_generators, update_battery, anderson_acceleration, anderson_depth, 
            anderson_beta_default, anderson_beta_max, anderson_T)
    end
end

"""
    PolicyFunction

Enum for different operational policy functions.
"""
@enum PolicyFunction PerfectForesight DLAC_i

# =============================================================================
# ANDERSON ACCELERATION FUNCTIONS
# =============================================================================

"""
    anderson_acceleration_aaopt1(x_history, g_history, iteration, anderson_params)

AAopt1_T Anderson acceleration with optimal relaxation parameter computation.
Implements Algorithm 2 from the Anderson acceleration literature.

Args:
- x_history: History of x iterates 
- g_history: History of g(x) evaluates
- iteration: Current iteration number
- anderson_params: Struct with anderson_depth, anderson_beta_default, anderson_beta_max, anderson_T

Returns:
- (next_x, beta_used): Next iterate and the beta parameter used
"""
function anderson_acceleration_aaopt1(x_history::Vector{Vector{Float64}}, 
                                     g_history::Vector{Vector{Float64}},
                                     iteration::Int,
                                     anderson_params)
    
    k = length(x_history)  # Current iteration index (1-based)
    
    if k == 1
        # First iteration: just return g(x_0)
        return g_history[1], anderson_params.anderson_beta_default
    end
    
    # Determine m_k (number of iterates to use)
    m_k = min(k, anderson_params.anderson_depth)
    
    try
        # Step 4: Solve for α coefficients
        # min ||sum(α_i * f(x_{k-m_k+i}))||^2 subject to sum(α_i) = 1
        # where f(x) = g(x) - x
        
        n = length(x_history[1])
        F = zeros(n, m_k)
        
        # Build residual matrix F where F[:, i] = g(x_{k-m_k+i}) - x_{k-m_k+i}
        for i in 1:m_k
            idx = k - m_k + i
            F[:, i] = g_history[idx] - x_history[idx]
        end
        
        # Solve constrained least squares: min ||F*α||^2 subject to 1^T α = 1
        ones_vec = ones(m_k)
        
        # Use regularized approach for numerical stability
        regularization = 1e-12
        FtF = F' * F + regularization * I
        
        # Solve: [F'F  1; 1'  0] [α; λ] = [0; 1]
        augmented_matrix = [FtF ones_vec; ones_vec' 0.0]
        rhs = [zeros(m_k); 1.0]
        
        solution = augmented_matrix \ rhs
        α = solution[1:m_k]
        
        # Step 5: Compute x̄_k and ȳ_k
        x_bar_k = zeros(n)
        y_bar_k = zeros(n)
        
        for i in 1:m_k
            idx = k - m_k + i
            x_bar_k += α[i] * x_history[idx]
            y_bar_k += α[i] * g_history[idx]
        end
        
        # Step 6-18: Compute optimal β or use previous β
        beta_k = anderson_params.anderson_beta_default
        
        if iteration == 1 || (iteration % anderson_params.anderson_T) == 0
            # Recompute optimal β
            
            # We need g(x̄_k) and g(ȳ_k) - these would require additional function evaluations
            # For computational efficiency, we'll approximate using the current data
            # This is a practical modification of the algorithm
            
            # Step 8: Compute β_k* = -<f(ȳ_k) - f(x̄_k), f(x̄_k)> / ||f(ȳ_k) - f(x̄_k)||^2
            f_x_bar_k = y_bar_k - x_bar_k  # f(x̄_k) = g(x̄_k) - x̄_k ≈ ȳ_k - x̄_k
            
            # For f(ȳ_k), we approximate it as the residual direction from recent iterates
            if m_k >= 2
                # Approximate f(ȳ_k) using the trend in residuals
                recent_residual = g_history[k] - x_history[k]
                f_y_bar_k = recent_residual  # Approximation
                
                residual_diff = f_y_bar_k - f_x_bar_k
                residual_diff_norm_sq = dot(residual_diff, residual_diff)
                
                if residual_diff_norm_sq > 1e-12
                    beta_k_star = -dot(residual_diff, f_x_bar_k) / residual_diff_norm_sq
                    
                    if beta_k_star <= 0
                        beta_k = anderson_params.anderson_beta_default
                    else
                        beta_k = min(beta_k_star, anderson_params.anderson_beta_max)
                    end
                else
                    beta_k = anderson_params.anderson_beta_default
                end
            end
            
            # Step 14: x_{k+1} = x̄_k + β_k * (ȳ_k - x̄_k)  [Simplified form]
            x_next = x_bar_k + beta_k * (y_bar_k - x_bar_k)
        else
            # Step 16-17: Use previous β
            # x_{k+1} = x̄_k + β_k * (ȳ_k - x̄_k)
            x_next = x_bar_k + beta_k * (y_bar_k - x_bar_k)
        end
        
        println("  AAopt1: Used $m_k iterates, β=$(round(beta_k, digits=3)), α=$(round.(α, digits=3))")
        return x_next, beta_k
        
    catch e
        println("  AAopt1: Failed with error $e, using simple update")
        # Fallback to simple fixed-point iteration
        return g_history[end], anderson_params.anderson_beta_default
    end
end


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
# OSCILLATION DETECTION
# =============================================================================

"""
    detect_magnitude_oscillation(pmr_history, threshold=10.0)

Detect oscillations based on magnitude flip-flops in individual PMR values.
Returns true if any PMR component shows large changes in opposite directions.
"""
function detect_magnitude_oscillation(pmr_history, threshold=10.0)
    if length(pmr_history) < 3
        return false
    end
    
    # Get last 3 PMR vectors
    recent_pmrs = pmr_history[end-2:end]
    n_components = length(recent_pmrs[1])
    
    # Check each PMR component for oscillations
    for i in 1:n_components
        # Extract PMR values for this component across last 3 iterations
        pmr_series = [recent_pmrs[j][i] for j in 1:3]
        
        # Calculate consecutive changes
        change_1 = pmr_series[2] - pmr_series[1]  # From iter n-2 to n-1
        change_2 = pmr_series[3] - pmr_series[2]  # From iter n-1 to n
        
        # Check for large magnitude changes in opposite directions
        large_changes = abs(change_1) > threshold && abs(change_2) > threshold
        opposite_directions = sign(change_1) != sign(change_2)
        
        if large_changes && opposite_directions
            println("  Oscillation detected in component $i: $(round(pmr_series[1], digits=2))% → $(round(pmr_series[2], digits=2))% → $(round(pmr_series[3], digits=2))%")
            return true
        end
    end
    
    # Also check maximum absolute PMR oscillations (original logic)
    recent_max_pmrs = [maximum(abs.(pmr)) for pmr in recent_pmrs]
    change_1 = recent_max_pmrs[2] - recent_max_pmrs[1]
    change_2 = recent_max_pmrs[3] - recent_max_pmrs[2]
    
    large_changes = abs(change_1) > threshold && abs(change_2) > threshold
    opposite_directions = sign(change_1) != sign(change_2)
    
    if large_changes && opposite_directions
        println("  Oscillation detected in max PMR: $(round(recent_max_pmrs[1], digits=2))% → $(round(recent_max_pmrs[2], digits=2))% → $(round(recent_max_pmrs[3], digits=2))%")
        return true
    end
    
    return false
end

# =============================================================================
# CAPACITY UPDATE MECHANISM
# =============================================================================

"""
    softplus(x, β=10.0)

Compute the softplus function as a smooth approximation to max(0, x).
The parameter β controls the sharpness of the approximation.
Uses numerically stable implementation for large inputs.
"""
function softplus(x, β=10.0)
    # For large inputs, softplus(x) ≈ x to avoid overflow
    # Use the identity: log(1 + exp(βx)) = βx + log(1 + exp(-βx)) for βx > 0
    βx = β * x
    if βx > 20.0  # exp(20) ≈ 5e8, beyond this we risk overflow
        return x  # For large x, softplus(x) ≈ x
    elseif βx < -20.0  # For very negative x, softplus(x) ≈ 0
        return 0.0
    else
        return (1.0 / β) * log(1.0 + exp(βx))
    end
end

"""
    update_all_capacities(current_capacities, current_battery_power_cap, current_battery_energy_cap,
                         pmr_values, generators, battery, step_size, min_threshold, pmr_history, 
                         update_generators, update_battery)

Selective capacity update with adaptive step size: new_capacity = current_capacity + step_size * PMR/100
Only updates capacities where the corresponding update flag is true.
"""
function update_all_capacities(current_capacities, current_battery_power_cap, current_battery_energy_cap,
                               pmr_values, generators, battery, step_size, min_threshold, pmr_history=nothing,
                               update_generators::Vector{Bool}=Bool[], update_battery::Bool=true)
    G = length(generators)
    
    # If update_generators is empty, default to updating all generators
    if isempty(update_generators)
        update_generators = fill(true, G)
    end
    
    println("DEBUG - Selective Capacity Update:")
    println("  Using step size: $step_size")
    println("  Generator update flags: $update_generators")
    println("  Battery update flag: $update_battery")
    
    # Update generator capacities (only if flagged for update)
    new_capacities = Float64.(current_capacities)  # Ensure Float64 array
    for g in 1:G
        current_cap = current_capacities[g]
        pmr = pmr_values[g]
        
        if update_generators[g]
            # Additive update: capacity + step_size * PMR / 100
            new_cap = Float64(current_cap) + Float64(step_size) * (Float64(pmr) / 100.0)
            new_capacities[g] = Float64(softplus(new_cap))
            
            println("  Gen $g ($(generators[g].name)) - UPDATING:")
            println("    Current: $(round(current_cap, digits=2)) MW, PMR: $(round(pmr, digits=2))%")
            println("    Update: $(round(new_capacities[g], digits=2)) MW")
        else
            # Keep current capacity frozen
            new_capacities[g] = Float64(current_cap)
            
            println("  Gen $g ($(generators[g].name)) - FROZEN:")
            println("    Capacity: $(round(current_cap, digits=2)) MW, PMR: $(round(pmr, digits=2))%")
        end
    end
    
    # Update battery capacity (only if flagged for update)
    battery_pmr = pmr_values[G+1]
    
    if update_battery
        # Additive update for battery: capacity + step_size * PMR / 100
        new_battery_power_cap = Float64(current_battery_power_cap) + Float64(step_size) * (Float64(battery_pmr) / 100.0)
        new_battery_power_cap = Float64(softplus(new_battery_power_cap))
        
        println("  Battery - UPDATING:")
        println("    Current: $(round(current_battery_power_cap, digits=2)) MW, PMR: $(round(battery_pmr, digits=2))%")
        println("    Update: $(round(new_battery_power_cap, digits=2)) MW")
    else
        # Keep current battery capacity frozen
        new_battery_power_cap = Float64(current_battery_power_cap)
        
        println("  Battery - FROZEN:")
        println("    Capacity: $(round(current_battery_power_cap, digits=2)) MW, PMR: $(round(battery_pmr, digits=2))%")
    end
    
    # Energy capacity follows power capacity
    new_battery_energy_cap = new_battery_power_cap * battery.duration
    println("    Energy cap: $(round(new_battery_energy_cap, digits=2)) MWh")
    
    return new_capacities, new_battery_power_cap, new_battery_energy_cap
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
                     equilibrium_params=EquilibriumParameters(), output_dir="results/equilibrium",
                     resume=false)

Solve for capacity equilibrium using fixed-point iteration with the specified policy function.
Saves iteration logs in CSV format compatible with equilibrium_plots.py script.

Args:
- resume: If true, attempts to resume from existing log file. If false, starts fresh.

Returns:
- Dictionary with final capacities, convergence info, and iteration history
"""
function solve_equilibrium(generators, battery, initial_capacities, initial_battery_power_cap, 
                          initial_battery_energy_cap, profiles::SystemProfiles, policy::PolicyFunction;
                          equilibrium_params::EquilibriumParameters = EquilibriumParameters(),
                          output_dir="results/equilibrium", resume::Bool = false)
    
    println("="^80)
    println("EQUILIBRIUM SOLVER: $(policy) Policy")
    println("="^80)
    println("Parameters:")
    println("  Max iterations: $(equilibrium_params.max_iterations)")
    println("  Tolerance: $(equilibrium_params.tolerance)")
    println("  Step size: $(equilibrium_params.step_size)")
    if equilibrium_params.anderson_acceleration
        println("  AAopt1_T acceleration: depth=$(equilibrium_params.anderson_depth), β_default=$(equilibrium_params.anderson_beta_default), β_max=$(equilibrium_params.anderson_beta_max), T=$(equilibrium_params.anderson_T)")
    end
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
    
    # Anderson acceleration tracking for AAopt1_T
    x_history = Vector{Vector{Float64}}()  # History of x iterates (capacity vectors)
    g_history = Vector{Vector{Float64}}()  # History of g(x) evaluates (updated capacity vectors)
    anderson_beta_history = Float64[]      # History of beta values used
    
    converged = false
    iteration = 0
    
    # Set up real-time CSV logging (compatible with equilibrium_plots.py)
    mkpath(output_dir)
    log_file = joinpath(output_dir, "equilibrium_log.csv")
    
    # Handle resume logic
    starting_iteration = 1
    if resume
        resume_data = resume_from_log(log_file, generators, battery)
        if resume_data !== nothing
            last_iteration, resume_capacities, resume_battery_power_cap, resume_battery_energy_cap = resume_data
            current_capacities = resume_capacities
            current_battery_power_cap = resume_battery_power_cap
            current_battery_energy_cap = resume_battery_energy_cap
            starting_iteration = last_iteration + 1
            println("Resuming from iteration $starting_iteration")
        else
            println("Could not resume from log file, starting fresh")
        end
    end
    
    # Create CSV headers compatible with plotting script
    csv_headers = ["Iteration"]
    for g in 1:G
        push!(csv_headers, "$(generators[g].name)_capacity_MW")
    end
    push!(csv_headers, "Battery_capacity_MW")  # Power capacity for battery
    for g in 1:G
        push!(csv_headers, "$(generators[g].name)_pmr")
    end
    push!(csv_headers, "Battery_pmr")
    push!(csv_headers, "max_pmr")
    push!(csv_headers, "step_size")
    push!(csv_headers, "total_cost")
    
    # Initialize CSV file with headers (only if starting fresh or resume failed)
    if !resume || starting_iteration == 1
        open(log_file, "w") do f
            println(f, join(csv_headers, ","))
        end
    end
    
    println("Starting fixed-point iteration...")
    
    for iter in starting_iteration:(starting_iteration + equilibrium_params.max_iterations - 1)
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
        
        # Use fixed step size
        step_size = equilibrium_params.step_size
        println("  Step size: $(round(step_size, digits=6))")
        
        # Log current iteration to CSV (real-time logging)
        max_abs_pmr = maximum(abs.(pmr_values))
        # if max_abs_pmr < 100
        #     step_size = step_size * 0.1
        # end
        csv_row = Any[iter]  # Start with Any array to handle mixed types
        
        # Add capacities
        for g in 1:G
            push!(csv_row, current_capacities[g])
        end
        push!(csv_row, current_battery_power_cap)  # Battery power capacity
        
        # Add PMR values
        for g in 1:G
            push!(csv_row, pmr_values[g])
        end
        push!(csv_row, pmr_values[G+1])  # Battery PMR
        push!(csv_row, max_abs_pmr)      # Max absolute PMR
        push!(csv_row, step_size)        # Current step size
        
        # Add objective function (total cost)
        push!(csv_row, operational_result["total_cost"])
        
        # Append to CSV file
        open(log_file, "a") do f
            println(f, join(csv_row, ","))
        end
        
        # Check convergence
        if check_convergence(pmr_values, equilibrium_params.tolerance)
            println("\n✓ CONVERGENCE ACHIEVED at iteration $iter")
            println("  Max |PMR|: $(round(maximum(abs.(pmr_values)), digits=3))%")
            converged = true
            break
        end
        
        # Store current capacities for Anderson acceleration
        current_x = vcat(current_capacities, [current_battery_power_cap])
        push!(x_history, current_x)
        
        # Compute capacity updates using standard fixed-point iteration
        proposed_capacities, proposed_battery_power_cap, proposed_battery_energy_cap = update_all_capacities(
            current_capacities, current_battery_power_cap, current_battery_energy_cap,
            pmr_values, generators, battery, step_size,
            equilibrium_params.min_capacity_threshold, pmr_history,
            equilibrium_params.update_generators, equilibrium_params.update_battery
        )
        
        # Store g(x) for Anderson acceleration  
        proposed_x = vcat(proposed_capacities, [proposed_battery_power_cap])
        push!(g_history, proposed_x)
        
        # Apply AAopt1_T Anderson acceleration if enabled
        if equilibrium_params.anderson_acceleration && length(x_history) >= 1
            println("  Applying AAopt1_T Anderson acceleration...")
            accelerated_x, beta_used = anderson_acceleration_aaopt1(
                x_history, g_history, iter, equilibrium_params
            )
            push!(anderson_beta_history, beta_used)
            
            # Extract accelerated capacities
            new_capacities = accelerated_x[1:G]
            new_battery_power_cap = accelerated_x[G+1]
            
            # Apply softplus smoothing to ensure non-negativity
            for g in 1:G
                new_capacities[g] = softplus(new_capacities[g], equilibrium_params.smoothing_beta)
            end
            new_battery_power_cap = softplus(new_battery_power_cap, equilibrium_params.smoothing_beta)
            new_battery_energy_cap = new_battery_power_cap * battery.duration
        else
            # Use standard update without acceleration
            new_capacities = proposed_capacities
            new_battery_power_cap = proposed_battery_power_cap
            new_battery_energy_cap = proposed_battery_energy_cap
            push!(anderson_beta_history, equilibrium_params.anderson_beta_default)
        end
        
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
        "equilibrium_params" => equilibrium_params,
        "log_file" => log_file
    )
    
    println("\n" * "="^80)
    if converged
        println("EQUILIBRIUM CONVERGED in $iteration iterations")
        println("Final max |PMR|: $(round(maximum(abs.(final_pmr)), digits=3))%")
    else
        println("EQUILIBRIUM DID NOT CONVERGE in $(equilibrium_params.max_iterations) iterations")
        println("Final max |PMR|: $(round(maximum(abs.(final_pmr)), digits=3))%")
    end
    println("Iteration log saved to: $log_file")
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
        Initial_Step_Size = [equilibrium_result["equilibrium_params"].initial_step_size],
        Adaptive_Step_Size = [equilibrium_result["equilibrium_params"].adaptive_step_size],
        Step_Size_Decay = [equilibrium_result["equilibrium_params"].step_size_decay],
        Min_Step_Size = [equilibrium_result["equilibrium_params"].min_step_size]
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
            n_conv_metrics = length(equilibrium_result["convergence_metrics_history"])
            convergence_df = DataFrame(
                Iteration = 2:(n_conv_metrics+1),  # Convergence metrics start from iteration 2
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
                          equilibrium_params=EquilibriumParameters(), base_output_dir="results", resume=false)

Complete equilibrium run with automatic directory creation and result saving.
"""
function run_policy_equilibrium(generators, battery, initial_capacities, initial_battery_power_cap, 
                               initial_battery_energy_cap, profiles::SystemProfiles, policy::PolicyFunction;
                               equilibrium_params::EquilibriumParameters = EquilibriumParameters(),
                               base_output_dir="results", resume::Bool = false)
    
    # Create equilibrium-specific output directory
    equilibrium_dir = joinpath(base_output_dir, "equilibrium")
    policy_dir = joinpath(equilibrium_dir, lowercase(string(policy)))
    mkpath(policy_dir)
    
    # Solve equilibrium
    result = solve_equilibrium(generators, battery, initial_capacities, initial_battery_power_cap, 
                              initial_battery_energy_cap, profiles, policy;
                              equilibrium_params=equilibrium_params, output_dir=policy_dir, resume=resume)
    
    # Save results
    save_equilibrium_results(result, generators, battery, policy_dir)
    
    return result
end

end # module EquilibriumModule