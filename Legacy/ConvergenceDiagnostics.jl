"""
ConvergenceDiagnostics.jl

Convergence diagnostics and equilibrium analysis module for ToySystemQuad.jl
Provides tools for analyzing fixed point iteration convergence and stability.
"""

module ConvergenceDiagnostics

using LinearAlgebra, Statistics
using ..SystemConfig: Generator, Battery

export ConvergenceMetrics
export compute_pmr, analyze_convergence_properties, diagnose_convergence_issues
export create_convergence_summary, compute_equilibrium_jacobian

# =============================================================================
# DATA STRUCTURES
# =============================================================================

struct ConvergenceMetrics
    iteration::Int
    max_pmr::Float64
    capacity_change_norm::Float64
    profit_gradient_norm::Float64
    step_size::Float64
    oscillation_metric::Float64
    convergence_rate::Float64
end

# =============================================================================
# PMR CALCULATION
# =============================================================================

"""
    compute_pmr(operational_results, generators, battery, capacities, battery_power_cap, battery_energy_cap)

Compute Profit-to-Market-Rate (PMR) for each generator and battery storage.
PMR = (Revenue - OpCost - FixedCost - InvestCost) / (InvestCost + FixedCost) * 100
"""
function compute_pmr(operational_results, generators, battery, capacities, battery_power_cap, battery_energy_cap)
    G = length(generators)
    T = length(operational_results["prices"])
    pmr = zeros(G + 1)  # +1 for battery
    
    # Generator PMRs
    for g in 1:G
        if capacities[g] > 1e-6  # Only compute for non-zero capacities
            # Revenue
            energy_revenue = sum(operational_results["prices"][t] * operational_results["generation"][g,t] for t in 1:T)
            
            # Costs
            fuel_costs = sum(generators[g].fuel_cost * operational_results["generation"][g,t] for t in 1:T)
            vom_costs = sum(generators[g].var_om_cost * operational_results["generation"][g,t] for t in 1:T)
            startup_costs = sum(generators[g].startup_cost * operational_results["startup"][g,t] for t in 1:T)
            fixed_om_costs = generators[g].fixed_om_cost * capacities[g]
            investment_costs = generators[g].inv_cost * capacities[g]
            
            operating_profit = energy_revenue - fuel_costs - vom_costs - startup_costs
            net_profit = operating_profit - (investment_costs + fixed_om_costs)
            
            # PMR calculation
            total_fixed_costs = investment_costs + fixed_om_costs
            if total_fixed_costs > 1e-6
                pmr[g] = (net_profit / total_fixed_costs) * 100
            end
        end
    end
    
    # Battery PMR
    if battery_power_cap > 1e-6
        # Revenue from arbitrage
        battery_energy_revenue = sum(operational_results["prices"][t] * operational_results["battery_discharge"][t] for t in 1:T)
        battery_energy_costs = sum(operational_results["prices"][t] * operational_results["battery_charge"][t] for t in 1:T)
        battery_net_energy_revenue = battery_energy_revenue - battery_energy_costs
        
        # Costs
        battery_vom_costs = sum(battery.var_om_cost * (operational_results["battery_charge"][t] + 
                               operational_results["battery_discharge"][t]) for t in 1:T)
        battery_fixed_costs = battery.fixed_om_cost * battery_power_cap
        battery_investment_costs = battery.inv_cost_power * battery_power_cap + battery.inv_cost_energy * battery_energy_cap
        
        battery_operating_profit = battery_net_energy_revenue - battery_vom_costs
        battery_net_profit = battery_operating_profit - (battery_investment_costs + battery_fixed_costs)
        
        # Battery PMR
        battery_total_fixed_costs = battery_investment_costs + battery_fixed_costs
        if battery_total_fixed_costs > 1e-6
            pmr[G + 1] = (battery_net_profit / battery_total_fixed_costs) * 100
        end
    end
    
    return pmr
end

# =============================================================================
# CONVERGENCE ANALYSIS
# =============================================================================

"""
    analyze_convergence_properties(capacity_history, pmr_history, step_size_history)

Analyze convergence properties including oscillation detection and convergence rate estimation.
"""
function analyze_convergence_properties(capacity_history, pmr_history, step_size_history)
    n_iter = length(capacity_history)
    if n_iter < 3
        return Dict("status" => "insufficient_data")
    end
    
    # Compute capacity change norms
    capacity_change_norms = []
    for i in 2:n_iter
        change = norm(capacity_history[i] - capacity_history[i-1])
        push!(capacity_change_norms, change)
    end
    
    # Compute max PMR evolution
    max_pmr_evolution = [maximum(abs.(pmr)) for pmr in pmr_history]
    
    # Oscillation detection (look for alternating signs in capacity changes)
    oscillation_metric = 0.0
    if length(capacity_change_norms) >= 4
        # Check for oscillatory behavior in the last few iterations
        recent_changes = capacity_change_norms[end-3:end]
        if all(recent_changes .> 1e-6)  # Non-trivial changes
            # Simple oscillation metric: variance in change direction
            directions = sign.(recent_changes[2:end] - recent_changes[1:end-1])
            oscillation_metric = var(directions)
        end
    end
    
    # Convergence rate estimation (exponential fit to max PMR)
    convergence_rate = NaN
    if length(max_pmr_evolution) >= 5
        recent_pmr = max_pmr_evolution[end-4:end]
        if all(recent_pmr .> 1e-10)
            # Fit exponential decay: PMR(k) ≈ PMR(0) * exp(-rate * k)
            log_pmr = log.(recent_pmr)
            k_vals = collect(1:length(log_pmr))
            if length(k_vals) > 1 && var(log_pmr) > 1e-10
                # Simple linear regression for log(PMR) vs iteration
                convergence_rate = -cov(k_vals, log_pmr) / var(k_vals)
            end
        end
    end
    
    return Dict(
        "capacity_change_norms" => capacity_change_norms,
        "max_pmr_evolution" => max_pmr_evolution,
        "oscillation_metric" => oscillation_metric,
        "convergence_rate" => convergence_rate,
        "final_max_pmr" => max_pmr_evolution[end],
        "final_capacity_change" => capacity_change_norms[end]
    )
end

"""
    compute_equilibrium_jacobian(generators, battery, capacities, battery_power_cap, battery_energy_cap;
                                 perturbation=1e-4)

Estimate the Jacobian matrix of the PMR function around current capacities using finite differences.
This helps analyze stability properties of the equilibrium.
"""
function compute_equilibrium_jacobian(generators, battery, capacities, battery_power_cap, battery_energy_cap;
                                     perturbation=1e-4)
    n_techs = length(generators) + 1  # +1 for battery
    jacobian = zeros(n_techs, n_techs)
    
    # Get baseline PMR
    # Note: This would require re-running operations model, so we'll return a placeholder for now
    # In practice, this would call a streamlined operations model
    
    println("⚠️  Jacobian computation requires operations model integration")
    return jacobian
end

# =============================================================================
# DIAGNOSTICS AND ISSUE IDENTIFICATION
# =============================================================================

"""
    diagnose_convergence_issues(capacity_history, pmr_history, step_size_history)

Diagnose potential convergence issues and suggest remedies.
"""
function diagnose_convergence_issues(capacity_history, pmr_history, step_size_history)
    analysis = analyze_convergence_properties(capacity_history, pmr_history, step_size_history)
    
    if analysis["status"] == "insufficient_data"
        return ["Need more iterations for diagnosis"]
    end
    
    issues = String[]
    suggestions = String[]
    
    # Check for slow convergence
    if analysis["final_max_pmr"] > 5.0 && length(pmr_history) > 20
        push!(issues, "Slow convergence: Max PMR still $(round(analysis["final_max_pmr"], digits=2))% after $(length(pmr_history)) iterations")
        push!(suggestions, "Consider increasing step size or using Anderson acceleration")
    end
    
    # Check for oscillation
    if analysis["oscillation_metric"] > 0.5
        push!(issues, "Oscillatory behavior detected (metric: $(round(analysis["oscillation_metric"], digits=3)))")
        push!(suggestions, "Consider reducing step size or adding damping")
    end
    
    # Check for stagnation
    recent_changes = analysis["capacity_change_norms"][max(1, end-4):end]
    if length(recent_changes) >= 3 && maximum(recent_changes) < 1e-6
        push!(issues, "Capacity changes very small, possible stagnation")
        push!(suggestions, "Check for binding constraints or numerical issues")
    end
    
    # Check convergence rate
    if !isnan(analysis["convergence_rate"]) && analysis["convergence_rate"] < 0.1
        push!(issues, "Very slow convergence rate: $(round(analysis["convergence_rate"], digits=4))")
        push!(suggestions, "Consider alternative iteration schemes or different step size adaptation")
    end
    
    return Dict("issues" => issues, "suggestions" => suggestions, "analysis" => analysis)
end

# =============================================================================
# REPORTING AND SUMMARIES
# =============================================================================

"""
    create_convergence_summary(capacity_history, pmr_history, step_size_history, generators, battery)

Create a comprehensive convergence summary report.
"""
function create_convergence_summary(capacity_history, pmr_history, step_size_history, generators, battery)
    n_iter = length(capacity_history)
    
    if n_iter == 0
        return "No convergence data available"
    end
    
    # Get final state
    final_capacities = capacity_history[end]
    final_pmr = pmr_history[end]
    final_step_size = length(step_size_history) > 0 ? step_size_history[end] : NaN
    
    # Run diagnostics
    diagnostics = diagnose_convergence_issues(capacity_history, pmr_history, step_size_history)
    
    # Create summary
    summary = """
    CONVERGENCE SUMMARY
    ==================
    
    Iterations completed: $n_iter
    Final max PMR: $(round(maximum(abs.(final_pmr)), digits=3))%
    Final step size: $(round(final_step_size, digits=6))
    
    Final Capacities:
    """
    
    for (g, gen) in enumerate(generators)
        capacity_mw = round(final_capacities[g], digits=2)
        pmr_pct = round(final_pmr[g], digits=2)
        summary *= "    $(gen.name): $(capacity_mw) MW (PMR: $(pmr_pct)%)\n"
    end
    
    battery_power = round(final_capacities[end], digits=2)  # Assuming battery power is last
    battery_pmr = round(final_pmr[end], digits=2)
    summary *= "    Battery: $(battery_power) MW (PMR: $(battery_pmr)%)\n"
    
    # Add diagnostics
    if haskey(diagnostics, "issues") && !isempty(diagnostics["issues"])
        summary *= "\n  Issues Identified:\n"
        for issue in diagnostics["issues"]
            summary *= "    ⚠️  $issue\n"
        end
    end
    
    if haskey(diagnostics, "suggestions") && !isempty(diagnostics["suggestions"])
        summary *= "\n  Suggestions:\n"
        for suggestion in diagnostics["suggestions"]
            summary *= "    💡 $suggestion\n"
        end
    end
    
    return summary
end

end # module ConvergenceDiagnostics