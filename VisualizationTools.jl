"""
VisualizationTools.jl

Visualization tools module for ToySystemQuad.jl
Provides plotting and analysis functions for equilibrium convergence and system results.
"""

module VisualizationTools

using Plots, Statistics
using ..SystemConfig: Generator, Battery

export plot_convergence_evolution, plot_capacity_evolution, plot_pmr_evolution
export plot_demand_wind_profiles, plot_price_duration_curve, plot_generation_stack
export create_convergence_dashboard

# =============================================================================
# CONVERGENCE VISUALIZATION
# =============================================================================

"""
    plot_convergence_evolution(capacity_history, pmr_history, step_size_history; save_path=nothing)

Plot the evolution of capacities, PMR, and step sizes during equilibrium iteration.
"""
function plot_convergence_evolution(capacity_history, pmr_history, step_size_history; save_path=nothing)
    n_iter = length(capacity_history)
    if n_iter < 2
        println("⚠️  Need at least 2 iterations for convergence plots")
        return nothing
    end
    
    # Extract max PMR evolution
    max_pmr_evolution = [maximum(abs.(pmr)) for pmr in pmr_history]
    
    # Create subplots
    p1 = plot(1:n_iter, max_pmr_evolution, 
              title="Max PMR Evolution", xlabel="Iteration", ylabel="Max |PMR| (%)",
              lw=2, color=:red, legend=false)
    hline!([1.0], color=:black, linestyle=:dash, alpha=0.5)  # 1% convergence threshold
    
    p2 = plot(title="Step Size Evolution", xlabel="Iteration", ylabel="Step Size",
              legend=false)
    if !isempty(step_size_history)
        plot!(p2, 1:length(step_size_history), step_size_history, lw=2, color=:blue)
    end
    
    # Capacity change norms
    if n_iter > 1
        capacity_changes = [norm(capacity_history[i] - capacity_history[i-1]) for i in 2:n_iter]
        p3 = plot(2:n_iter, capacity_changes,
                  title="Capacity Change Norm", xlabel="Iteration", ylabel="||ΔCapacity||",
                  lw=2, color=:green, legend=false, yscale=:log10)
    else
        p3 = plot(title="Capacity Change Norm (Need >1 iteration)")
    end
    
    # Combine plots
    combined_plot = plot(p1, p2, p3, layout=(3,1), size=(800, 600))
    
    if save_path !== nothing
        savefig(combined_plot, save_path)
        println("📊 Convergence evolution plot saved to: $save_path")
    end
    
    return combined_plot
end

"""
    plot_capacity_evolution(capacity_history, generators, battery; save_path=nothing)

Plot the evolution of individual technology capacities during equilibrium iteration.
"""
function plot_capacity_evolution(capacity_history, generators, battery; save_path=nothing)
    n_iter = length(capacity_history)
    if n_iter < 2
        println("⚠️  Need at least 2 iterations for capacity evolution plots")
        return nothing
    end
    
    # Extract capacity trajectories
    n_techs = length(capacity_history[1])
    tech_names = [gen.name for gen in generators]
    push!(tech_names, "Battery")
    
    p = plot(title="Capacity Evolution", xlabel="Iteration", ylabel="Capacity (MW)",
             size=(800, 500), legend=:outertopright)
    
    for tech in 1:n_techs
        capacity_trajectory = [capacity_history[iter][tech] for iter in 1:n_iter]
        plot!(p, 1:n_iter, capacity_trajectory, 
              label=tech_names[tech], lw=2, marker=:circle, markersize=3)
    end
    
    if save_path !== nothing
        savefig(p, save_path)
        println("📊 Capacity evolution plot saved to: $save_path")
    end
    
    return p
end

"""
    plot_pmr_evolution(pmr_history, generators, battery; save_path=nothing)

Plot the evolution of PMR for each technology during equilibrium iteration.
"""
function plot_pmr_evolution(pmr_history, generators, battery; save_path=nothing)
    n_iter = length(pmr_history)
    if n_iter < 2
        println("⚠️  Need at least 2 iterations for PMR evolution plots")
        return nothing
    end
    
    # Extract PMR trajectories
    n_techs = length(pmr_history[1])
    tech_names = [gen.name for gen in generators]
    push!(tech_names, "Battery")
    
    p = plot(title="PMR Evolution", xlabel="Iteration", ylabel="PMR (%)",
             size=(800, 500), legend=:outertopright)
    
    for tech in 1:n_techs
        pmr_trajectory = [pmr_history[iter][tech] for iter in 1:n_iter]
        plot!(p, 1:n_iter, pmr_trajectory, 
              label=tech_names[tech], lw=2, marker=:circle, markersize=3)
    end
    
    # Add zero line
    hline!([0.0], color=:black, linestyle=:dash, alpha=0.5, label="Equilibrium")
    
    if save_path !== nothing
        savefig(p, save_path)
        println("📊 PMR evolution plot saved to: $save_path")
    end
    
    return p
end

# =============================================================================
# SYSTEM ANALYSIS VISUALIZATION
# =============================================================================

"""
    plot_demand_wind_profiles(actual_demand, actual_wind, nuclear_availability, gas_availability; save_path=nothing)

Plot demand and wind profiles along with thermal availability.
"""
function plot_demand_wind_profiles(actual_demand, actual_wind, nuclear_availability, gas_availability; save_path=nothing)
    T = length(actual_demand)
    hours = 1:T
    
    # Create subplots
    p1 = plot(hours, actual_demand, title="Demand Profile", xlabel="Hour", ylabel="Demand (MW)",
              lw=2, color=:blue, legend=false)
    
    p2 = plot(hours, actual_wind, title="Wind Capacity Factor", xlabel="Hour", ylabel="Capacity Factor",
              lw=2, color=:green, legend=false, ylims=(0, 1))
    
    p3 = plot(hours, nuclear_availability, title="Nuclear Availability", xlabel="Hour", ylabel="Available",
              lw=2, color=:red, legend=false, ylims=(0, 1.1))
    
    p4 = plot(hours, gas_availability, title="Gas Availability", xlabel="Hour", ylabel="Available",
              lw=2, color=:orange, legend=false, ylims=(0, 1.1))
    
    combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 600))
    
    if save_path !== nothing
        savefig(combined_plot, save_path)
        println("📊 Demand and wind profiles plot saved to: $save_path")
    end
    
    return combined_plot
end

"""
    plot_price_duration_curve(prices; save_path=nothing)

Plot price duration curve from operational results.
"""
function plot_price_duration_curve(prices; save_path=nothing)
    sorted_prices = sort(prices, rev=true)
    hours = 1:length(sorted_prices)
    
    p = plot(hours, sorted_prices, title="Price Duration Curve", 
             xlabel="Hours", ylabel="Price ($/MWh)",
             lw=2, color=:purple, legend=false, size=(800, 500))
    
    # Add statistics
    avg_price = mean(prices)
    max_price = maximum(prices)
    hline!([avg_price], color=:red, linestyle=:dash, alpha=0.7, 
           label="Average: \$$(round(avg_price, digits=2))/MWh")
    
    if save_path !== nothing
        savefig(p, save_path)
        println("📊 Price duration curve plot saved to: $save_path")
    end
    
    return p
end

"""
    plot_generation_stack(generation, generators, battery_discharge, actual_demand; save_path=nothing)

Plot generation stack showing dispatch by technology.
"""
function plot_generation_stack(generation, generators, battery_discharge, actual_demand; save_path=nothing)
    T = size(generation, 2)
    hours = 1:T
    
    # Stack generation by technology
    tech_names = [gen.name for gen in generators]
    colors = [:red, :green, :orange, :purple, :blue]  # Nuclear, Wind, Gas, Battery, Demand
    
    p = plot(title="Generation Stack", xlabel="Hour", ylabel="Power (MW)",
             size=(1000, 600), legend=:outertopleft)
    
    # Plot stacked generation
    cumulative = zeros(T)
    for (g, gen_name) in enumerate(tech_names)
        gen_profile = generation[g, :]
        plot!(p, hours, cumulative + gen_profile, fillrange=cumulative, 
              label=gen_name, color=colors[g], alpha=0.7)
        cumulative += gen_profile
    end
    
    # Add battery discharge
    plot!(p, hours, cumulative + battery_discharge, fillrange=cumulative,
          label="Battery", color=colors[4], alpha=0.7)
    cumulative += battery_discharge
    
    # Add demand line
    plot!(p, hours, actual_demand, label="Demand", color=:black, lw=3, linestyle=:dash)
    
    if save_path !== nothing
        savefig(p, save_path)
        println("📊 Generation stack plot saved to: $save_path")
    end
    
    return p
end

# =============================================================================
# DASHBOARD CREATION
# =============================================================================

"""
    create_convergence_dashboard(capacity_history, pmr_history, step_size_history, 
                                generators, battery; save_dir="plots")

Create a comprehensive dashboard with multiple convergence analysis plots.
"""
function create_convergence_dashboard(capacity_history, pmr_history, step_size_history,
                                     generators, battery; save_dir="plots")
    mkpath(save_dir)
    
    plots_created = []
    
    # Convergence evolution
    conv_plot = plot_convergence_evolution(capacity_history, pmr_history, step_size_history;
                                          save_path=joinpath(save_dir, "convergence_evolution.png"))
    if conv_plot !== nothing
        push!(plots_created, "convergence_evolution.png")
    end
    
    # Capacity evolution
    cap_plot = plot_capacity_evolution(capacity_history, generators, battery;
                                      save_path=joinpath(save_dir, "capacity_evolution.png"))
    if cap_plot !== nothing
        push!(plots_created, "capacity_evolution.png")
    end
    
    # PMR evolution
    pmr_plot = plot_pmr_evolution(pmr_history, generators, battery;
                                 save_path=joinpath(save_dir, "pmr_evolution.png"))
    if pmr_plot !== nothing
        push!(plots_created, "pmr_evolution.png")
    end
    
    println("📊 Convergence dashboard created with $(length(plots_created)) plots:")
    for plot_name in plots_created
        println("   - $plot_name")
    end
    
    return plots_created
end

end # module VisualizationTools