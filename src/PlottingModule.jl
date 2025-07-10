"""
PlottingModule.jl

Comprehensive plotting module for ToySystemQuad.jl results analysis.
Handles price analysis, generation stacks, and model comparisons.
"""

module PlottingModule

using Plots, Statistics, CSV, DataFrames
using ..SystemConfig: Generator, Battery, SystemProfiles

# Paper-quality plotting settings optimized for publication
function setup_paper_quality_plots()
    # Use GR backend for better performance and quality
    gr()
    
    # Set default plot attributes for publication quality
    default(fontfamily="Times New Roman",
            titlefontsize=9,     # Much smaller titles
            labelfontsize=8,     # Smaller axis labels
            tickfontsize=7,      # Smaller tick labels
            legendfontsize=7,    # Compact legend
            linewidth=1.5,       # Thinner lines
            dpi=300,
            background_color=:white,
            foreground_color=:black,
            grid=false,
            framestyle=:box,
            margin=3Plots.mm)    # Tighter margins
end

# Professional color palette (colorblind-friendly)
const PAPER_COLORS = [
    RGB(0.0, 0.4, 0.8),     # Blue
    RGB(0.8, 0.2, 0.2),     # Red  
    RGB(0.0, 0.6, 0.3),     # Green
    RGB(0.9, 0.6, 0.0),     # Orange
    RGB(0.6, 0.0, 0.8),     # Purple
    RGB(0.4, 0.4, 0.4),     # Gray
    RGB(0.8, 0.8, 0.0),     # Yellow
    RGB(0.0, 0.8, 0.8)      # Cyan
]

# Line styles for different models
const PAPER_LINESTYLES = [:solid, :dash, :dot, :dashdot]

# Initialize paper quality settings when module is loaded
function __init__()
    setup_paper_quality_plots()
end

export plot_price_time_series, plot_price_duration_curves, plot_combined_price_analysis
export plot_generation_stacks, plot_system_profiles, plot_capacity_comparison
export plot_battery_operations, plot_battery_soc_comparison
export generate_all_plots, save_price_analysis
export plot_capacity_mix_differences, plot_capacity_mix_stacked, calculate_operational_results_comparison

"""
    plot_price_time_series(prices, model_name; save_path=nothing)

Plot price time series for a single model over the entire time horizon.
"""
function plot_price_time_series(prices, model_name; save_path=nothing)
    T = length(prices)
    hours = 1:T
    
    p = plot(hours, prices, 
             title="Price Time Series - $model_name",
             xlabel="Hour", 
             ylabel="Price (\$/MWh)",
             color=PAPER_COLORS[1],
             size=(8*72, 4*72),  # 8x4 inches at 72 DPI base
             legend=false)
    
    # Add statistics annotations
    avg_price = mean(prices)
    max_price = maximum(prices)
    min_price = minimum(prices)
    
    hline!([avg_price], color=PAPER_COLORS[2], linestyle=:dash, alpha=0.8, 
           label="Average: \$$(round(avg_price, digits=2))/MWh")
    
    # Add text annotation with stats
    annotate!(T*0.7, max_price*0.9, 
             text("Avg: \$$(round(avg_price, digits=1))\nMax: \$$(round(max_price, digits=1))\nMin: \$$(round(min_price, digits=1))", 
                  :left, 8, :black))
    
    if save_path !== nothing
        savefig(p, save_path)
        println("Price time series plot saved: $save_path")
    end
    
    return p
end

"""
    plot_price_duration_curves(price_results; save_path=nothing)

Plot price duration curves for all models (Perfect Foresight, DLAC-i, SLAC).
"""
function plot_price_duration_curves(price_results; save_path=nothing)
    p = plot(title="Price Duration Curves Comparison", 
             xlabel="Hours", 
             ylabel="Price (\$/MWh)",
             size=(8*72, 5*72),
             legend=:topright)
    
    colors = PAPER_COLORS[1:3]
    linestyles = PAPER_LINESTYLES[1:3]
    
    for (i, (model_name, prices)) in enumerate(price_results)
        sorted_prices = sort(prices, rev=true)
        hours = 1:length(sorted_prices)
        
        plot!(p, hours, sorted_prices, 
              label=model_name, 
              color=colors[i], 
              linestyle=linestyles[i])
    end
    
    if save_path !== nothing
        savefig(p, save_path)
        println("Price duration curves plot saved: $save_path")
    end
    
    return p
end

"""
    plot_combined_price_analysis(pf_prices, dlac_prices, slac_prices; save_path=nothing)

Create comprehensive price analysis with time series and duration curves.
"""
function plot_combined_price_analysis(pf_prices, dlac_prices, slac_prices; save_path=nothing)
    T = length(pf_prices)
    hours = 1:T
    
    # Time series comparison
    p1 = plot(title="Price Time Series Comparison", 
              xlabel="Hour", 
              ylabel="Price (\$/MWh)",
              size=(8*72, 3*72),
              legend=:topright)
    
    plot!(p1, hours, pf_prices, label="Perfect Foresight", color=PAPER_COLORS[2], linestyle=:dash)
    plot!(p1, hours, dlac_prices, label="DLAC-i", color=PAPER_COLORS[3], linestyle=:dot)
    plot!(p1, hours, slac_prices, label="SLAC", color=PAPER_COLORS[1], linestyle=:solid)
    
    # Duration curves
    p2 = plot(title="Price Duration Curves", 
              xlabel="Hours", 
              ylabel="Price (\$/MWh)",
              size=(8*72, 3*72),
              legend=:topright)
    
    sorted_pf = sort(pf_prices, rev=true)
    sorted_dlac = sort(dlac_prices, rev=true)
    sorted_slac = sort(slac_prices, rev=true)
    
    plot!(p2, hours, sorted_pf, label="Perfect Foresight", color=PAPER_COLORS[2], linestyle=:dash)
    plot!(p2, hours, sorted_dlac, label="DLAC-i", color=PAPER_COLORS[3], linestyle=:dot)
    plot!(p2, hours, sorted_slac, label="SLAC", color=PAPER_COLORS[1], linestyle=:solid)
    
    # Price difference analysis
    p3 = plot(title="Price Differences from Perfect Foresight", 
              xlabel="Hour", 
              ylabel="Price Difference (\$/MWh)",
              size=(8*72, 3*72),
              legend=:topright,
              titlefontsize=9,
              labelfontsize=8,
              tickfontsize=7,
              legendfontsize=7,
              left_margin=8Plots.mm,
              bottom_margin=6Plots.mm)
    
    plot!(p3, hours, dlac_prices - pf_prices, label="DLAC-i - PF", color=PAPER_COLORS[3])
    plot!(p3, hours, slac_prices - pf_prices, label="SLAC - PF", color=PAPER_COLORS[1])
    hline!(p3, [0], color=:black, linestyle=:dash, alpha=0.7, label="Zero Difference")
    
    # Combine all plots
    combined_plot = plot(p1, p2, p3, layout=(3,1), size=(8*72, 9*72))
    
    if save_path !== nothing
        savefig(combined_plot, save_path)
        println("Combined price analysis plot saved: $save_path")
    end
    
    return combined_plot
end

"""
    plot_generation_stacks(generation_results, battery_results, demand, generators; save_path=nothing)

Plot generation stack for each model showing dispatch by technology.
"""
function plot_generation_stacks(generation_results, battery_results, demand, generators; save_path=nothing)
    plots_list = []
    
    for (model_name, generation) in generation_results
        T = size(generation, 2)
        hours = 1:T
        
        # Technology colors - use paper-quality palette
        colors = PAPER_COLORS[1:4]  # Nuclear, Wind, Gas, Battery
        
        p = plot(title="Generation Stack - $model_name", 
                 xlabel="Hour", 
                 ylabel="Power (MW)",
                 size=(6*72, 4*72),  # Optimized for half-page width
                 legend=:outertopleft,
                 left_margin=10Plots.mm,
                 bottom_margin=10Plots.mm)
        
        # Plot stacked generation
        cumulative = zeros(T)
        for (g, gen_name) in enumerate([gen.name for gen in generators])
            gen_profile = generation[g, :]
            plot!(p, hours, cumulative + gen_profile, fillrange=cumulative, 
                  label=gen_name, color=colors[g], alpha=0.7)
            cumulative += gen_profile
        end
        
        # Add battery discharge if available
        if haskey(battery_results, model_name)
            battery_discharge = battery_results[model_name]
            plot!(p, hours, cumulative + battery_discharge, fillrange=cumulative,
                  label="Battery", color=colors[4], alpha=0.7)
            cumulative += battery_discharge
        end
        
        # Add demand line
        plot!(p, hours, demand, label="Demand", color=:black, linewidth=3, linestyle=:dash)
        
        push!(plots_list, p)
    end
    
    # Combine all generation stacks
    if length(plots_list) == 3
        combined_plot = plot(plots_list..., layout=(3,1), size=(6*72, 12*72))
    else
        combined_plot = plot(plots_list..., layout=(length(plots_list),1), size=(6*72, 4*72*length(plots_list)))
    end
    
    if save_path !== nothing
        savefig(combined_plot, save_path)
        println("Generation stacks plot saved: $save_path")
    end
    
    return combined_plot
end

"""
    plot_system_profiles(profiles::SystemProfiles; save_path=nothing)

Plot system demand and availability profiles from SystemProfiles struct.
"""
function plot_system_profiles(profiles::SystemProfiles; save_path=nothing)
    T = length(profiles.actual_demand)
    hours = 1:T
    
    # Create subplots with paper-quality styling
    p1 = plot(hours, profiles.actual_demand, title="Demand Profile", xlabel="Hour", ylabel="Demand (MW)",
              color=PAPER_COLORS[1], legend=false, left_margin=6Plots.mm, bottom_margin=6Plots.mm)
    
    p2 = plot(hours, profiles.actual_wind, title="Wind Capacity Factor", xlabel="Hour", ylabel="Capacity Factor",
              color=PAPER_COLORS[3], legend=false, ylims=(0, 1), left_margin=6Plots.mm, bottom_margin=6Plots.mm)
    
    p3 = plot(hours, profiles.actual_nuclear_availability, title="Nuclear Availability", xlabel="Hour", ylabel="Available",
              color=PAPER_COLORS[2], legend=false, ylims=(0, 1.1), left_margin=6Plots.mm, bottom_margin=6Plots.mm)
    
    p4 = plot(hours, profiles.actual_gas_availability, title="Gas Availability", xlabel="Hour", ylabel="Available",
              color=PAPER_COLORS[4], legend=false, ylims=(0, 1.1), left_margin=6Plots.mm, bottom_margin=6Plots.mm)
    
    combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(8*72, 6*72))
    
    if save_path !== nothing
        savefig(combined_plot, save_path)
        println("System profiles plot saved: $save_path")
    end
    
    return combined_plot
end

"""
    plot_system_profiles(actual_demand, actual_wind, nuclear_availability, gas_availability; save_path=nothing)

Plot system demand and availability profiles (legacy method for backwards compatibility).
"""
function plot_system_profiles(actual_demand, actual_wind, nuclear_availability, gas_availability; save_path=nothing)
    T = length(actual_demand)
    hours = 1:T
    
    # Create subplots with paper-quality styling
    p1 = plot(hours, actual_demand, title="Demand Profile", xlabel="Hour", ylabel="Demand (MW)",
              color=PAPER_COLORS[1], legend=false, left_margin=6Plots.mm, bottom_margin=6Plots.mm)
    
    p2 = plot(hours, actual_wind, title="Wind Capacity Factor", xlabel="Hour", ylabel="Capacity Factor",
              color=PAPER_COLORS[3], legend=false, ylims=(0, 1), left_margin=6Plots.mm, bottom_margin=6Plots.mm)
    
    p3 = plot(hours, nuclear_availability, title="Nuclear Availability", xlabel="Hour", ylabel="Available",
              color=PAPER_COLORS[2], legend=false, ylims=(0, 1.1), left_margin=6Plots.mm, bottom_margin=6Plots.mm)
    
    p4 = plot(hours, gas_availability, title="Gas Availability", xlabel="Hour", ylabel="Available",
              color=PAPER_COLORS[4], legend=false, ylims=(0, 1.1), left_margin=6Plots.mm, bottom_margin=6Plots.mm)
    
    combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(8*72, 6*72))
    
    if save_path !== nothing
        savefig(combined_plot, save_path)
        println("System profiles plot saved: $save_path")
    end
    
    return combined_plot
end

"""
    plot_capacity_comparison(generators, battery, optimal_capacities, optimal_battery_power; save_path=nothing)

Plot capacity comparison showing optimal investments.
"""
function plot_capacity_comparison(generators, battery, optimal_capacities, optimal_battery_power; save_path=nothing)
    tech_names = [gen.name for gen in generators]
    push!(tech_names, "Battery")
    
    capacities = [optimal_capacities; optimal_battery_power]
    colors = PAPER_COLORS[1:length(capacities)]
    
    p = bar(tech_names, capacities,
            title="Optimal Capacity Investments",
            xlabel="Technology",
            ylabel="Capacity (MW)",
            color=colors,
            legend=false,
            size=(6*72, 4*72),
            left_margin=10Plots.mm,
            bottom_margin=10Plots.mm)
    
    # Add capacity values on top of bars
    for (i, cap) in enumerate(capacities)
        annotate!(i, cap + maximum(capacities)*0.02, text("$(round(cap, digits=1)) MW", :center, 8, :black))
    end
    
    if save_path !== nothing
        savefig(p, save_path)
        println("Capacity comparison plot saved: $save_path")
    end
    
    return p
end

"""
    plot_battery_operations(battery_results, model_names; save_path=nothing)

Plot battery charge/discharge operations for all models.
"""
function plot_battery_operations(battery_results, model_names; save_path=nothing)
    plots_list = []
    colors = PAPER_COLORS[1:3]
    
    for (i, model_name) in enumerate(model_names)
        if haskey(battery_results, model_name)
            charge = battery_results[model_name]["charge"]
            discharge = battery_results[model_name]["discharge"]
            T = length(charge)
            hours = 1:T
            
            p = plot(title="Battery Operations - $model_name",
                    xlabel="Hour",
                    ylabel="Power (MW)",
                    size=(6*72, 3*72),
                    legend=:topright,
                    left_margin=8Plots.mm,
                    bottom_margin=8Plots.mm)
            
            # Plot charging (negative values)
            plot!(p, hours, -charge, label="Charging", color=colors[i], alpha=0.7, fillrange=0)
            
            # Plot discharging (positive values)
            plot!(p, hours, discharge, label="Discharging", color=colors[i], alpha=0.9, fillrange=0)
            
            # Add zero line
            hline!(p, [0], color=:black, linestyle=:dash, alpha=0.5, label="")
            
            push!(plots_list, p)
        end
    end
    
    # Combine all battery operation plots
    if length(plots_list) == 3
        combined_plot = plot(plots_list..., layout=(3,1), size=(6*72, 9*72))
    else
        combined_plot = plot(plots_list..., layout=(length(plots_list),1), size=(6*72, 3*72*length(plots_list)))
    end
    
    if save_path !== nothing
        savefig(combined_plot, save_path)
        println("Battery operations plot saved: $save_path")
    end
    
    return combined_plot
end

"""
    plot_battery_soc_comparison(battery_results, model_names; save_path=nothing)

Plot battery state of charge comparison across models.
"""
function plot_battery_soc_comparison(battery_results, model_names; save_path=nothing)
    colors = PAPER_COLORS[1:3]
    linestyles = PAPER_LINESTYLES[1:3]
    
    p = plot(title="Battery State of Charge Comparison",
             xlabel="Hour",
             ylabel="SOC (MWh)",
             size=(8*72, 4*72),
             legend=:topright,
             left_margin=10Plots.mm,
             bottom_margin=10Plots.mm)
    
    for (i, model_name) in enumerate(model_names)
        if haskey(battery_results, model_name)
            soc = battery_results[model_name]["soc"]
            T = length(soc)
            hours = 1:T
            
            plot!(p, hours, soc,
                  label=model_name,
                  color=colors[i],
                  linestyle=linestyles[i])
        end
    end
    
    if save_path !== nothing
        savefig(p, save_path)
        println("Battery SOC comparison plot saved: $save_path")
    end
    
    return p
end

"""
    save_price_analysis(pf_prices, dlac_prices, slac_prices, output_dir)

Save detailed price analysis to CSV files.
"""
function save_price_analysis(pf_prices, dlac_prices, slac_prices, output_dir)
    T = length(pf_prices)
    
    # Detailed price analysis
    price_analysis_df = DataFrame(
        Hour = 1:T,
        PF_Price = pf_prices,
        DLAC_i_Price = dlac_prices,
        SLAC_Price = slac_prices,
        DLAC_i_vs_PF_Diff = dlac_prices - pf_prices,
        SLAC_vs_PF_Diff = slac_prices - pf_prices,
        SLAC_vs_DLAC_i_Diff = slac_prices - dlac_prices
    )
    CSV.write(joinpath(output_dir, "comprehensive_price_analysis.csv"), price_analysis_df)
    
    # Price statistics summary
    price_stats_df = DataFrame(
        Model = ["Perfect_Foresight", "DLAC_i", "SLAC"],
        Mean_Price = [mean(pf_prices), mean(dlac_prices), mean(slac_prices)],
        Max_Price = [maximum(pf_prices), maximum(dlac_prices), maximum(slac_prices)],
        Min_Price = [minimum(pf_prices), minimum(dlac_prices), minimum(slac_prices)],
        Std_Price = [std(pf_prices), std(dlac_prices), std(slac_prices)],
        Price_Volatility = [std(pf_prices)/mean(pf_prices), std(dlac_prices)/mean(dlac_prices), std(slac_prices)/mean(slac_prices)]
    )
    CSV.write(joinpath(output_dir, "price_statistics_summary.csv"), price_stats_df)
    
    println("Price analysis CSV files saved")
end

"""
    generate_all_plots(pf_result, dlac_result, slac_result, profiles::SystemProfiles,
                      generators, battery, optimal_capacities, optimal_battery_power, output_dir)

Generate all comprehensive plots for the system analysis using SystemProfiles.
"""
function generate_all_plots(pf_result, dlac_result, slac_result, profiles::SystemProfiles,
                           generators, battery, optimal_capacities, optimal_battery_power, output_dir)
    plots_dir = joinpath(output_dir, "plots")
    mkpath(plots_dir)
    
    try
        # Extract prices
        pf_prices = pf_result["prices"]
        dlac_prices = dlac_result["prices"]
        slac_prices = slac_result["prices"]
        
        # Individual price time series
        plot_price_time_series(pf_prices, "Perfect Foresight"; 
                              save_path=joinpath(plots_dir, "pf_price_time_series.png"))
        plot_price_time_series(dlac_prices, "DLAC-i"; 
                              save_path=joinpath(plots_dir, "dlac_i_price_time_series.png"))
        plot_price_time_series(slac_prices, "SLAC"; 
                              save_path=joinpath(plots_dir, "slac_price_time_series.png"))
        
        # Price duration curves (all models)
        price_results = [
            ("Perfect Foresight", pf_prices),
            ("DLAC-i", dlac_prices),
            ("SLAC", slac_prices)
        ]
        plot_price_duration_curves(price_results; 
                                  save_path=joinpath(plots_dir, "price_duration_curves.png"))
        
        # Combined comprehensive price analysis
        plot_combined_price_analysis(pf_prices, dlac_prices, slac_prices; 
                                    save_path=joinpath(plots_dir, "comprehensive_price_analysis.png"))
        
        # Generation stacks
        generation_results = [
            ("Perfect Foresight", pf_result["generation"]),
            ("DLAC-i", dlac_result["generation"]),
            ("SLAC", slac_result["generation"])
        ]
        
        battery_discharge_results = Dict(
            "SLAC" => slac_result["battery_discharge"],
            "Perfect Foresight" => pf_result["battery_discharge"],
            "DLAC-i" => dlac_result["battery_discharge"]
        )
        
        plot_generation_stacks(generation_results, battery_discharge_results, profiles.actual_demand, generators;
                              save_path=joinpath(plots_dir, "generation_stacks.png"))
        
        # Battery operations plots
        battery_detailed_results = Dict(
            "SLAC" => Dict(
                "charge" => slac_result["battery_charge"],
                "discharge" => slac_result["battery_discharge"],
                "soc" => slac_result["battery_soc"]
            ),
            "Perfect Foresight" => Dict(
                "charge" => pf_result["battery_charge"],
                "discharge" => pf_result["battery_discharge"],
                "soc" => pf_result["battery_soc"]
            ),
            "DLAC-i" => Dict(
                "charge" => dlac_result["battery_charge"],
                "discharge" => dlac_result["battery_discharge"],
                "soc" => dlac_result["battery_soc"]
            )
        )
        
        model_names = ["Perfect Foresight", "DLAC-i", "SLAC"]
        
        # Plot battery operations (charge/discharge)
        plot_battery_operations(battery_detailed_results, model_names;
                               save_path=joinpath(plots_dir, "battery_operations.png"))
        
        # Plot battery SOC comparison
        plot_battery_soc_comparison(battery_detailed_results, model_names;
                                   save_path=joinpath(plots_dir, "battery_soc_comparison.png"))
        
        # System profiles
        plot_system_profiles(profiles; save_path=joinpath(plots_dir, "system_profiles.png"))
        
        # Capacity comparison
        plot_capacity_comparison(generators, battery, optimal_capacities, optimal_battery_power;
                                save_path=joinpath(plots_dir, "capacity_comparison.png"))
        
        # Save price analysis to CSV
        save_price_analysis(pf_prices, dlac_prices, slac_prices, output_dir)
        
        println("All plots generated successfully in: $plots_dir")
        
        return true
        
    catch e
        println("Plot generation failed (likely missing Plots.jl): $e")
        println("Results are still available in CSV files")
        return false
    end
end

"""
    generate_all_plots(pf_result, dlac_result, slac_result, actual_demand, actual_wind,
                      nuclear_availability, gas_availability, generators, battery,
                      optimal_capacities, optimal_battery_power, output_dir)

Generate all comprehensive plots (legacy method for backwards compatibility).
"""
function generate_all_plots(pf_result, dlac_result, slac_result, actual_demand, actual_wind,
                           nuclear_availability, gas_availability, generators, battery,
                           optimal_capacities, optimal_battery_power, output_dir)
    plots_dir = joinpath(output_dir, "plots")
    mkpath(plots_dir)
    
    try
        # Extract prices
        pf_prices = pf_result["prices"]
        dlac_prices = dlac_result["prices"]
        slac_prices = slac_result["prices"]
        
        # Individual price time series
        plot_price_time_series(pf_prices, "Perfect Foresight"; 
                              save_path=joinpath(plots_dir, "pf_price_time_series.png"))
        plot_price_time_series(dlac_prices, "DLAC-i"; 
                              save_path=joinpath(plots_dir, "dlac_i_price_time_series.png"))
        plot_price_time_series(slac_prices, "SLAC"; 
                              save_path=joinpath(plots_dir, "slac_price_time_series.png"))
        
        # Price duration curves (all models)
        price_results = [
            ("Perfect Foresight", pf_prices),
            ("DLAC-i", dlac_prices),
            ("SLAC", slac_prices)
        ]
        plot_price_duration_curves(price_results; 
                                  save_path=joinpath(plots_dir, "price_duration_curves.png"))
        
        # Combined comprehensive price analysis
        plot_combined_price_analysis(pf_prices, dlac_prices, slac_prices; 
                                    save_path=joinpath(plots_dir, "comprehensive_price_analysis.png"))
        
        # Generation stacks
        generation_results = [
            ("Perfect Foresight", pf_result["generation"]),
            ("DLAC-i", dlac_result["generation"]),
            ("SLAC", slac_result["generation"])
        ]
        
        battery_discharge_results = Dict(
            "SLAC" => slac_result["battery_discharge"],
            "Perfect Foresight" => pf_result["battery_discharge"],
            "DLAC-i" => dlac_result["battery_discharge"]
        )
        
        plot_generation_stacks(generation_results, battery_discharge_results, actual_demand, generators;
                              save_path=joinpath(plots_dir, "generation_stacks.png"))
        
        # Battery operations plots
        battery_detailed_results = Dict(
            "SLAC" => Dict(
                "charge" => slac_result["battery_charge"],
                "discharge" => slac_result["battery_discharge"],
                "soc" => slac_result["battery_soc"]
            ),
            "Perfect Foresight" => Dict(
                "charge" => pf_result["battery_charge"],
                "discharge" => pf_result["battery_discharge"],
                "soc" => pf_result["battery_soc"]
            ),
            "DLAC-i" => Dict(
                "charge" => dlac_result["battery_charge"],
                "discharge" => dlac_result["battery_discharge"],
                "soc" => dlac_result["battery_soc"]
            )
        )
        
        model_names = ["Perfect Foresight", "DLAC-i", "SLAC"]
        
        # Plot battery operations (charge/discharge)
        plot_battery_operations(battery_detailed_results, model_names;
                               save_path=joinpath(plots_dir, "battery_operations.png"))
        
        # Plot battery SOC comparison
        plot_battery_soc_comparison(battery_detailed_results, model_names;
                                   save_path=joinpath(plots_dir, "battery_soc_comparison.png"))
        
        # System profiles
        plot_system_profiles(actual_demand, actual_wind, nuclear_availability, gas_availability;
                            save_path=joinpath(plots_dir, "system_profiles.png"))
        
        # Capacity comparison
        plot_capacity_comparison(generators, battery, optimal_capacities, optimal_battery_power;
                                save_path=joinpath(plots_dir, "capacity_comparison.png"))
        
        # Save price analysis to CSV
        save_price_analysis(pf_prices, dlac_prices, slac_prices, output_dir)
        
        println("All plots generated successfully in: $plots_dir")
        
        return true
        
    catch e
        println("Plot generation failed (likely missing Plots.jl): $e")
        println("Results are still available in CSV files")
        return false
    end
end

"""
    plot_capacity_mix_differences(equilibrium_results_dir; save_path=nothing)

Plot capacity mix differences from equilibrium results for DLAC-i, SLAC, and Perfect Foresight policies.
Uses the last row of equilibrium_log.csv files for each policy.
"""
function plot_capacity_mix_differences(equilibrium_results_dir; save_path=nothing)
    # Define paths to equilibrium log files
    slac_path = joinpath(equilibrium_results_dir, "equilibrium", "slac", "equilibrium_log.csv")
    dlac_path = joinpath(equilibrium_results_dir, "equilibrium", "dlac_i", "equilibrium_log.csv")
    pf_path = joinpath(equilibrium_results_dir, "validation", "equilibrium", "perfectforesight", "equilibrium_log.csv")
    
    # Read the last row of each equilibrium log file
    function read_last_row(filepath)
        df = CSV.read(filepath, DataFrame)
        return df[end, :]
    end
    
    # Extract capacity data
    slac_data = read_last_row(slac_path)
    dlac_data = read_last_row(dlac_path)
    pf_data = read_last_row(pf_path)
    
    # Technology names
    tech_names = ["Nuclear", "Wind", "Gas", "Battery"]
    
    # Extract capacities (MW)
    slac_capacities = [slac_data.Nuclear_capacity_MW, slac_data.Wind_capacity_MW, 
                      slac_data.Gas_capacity_MW, slac_data.Battery_capacity_MW]
    dlac_capacities = [dlac_data.Nuclear_capacity_MW, dlac_data.Wind_capacity_MW, 
                      dlac_data.Gas_capacity_MW, dlac_data.Battery_capacity_MW]
    pf_capacities = [pf_data.Nuclear_capacity_MW, pf_data.Wind_capacity_MW, 
                    pf_data.Gas_capacity_MW, pf_data.Battery_capacity_MW]
    
    # Create grouped bar chart with publication-quality formatting
    x_positions = 1:length(tech_names)
    bar_width = 0.25
    
    p = plot(title="Equilibrium Capacity Mix by Policy",
             xlabel="Technology",
             ylabel="Installed Capacity (MW)",
             size=(10*72, 6*72),
             legend=:topright)
    
    # Use distinct colors for better visibility
    colors = [PAPER_COLORS[1], PAPER_COLORS[2], PAPER_COLORS[3]]  # Blue, Red, Green
    
    # Plot bars for each policy with better styling
    bar!(p, x_positions .- bar_width, dlac_capacities, 
         width=bar_width, label="DLAC-i", color=colors[1], alpha=0.7, 
         linewidth=1, linecolor=colors[1])
    bar!(p, x_positions, slac_capacities, 
         width=bar_width, label="SLAC", color=colors[2], alpha=0.7,
         linewidth=1, linecolor=colors[2])
    bar!(p, x_positions .+ bar_width, pf_capacities, 
         width=bar_width, label="Perfect Foresight", color=colors[3], alpha=0.7,
         linewidth=1, linecolor=colors[3])
    
    # Set x-axis labels with better spacing
    plot!(p, xticks=(x_positions, tech_names), xrotation=0)
    
    # Add subtle grid for better readability
    plot!(p, grid=true, gridwidth=0.5, gridcolor=:lightgray, gridalpha=0.4)
    
    # Add capacity values on top of bars with smaller, cleaner text
    for (i, tech) in enumerate(tech_names)
        max_height = maximum([dlac_capacities[i], slac_capacities[i], pf_capacities[i]])
        y_offset = max_height * 0.03  # Smaller offset for cleaner look
        
        # DLAC-i values
        if dlac_capacities[i] > 0
            annotate!(p, i - bar_width, dlac_capacities[i] + y_offset, 
                     text("$(round(dlac_capacities[i], digits=0))", :center, :black))
        end
        
        # SLAC values
        if slac_capacities[i] > 0
            annotate!(p, i, slac_capacities[i] + y_offset, 
                     text("$(round(slac_capacities[i], digits=0))", :center, :black))
        end
        
        # PF values
        if pf_capacities[i] > 0
            annotate!(p, i + bar_width, pf_capacities[i] + y_offset, 
                     text("$(round(pf_capacities[i], digits=0))", :center, :black))
        end
    end
    
    # Set y-axis to start from 0 and add some headroom
    max_capacity = maximum([maximum(dlac_capacities), maximum(slac_capacities), maximum(pf_capacities)])
    ylims!(p, 0, max_capacity * 1.15)
    
    # Clean title without subtitle
    plot!(p, title="Equilibrium Capacity Mix by Policy")
    
    if save_path !== nothing
        savefig(p, save_path)
        println("Capacity mix differences plot saved: $save_path")
    end
    
    return p, Dict("DLAC-i" => dlac_capacities, "SLAC" => slac_capacities, "Perfect Foresight" => pf_capacities)
end

"""
    plot_capacity_mix_stacked(equilibrium_results_dir; save_path=nothing)

Plot stacked bar chart showing capacity mix proportions for each policy.
"""
function plot_capacity_mix_stacked(equilibrium_results_dir; save_path=nothing)
    # Get capacity data using the existing function
    _, capacity_data = plot_capacity_mix_differences(equilibrium_results_dir; save_path=nothing)
    
    # Technology names and colors - use very distinct colors
    tech_names = ["Nuclear", "Wind", "Gas", "Battery"]
    colors = [
        RGB(0.1, 0.1, 0.8),     # Blue - Nuclear
        RGB(0.0, 0.8, 0.0),     # Bright Green - Wind  
        RGB(0.9, 0.5, 0.0),     # Orange - Gas
        RGB(0.8, 0.0, 0.8)      # Magenta - Battery
    ]
    
    # Policy names and data
    policies = ["DLAC-i", "SLAC", "Perfect Foresight"]
    policy_data = [capacity_data["DLAC-i"], capacity_data["SLAC"], capacity_data["Perfect Foresight"]]
    
    # Calculate total capacities for percentages
    totals = [sum(data) for data in policy_data]
    
    # Create simple stacked bar chart using StatsPlots approach
    # Reshape data for plotting: each column is a policy, each row is a technology
    data_matrix = hcat(policy_data...)  # 4×3 matrix
    
    # Create the stacked bar chart using manual stacking since GR doesn't support bar_position=:stack
    p = plot(title="Capacity Mix by Policy",
             xlabel="Policy",
             ylabel="Installed Capacity (MW)",
             size=(8*72, 6*72),
             legend=:topright)
    
    # Create stacked bars manually using the bottom parameter
    x_positions = 1:length(policies)
    bar_width = 0.6
    
    # Start with the first technology (Nuclear) at the bottom
    bar!(p, x_positions, [data[1] for data in policy_data], 
         width=bar_width, label=tech_names[1], color=colors[1], alpha=0.8)
    
    # Add subsequent technologies on top
    for i in 2:length(tech_names)
        bottom_values = [sum(data[1:i-1]) for data in policy_data]
        bar!(p, x_positions, [data[i] for data in policy_data], 
             width=bar_width, label=tech_names[i], color=colors[i], alpha=0.8,
             bottom=bottom_values)
    end
    
    # Set x-axis labels
    plot!(p, xticks=(1:length(policies), policies), xrotation=0)
    
    # Add subtle grid for better readability
    plot!(p, grid=true, gridwidth=0.5, gridcolor=:lightgray, gridalpha=0.3)
    
    # Remove total capacity labels on top for cleaner look
    
    # Set y-axis limits
    max_total = maximum(totals)
    ylims!(p, 0, max_total * 1.1)
    
    if save_path !== nothing
        savefig(p, save_path)
        println("Stacked capacity mix plot saved: $save_path")
    end
    
    return p
end

"""
    calculate_operational_results_comparison(equilibrium_results_dir; save_path=nothing)

Calculate operational results and total system costs for each policy's capacity mix.
Returns operational metrics and total system costs (investment + fixed O&M + operations).
"""
function calculate_operational_results_comparison(equilibrium_results_dir; save_path=nothing)
    # Define paths to operations files
    slac_ops_path = joinpath(equilibrium_results_dir, "equilibrium", "slac", "slac_operations.csv")
    dlac_ops_path = joinpath(equilibrium_results_dir, "equilibrium", "dlac_i", "dlac_i_operations.csv")
    pf_ops_path = joinpath(equilibrium_results_dir, "validation", "equilibrium", "perfectforesight", "perfect_foresight_operations.csv")
    
    # Define paths to equilibrium log files for total costs
    slac_log_path = joinpath(equilibrium_results_dir, "equilibrium", "slac", "equilibrium_log.csv")
    dlac_log_path = joinpath(equilibrium_results_dir, "equilibrium", "dlac_i", "equilibrium_log.csv")
    pf_log_path = joinpath(equilibrium_results_dir, "validation", "equilibrium", "perfectforesight", "equilibrium_log.csv")
    
    # Read operational results
    slac_ops = CSV.read(slac_ops_path, DataFrame)
    dlac_ops = CSV.read(dlac_ops_path, DataFrame)
    pf_ops = CSV.read(pf_ops_path, DataFrame)
    
    # Read total costs from equilibrium logs (last row)
    slac_log = CSV.read(slac_log_path, DataFrame)
    dlac_log = CSV.read(dlac_log_path, DataFrame)
    pf_log = CSV.read(pf_log_path, DataFrame)
    
    # Get capacity data for cost calculations
    slac_capacity_data = slac_log[end, :]
    dlac_capacity_data = dlac_log[end, :]
    pf_capacity_data = pf_log[end, :]
    
    # Cost parameters from SystemConfig.jl
    # Investment costs ($/MW/year)
    nuclear_inv_cost = 120_000.0
    wind_inv_cost = 85_000.0  
    gas_inv_cost = 70_000.0
    battery_power_inv_cost = 95_000.0
    battery_energy_inv_cost = 100.0  # $/MWh/year
    
    # Fixed O&M costs ($/MW/year)
    nuclear_fixed_om = 35_000.0
    wind_fixed_om = 12_000.0
    gas_fixed_om = 12_000.0
    battery_fixed_om = 6_000.0
    
    # Battery duration (hours) - from SystemConfig.jl
    battery_duration = 4.0  # hours
    
    # Function to calculate total system cost
    function calculate_total_system_cost(capacity_data, operational_cost)
        # Extract capacities
        nuclear_cap = capacity_data.Nuclear_capacity_MW
        wind_cap = capacity_data.Wind_capacity_MW
        gas_cap = capacity_data.Gas_capacity_MW
        battery_power_cap = capacity_data.Battery_capacity_MW
        battery_energy_cap = battery_power_cap * battery_duration  # MWh
        
        # Calculate investment costs
        investment_cost = (nuclear_inv_cost * nuclear_cap + 
                          wind_inv_cost * wind_cap + 
                          gas_inv_cost * gas_cap + 
                          battery_power_inv_cost * battery_power_cap + 
                          battery_energy_inv_cost * battery_energy_cap)
        
        # Calculate fixed O&M costs
        fixed_om_cost = (nuclear_fixed_om * nuclear_cap + 
                        wind_fixed_om * wind_cap + 
                        gas_fixed_om * gas_cap + 
                        battery_fixed_om * battery_power_cap)
        
        # Total system cost = operational + investment + fixed O&M
        return operational_cost + investment_cost + fixed_om_cost
    end
    
    # Calculate operational metrics
    function calculate_metrics(ops_df, policy_name)
        # Calculate total generation (excluding battery discharge)
        total_generation = ops_df.Nuclear_Generation + ops_df.Wind_Generation + ops_df.Gas_Generation +
                          ops_df.Nuclear_Generation_Flex + ops_df.Wind_Generation_Flex + ops_df.Gas_Generation_Flex 
        
        # The actual demand is fixed and should be the same for all policies
        # It equals total generation + battery discharge - battery charge + load shed
        # But more simply: actual_demand = total_generation + battery_discharge + load_shed
        actual_demand = total_generation + ops_df.Battery_Discharge -ops_df.Battery_Charge + ops_df.Load_Shed
        
        # Calculate demand-weighted average price
        weighted_avg_price = sum(ops_df.Price .* actual_demand) / sum(actual_demand)
        
        return Dict(
            "policy" => policy_name,
            "total_generation_MWh" => sum(ops_df.Nuclear_Generation + ops_df.Wind_Generation + ops_df.Gas_Generation +
                                         ops_df.Nuclear_Generation_Flex + ops_df.Wind_Generation_Flex + ops_df.Gas_Generation_Flex),
            "battery_discharge_MWh" => sum(ops_df.Battery_Discharge),
            "battery_charge_MWh" => sum(ops_df.Battery_Charge),
            "unmet_demand_MWh" => sum(ops_df.Load_Shed),
            "unmet_demand_fixed_MWh" => sum(ops_df.Load_Shed_Fixed),
            "unmet_demand_flex_MWh" => sum(ops_df.Load_Shed_Flex),
            "total_demand_MWh" => sum(actual_demand),
            "demand_weighted_avg_price" => weighted_avg_price,
            "max_price" => maximum(ops_df.Price),
            "min_price" => minimum(ops_df.Price),
            "price_volatility" => std(ops_df.Price) / mean(ops_df.Price),
            "unmet_demand_rate" => sum(ops_df.Load_Shed) / sum(actual_demand) * 100  # Percentage
        )
    end
    
    # Calculate metrics for each policy
    slac_metrics = calculate_metrics(slac_ops, "SLAC")
    slac_operational_cost = slac_log[end, :total_cost]  # This is the operational cost from the equilibrium
    slac_total_system_cost = calculate_total_system_cost(slac_capacity_data, slac_operational_cost)
    slac_metrics["total_cost_M"] = round(slac_total_system_cost / 1e6, digits=2)
    
    dlac_metrics = calculate_metrics(dlac_ops, "DLAC-i")
    dlac_operational_cost = dlac_log[end, :total_cost]  # This is the operational cost from the equilibrium
    dlac_total_system_cost = calculate_total_system_cost(dlac_capacity_data, dlac_operational_cost)
    dlac_metrics["total_cost_M"] = round(dlac_total_system_cost / 1e6, digits=2)
    
    pf_metrics = calculate_metrics(pf_ops, "Perfect Foresight")
    pf_operational_cost = pf_log[end, :total_cost]  # This is the operational cost from the equilibrium  
    pf_total_system_cost = calculate_total_system_cost(pf_capacity_data, pf_operational_cost)
    pf_metrics["total_cost_M"] = round(pf_total_system_cost / 1e6, digits=2)
    
    # Create comparison DataFrame
    comparison_df = DataFrame(
        Policy = ["DLAC-i", "SLAC", "Perfect Foresight"],
        Total_Cost_M = [dlac_metrics["total_cost_M"], slac_metrics["total_cost_M"], pf_metrics["total_cost_M"]],
        Total_Demand_MWh = [dlac_metrics["total_demand_MWh"], slac_metrics["total_demand_MWh"], pf_metrics["total_demand_MWh"]],
        Total_Generation_MWh = [dlac_metrics["total_generation_MWh"], slac_metrics["total_generation_MWh"], pf_metrics["total_generation_MWh"]],
        Unmet_Demand_MWh = [dlac_metrics["unmet_demand_MWh"], slac_metrics["unmet_demand_MWh"], pf_metrics["unmet_demand_MWh"]],
        Unmet_Demand_Rate_Pct = [dlac_metrics["unmet_demand_rate"], slac_metrics["unmet_demand_rate"], pf_metrics["unmet_demand_rate"]],
        Battery_Discharge_MWh = [dlac_metrics["battery_discharge_MWh"], slac_metrics["battery_discharge_MWh"], pf_metrics["battery_discharge_MWh"]],
        Battery_Charge_MWh = [dlac_metrics["battery_charge_MWh"], slac_metrics["battery_charge_MWh"], pf_metrics["battery_charge_MWh"]],
        Demand_Weighted_Avg_Price = [dlac_metrics["demand_weighted_avg_price"], slac_metrics["demand_weighted_avg_price"], pf_metrics["demand_weighted_avg_price"]],
        Max_Price = [dlac_metrics["max_price"], slac_metrics["max_price"], pf_metrics["max_price"]],
        Price_Volatility = [dlac_metrics["price_volatility"], slac_metrics["price_volatility"], pf_metrics["price_volatility"]]
    )
    
    # Save comparison results
    if save_path !== nothing
        CSV.write(save_path, comparison_df)
        println("Operational results comparison saved: $save_path")
    end
    
    # Print summary
    println("\n" * "="^80)
    println("OPERATIONAL RESULTS COMPARISON")
    println("="^80)
    println("Policy\t\tTotal System Cost (M\$)\tUnmet Demand (MWh)\tUnmet Rate (%)\tWeighted Avg Price (\$/MWh)")
    println("-"^80)
    for row in eachrow(comparison_df)
        println("$(row.Policy)\t$(row.Total_Cost_M)\t\t\t$(round(row.Unmet_Demand_MWh, digits=2))\t\t$(round(row.Unmet_Demand_Rate_Pct, digits=3))\t\t$(round(row.Demand_Weighted_Avg_Price, digits=2))")
    end
    println("-"^80)
    
    # Print cost breakdown
    println("\n" * "="^80)
    println("COST BREAKDOWN")
    println("="^80)
    println("Policy\t\tOperational (M\$)\tInvestment (M\$)\tFixed O&M (M\$)\tTotal System (M\$)")
    println("-"^80)
    
    policies = ["DLAC-i", "SLAC", "Perfect Foresight"]
    operational_costs = [dlac_operational_cost, slac_operational_cost, pf_operational_cost]
    capacity_data = [dlac_capacity_data, slac_capacity_data, pf_capacity_data]
    
    for (i, policy) in enumerate(policies)
        cap_data = capacity_data[i]
        op_cost = operational_costs[i]
        
        # Calculate investment and fixed O&M costs
        nuclear_cap = cap_data.Nuclear_capacity_MW
        wind_cap = cap_data.Wind_capacity_MW
        gas_cap = cap_data.Gas_capacity_MW
        battery_power_cap = cap_data.Battery_capacity_MW
        battery_energy_cap = battery_power_cap * battery_duration
        
        investment_cost = (nuclear_inv_cost * nuclear_cap + 
                          wind_inv_cost * wind_cap + 
                          gas_inv_cost * gas_cap + 
                          battery_power_inv_cost * battery_power_cap + 
                          battery_energy_inv_cost * battery_energy_cap)
        
        fixed_om_cost = (nuclear_fixed_om * nuclear_cap + 
                        wind_fixed_om * wind_cap + 
                        gas_fixed_om * gas_cap + 
                        battery_fixed_om * battery_power_cap)
        
        total_system_cost = op_cost + investment_cost + fixed_om_cost
        
        println("$(policy)\t$(round(op_cost / 1e6, digits=2))\t\t\t$(round(investment_cost / 1e6, digits=2))\t\t\t$(round(fixed_om_cost / 1e6, digits=2))\t\t\t$(round(total_system_cost / 1e6, digits=2))")
    end
    println("-"^80)
    
    return comparison_df, Dict("DLAC-i" => dlac_metrics, "SLAC" => slac_metrics, "Perfect Foresight" => pf_metrics)
end

end # module PlottingModule