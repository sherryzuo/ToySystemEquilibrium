"""
PlottingModule.jl

Comprehensive plotting module for ToySystemQuad.jl results analysis.
Handles price analysis, generation stacks, and model comparisons.
"""

module PlottingModule

using Plots, Statistics, CSV, DataFrames
using ..SystemConfig: Generator, Battery, SystemProfiles

export plot_price_time_series, plot_price_duration_curves, plot_combined_price_analysis
export plot_generation_stacks, plot_system_profiles, plot_capacity_comparison
export plot_battery_operations, plot_battery_soc_comparison
export generate_all_plots, save_price_analysis

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
             lw=2, 
             size=(1200, 600),
             legend=false)
    
    # Add statistics annotations
    avg_price = mean(prices)
    max_price = maximum(prices)
    min_price = minimum(prices)
    
    hline!([avg_price], color=:red, linestyle=:dash, alpha=0.7, 
           label="Average: \$$(round(avg_price, digits=2))/MWh")
    
    # Add text annotation with stats
    annotate!(T*0.7, max_price*0.9, 
             text("Avg: \$$(round(avg_price, digits=1))\nMax: \$$(round(max_price, digits=1))\nMin: \$$(round(min_price, digits=1))", 
                  :left, 10))
    
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
             size=(1000, 700), 
             legend=:topright)
    
    colors = [:blue, :red, :green]
    linestyles = [:solid, :dash, :dot]
    
    for (i, (model_name, prices)) in enumerate(price_results)
        sorted_prices = sort(prices, rev=true)
        hours = 1:length(sorted_prices)
        
        plot!(p, hours, sorted_prices, 
              label=model_name, 
              lw=2, 
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
              size=(1200, 400),
              legend=:topright)
    
    plot!(p1, hours, pf_prices, label="Perfect Foresight", lw=2, color=:red, linestyle=:dash)
    plot!(p1, hours, dlac_prices, label="DLAC-i", lw=2, color=:green, linestyle=:dot)
    plot!(p1, hours, slac_prices, label="SLAC", lw=2, color=:blue)
    
    # Duration curves
    p2 = plot(title="Price Duration Curves", 
              xlabel="Hours", 
              ylabel="Price (\$/MWh)",
              size=(1200, 400),
              legend=:topright)
    
    sorted_pf = sort(pf_prices, rev=true)
    sorted_dlac = sort(dlac_prices, rev=true)
    sorted_slac = sort(slac_prices, rev=true)
    
    plot!(p2, hours, sorted_pf, label="Perfect Foresight", lw=2, color=:red, linestyle=:dash)
    plot!(p2, hours, sorted_dlac, label="DLAC-i", lw=2, color=:green, linestyle=:dot)
    plot!(p2, hours, sorted_slac, label="SLAC", lw=2, color=:blue)
    
    # Price difference analysis
    p3 = plot(title="Price Differences from Perfect Foresight", 
              xlabel="Hour", 
              ylabel="Price Difference (\$/MWh)",
              size=(1200, 400),
              legend=:topright)
    
    plot!(p3, hours, dlac_prices - pf_prices, label="DLAC-i - PF", lw=2, color=:green)
    plot!(p3, hours, slac_prices - pf_prices, label="SLAC - PF", lw=2, color=:blue)
    hline!(p3, [0], color=:black, linestyle=:dash, alpha=0.5, label="Zero Difference")
    
    # Combine all plots
    combined_plot = plot(p1, p2, p3, layout=(3,1), size=(1200, 1200))
    
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
        
        # Technology colors
        colors = [:red, :green, :orange, :purple]  # Nuclear, Wind, Gas, Battery
        
        p = plot(title="Generation Stack - $model_name", 
                 xlabel="Hour", 
                 ylabel="Power (MW)",
                 size=(1000, 500), 
                 legend=:outertopleft)
        
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
        plot!(p, hours, demand, label="Demand", color=:black, lw=3, linestyle=:dash)
        
        push!(plots_list, p)
    end
    
    # Combine all generation stacks
    if length(plots_list) == 3
        combined_plot = plot(plots_list..., layout=(3,1), size=(1000, 1500))
    else
        combined_plot = plot(plots_list..., layout=(length(plots_list),1), size=(1000, 500*length(plots_list)))
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
    
    # Create subplots
    p1 = plot(hours, profiles.actual_demand, title="Demand Profile", xlabel="Hour", ylabel="Demand (MW)",
              lw=2, color=:blue, legend=false)
    
    p2 = plot(hours, profiles.actual_wind, title="Wind Capacity Factor", xlabel="Hour", ylabel="Capacity Factor",
              lw=2, color=:green, legend=false, ylims=(0, 1))
    
    p3 = plot(hours, profiles.actual_nuclear_availability, title="Nuclear Availability", xlabel="Hour", ylabel="Available",
              lw=2, color=:red, legend=false, ylims=(0, 1.1))
    
    p4 = plot(hours, profiles.actual_gas_availability, title="Gas Availability", xlabel="Hour", ylabel="Available",
              lw=2, color=:orange, legend=false, ylims=(0, 1.1))
    
    combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800))
    
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
    
    # Create subplots
    p1 = plot(hours, actual_demand, title="Demand Profile", xlabel="Hour", ylabel="Demand (MW)",
              lw=2, color=:blue, legend=false)
    
    p2 = plot(hours, actual_wind, title="Wind Capacity Factor", xlabel="Hour", ylabel="Capacity Factor",
              lw=2, color=:green, legend=false, ylims=(0, 1))
    
    p3 = plot(hours, nuclear_availability, title="Nuclear Availability", xlabel="Hour", ylabel="Available",
              lw=2, color=:red, legend=false, ylims=(0, 1.1))
    
    p4 = plot(hours, gas_availability, title="Gas Availability", xlabel="Hour", ylabel="Available",
              lw=2, color=:orange, legend=false, ylims=(0, 1.1))
    
    combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800))
    
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
    colors = [:red, :green, :orange, :purple]
    
    p = bar(tech_names, capacities,
            title="Optimal Capacity Investments",
            xlabel="Technology",
            ylabel="Capacity (MW)",
            color=colors,
            legend=false,
            size=(800, 600))
    
    # Add capacity values on top of bars
    for (i, cap) in enumerate(capacities)
        annotate!(i, cap + maximum(capacities)*0.02, text("$(round(cap, digits=1)) MW", :center, 8))
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
    colors = [:blue, :red, :green]
    
    for (i, model_name) in enumerate(model_names)
        if haskey(battery_results, model_name)
            charge = battery_results[model_name]["charge"]
            discharge = battery_results[model_name]["discharge"]
            T = length(charge)
            hours = 1:T
            
            p = plot(title="Battery Operations - $model_name",
                    xlabel="Hour",
                    ylabel="Power (MW)",
                    size=(1000, 400),
                    legend=:topright)
            
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
        combined_plot = plot(plots_list..., layout=(3,1), size=(1000, 1200))
    else
        combined_plot = plot(plots_list..., layout=(length(plots_list),1), size=(1000, 400*length(plots_list)))
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
    colors = [:blue, :red, :green]
    linestyles = [:solid, :dash, :dot]
    
    p = plot(title="Battery State of Charge Comparison",
             xlabel="Hour",
             ylabel="SOC (MWh)",
             size=(1200, 600),
             legend=:topright)
    
    for (i, model_name) in enumerate(model_names)
        if haskey(battery_results, model_name)
            soc = battery_results[model_name]["soc"]
            T = length(soc)
            hours = 1:T
            
            plot!(p, hours, soc,
                  label=model_name,
                  color=colors[i],
                  linestyle=linestyles[i],
                  linewidth=2)
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

end # module PlottingModule