#!/usr/bin/env julia

"""
compare_iterations.jl

Compare two iterations from an equilibrium log file by running operations 
with those capacities and plotting the resulting price differences.

Usage:
    julia compare_iterations.jl <log_file> <iteration1> <iteration2>

Example:
    julia compare_iterations.jl results/validation/equilibrium/perfectforesight/equilibrium_log.csv 100 200
"""

include("src/ToySystemQuad.jl")
using .ToySystemQuad
using CSV, DataFrames
using Plots, Statistics, StatsPlots

"""
    read_iteration_capacities(log_file, iteration)

Read capacities from a specific iteration in the equilibrium log file.
Returns (capacities, battery_power_cap, battery_energy_cap) or nothing if iteration not found.
"""
function read_iteration_capacities(log_file, iteration)
    if !isfile(log_file)
        println("Error: Log file not found: $log_file")
        return nothing
    end
    
    try
        df = CSV.read(log_file, DataFrame)
        
        # Find the row for this iteration
        iteration_row = df[df.Iteration .== iteration, :]
        
        if nrow(iteration_row) == 0
            println("Error: Iteration $iteration not found in log file")
            return nothing
        end
        
        row = iteration_row[1, :]
        
        # Extract generator capacities (assume Nuclear, Wind, Gas order)
        capacities = [
            row.Nuclear_capacity_MW,
            row.Wind_capacity_MW, 
            row.Gas_capacity_MW
        ]
        
        battery_power_cap = row.Battery_capacity_MW
        battery_energy_cap = battery_power_cap * 4.0  # Assume 4-hour duration
        
        return capacities, battery_power_cap, battery_energy_cap
        
    catch e
        println("Error reading log file: $e")
        return nothing
    end
end

"""
    run_operations_comparison(iteration1, iteration2, capacities1, battery_power1, battery_energy1,
                             capacities2, battery_power2, battery_energy2, generators, battery, profiles)

Run operations for both capacity sets and return results for comparison.
"""
function run_operations_comparison(iteration1, iteration2, 
                                 capacities1, battery_power1, battery_energy1,
                                 capacities2, battery_power2, battery_energy2,
                                 generators, battery, profiles)
    
    println("="^80)
    println("RUNNING OPERATIONS COMPARISON")
    println("="^80)
    
    # Run operations for iteration 1
    println("\nRunning operations for Iteration $iteration1...")
    println("Capacities 1:")
    for (i, gen) in enumerate(generators)
        println("  $(gen.name): $(round(capacities1[i], digits=1)) MW")
    end
    println("  Battery Power: $(round(battery_power1, digits=1)) MW")
    println("  Battery Energy: $(round(battery_energy1, digits=1)) MWh")
    
    result1 = solve_perfect_foresight_operations(
        generators, battery, capacities1, battery_power1, battery_energy1, profiles;
        output_dir="results/comparison/iteration_$(iteration1)"
    )
    
    # Run operations for iteration 2
    println("\nRunning operations for Iteration $iteration2...")
    println("Capacities 2:")
    for (i, gen) in enumerate(generators)
        println("  $(gen.name): $(round(capacities2[i], digits=1)) MW")
    end
    println("  Battery Power: $(round(battery_power2, digits=1)) MW")
    println("  Battery Energy: $(round(battery_energy2, digits=1)) MWh")
    
    result2 = solve_perfect_foresight_operations(
        generators, battery, capacities2, battery_power2, battery_energy2, profiles;
        output_dir="results/comparison/iteration_$(iteration2)"
    )
    
    return result1, result2
end

"""
    plot_price_comparison(iteration1, iteration2, result1, result2, output_dir="results/comparison")

Create comparison plots of electricity prices between two iterations.
"""
function plot_price_comparison(iteration1, iteration2, result1, result2, output_dir="results/comparison")
    mkpath(output_dir)
    
    prices1 = result1["prices"]
    prices2 = result2["prices"]
    hours = 1:length(prices1)
    
    # Create time series comparison plot
    p1 = plot(hours, prices1, 
              label="Iteration $iteration1", 
              linewidth=2, 
              color=:blue,
              title="Electricity Price Comparison",
              xlabel="Hour",
              ylabel="Price (\$/MWh)")
    plot!(p1, hours, prices2, 
          label="Iteration $iteration2", 
          linewidth=2, 
          color=:red)
    
    # Create price difference plot
    price_diff = prices2 .- prices1
    p2 = plot(hours, price_diff,
              label="Price Difference (Iter $iteration2 - Iter $iteration1)",
              linewidth=2,
              color=:green,
              title="Price Difference Between Iterations",
              xlabel="Hour", 
              ylabel="Price Difference (\$/MWh)")
    hline!(p2, [0], color=:black, linestyle=:dash, alpha=0.5, label="")
    
    # Create price duration curves
    sorted_prices1 = sort(prices1, rev=true)
    sorted_prices2 = sort(prices2, rev=true)
    percentiles = (1:length(prices1)) ./ length(prices1) .* 100
    
    p3 = plot(percentiles, sorted_prices1,
              label="Iteration $iteration1",
              linewidth=2,
              color=:blue,
              title="Price Duration Curves",
              xlabel="Percentile (%)",
              ylabel="Price (\$/MWh)")
    plot!(p3, percentiles, sorted_prices2,
          label="Iteration $iteration2",
          linewidth=2,
          color=:red)
    
    # Create summary statistics plot
    stats1 = [mean(prices1), median(prices1), maximum(prices1), minimum(prices1)]
    stats2 = [mean(prices2), median(prices2), maximum(prices2), minimum(prices2)]
    stat_names = ["Mean", "Median", "Max", "Min"]
    
    p4 = groupedbar([stats1 stats2],
                    bar_position=:dodge,
                    label=["Iteration $iteration1" "Iteration $iteration2"],
                    title="Price Statistics Comparison",
                    xlabel="Statistic",
                    ylabel="Price (\$/MWh)",
                    xticks=(1:4, stat_names))
    
    # Combine all plots
    combined_plot = plot(p1, p2, p3, p4, 
                        layout=(2,2), 
                        size=(1200, 800),
                        plot_title="Electricity Price Comparison: Iteration $iteration1 vs $iteration2")
    
    # Save plots
    price_comparison_file = joinpath(output_dir, "price_comparison_$(iteration1)_vs_$(iteration2).png")
    savefig(combined_plot, price_comparison_file)
    
    # Save individual plots
    savefig(p1, joinpath(output_dir, "price_timeseries_$(iteration1)_vs_$(iteration2).png"))
    savefig(p2, joinpath(output_dir, "price_difference_$(iteration1)_vs_$(iteration2).png"))
    savefig(p3, joinpath(output_dir, "price_duration_$(iteration1)_vs_$(iteration2).png"))
    savefig(p4, joinpath(output_dir, "price_statistics_$(iteration1)_vs_$(iteration2).png"))
    
    println("Plots saved to: $output_dir")
    println("  - Combined: $price_comparison_file")
    println("  - Time series: $(joinpath(output_dir, "price_timeseries_$(iteration1)_vs_$(iteration2).png"))")
    println("  - Difference: $(joinpath(output_dir, "price_difference_$(iteration1)_vs_$(iteration2).png"))")
    println("  - Duration curves: $(joinpath(output_dir, "price_duration_$(iteration1)_vs_$(iteration2).png"))")
    println("  - Statistics: $(joinpath(output_dir, "price_statistics_$(iteration1)_vs_$(iteration2).png"))")
    
    return combined_plot
end

"""
    print_summary_comparison(iteration1, iteration2, result1, result2, capacities1, capacities2)

Print a summary comparison of the two iterations.
"""
function print_summary_comparison(iteration1, iteration2, result1, result2, capacities1, capacities2)
    println("\n" * "="^80)
    println("COMPARISON SUMMARY")
    println("="^80)
    
    prices1 = result1["prices"]
    prices2 = result2["prices"]
    
    println("Iteration $iteration1:")
    println("  Average price: \$$(round(mean(prices1), digits=2))/MWh")
    println("  Max price: \$$(round(maximum(prices1), digits=2))/MWh")
    println("  Total cost: \$$(round(result1["total_cost"], digits=0))")
    println("  Load shed: $(round(result1["total_load_shed"], digits=1)) MWh")
    
    println("\nIteration $iteration2:")
    println("  Average price: \$$(round(mean(prices2), digits=2))/MWh")
    println("  Max price: \$$(round(maximum(prices2), digits=2))/MWh")
    println("  Total cost: \$$(round(result2["total_cost"], digits=0))")
    println("  Load shed: $(round(result2["total_load_shed"], digits=1)) MWh")
    
    println("\nDifferences (Iteration $iteration2 - Iteration $iteration1):")
    println("  Average price: \$$(round(mean(prices2) - mean(prices1), digits=2))/MWh")
    println("  Max price: \$$(round(maximum(prices2) - maximum(prices1), digits=2))/MWh")
    println("  Total cost: \$$(round(result2["total_cost"] - result1["total_cost"], digits=0))")
    println("  Load shed: $(round(result2["total_load_shed"] - result1["total_load_shed"], digits=1)) MWh")
    
    println("\nCapacity Differences:")
    capacity_names = ["Nuclear", "Wind", "Gas"]
    for i in 1:length(capacities1)
        diff = capacities2[i] - capacities1[i]
        println("  $(capacity_names[i]): $(round(diff, digits=1)) MW")
    end
end

"""
    main()

Main function to run the iteration comparison.
"""
function main()
    if length(ARGS) < 3
        println("Usage: julia compare_iterations.jl <log_file> <iteration1> <iteration2>")
        println("\nExample:")
        println("  julia compare_iterations.jl results/validation/equilibrium/perfectforesight/equilibrium_log.csv 100 200")
        return
    end
    
    log_file = ARGS[1]
    iteration1 = parse(Int, ARGS[2])
    iteration2 = parse(Int, ARGS[3])
    
    println("="^80)
    println("ITERATION COMPARISON TOOL")
    println("="^80)
    println("Log file: $log_file")
    println("Comparing iterations: $iteration1 vs $iteration2")
    
    # Read capacities from both iterations
    result1_caps = read_iteration_capacities(log_file, iteration1)
    result2_caps = read_iteration_capacities(log_file, iteration2)
    
    if result1_caps === nothing || result2_caps === nothing
        println("Error: Could not read capacities from one or both iterations")
        return
    end
    
    capacities1, battery_power1, battery_energy1 = result1_caps
    capacities2, battery_power2, battery_energy2 = result2_caps
    
    # Set up system configuration (same as validation test)
    params = SystemParameters(
        720,     # hours (30 days)
        30,      # days  
        5,       # N (number of generators per technology fleet)
        42,      # random_seed
        10000.0, # load_shed_penalty ($/MWh)
        0.001    # load_shed_quad
    )
    
    generators, battery, profiles = create_complete_toy_system(params)
    
    # Run operations comparison
    result1, result2 = run_operations_comparison(
        iteration1, iteration2, 
        capacities1, battery_power1, battery_energy1,
        capacities2, battery_power2, battery_energy2,
        generators, battery, profiles
    )
    
    # Create comparison plots
    plot_price_comparison(iteration1, iteration2, result1, result2)
    
    # Print summary
    print_summary_comparison(iteration1, iteration2, result1, result2, capacities1, capacities2)
    
    println("\n" * "="^80)
    println("COMPARISON COMPLETE")
    println("="^80)
end

# Run the main function
main()