#!/usr/bin/env julia

using Pkg
Pkg.activate(".")
using ToySystemQuad
using .PlottingModule

# Call the existing function to calculate operational results comparison
println("Calculating system metrics for equilibrium results...")
# Updated to point directly to the anderson directory
results_dir = "results/equilibrium/anderson"

try
    comparison_df, metrics_dict = calculate_operational_results_comparison(results_dir)
    
    println("\n" * "="^80)
    println("EQUILIBRIUM SYSTEM METRICS SUMMARY")
    println("="^80)
    
    for (policy, metrics) in metrics_dict
        println("\n$(policy):")
        println("  Total System Cost: \$$(metrics["total_cost_M"]) Million")
        println("  Unmet Demand: $(round(metrics["unmet_demand_MWh"], digits=1)) MWh") 
        println("  Unmet Demand Rate: $(round(metrics["unmet_demand_rate"], digits=3))%")
        println("  Demand-Weighted Avg Price: \$$(round(metrics["demand_weighted_avg_price"], digits=2))/MWh")
        println("  Max Price: \$$(round(metrics["max_price"], digits=2))/MWh")
        println("  Price Volatility: $(round(metrics["price_volatility"], digits=3))")
    end
    
    println("\n" * "="^80)
    
catch e
    println("Error calculating metrics: $e")
    println("Make sure equilibrium results exist in: $results_dir")
end