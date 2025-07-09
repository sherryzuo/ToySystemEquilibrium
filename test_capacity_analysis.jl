#!/usr/bin/env julia

"""
test_capacity_analysis.jl

Test script to run the new capacity mix analysis functions.
"""

using ToySystemQuad
using ToySystemQuad.PlottingModule

# Set the results directory
results_dir = "results"

println("="^80)
println("CAPACITY MIX ANALYSIS TEST")
println("="^80)

# Test 1: Plot capacity mix differences (grouped bars)
println("\n1. Testing capacity mix differences plotting...")
try
    plot_result, capacity_data = plot_capacity_mix_differences(results_dir; save_path="results/capacity_mix_differences.png")
    println("✓ Capacity mix differences plot created successfully")
    
    # Display capacity data
    println("\nCapacity Data (MW):")
    println("-"^50)
    for (policy, capacities) in capacity_data
        total = sum(capacities)
        println("$policy (Total: $(round(total, digits=0)) MW):")
        println("  Nuclear: $(round(capacities[1], digits=1)) MW ($(round(capacities[1]/total*100, digits=1))%)")
        println("  Wind: $(round(capacities[2], digits=1)) MW ($(round(capacities[2]/total*100, digits=1))%)")
        println("  Gas: $(round(capacities[3], digits=1)) MW ($(round(capacities[3]/total*100, digits=1))%)")
        println("  Battery: $(round(capacities[4], digits=1)) MW ($(round(capacities[4]/total*100, digits=1))%)")
        println()
    end
    
catch e
    println("✗ Error in capacity mix plotting: $e")
end

# Test 1b: Plot stacked capacity mix 
println("\n1b. Testing stacked capacity mix plotting...")
try
    stacked_plot = plot_capacity_mix_stacked(results_dir; save_path="results/capacity_mix_stacked.png")
    println("✓ Stacked capacity mix plot created successfully")
    
catch e
    println("✗ Error in stacked capacity mix plotting: $e")
end

# Test 2: Calculate operational results comparison
println("\n2. Testing operational results comparison...")
try
    comparison_df, detailed_metrics = calculate_operational_results_comparison(results_dir; save_path="results/operational_comparison.csv")
    println("✓ Operational results comparison completed successfully")
    
    # Display key results
    println("\nOperational Results Summary:")
    println(comparison_df)
    
catch e
    println("✗ Error in operational results calculation: $e")
end

println("\n" * "="^80)
println("CAPACITY MIX ANALYSIS TEST COMPLETE")
println("Results saved in: results/")
println("="^80)