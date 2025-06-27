#!/usr/bin/env julia

"""
run_complete_test.jl

Main script to run the complete ToySystemQuad test system.
Executes all three optimization models with full system parameters.
"""

using Revise

# Load the local ToySystemQuad module
push!(LOAD_PATH, @__DIR__)
include("src/ToySystemQuad.jl")
using .ToySystemQuad

function main()
    println("Starting Complete ToySystemQuad Test System")
    println("This will run all three optimization models with:")
    println("  - 720-hour horizon (30 days)")
    println("  - Fleet-based thermal generation (5 generators per technology)")
    println("  - 5 stochastic scenarios for DLAC-i operations")
    println("  - Realistic wind forecast error modeling")
    println()
    
    # Run the complete test system
    results = run_complete_test_system(output_dir="results")
    
    if results["status"] == "success"
        println("\n✅ SUCCESS: All models completed successfully!")
        println("\nKey Results Summary:")
        println("- Capacity Expansion: \$$(round(results["cem"]["total_cost"]/1e6, digits=2))M total cost")
        println("- Perfect Foresight: \$$(round(results["perfect_foresight"]["total_cost"]/1e3, digits=0))k operational cost")  
        println("- DLAC-i Operations: \$$(round(results["dlac_i"]["total_cost"]/1e3, digits=0))k operational cost")
        
        println("\nFiles created in results/:")
        println("- capacity_expansion_results.csv, capacity_expansion_operations.csv, capacity_expansion_summary.csv, capacity_expansion_profits.csv")
        println("- perfect_foresight_operations.csv, perfect_foresight_summary.csv, perfect_foresight_profits.csv")
        println("- dlac_i_operations.csv, dlac_i_summary.csv, dlac_i_profits.csv")
        println("- three_model_comprehensive_comparison.csv")
        println("- pf_vs_dlac_i_comprehensive_comparison.csv")
        println("- comprehensive_forecast_quality_analysis.csv, comprehensive_price_analysis.csv")
        println("- demand_wind_profiles.csv (actuals), demand_wind_outage_profiles.csv (5 scenarios)")
        println("- price_statistics_summary.csv")
        println("\nPlots created in results/plots/:")
        println("- Individual price time series: cem_price_time_series.png, pf_price_time_series.png, dlac_i_price_time_series.png")
        println("- Price duration curves: price_duration_curves.png")
        println("- Comprehensive price analysis: comprehensive_price_analysis.png")
        println("- Generation stacks: generation_stacks.png")
        println("- System profiles: system_profiles.png")
        println("- Capacity comparison: capacity_comparison.png")
        
    else
        println("\n❌ FAILURE: Test failed at stage $(results["stage"])")
        println("Check error details above")
    end
    
    return results
end

# Execute if run directly or in REPL
results = main()