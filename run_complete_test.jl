#!/usr/bin/env julia

"""
run_complete_test.jl

Main script to run the complete NYISO system test.
Executes all three optimization models with real NYISO data and full-year analysis.
"""

using Revise

using ToySystemQuad

function main()
    println("Starting Complete NYISO System Test")
    
    # Use NYISO system parameters (full year analysis)
    params = get_nyiso_system_parameters()
    
    println("This will run all three optimization models with NYISO data:")
    println("  - $(params.hours)-hour horizon ($(params.days) days) - FULL YEAR")
    println("  - Real NYISO generator data with 7 technology types")
    println("  - $(params.N) stochastic scenarios for DLAC-i and SLAC operations")
    println("  - Real wind, solar, and load profiles from NYISO")
    println("  - Flexible demand: $(params.flex_demand_mw) MW with quadratic pricing")
    println("  - Random seed: $(params.random_seed)")
    println()
    
    # Create NYISO system
    println("Creating NYISO system...")
    generators, battery, profiles = create_nyiso_system(params)
    
    println("NYISO system created with:")
    println("  - $(length(generators)) generator types: $(join([g.name for g in generators], ", "))")
    println("  - Battery storage: $(battery.name)")
    println("  - Peak demand: $(round(maximum(profiles.actual_demand), digits=0)) MW")
    println()
    
    # Run the complete test system with NYISO data
    results = run_complete_test_system_nyiso(generators, battery, profiles, output_dir="results")
    
    if results["status"] == "success"
        println("\n✅ SUCCESS: All models completed successfully!")
        println("\nKey Results Summary:")
        println("- Capacity Expansion: \$$(round(results["cem"]["total_cost"]/1e6, digits=2))M total cost")
        println("- Perfect Foresight: \$$(round(results["perfect_foresight"]["total_cost"]/1e3, digits=0))k operational cost")  
        println("- DLAC-i Operations: \$$(round(results["dlac_i"]["total_cost"]/1e3, digits=0))k operational cost")
        println("- SLAC Operations: \$$(round(results["slac"]["total_cost"]/1e3, digits=0))k operational cost")
        
        println("\nFiles created in results/:")
        println("- capacity_expansion_results.csv, capacity_expansion_operations.csv, capacity_expansion_summary.csv, capacity_expansion_profits.csv")
        println("- perfect_foresight_operations.csv, perfect_foresight_summary.csv, perfect_foresight_profits.csv")
        println("- dlac_i_operations.csv, dlac_i_summary.csv, dlac_i_profits.csv")
        println("- slac_operations.csv, slac_summary.csv, slac_profits.csv")
        println("- four_model_comprehensive_comparison.csv")
        println("- four_model_detailed_comparison.csv")
        println("- comprehensive_forecast_quality_analysis.csv, comprehensive_price_analysis.csv")
        println("- demand_wind_profiles.csv (actuals), demand_wind_outage_profiles.csv (5 scenarios)")
        println("- price_statistics_summary.csv")
        println("\nPlots created in results/plots/:")
        println("- Individual price time series: cem_price_time_series.png, pf_price_time_series.png, dlac_i_price_time_series.png, slac_price_time_series.png")
        println("- Price duration curves: price_duration_curves.png")
        println("- Comprehensive price analysis: comprehensive_price_analysis.png")
        println("- Generation stacks: generation_stacks.png")
        println("- Battery operations: battery_operations.png (charge/discharge by model)")
        println("- Battery SOC comparison: battery_soc_comparison.png (state of charge across models)")
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