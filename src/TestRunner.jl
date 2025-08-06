"""
TestRunner.jl

Complete test system runner for ToySystemQuad.jl modular implementation.
Runs all three optimization models with full system parameters and saves detailed results.
"""

module TestRunner

# Use modules from parent
using ..SystemConfig
using ..ProfileGeneration
using ..OptimizationModels
using ..PlottingModule
using CSV, DataFrames, Plots, Statistics

export run_complete_test_system
export compare_models_analysis
export run_performance_test_system
export validate_cached_results_match_original

"""
    run_complete_test_system(; params=nothing, output_dir="results")

Run complete test system with all three optimization models:
1. Capacity Expansion Model (CEM) 
2. Perfect Foresight Operations (DLAC-p)
3. DLAC-i Operations (rolling horizon)

Args:
- params: SystemParameters to use (if nothing, uses defaults)
- output_dir: Directory to save results

Saves detailed results and comparisons to CSV files.
"""
function run_complete_test_system(; params=nothing, output_dir="results", use_cached=false)
    cache_suffix = use_cached ? " (with Model Reuse Cache)" : ""
    println("🚀 Running Complete ToySystemQuad Test System$cache_suffix")
    println(repeat("=", 60))
    
    # Use provided parameters or defaults
    if params === nothing
        params = get_default_system_parameters()
    end
    
    # Create complete system with specified profiles
    generators, battery, profiles = create_complete_toy_system(params)
    
    println("System Configuration:")
    println("  Time horizon: $(params.hours) hours ($(params.days) days)")
    println("  Technologies:")
    for (i, gen) in enumerate(generators)
        println("    $i. $(gen.name): Fuel \$$(gen.fuel_cost)/MWh, Investment \$$(gen.inv_cost)/MW/yr")
    end
    println("    4. $(battery.name): Power \$$(battery.inv_cost_power)/MW/yr, Energy \$$(battery.inv_cost_energy)/MWh/yr")
    
    println("\n📊 System profiles generated with $(profiles.n_scenarios) scenarios")
    
    # Save profiles to CSV
    save_system_profiles(profiles, output_dir)
    
    # STEP 1: Capacity Expansion Model
    println("\n" * repeat("=", 25) * " CAPACITY EXPANSION MODEL " * repeat("=", 25))
    println("TESTING")
    cem_result = solve_capacity_expansion_model(generators, battery, profiles; 
                                               output_dir=output_dir)
    
    if cem_result["status"] != "optimal"
        println("❌ Capacity Expansion Model failed: $(cem_result["status"])")
        return Dict("status" => "failed", "stage" => "CEM", "result" => cem_result)
    end
    
    println("✅ Capacity Expansion Model solved successfully!")
    optimal_capacities = cem_result["capacity"]
    optimal_battery_power = cem_result["battery_power_cap"]
    optimal_battery_energy = cem_result["battery_energy_cap"]
    
    print_capacity_results(generators, battery, optimal_capacities, optimal_battery_power, optimal_battery_energy)
    
    # Initialize model cache if using cached operations
    model_cache = nothing
    if use_cached
        println("\n✓ Initializing model cache for operations...")
        model_cache = ModelCache(24, profiles.n_scenarios)  # 24-hour lookahead
    end
    
    # Save capacity expansion operations and profits
    calculate_profits_and_save(generators, battery, cem_result,
                               optimal_capacities, optimal_battery_power, optimal_battery_energy,
                               "capacity_expansion", output_dir)
    
    # STEP 2: Perfect Foresight Operations
    println("\n" * repeat("=", 25) * " PERFECT FORESIGHT OPERATIONS " * repeat("=", 21))
    
    pf_result = solve_perfect_foresight_operations(generators, battery, 
                                                  optimal_capacities,
                                                  optimal_battery_power, 
                                                  optimal_battery_energy,
                                                  profiles;
                                                  output_dir=output_dir)
    
    if pf_result["status"] != "optimal"
        println("❌ Perfect Foresight Operations failed: $(pf_result["status"])")
        return Dict("status" => "failed", "stage" => "PF", "result" => pf_result)
    end
    
    println("✅ Perfect Foresight Operations solved successfully!")
    
    # STEP 3: DLAC-i Operations
    cache_label = use_cached ? " (CACHED)" : ""
    println("\n" * repeat("=", 25) * " DLAC-I OPERATIONS$cache_label " * repeat("=", 27))
    
    if use_cached && model_cache !== nothing
        dlac_result = solve_dlac_i_operations_cached(generators, battery,
                                                   optimal_capacities,
                                                   optimal_battery_power,
                                                   optimal_battery_energy,
                                                   profiles, model_cache;
                                                   lookahead_hours=24,
                                                   output_dir=output_dir)
    else
        dlac_result = solve_dlac_i_operations(generators, battery,
                                             optimal_capacities,
                                             optimal_battery_power,
                                             optimal_battery_energy,
                                             profiles;
                                             lookahead_hours=24,
                                             output_dir=output_dir)
    end
    
    if dlac_result["status"] != "optimal"
        println("❌ DLAC-i Operations failed: $(dlac_result["status"])")
        return Dict("status" => "failed", "stage" => "DLAC-i", "result" => dlac_result)
    end
    
    println("✅ DLAC-i Operations solved successfully!")
    
    # STEP 4: SLAC Operations
    println("\n" * repeat("=", 25) * " SLAC OPERATIONS$cache_label " * repeat("=", 29))
    
    if use_cached && model_cache !== nothing
        slac_result = solve_slac_operations_cached(generators, battery,
                                                 optimal_capacities,
                                                 optimal_battery_power,
                                                 optimal_battery_energy,
                                                 profiles, model_cache;
                                                 lookahead_hours=24,
                                                 output_dir=output_dir)
    else
        slac_result = solve_slac_operations(generators, battery,
                                           optimal_capacities,
                                           optimal_battery_power,
                                           optimal_battery_energy,
                                           profiles;
                                           lookahead_hours=24,
                                           output_dir=output_dir)
    end
    
    if slac_result["status"] != "optimal"
        println("❌ SLAC Operations failed: $(slac_result["status"])")
        return Dict("status" => "failed", "stage" => "SLAC", "result" => slac_result)
    end
    
    println("✅ SLAC Operations solved successfully!")
    
    # STEP 5: Profit Analysis and PMR Calculation
    println("\n" * repeat("=", 25) * " PROFIT ANALYSIS " * repeat("=", 29))
    
    pf_profits = calculate_profits_and_save(generators, battery, pf_result,
                                           optimal_capacities, optimal_battery_power, optimal_battery_energy,
                                           "perfect_foresight", output_dir)
    
    dlac_profits = calculate_profits_and_save(generators, battery, dlac_result,
                                             optimal_capacities, optimal_battery_power, optimal_battery_energy,
                                             "dlac_i", output_dir)
    
    slac_profits = calculate_profits_and_save(generators, battery, slac_result,
                                             optimal_capacities, optimal_battery_power, optimal_battery_energy,
                                             "slac", output_dir)
    
    # Calculate PMRs
    pf_pmr = compute_pmr(pf_result, generators, battery, 
                        optimal_capacities, optimal_battery_power, optimal_battery_energy)
    dlac_pmr = compute_pmr(dlac_result, generators, battery,
                          optimal_capacities, optimal_battery_power, optimal_battery_energy)
    slac_pmr = compute_pmr(slac_result, generators, battery,
                          optimal_capacities, optimal_battery_power, optimal_battery_energy)
    
    println("Perfect Foresight PMRs (%): $(round.(pf_pmr, digits=2))")
    println("DLAC-i PMRs (%): $(round.(dlac_pmr, digits=2))")
    println("SLAC PMRs (%): $(round.(slac_pmr, digits=2))")
    
    # STEP 6: Comprehensive Model Comparison
    println("\n" * repeat("=", 25) * " MODEL COMPARISON " * repeat("=", 28))
    
    comparison_results = compare_models_analysis(cem_result, pf_result, dlac_result, slac_result,
                                               generators, battery, optimal_capacities,
                                               optimal_battery_power, optimal_battery_energy,
                                               output_dir)
    
    # STEP 7: Generate Comprehensive Plots
    println("\n" * repeat("=", 25) * " GENERATING PLOTS " * repeat("=", 28))
    
    generate_all_plots(pf_result, dlac_result, slac_result, profiles,
                      generators, battery, optimal_capacities, optimal_battery_power, output_dir)
    
    println("\nComplete test system finished successfully!")
    println("All results saved to: $(output_dir)/")
    
    return Dict(
        "status" => "success",
        "cem" => cem_result,
        "perfect_foresight" => pf_result,
        "dlac_i" => dlac_result,
        "slac" => slac_result,
        "comparison" => comparison_results
    )
end

"""
    save_system_profiles(profiles::SystemProfiles, output_dir)

Save system profiles and forecast scenarios to CSV files.
"""
function save_system_profiles(profiles::SystemProfiles, output_dir)
    mkpath(output_dir)
    
    T = length(profiles.actual_demand)
    
    # Save actual profiles
    profiles_df = DataFrame(
        Hour = 1:T,
        Demand_MW = profiles.actual_demand,
        Wind_CF = profiles.actual_wind,
        Nuclear_Available = profiles.actual_nuclear_availability,
        Gas_Available = profiles.actual_gas_availability
    )
    CSV.write(joinpath(output_dir, "demand_wind_profiles.csv"), profiles_df)
    
    # Save scenarios for DLAC-i analysis
    scenarios_df = DataFrame(
        Hour = repeat(1:T, profiles.n_scenarios),
        Scenario = vcat([fill(s, T) for s in 1:profiles.n_scenarios]...),
        Demand_MW = vcat(profiles.demand_scenarios...),
        Wind_CF = vcat(profiles.wind_scenarios...),
        Nuclear_Available = vcat(profiles.nuclear_availability_scenarios...),
        Gas_Available = vcat(profiles.gas_availability_scenarios...)
    )
    CSV.write(joinpath(output_dir, "demand_wind_outage_profiles.csv"), scenarios_df)
    
    println("System profiles saved to CSV files")
end

"""
    compare_models_analysis(cem_result, pf_result, dlac_result, generators, battery, 
                           optimal_capacities, optimal_battery_power, optimal_battery_energy, output_dir)

Comprehensive comparison analysis between the three models.
"""
function compare_models_analysis(cem_result, pf_result, dlac_result, slac_result, generators, battery,
                                optimal_capacities, optimal_battery_power, optimal_battery_energy, output_dir)
    
    # Calculate PMRs for each model
    pf_pmr = compute_pmr(pf_result, generators, battery, optimal_capacities, optimal_battery_power, optimal_battery_energy)
    dlac_pmr = compute_pmr(dlac_result, generators, battery, optimal_capacities, optimal_battery_power, optimal_battery_energy)
    slac_pmr = compute_pmr(slac_result, generators, battery, optimal_capacities, optimal_battery_power, optimal_battery_energy)
    
    # Three-model comparison
    n_technologies = length(generators) + 1  # +1 for battery
    tech_names = [gen.name for gen in generators]
    push!(tech_names, "Battery")
    capacities_array = copy(optimal_capacities)
    push!(capacities_array, optimal_battery_power)
    
    comparison_df = DataFrame(
        Technology = tech_names,
        Capacity_MW = capacities_array,
        PF_PMR_Percent = pf_pmr,
        DLAC_i_PMR_Percent = dlac_pmr,
        SLAC_PMR_Percent = slac_pmr
    )
    
    # Add summary costs as a separate DataFrame
    summary_df = DataFrame(
        Model = ["CEM", "Perfect_Foresight", "DLAC_i", "SLAC"],
        Total_Cost = [cem_result["total_cost"], pf_result["total_cost"], dlac_result["total_cost"], slac_result["total_cost"]]
    )
    CSV.write(joinpath(output_dir, "four_model_comprehensive_comparison.csv"), comparison_df)
    CSV.write(joinpath(output_dir, "four_model_cost_summary.csv"), summary_df)
    
    # Detailed four-model comparison
    T = length(pf_result["prices"])
    detailed_comparison_df = DataFrame(
        Hour = 1:T,
        PF_Price = pf_result["prices"],
        DLAC_i_Price = dlac_result["prices"],
        SLAC_Price = slac_result["prices"],
        DLAC_PF_Price_Diff = dlac_result["prices"] - pf_result["prices"],
        SLAC_PF_Price_Diff = slac_result["prices"] - pf_result["prices"],
        SLAC_DLAC_Price_Diff = slac_result["prices"] - dlac_result["prices"],
        PF_Load_Shed = pf_result["load_shed"],
        DLAC_i_Load_Shed = dlac_result["load_shed"],
        SLAC_Load_Shed = slac_result["load_shed"],
        DLAC_PF_Load_Shed_Diff = dlac_result["load_shed"] - pf_result["load_shed"],
        SLAC_PF_Load_Shed_Diff = slac_result["load_shed"] - pf_result["load_shed"],
        SLAC_DLAC_Load_Shed_Diff = slac_result["load_shed"] - dlac_result["load_shed"]
    )
    
    # Add generation differences
    for (g, gen) in enumerate(generators)
        detailed_comparison_df[!, "PF_$(gen.name)_Gen"] = pf_result["generation"][g, :]
        detailed_comparison_df[!, "DLAC_i_$(gen.name)_Gen"] = dlac_result["generation"][g, :]
        detailed_comparison_df[!, "$(gen.name)_Gen_Diff"] = dlac_result["generation"][g, :] - pf_result["generation"][g, :]
    end
    
    CSV.write(joinpath(output_dir, "four_model_detailed_comparison.csv"), detailed_comparison_df)
    
    # Summary statistics
    summary_stats = DataFrame(
        Metric = ["PF_Total_Cost", "DLAC_i_Total_Cost", "Cost_Difference", 
                 "PF_Total_Load_Shed", "DLAC_i_Total_Load_Shed", "Load_Shed_Difference",
                 "PF_Avg_Price", "DLAC_i_Avg_Price", "Price_Difference",
                 "PF_Max_Price", "DLAC_i_Max_Price"],
        Value = [pf_result["total_cost"], dlac_result["total_cost"], 
                dlac_result["total_cost"] - pf_result["total_cost"],
                sum(pf_result["load_shed"]), sum(dlac_result["load_shed"]),
                sum(dlac_result["load_shed"]) - sum(pf_result["load_shed"]),
                mean(pf_result["prices"]), mean(dlac_result["prices"]),
                mean(dlac_result["prices"]) - mean(pf_result["prices"]),
                maximum(pf_result["prices"]), maximum(dlac_result["prices"])]
    )
    CSV.write(joinpath(output_dir, "comprehensive_forecast_quality_analysis.csv"), summary_stats)
    
    println("Comprehensive model comparison saved to CSV files")
    
    return Dict(
        "comparison" => comparison_df,
        "cost_summary" => summary_df,
        "detailed_comparison" => detailed_comparison_df,
        "summary_stats" => summary_stats
    )
end

"""
    print_capacity_results(generators, battery, capacities, battery_power_cap, battery_energy_cap)

Print formatted capacity results.
"""
function print_capacity_results(generators, battery, capacities, battery_power_cap, battery_energy_cap)
    println("\nOptimal Capacities:")
    total_investment = 0.0
    
    for (i, gen) in enumerate(generators)
        capacity_mw = round(capacities[i], digits=1)
        investment_cost = gen.inv_cost * capacities[i]
        total_investment += investment_cost
        println("  $(gen.name): $(capacity_mw) MW (\$$(round(investment_cost/1e6, digits=1))M investment)")
    end
    
    battery_investment = battery.inv_cost_power * battery_power_cap + battery.inv_cost_energy * battery_energy_cap
    total_investment += battery_investment
    
    println("  Battery: $(round(battery_power_cap, digits=1)) MW / $(round(battery_energy_cap, digits=1)) MWh")
    println("           (\$$(round(battery_investment/1e6, digits=1))M investment)")
    println("  Total Investment: \$$(round(total_investment/1e6, digits=1))M")
end

"""
    run_performance_test_system(; params=nothing, output_dir="performance_results")

Run performance comparison between original and optimized (cached) DLAC-i and SLAC operations.
Tests model reuse with warm starts against original implementations.
"""
function run_performance_test_system(; params=nothing, output_dir="performance_results")
    println("🚀 Running Model Reuse Performance Test System")
    println(repeat("=", 60))
    
    # Use provided parameters or defaults (same as complete test)
    if params === nothing
        params = get_default_system_parameters()
    end
    
    # Create complete system with specified profiles
    generators, battery, profiles = create_complete_toy_system(params)
    
    println("Performance Test Configuration:")
    println("  Time horizon: $(params.hours) hours ($(params.days) days)")
    println("  Scenarios: $(profiles.n_scenarios)")
    
    # STEP 1: Run Capacity Expansion Model to get optimal capacities (same as complete test)
    println("\n" * repeat("=", 25) * " CAPACITY EXPANSION MODEL " * repeat("=", 25))
    cem_result = solve_capacity_expansion_model(generators, battery, profiles; 
                                               output_dir=output_dir)
    
    if cem_result["status"] != "optimal"
        println("❌ Capacity Expansion Model failed: $(cem_result["status"])")
        return Dict("status" => "failed", "stage" => "CEM", "result" => cem_result)
    end
    
    # Use optimal capacities from CEM (not hardcoded values)
    capacities = cem_result["capacity"]
    battery_power_cap = cem_result["battery_power_cap"]
    battery_energy_cap = cem_result["battery_energy_cap"]
    lookahead_hours = 24  # Use same as complete test
    
    println("✅ Capacity Expansion Model solved successfully!")
    print_capacity_results(generators, battery, capacities, battery_power_cap, battery_energy_cap)
    println("  Using lookahead: $lookahead_hours hours")
    
    # Create output directories
    mkpath(output_dir)
    mkpath(joinpath(output_dir, "original"))
    mkpath(joinpath(output_dir, "cached"))
    
    println("\n" * repeat("=", 60))
    println("PERFORMANCE COMPARISON TESTS")
    println(repeat("=", 60))
    
    # Create model cache for cached tests
    model_cache = ModelCache(lookahead_hours, profiles.n_scenarios)
    
    # Test DLAC-i Performance
    println("\nTesting DLAC-i Performance...")
    dlac_original_time = @elapsed solve_dlac_i_operations(
        generators, battery, capacities, battery_power_cap, battery_energy_cap,
        profiles; lookahead_hours=lookahead_hours, output_dir=joinpath(output_dir, "original")
    )
    
    dlac_cached_time = @elapsed solve_dlac_i_operations_cached(
        generators, battery, capacities, battery_power_cap, battery_energy_cap,
        profiles, model_cache; lookahead_hours=lookahead_hours, output_dir=joinpath(output_dir, "cached")
    )
    
    # Reset cache for SLAC test
    model_cache = ModelCache(lookahead_hours, profiles.n_scenarios)
    
    # Test SLAC Performance  
    println("\nTesting SLAC Performance...")
    slac_original_time = @elapsed solve_slac_operations(
        generators, battery, capacities, battery_power_cap, battery_energy_cap,
        profiles; lookahead_hours=lookahead_hours, output_dir=joinpath(output_dir, "original")
    )
    
    slac_cached_time = @elapsed solve_slac_operations_cached(
        generators, battery, capacities, battery_power_cap, battery_energy_cap,
        profiles, model_cache; lookahead_hours=lookahead_hours, output_dir=joinpath(output_dir, "cached")
    )
    
    # Calculate performance improvements
    dlac_speedup = dlac_original_time / dlac_cached_time
    dlac_improvement = (1 - dlac_cached_time/dlac_original_time) * 100
    
    slac_speedup = slac_original_time / slac_cached_time
    slac_improvement = (1 - slac_cached_time/slac_original_time) * 100
    
    avg_speedup = (dlac_speedup + slac_speedup) / 2
    
    # Create performance results summary
    performance_df = DataFrame(
        Method = ["DLAC-i", "SLAC"],
        Original_Time_s = [dlac_original_time, slac_original_time],
        Cached_Time_s = [dlac_cached_time, slac_cached_time],
        Speedup = [dlac_speedup, slac_speedup],
        Improvement_Percent = [dlac_improvement, slac_improvement]
    )
    
    CSV.write(joinpath(output_dir, "performance_comparison_results.csv"), performance_df)
    
    # Print results
    println("\n" * repeat("=", 60))
    println("PERFORMANCE RESULTS SUMMARY")
    println(repeat("=", 60))
    
    println("DLAC-i Results:")
    println("  Original:  $(round(dlac_original_time, digits=2))s")
    println("  Cached:    $(round(dlac_cached_time, digits=2))s")
    println("  Speedup:   $(round(dlac_speedup, digits=2))x ($(round(dlac_improvement, digits=1))% improvement)")
    
    println("\nSLAC Results:")
    println("  Original:  $(round(slac_original_time, digits=2))s")
    println("  Cached:    $(round(slac_cached_time, digits=2))s")
    println("  Speedup:   $(round(slac_speedup, digits=2))x ($(round(slac_improvement, digits=1))% improvement)")
    
    println("\nOverall Performance:")
    println("  Average Speedup: $(round(avg_speedup, digits=2))x")
    
    if avg_speedup >= 2.0
        println("\n🎉 Excellent performance gains! Model reuse is working effectively.")
    elseif avg_speedup >= 1.5
        println("\n✨ Good performance improvements achieved!")
    else
        println("\n⚠️  Performance gains are modest. Consider larger problems for better demonstration.")
    end
    
    println("\nKey Benefits Achieved:")
    println("• Model structure built once, reused across rolling horizon")
    println("• Warm starts accelerate optimization convergence")
    println("• Constraint RHS updates instead of model rebuilding")
    println("• Equilibrium-aware model caching for multiple iterations")
    
    return Dict(
        "status" => "success",
        "performance_results" => performance_df,
        "dlac_speedup" => dlac_speedup,
        "slac_speedup" => slac_speedup,
        "average_speedup" => avg_speedup
    )
end

"""
    validate_cached_results_match_original(; params=nothing, output_dir="validation_results")

Validate that cached operations produce identical results to original operations.
Tests both DLAC-i and SLAC with same capacities and profiles.
"""
function validate_cached_results_match_original(; params=nothing, output_dir="validation_results")
    println("🔍 Validating Cached Operations Match Original Results")
    println(repeat("=", 60))
    
    # Use provided parameters or defaults  
    if params === nothing
        params = get_default_system_parameters()
    end
    
    # Create complete system
    generators, battery, profiles = create_complete_toy_system(params)
    
    # Run capacity expansion to get optimal capacities
    cem_result = solve_capacity_expansion_model(generators, battery, profiles; 
                                               output_dir=output_dir)
    
    if cem_result["status"] != "optimal"
        println("❌ Capacity Expansion failed, cannot validate")
        return Dict("status" => "failed", "reason" => "CEM failed")
    end
    
    capacities = cem_result["capacity"]
    battery_power_cap = cem_result["battery_power_cap"]
    battery_energy_cap = cem_result["battery_energy_cap"]
    lookahead_hours = 24
    
    println("✓ Using optimal capacities from CEM for validation")
    print_capacity_results(generators, battery, capacities, battery_power_cap, battery_energy_cap)
    
    # Initialize model cache
    model_cache = ModelCache(lookahead_hours, profiles.n_scenarios)
    
    println("\n" * repeat("=", 60))
    println("VALIDATION TESTS")
    println(repeat("=", 60))
    
    # Test DLAC-i
    println("\n🧪 Testing DLAC-i: Original vs Cached")
    
    dlac_original = solve_dlac_i_operations(generators, battery, capacities, 
                                           battery_power_cap, battery_energy_cap, profiles;
                                           lookahead_hours=lookahead_hours, 
                                           output_dir=joinpath(output_dir, "original"))
    
    dlac_cached = solve_dlac_i_operations_cached(generators, battery, capacities,
                                               battery_power_cap, battery_energy_cap, profiles, model_cache;
                                               lookahead_hours=lookahead_hours,
                                               output_dir=joinpath(output_dir, "cached"))
    
    dlac_match = validate_results_identical(dlac_original, dlac_cached, "DLAC-i")
    
    # Reset cache for SLAC test
    model_cache = ModelCache(lookahead_hours, profiles.n_scenarios)
    
    # Test SLAC  
    println("\n🧪 Testing SLAC: Original vs Cached")
    
    slac_original = solve_slac_operations(generators, battery, capacities,
                                         battery_power_cap, battery_energy_cap, profiles;
                                         lookahead_hours=lookahead_hours,
                                         output_dir=joinpath(output_dir, "original"))
    
    slac_cached = solve_slac_operations_cached(generators, battery, capacities,
                                             battery_power_cap, battery_energy_cap, profiles, model_cache;
                                             lookahead_hours=lookahead_hours,
                                             output_dir=joinpath(output_dir, "cached"))
    
    slac_match = validate_results_identical(slac_original, slac_cached, "SLAC")
    
    # Summary
    println("\n" * repeat("=", 60))
    println("VALIDATION RESULTS SUMMARY")
    println(repeat("=", 60))
    
    if dlac_match && slac_match
        println("✅ SUCCESS: All cached operations produce identical results!")
        println("  DLAC-i: ✓ Identical")
        println("  SLAC:   ✓ Identical")
        return Dict("status" => "success", "dlac_match" => true, "slac_match" => true)
    else
        println("❌ FAILURE: Cached operations do not match original")
        println("  DLAC-i: $(dlac_match ? "✓" : "❌")")
        println("  SLAC:   $(slac_match ? "✓" : "❌")")
        return Dict("status" => "failed", "dlac_match" => dlac_match, "slac_match" => slac_match)
    end
end

"""
    validate_results_identical(result1, result2, model_name)

Compare two optimization results dictionaries for identical values.
Returns true if all key metrics match within numerical precision.
"""
function validate_results_identical(result1::Dict, result2::Dict, model_name::String)
    tolerance = 1e-10  # Very tight tolerance for exact match
    
    # Key metrics to compare
    metrics = ["total_cost", "generation", "generation_flex", "battery_charge", "battery_discharge", 
               "battery_soc", "load_shed", "prices"]
    
    all_match = true
    
    for metric in metrics
        if haskey(result1, metric) && haskey(result2, metric)
            val1 = result1[metric]
            val2 = result2[metric]
            
            if isa(val1, Array) && isa(val2, Array)
                if size(val1) != size(val2)
                    println("  ❌ $metric: Array sizes differ - $(size(val1)) vs $(size(val2))")
                    all_match = false
                    continue
                end
                
                max_diff = maximum(abs.(val1 - val2))
                if max_diff > tolerance
                    println("  ❌ $metric: Max difference $(max_diff) > tolerance $(tolerance)")
                    all_match = false
                else
                    println("  ✓ $metric: Identical (max diff: $(max_diff))")
                end
            else
                diff = abs(val1 - val2)
                if diff > tolerance
                    println("  ❌ $metric: Difference $(diff) > tolerance $(tolerance)")
                    all_match = false
                else
                    println("  ✓ $metric: Identical (diff: $(diff))")
                end
            end
        else
            println("  ⚠️  $metric: Missing in one result")
        end
    end
    
    return all_match
end

end # module TestRunner