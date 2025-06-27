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

"""
    run_complete_test_system(; output_dir="results")

Run complete test system with all three optimization models:
1. Capacity Expansion Model (CEM) 
2. Perfect Foresight Operations (DLAC-p)
3. DLAC-i Operations (rolling horizon)

Saves detailed results and comparisons to CSV files.
"""
function run_complete_test_system(; output_dir="results")
    println("🚀 Running Complete ToySystemQuad Test System")
    println(repeat("=", 60))
    
    # Create complete system with profiles
    generators, battery, profiles = create_complete_toy_system()
    params = profiles.params
    
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
    println("\n" * repeat("=", 25) * " DLAC-I OPERATIONS " * repeat("=", 27))
    
    dlac_result = solve_dlac_i_operations(generators, battery,
                                         optimal_capacities,
                                         optimal_battery_power,
                                         optimal_battery_energy,
                                         profiles;
                                         lookahead_hours=24,
                                         output_dir=output_dir)
    
    if dlac_result["status"] != "optimal"
        println("❌ DLAC-i Operations failed: $(dlac_result["status"])")
        return Dict("status" => "failed", "stage" => "DLAC-i", "result" => dlac_result)
    end
    
    println("✅ DLAC-i Operations solved successfully!")
    
    # STEP 4: Profit Analysis and PMR Calculation
    println("\n" * repeat("=", 25) * " PROFIT ANALYSIS " * repeat("=", 29))
    
    pf_profits = calculate_profits_and_save(generators, battery, pf_result,
                                           optimal_capacities, optimal_battery_power, optimal_battery_energy,
                                           "perfect_foresight", output_dir)
    
    dlac_profits = calculate_profits_and_save(generators, battery, dlac_result,
                                             optimal_capacities, optimal_battery_power, optimal_battery_energy,
                                             "dlac_i", output_dir)
    
    # Calculate PMRs
    pf_pmr = compute_pmr(pf_result, generators, battery, 
                        optimal_capacities, optimal_battery_power, optimal_battery_energy)
    dlac_pmr = compute_pmr(dlac_result, generators, battery,
                          optimal_capacities, optimal_battery_power, optimal_battery_energy)
    
    println("Perfect Foresight PMRs (%): $(round.(pf_pmr, digits=2))")
    println("DLAC-i PMRs (%): $(round.(dlac_pmr, digits=2))")
    
    # STEP 5: Comprehensive Model Comparison
    println("\n" * repeat("=", 25) * " MODEL COMPARISON " * repeat("=", 28))
    
    comparison_results = compare_models_analysis(cem_result, pf_result, dlac_result, 
                                               generators, battery, optimal_capacities,
                                               optimal_battery_power, optimal_battery_energy,
                                               output_dir)
    
    # STEP 6: Generate Comprehensive Plots
    println("\n" * repeat("=", 25) * " GENERATING PLOTS " * repeat("=", 28))
    
    generate_all_plots(cem_result, pf_result, dlac_result, profiles,
                      generators, battery, optimal_capacities, optimal_battery_power, output_dir)
    
    println("\nComplete test system finished successfully!")
    println("All results saved to: $(output_dir)/")
    
    return Dict(
        "status" => "success",
        "cem" => cem_result,
        "perfect_foresight" => pf_result,
        "dlac_i" => dlac_result,
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
function compare_models_analysis(cem_result, pf_result, dlac_result, generators, battery,
                                optimal_capacities, optimal_battery_power, optimal_battery_energy, output_dir)
    
    # Calculate PMRs for each model
    pf_pmr = compute_pmr(pf_result, generators, battery, optimal_capacities, optimal_battery_power, optimal_battery_energy)
    dlac_pmr = compute_pmr(dlac_result, generators, battery, optimal_capacities, optimal_battery_power, optimal_battery_energy)
    
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
        DLAC_i_PMR_Percent = dlac_pmr
    )
    
    # Add summary costs as a separate DataFrame
    summary_df = DataFrame(
        Model = ["CEM", "Perfect_Foresight", "DLAC_i"],
        Total_Cost = [cem_result["total_cost"], pf_result["total_cost"], dlac_result["total_cost"]]
    )
    CSV.write(joinpath(output_dir, "three_model_comprehensive_comparison.csv"), comparison_df)
    CSV.write(joinpath(output_dir, "three_model_cost_summary.csv"), summary_df)
    
    # Detailed PF vs DLAC-i comparison
    T = length(pf_result["prices"])
    detailed_comparison_df = DataFrame(
        Hour = 1:T,
        PF_Price = pf_result["prices"],
        DLAC_i_Price = dlac_result["prices"],
        Price_Difference = dlac_result["prices"] - pf_result["prices"],
        PF_Load_Shed = pf_result["load_shed"],
        DLAC_i_Load_Shed = dlac_result["load_shed"],
        Load_Shed_Difference = dlac_result["load_shed"] - pf_result["load_shed"]
    )
    
    # Add generation differences
    for (g, gen) in enumerate(generators)
        detailed_comparison_df[!, "PF_$(gen.name)_Gen"] = pf_result["generation"][g, :]
        detailed_comparison_df[!, "DLAC_i_$(gen.name)_Gen"] = dlac_result["generation"][g, :]
        detailed_comparison_df[!, "$(gen.name)_Gen_Diff"] = dlac_result["generation"][g, :] - pf_result["generation"][g, :]
    end
    
    CSV.write(joinpath(output_dir, "pf_vs_dlac_i_comprehensive_comparison.csv"), detailed_comparison_df)
    
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

end # module TestRunner