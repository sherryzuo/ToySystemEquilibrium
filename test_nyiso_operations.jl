#!/usr/bin/env julia

using Pkg; Pkg.activate(".")
include("src/ToySystemQuad.jl")
using .ToySystemQuad

# Test NYISO operational models with existing capacities
println("Creating NYISO system with full year data...")
generators, battery, profiles = create_nyiso_system()

println("System created:")
println("  Time horizon: $(profiles.params.hours) hours ($(profiles.params.days) days)")
println("  Generators ($(length(generators))):")
for (i, gen) in enumerate(generators)
    println("    $i. $(gen.name): Existing=$(round(gen.existing_capacity, digits=0)) MW, VarCost=\$$(round(gen.var_om_cost + gen.fuel_cost, digits=1))/MWh")
end
println("  Demand profile: peak=$(round(maximum(profiles.actual_demand), digits=0)) MW")

# Use existing capacities as fixed capacities for operational models
existing_capacities = [gen.existing_capacity for gen in generators]
existing_battery_power = battery.existing_power_capacity
existing_battery_energy = battery.existing_power_capacity * battery.duration

println("\nUsing existing capacities for operational testing:")
for (i, gen) in enumerate(generators)
    println("  $(gen.name): $(round(existing_capacities[i], digits=1)) MW")
end
println("  Battery Power: $(round(existing_battery_power, digits=1)) MW")
println("  Battery Energy: $(round(existing_battery_energy, digits=1)) MWh")

# Test Perfect Foresight Operations
println("\n" * "="^80)
println("TESTING PERFECT FORESIGHT OPERATIONS")
println("="^80)
try
    pf_result = solve_perfect_foresight_operations(generators, battery, existing_capacities, 
                                                   existing_battery_power, existing_battery_energy, profiles)
    if pf_result["status"] == "optimal"
        println("✅ Perfect Foresight solved successfully")
        total_generation = sum(pf_result["generation"]) + sum(pf_result["generation_flex"])
        total_demand = sum(profiles.actual_demand) + profiles.params.flex_demand_mw * profiles.params.hours
        total_load_shed = sum(pf_result["load_shed"])
        
        println("  Total generation: $(round(total_generation, digits=1)) MWh")
        println("  Total demand: $(round(total_demand, digits=1)) MWh")
        println("  Total load shed: $(round(total_load_shed, digits=1)) MWh ($(round(100*total_load_shed/total_demand, digits=2))%)")
        println("  Total cost: \$$(round(pf_result["total_cost"]/1e6, digits=1))M")
    else
        println("❌ Perfect Foresight failed: $(pf_result["status"])")
    end
catch e
    println("❌ Perfect Foresight ERROR: ", e)
end

# Test DLAC-i Operations  
println("\n" * "="^80)
println("TESTING DLAC-I OPERATIONS")
println("="^80)
try
    dlac_cache = ModelCache(24, profiles.n_scenarios)  # 24-hour lookahead, n_scenarios from profiles
    dlac_result = solve_dlac_i_operations_cached(generators, battery, existing_capacities, 
                                                 existing_battery_power, existing_battery_energy, profiles, dlac_cache)
    if dlac_result["status"] == "optimal"
        println("✅ DLAC-i solved successfully")
        total_generation = sum(dlac_result["generation"]) + sum(dlac_result["generation_flex"])
        total_demand = sum(profiles.actual_demand) + profiles.params.flex_demand_mw * profiles.params.hours
        total_load_shed = sum(dlac_result["load_shed"])
        
        println("  Total generation: $(round(total_generation, digits=1)) MWh")
        println("  Total demand: $(round(total_demand, digits=1)) MWh")
        println("  Total load shed: $(round(total_load_shed, digits=1)) MWh ($(round(100*total_load_shed/total_demand, digits=2))%)")
        println("  Total cost: \$$(round(dlac_result["total_cost"]/1e6, digits=1))M")
    else
        println("❌ DLAC-i failed: $(dlac_result["status"])")
    end
catch e
    println("❌ DLAC-i ERROR: ", e)
end

# Test SLAC Operations
println("\n" * "="^80)
println("TESTING SLAC OPERATIONS")
println("="^80)
try
    slac_cache = ModelCache(24, profiles.n_scenarios)  # 24-hour lookahead, n_scenarios from profiles
    slac_result = solve_slac_operations_cached(generators, battery, existing_capacities, 
                                               existing_battery_power, existing_battery_energy, profiles, slac_cache)
    if slac_result["status"] == "optimal"
        println("✅ SLAC solved successfully")
        total_generation = sum(slac_result["generation"]) + sum(slac_result["generation_flex"])
        total_demand = sum(profiles.actual_demand) + profiles.params.flex_demand_mw * profiles.params.hours
        total_load_shed = sum(slac_result["load_shed"])
        
        println("  Total generation: $(round(total_generation, digits=1)) MWh")
        println("  Total demand: $(round(total_demand, digits=1)) MWh")
        println("  Total load shed: $(round(total_load_shed, digits=1)) MWh ($(round(100*total_load_shed/total_demand, digits=2))%)")
        println("  Total cost: \$$(round(slac_result["total_cost"]/1e6, digits=1))M")
    else
        println("❌ SLAC failed: $(slac_result["status"])")
    end
catch e
    println("❌ SLAC ERROR: ", e)
end

println("\n" * "="^80)
println("OPERATIONAL TESTING COMPLETE")
println("="^80)