using Pkg; Pkg.activate(".")
include("src/ToySystemQuad.jl")
using .ToySystemQuad

# Test full-year NYISO system with separate gas technologies
println("Creating NYISO system with full year data...")
generators, battery, profiles = create_nyiso_system()

println("System created:")
println("  Time horizon: $(profiles.params.hours) hours ($(profiles.params.days) days)")
println("  Generators ($(length(generators))):")
for (i, gen) in enumerate(generators)
    println("    $i. $(gen.name): Existing=$(round(gen.existing_capacity, digits=0)) MW, VarCost=\$$(round(gen.var_om_cost + gen.fuel_cost, digits=1))/MWh")
end
println("  Demand profile: peak=$(round(maximum(profiles.actual_demand), digits=0)) MW")

# Test capacity expansion
println("\nTesting Capacity Expansion Model with full-year NYISO data...")
try
    cem_result = solve_capacity_expansion_model(generators, battery, profiles)
    if cem_result["status"] == "optimal"
        println("✅ CEM solved successfully")
        for (i, gen) in enumerate(generators)
            println("  $(gen.name): $(round(cem_result["capacity"][i], digits=1)) MW")
        end
        println("  Battery: $(round(cem_result["battery_power_cap"], digits=1)) MW")
    else
        println("❌ CEM failed: $(cem_result["status"])")
    end
catch e
    println("❌ CEM ERROR: ", e)
end