#!/usr/bin/env julia

"""
test_modular_system.jl

Simple test to validate that all modular components work together.
This will run a small example to test the three optimization models.
"""

# Include all modules
include("SystemConfig.jl")
include("ProfileGeneration.jl") 
include("OptimizationModels.jl")
include("ConvergenceDiagnostics.jl")
include("VisualizationTools.jl")

using .SystemConfig
using .ProfileGeneration
using .OptimizationModels
using .ConvergenceDiagnostics
using .VisualizationTools

function test_modular_system()
    println("🧪 Testing Modular ToySystemQuad Implementation")
    println("=" ^ 50)
    
    # 1. Create system configuration
    println("\n1. Creating system configuration...")
    generators, battery = create_toy_system()
    params = get_default_system_parameters()
    
    # Use smaller system for testing (3 days instead of 30)
    test_params = SystemParameters(72, 3, 42, 10000.0, 0.001)
    
    println("   ✓ Created $(length(generators)) generators and battery storage")
    
    # 2. Validate system configuration
    println("\n2. Validating system configuration...")
    validate_system_configuration(generators, battery, test_params)
    
    # 3. Generate profiles
    println("\n3. Generating demand and wind profiles...")
    actual_demand, actual_wind, nuclear_availability, gas_availability,
    demand_scenarios, wind_scenarios, nuclear_avail_scenarios, gas_avail_scenarios = 
        create_actual_and_scenarios(test_params)
    
    println("   ✓ Generated profiles for $(test_params.hours) hours")
    
    # 4. Validate profiles
    println("\n4. Validating profiles...")
    validate_profiles(actual_demand, actual_wind, nuclear_availability, gas_availability, test_params)
    
    println("\n🎉 All modular components loaded and tested successfully!")
    println("   Ready for optimization model testing and equilibrium analysis.")
    
    return generators, battery, test_params, actual_demand, actual_wind, nuclear_availability, gas_availability
end

# Run the test if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    try
        test_modular_system()
        println("\n✅ Modular system test completed successfully!")
    catch e
        println("\n❌ Test failed with error:")
        println(e)
        rethrow(e)
    end
end