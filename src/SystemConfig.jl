"""
SystemConfig.jl

System configuration module for ToySystemQuad.jl
Defines technology parameters, system setup, and validation functions.
"""

module SystemConfig

using ..ProfileGeneration
using ..ProfileGeneration: SystemParameters

export Generator, Battery, SystemParameters, SystemProfiles
export get_default_system_parameters
export create_nuclear_generator, create_wind_generator, create_gas_generator
export create_battery_storage
export generate_system_profiles, create_complete_toy_system
export validate_system_configuration

# =============================================================================
# SYSTEM DATA STRUCTURES
# =============================================================================

struct Generator
    name::String
    fuel_cost::Float64      # $/MWh
    var_om_cost::Float64    # $/MWh  
    inv_cost::Float64       # $/MW/year
    fixed_om_cost::Float64  # $/MW/year
    max_capacity::Float64   # MW (for capacity expansion)
    min_stable_gen::Float64 # Minimum stable generation as fraction of capacity
    ramp_rate::Float64      # MW/h (as fraction of capacity)
    efficiency::Float64     # p.u.
    startup_cost::Float64   # $/startup
end

struct Battery
    name::String
    inv_cost_power::Float64    # $/MW/year (power capacity)
    inv_cost_energy::Float64   # $/MWh/year (energy capacity) 
    fixed_om_cost::Float64     # $/MW/year
    var_om_cost::Float64       # $/MWh
    max_power_capacity::Float64 # MW
    max_energy_capacity::Float64 # MWh
    efficiency_charge::Float64  # p.u.
    efficiency_discharge::Float64 # p.u.
    duration::Float64          # hours (energy/power ratio)
end

struct SystemProfiles
    # Actual profiles (realizations)
    actual_demand::Vector{Float64}
    actual_wind::Vector{Float64}
    actual_nuclear_availability::Vector{Float64}
    actual_gas_availability::Vector{Float64}
    
    # Forecast scenarios (for DLAC-i)
    demand_scenarios::Vector{Vector{Float64}}
    wind_scenarios::Vector{Vector{Float64}}
    nuclear_availability_scenarios::Vector{Vector{Float64}}
    gas_availability_scenarios::Vector{Vector{Float64}}
    
    # Metadata
    n_scenarios::Int
    params::SystemParameters
end

# =============================================================================
# SYSTEM CONFIGURATION FUNCTIONS
# =============================================================================

"""
    get_default_system_parameters()

Returns default system parameters for the toy system.
"""
function get_default_system_parameters()
    return SystemParameters(
        720,    # hours (30 days)
        30,     # days
        5,    # N (number of generators per technology)
        42,     # random_seed
        10000.0, # load_shed_penalty
        0.001   # load_shed_quad
    )
end

"""
    create_nuclear_generator()

Create nuclear generator with default parameters.
Nuclear: Low fuel cost, high investment, baseload operation.
"""
function create_nuclear_generator()
    return Generator(
        "Nuclear",     # name
        12.0,         # fuel_cost ($/MWh)
        2.0,          # var_om_cost ($/MWh)
        120000.0,     # inv_cost ($/MW/year)
        35000.0,      # fixed_om_cost ($/MW/year)
        1200.0,       # max_capacity (MW)
        0.9,          # min_stable_gen (fraction)
        0.1,          # ramp_rate (fraction/hour)
        0.33,         # efficiency
        2000.0        # startup_cost ($)
    )
end

"""
    create_wind_generator()

Create wind generator with default parameters.
Wind: Zero fuel cost, moderate investment, variable output.
"""
function create_wind_generator()
    return Generator(
        "Wind",       # name
        0.0,          # fuel_cost ($/MWh)
        0.0,          # var_om_cost ($/MWh)
        85000.0,      # inv_cost ($/MW/year)
        12000.0,      # fixed_om_cost ($/MW/year)
        1500.0,       # max_capacity (MW)
        0.0,          # min_stable_gen (fraction)
        1.0,          # ramp_rate (fraction/hour)
        1.0,          # efficiency
        0.0           # startup_cost ($)
    )
end

"""
    create_gas_generator()

Create gas generator with default parameters.
Gas: High fuel cost, low investment, flexible peaker.
"""
function create_gas_generator()
    return Generator(
        "Gas",        # name
        60.0,         # fuel_cost ($/MWh)
        8.0,          # var_om_cost ($/MWh)
        70000.0,      # inv_cost ($/MW/year)
        16000.0,      # fixed_om_cost ($/MW/year)
        1000.0,       # max_capacity (MW)
        0.2,          # min_stable_gen (fraction)
        1.0,          # ramp_rate (fraction/hour)
        0.45,         # efficiency
        80.0          # startup_cost ($)
    )
end

"""
    create_battery_storage()

Create battery storage with default parameters.
Battery: Medium investment, provides flexibility and arbitrage.
"""
function create_battery_storage()
    return Battery(
        "Battery",    # name
        95000.0,      # inv_cost_power ($/MW/year)
        100.0,        # inv_cost_energy ($/MWh/year)
        6000.0,       # fixed_om_cost ($/MW/year)
        1.5,          # var_om_cost ($/MWh)
        800.0,        # max_power_capacity (MW)
        3200.0,       # max_energy_capacity (MWh)
        0.90,         # efficiency_charge
        0.90,         # efficiency_discharge
        4.0           # duration (hours)
    )
end


"""
    validate_system_configuration(generators, battery, params)

Validate that system configuration is internally consistent and economically reasonable.
"""
function validate_system_configuration(generators, battery, params)
    # Check that generators have positive costs and capacities
    for gen in generators
        @assert gen.inv_cost > 0 "Generator $(gen.name) must have positive investment cost"
        @assert gen.max_capacity > 0 "Generator $(gen.name) must have positive max capacity"
        @assert 0 <= gen.min_stable_gen <= 1 "Generator $(gen.name) min stable gen must be in [0,1]"
        @assert 0 < gen.efficiency <= 1 "Generator $(gen.name) efficiency must be in (0,1]"
    end
    
    # Check battery parameters
    @assert battery.inv_cost_power > 0 "Battery must have positive power investment cost"
    @assert battery.inv_cost_energy > 0 "Battery must have positive energy investment cost"
    @assert battery.max_power_capacity > 0 "Battery must have positive max power capacity"
    @assert battery.max_energy_capacity > 0 "Battery must have positive max energy capacity"
    @assert 0 < battery.efficiency_charge <= 1 "Battery charge efficiency must be in (0,1]"
    @assert 0 < battery.efficiency_discharge <= 1 "Battery discharge efficiency must be in (0,1]"
    @assert battery.duration ≈ battery.max_energy_capacity / battery.max_power_capacity "Battery duration must match energy/power ratio"
    
    # Check system parameters
    @assert params.hours > 0 "System must have positive hours"
    @assert params.days > 0 "System must have positive days"
    @assert params.load_shed_penalty > 0 "Load shed penalty must be positive"
    
    println("✓ System configuration validated successfully")
    return true
end

"""
    generate_system_profiles(params::SystemParameters=get_default_system_parameters())

Generate both actual profiles and forecast scenarios for the system.
This centralizes all profile generation in SystemConfig rather than calling it repeatedly.
"""
function generate_system_profiles(params::SystemParameters=get_default_system_parameters())
    # Generate actual profiles
    actual_demand = ProfileGeneration.generate_demand_profile(params)
    actual_wind = ProfileGeneration.generate_wind_profile(params)
    actual_nuclear_availability = ProfileGeneration.generate_nuclear_availability(params)
    actual_gas_availability = ProfileGeneration.generate_gas_availability(params) 
    
    # Generate forecast scenarios (5 scenarios for DLAC-i)
    demand_scenarios, wind_scenarios, nuclear_availability_scenarios, gas_availability_scenarios = 
        ProfileGeneration.generate_scenarios(
            actual_demand, actual_wind, actual_nuclear_availability, actual_gas_availability, params; 
            n_scenarios=5
        )
    
    return SystemProfiles(
        actual_demand,
        actual_wind,
        actual_nuclear_availability,
        actual_gas_availability,
        demand_scenarios,
        wind_scenarios,
        nuclear_availability_scenarios,
        gas_availability_scenarios,
        5,  # n_scenarios
        params
    )
end

"""
    create_complete_toy_system(params::SystemParameters=get_default_system_parameters())

Create the complete toy system including technology parameters and all profiles.
Returns generators, battery, and system profiles.
"""
function create_complete_toy_system(params::SystemParameters=get_default_system_parameters())
    # Create technology components
    generators = [
        create_nuclear_generator(),
        create_wind_generator(), 
        create_gas_generator()
    ]
    battery = create_battery_storage()
    
    # Generate all profiles
    profiles = generate_system_profiles(params)
    
    # Validate everything
    validate_system_configuration(generators, battery, params)
    ProfileGeneration.validate_profiles(
        profiles.actual_demand, 
        profiles.actual_wind, 
        profiles.actual_nuclear_availability, 
        profiles.actual_gas_availability, 
        params
    )
    
    return generators, battery, profiles
end

end # module SystemConfig