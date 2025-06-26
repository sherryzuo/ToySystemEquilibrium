"""
SystemConfig.jl

System configuration module for ToySystemQuad.jl
Defines technology parameters, system setup, and validation functions.
"""

module SystemConfig

export Generator, Battery, SystemParameters
export get_default_system_parameters
export create_nuclear_generator, create_wind_generator, create_gas_generator
export create_battery_storage, create_toy_system
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

struct SystemParameters
    hours::Int              # Total simulation hours
    days::Int               # Number of days
    random_seed::Int        # For reproducibility
    load_shed_penalty::Float64  # $/MWh penalty for unserved energy
    load_shed_quad::Float64     # Quadratic load shed penalty coefficient
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
        3.0,          # var_om_cost ($/MWh)
        85000.0,      # inv_cost ($/MW/year)
        22000.0,      # fixed_om_cost ($/MW/year)
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
        90.0,         # fuel_cost ($/MWh)
        4.0,          # var_om_cost ($/MWh)
        55000.0,      # inv_cost ($/MW/year)
        12000.0,      # fixed_om_cost ($/MW/year)
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
    create_toy_system()

Create the complete 4-technology system with default parameters.
Returns generators array and battery storage.
"""
function create_toy_system()
    generators = [
        create_nuclear_generator(),
        create_wind_generator(), 
        create_gas_generator()
    ]
    
    battery = create_battery_storage()
    
    return generators, battery
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

end # module SystemConfig