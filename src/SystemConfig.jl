"""
SystemConfig.jl

System configuration module for ToySystemQuad.jl
Defines technology parameters, system setup, and validation functions.
"""

module SystemConfig

# SystemParameters moved here from ProfileGeneration since that's all we need
struct SystemParameters
    hours::Int              # Total simulation hours
    days::Int 
    N::Int              # Number of days
    random_seed::Int        # For reproducibility
    load_shed_penalty::Float64  # $/MWh penalty for unserved energy
    load_shed_quad::Float64     # Quadratic load shed penalty coefficient
    flex_demand_mw::Float64     # MW of flexible demand (rest is fixed)
end
using ..NYISODataLoader

export Generator, Battery, SystemParameters, SystemProfiles
export get_nyiso_system_parameters
export create_nyiso_system
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
    max_capacity::Float64   # MW (for capacity expansion) - not used as constraint
    min_stable_gen::Float64 # Minimum stable generation as fraction of capacity - not used
    ramp_rate::Float64      # MW/h (as fraction of capacity) - not used
    efficiency::Float64     # p.u. - not used
    startup_cost::Float64   # $/startup - not used
    existing_capacity::Float64  # MW - existing capacity from NYISO data for reference
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
    existing_power_capacity::Float64  # MW - existing power capacity from NYISO data
end

struct SystemProfiles
    # Actual profiles (realizations)
    actual_demand::Vector{Float64}
    generator_availabilities::Vector{Vector{Float64}}  # availabilities[g] for generator g
    
    # Forecast scenarios (indexed by scenario ω, then by generator g)
    demand_scenarios::Vector{Vector{Float64}}  # demand_scenarios[ω] for scenario ω
    generator_availability_scenarios::Vector{Vector{Vector{Float64}}}  # scenarios[ω][g] for scenario ω, generator g
    
    # Metadata
    n_scenarios::Int
    params::SystemParameters
end

# =============================================================================
# SYSTEM CONFIGURATION FUNCTIONS
# =============================================================================


"""
    get_nyiso_system_parameters()

Returns system parameters optimized for NYISO full-year analysis.
"""
function get_nyiso_system_parameters()
    return SystemParameters(
        8760,   # hours (full year)
        365,    # days (full year)
        1,      # N (individual generators, not fleets)
        42,     # random_seed
        10000.0, # load_shed_penalty
        0.001,   # load_shed_quad
        1000.0   # flex_demand_mw (scaled up for full system)
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
    create_nyiso_system(params::SystemParameters=get_nyiso_system_parameters(); nyiso_data_path::String="NYISO_System")

Create the complete NYISO system using real NYISO data including generator parameters and profiles.
Returns generators, battery, and system profiles loaded from NYISO data files.
"""
function create_nyiso_system(params::SystemParameters=get_nyiso_system_parameters(); nyiso_data_path::String="NYISO_System")
    # Load NYISO data from CSV files
    thermal_df, vre_df, hydro_df, storage_df = NYISODataLoader.load_nyiso_generators(nyiso_data_path)
    demand_df = NYISODataLoader.load_nyiso_demand(nyiso_data_path)
    fuels_df = NYISODataLoader.load_nyiso_fuels(nyiso_data_path)
    variability_df = NYISODataLoader.load_nyiso_variability(nyiso_data_path)
    
    # Process generators 
    thermal_generators = NYISODataLoader.process_nyiso_thermal_generators(thermal_df, fuels_df, Generator)
    renewable_generators = NYISODataLoader.process_nyiso_renewable_generators(vre_df, hydro_df, Generator)
    
    # Combine all generators
    generators = vcat(thermal_generators, renewable_generators)
    
    # Process storage
    battery = NYISODataLoader.process_nyiso_storage(storage_df, Battery)
    
    # Create system profiles from NYISO data
    profiles = NYISODataLoader.create_nyiso_system_profiles(demand_df, variability_df, generators, params, SystemProfiles)
    
    # Validate everything
    validate_system_configuration(generators, battery, params)
    
    println("✓ NYISO system created successfully with $(length(generators)) generators")
    println("  - Thermal generators: $(length(thermal_generators))")
    println("  - Renewable generators: $(length(renewable_generators))")
    println("  - Battery storage: $(battery.name)")
    
    return generators, battery, profiles
end

end # module SystemConfig