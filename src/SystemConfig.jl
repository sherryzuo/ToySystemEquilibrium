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
        8711,   # hours (full year)
        365,    # days (full year)
        1,      # N (individual generators, not fleets)
        42,     # random_seed
        10000.0, # load_shed_penalty
        0.001,   # load_shed_quad
        1000.0   #flex_demand_mw
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
    renewable_generators = NYISODataLoader.process_nyiso_renewable_generators(vre_df, hydro_df, storage_df, Generator)
    
    # Combine all generators
    generators = vcat(thermal_generators, renewable_generators)
    
    # Process storage
    battery = NYISODataLoader.process_nyiso_storage(storage_df, Battery)
    
    # Adjust investment costs for more diverse CEM results
    println("Adjusting investment costs for diverse capacity expansion...")
    
    # Increase gas plant investment costs by 1.5x (reflecting recent reports)
    gas_multiplier = 1.5
    for (i, gen) in enumerate(generators)
        if gen.name in ["CC", "CT", "ST"]  # Gas technologies
            old_cost = gen.inv_cost
            new_generator = Generator(
                gen.name,
                gen.fuel_cost,
                gen.var_om_cost,
                gen.inv_cost * gas_multiplier,  # Increase investment cost
                gen.fixed_om_cost,
                gen.max_capacity,
                gen.min_stable_gen,
                gen.ramp_rate,
                gen.efficiency,
                gen.startup_cost,
                gen.existing_capacity
            )
            generators[i] = new_generator
            new_cost = new_generator.inv_cost
            println("  - $(gen.name): Investment cost $(round(old_cost/1000))k → $(round(new_cost/1000))k/MW/yr (+$(round((gas_multiplier-1)*100))%)")
        end
    end
    
    # Reduce renewable investment costs by 0.7x (reflecting state subsidies)
    for (i, gen) in enumerate(generators)
        if gen.name in ["Wind", "Solar"]  # Renewable technologies
            if gen.name == "Wind"
                renewable_multiplier = 0.9  # Wind investment cost reduction
            elseif gen.name == "Solar"
                renewable_multiplier = 0.7  # Solar investment cost reduction
            end
            old_cost = gen.inv_cost
            new_generator = Generator(
                gen.name,
                gen.fuel_cost,
                gen.var_om_cost,
                gen.inv_cost * renewable_multiplier,  # Reduce investment cost
                gen.fixed_om_cost,
                gen.max_capacity,
                gen.min_stable_gen,
                gen.ramp_rate,
                gen.efficiency,
                gen.startup_cost,
                gen.existing_capacity
            )
            generators[i] = new_generator
            new_cost = new_generator.inv_cost
            println("  - $(gen.name): Investment cost $(round(old_cost/1000))k → $(round(new_cost/1000))k/MW/yr (-$(round((1-renewable_multiplier)*100))%)")
        end
    end
    
    # Reduce battery investment costs by 0.8x (reflecting subsidies and technology improvements)
    battery_multiplier = 0.8
    old_power_cost = battery.inv_cost_power
    old_energy_cost = battery.inv_cost_energy
    battery = Battery(
        battery.name,
        battery.inv_cost_power * battery_multiplier,  # Reduce power investment cost
        battery.inv_cost_energy * battery_multiplier,  # Reduce energy investment cost
        battery.fixed_om_cost,
        battery.var_om_cost,
        battery.max_power_capacity,
        battery.max_energy_capacity,
        battery.efficiency_charge,
        battery.efficiency_discharge,
        battery.duration,
        battery.existing_power_capacity
    )
    println("  - Battery Power: Investment cost $(round(old_power_cost/1000))k → $(round(battery.inv_cost_power/1000))k/MW/yr (-$(round((1-battery_multiplier)*100))%)")
    println("  - Battery Energy: Investment cost $(round(old_energy_cost/1000))k → $(round(battery.inv_cost_energy/1000))k/MWh/yr (-$(round((1-battery_multiplier)*100))%)")
    
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