"""
NYISODataLoader.jl

Data loading module for NYISO system data.
Loads and processes NYISO CSV files to create Generator, Battery, and SystemProfiles structs
compatible with the existing ToySystemQuad optimization framework.
"""

module NYISODataLoader

using CSV
using DataFrames
using Statistics

export load_nyiso_generators, load_nyiso_demand, load_nyiso_fuels, load_nyiso_variability
export create_nyiso_system_profiles, process_nyiso_thermal_generators
export process_nyiso_renewable_generators, process_nyiso_storage, interpolate_zero_values

# =============================================================================
# NYISO DATA LOADING FUNCTIONS
# =============================================================================

"""
    load_nyiso_generators(data_path::String)

Load all NYISO generator data from CSV files.
Returns DataFrames for thermal, VRE, hydro, and storage resources.
"""
function load_nyiso_generators(data_path::String)
    thermal_path = joinpath(data_path, "resources", "Thermal.csv")
    vre_path = joinpath(data_path, "resources", "Vre.csv")
    hydro_path = joinpath(data_path, "resources", "Hydro.csv")
    storage_path = joinpath(data_path, "resources", "Storage.csv")
    
    thermal_df = CSV.read(thermal_path, DataFrame)
    vre_df = CSV.read(vre_path, DataFrame)
    hydro_df = CSV.read(hydro_path, DataFrame)
    storage_df = CSV.read(storage_path, DataFrame)
    
    return thermal_df, vre_df, hydro_df, storage_df
end

"""
    load_nyiso_demand(data_path::String)

Load NYISO demand data from CSV file.
Returns DataFrame with hourly demand data.
"""
function load_nyiso_demand(data_path::String)
    demand_path = joinpath(data_path, "system", "Demand_data.csv")
    demand_df = CSV.read(demand_path, DataFrame)
    return demand_df
end

"""
    load_nyiso_fuels(data_path::String)

Load NYISO fuel price data from CSV file.
Returns DataFrame with fuel prices by time index.
"""
function load_nyiso_fuels(data_path::String)
    fuels_path = joinpath(data_path, "system", "Fuels_data.csv")
    fuels_df = CSV.read(fuels_path, DataFrame)
    return fuels_df
end

"""
    load_nyiso_variability(data_path::String)

Load NYISO generator variability data from CSV file.
Returns DataFrame with availability factors for all generators by time index.
"""
function load_nyiso_variability(data_path::String)
    variability_path = joinpath(data_path, "system", "Generators_variability.csv")
    variability_df = CSV.read(variability_path, DataFrame)
    return variability_df
end

# =============================================================================
# DATA PROCESSING UTILITY FUNCTIONS
# =============================================================================

"""
    interpolate_zero_values(data::Vector{Float64})

Replace zero values in time series data with linear interpolation between adjacent non-zero values.
This is useful for filling gaps in demand data where zeros indicate missing data points.
"""
function interpolate_zero_values(data::Vector{Float64})
    # Create a copy to avoid modifying the original
    interpolated = copy(data)
    n = length(interpolated)
    
    # Find all zero indices
    zero_indices = findall(x -> x == 0.0, interpolated)
    
    if isempty(zero_indices)
        return interpolated  # No zeros to interpolate
    end
    
    println("Found $(length(zero_indices)) zero values in data, interpolating...")
    
    for i in zero_indices
        # Find previous non-zero value
        prev_idx = i - 1
        while prev_idx >= 1 && interpolated[prev_idx] == 0.0
            prev_idx -= 1
        end
        
        # Find next non-zero value
        next_idx = i + 1
        while next_idx <= n && interpolated[next_idx] == 0.0
            next_idx += 1
        end
        
        # Interpolate based on available neighbors
        if prev_idx >= 1 && next_idx <= n
            # Linear interpolation between prev and next
            prev_val = interpolated[prev_idx]
            next_val = interpolated[next_idx]
            steps = next_idx - prev_idx
            step_size = (next_val - prev_val) / steps
            interpolated[i] = prev_val + step_size * (i - prev_idx)
        elseif prev_idx >= 1
            # Use previous value (extrapolate backwards)
            interpolated[i] = interpolated[prev_idx]
        elseif next_idx <= n
            # Use next value (extrapolate forwards)
            interpolated[i] = interpolated[next_idx]
        else
            # All values are zero, use a reasonable default (this shouldn't happen with real demand data)
            interpolated[i] = 1000.0  # Default 1000 MW
            println("Warning: All demand values are zero, using default value")
        end
    end
    
    return interpolated
end

# =============================================================================
# NYISO DATA PROCESSING FUNCTIONS
# =============================================================================

"""
    process_nyiso_thermal_generators(thermal_df::DataFrame, fuels_df::DataFrame, Generator)

Process NYISO thermal generators and combine variable O&M costs with fuel costs.
Returns Vector of Generator structs representing major technology types (Nuclear, Gas).
"""
function process_nyiso_thermal_generators(thermal_df::DataFrame, fuels_df::DataFrame, Generator)
    generators = Generator[]
    
    # Get average fuel prices (skip header row with units)
    avg_fuel_prices = Dict(
        "NG_NY" => mean(skipmissing(fuels_df[2:end, "NG_NY"])),
        "FO6_NY" => mean(skipmissing(fuels_df[2:end, "FO6_NY"])), 
        "FO2_NY" => mean(skipmissing(fuels_df[2:end, "FO2_NY"])),
        "None" => 0.0  # For nuclear
    )
    
    # Create representative generators for major technology types
    nuclear_existing = filter(row -> occursin("Nuclear", row.Resource) && !occursin("_New", row.Resource), thermal_df)
    nuclear_new = filter(row -> occursin("Nuclear", row.Resource) && occursin("_New", row.Resource), thermal_df)
    
    # Nuclear generator (use new for costs, existing for capacity reference)
    if nrow(nuclear_existing) > 0 && nrow(nuclear_new) > 0
        existing_row = nuclear_existing[1, :]
        new_row = nuclear_new[1, :]
        fuel_cost = avg_fuel_prices["None"] * new_row.Heat_Rate_MMBTU_per_MWh
        total_var_cost = new_row.Var_OM_Cost_per_MWh + fuel_cost
        
        nuclear_gen = Generator(
            "Nuclear",                      # name
            fuel_cost,                      # fuel_cost
            total_var_cost,                 # var_om_cost (includes fuel)
            new_row.Inv_Cost_per_MWyr,     # inv_cost (from new technology)
            new_row.Fixed_OM_Cost_per_MWyr, # fixed_om_cost (from new technology)
            50000.0,                       # max_capacity (not used as constraint)
            0.0,                           # min_stable_gen (not used)
            1.0,                           # ramp_rate (not used)
            1.0,                           # efficiency (not used)
            0.0,                           # startup_cost (not used)
            existing_row.Existing_Cap_MW   # existing_capacity
        )
        push!(generators, nuclear_gen)
    end
    
    # Gas technologies - separate CC, CT, ST with capacity-weighted fuel costs
    gas_technologies = [
        ("CombinedCycle", "CC"),
        ("CombustionTurbine", "CT"), 
        ("SteamTurbine", "ST")
    ]
    
    for (tech_pattern, tech_name) in gas_technologies
        existing_data = filter(row -> occursin(tech_pattern, row.Resource) && 
                                      occursin("NaturalGas", row.Resource) && 
                                      !occursin("_New", row.Resource), thermal_df)
        new_data = filter(row -> occursin(tech_pattern, row.Resource) && 
                                 occursin("NaturalGas", row.Resource) && 
                                 occursin("_New", row.Resource), thermal_df)
        
        if nrow(existing_data) > 0 && nrow(new_data) > 0
            new_row = new_data[1, :]  # Use new for costs
            total_existing_capacity = sum(existing_data.Existing_Cap_MW)
            
            # Calculate capacity-weighted fuel cost
            weighted_fuel_cost = 0.0
            total_capacity = 0.0
            for existing_row in eachrow(existing_data)
                capacity = existing_row.Existing_Cap_MW
                fuel_cost = avg_fuel_prices["NG_NY"] * existing_row.Heat_Rate_MMBTU_per_MWh
                weighted_fuel_cost += fuel_cost * capacity
                total_capacity += capacity
            end
            avg_fuel_cost = total_capacity > 0 ? weighted_fuel_cost / total_capacity : 0.0
            
            total_var_cost = new_row.Var_OM_Cost_per_MWh + avg_fuel_cost
            
            gas_gen = Generator(
                tech_name,                      # name (CC, CT, or ST)
                avg_fuel_cost,                  # fuel_cost (capacity-weighted)
                total_var_cost,                 # var_om_cost (includes fuel)
                new_row.Inv_Cost_per_MWyr,     # inv_cost (from new technology)
                new_row.Fixed_OM_Cost_per_MWyr, # fixed_om_cost (from new technology)
                50000.0,                       # max_capacity (not used as constraint)
                0.0,                           # min_stable_gen (not used)
                1.0,                           # ramp_rate (not used)
                1.0,                           # efficiency (not used)
                0.0,                           # startup_cost (not used)
                total_existing_capacity        # existing_capacity
            )
            push!(generators, gas_gen)
        end
    end
    
    return generators
end

"""
    process_nyiso_renewable_generators(vre_df::DataFrame, hydro_df::DataFrame, storage_df::DataFrame)

Process NYISO renewable generators (Wind, Solar, Hydro) into representative technology types.
Returns Vector of Generator structs for Wind, Solar, and Hydro.
Note: Uses PumpedHydro_NY from storage_df for hydro costs instead of hydro_df.
"""
function process_nyiso_renewable_generators(vre_df::DataFrame, hydro_df::DataFrame, storage_df::DataFrame, Generator)
    generators = Generator[]
    
    # Process Wind generators (use new for costs, existing for capacity reference)
    wind_existing = filter(row -> occursin("Wind", row.Resource) && !occursin("_New", row.Resource), vre_df)
    wind_new = filter(row -> occursin("Wind", row.Resource) && occursin("_New", row.Resource), vre_df)
    if nrow(wind_existing) > 0 && nrow(wind_new) > 0
        existing_wind = wind_existing[1, :]
        new_wind = wind_new[1, :]  # Use new wind for costs
        total_wind_capacity = sum(wind_existing.Existing_Cap_MW)
        
        wind_gen = Generator(
            "Wind",                         # name
            0.0,                            # fuel_cost (no fuel cost for renewables)
            new_wind.Var_OM_Cost_per_MWh,  # var_om_cost (from new technology)
            new_wind.Inv_Cost_per_MWyr,    # inv_cost (from new technology)
            new_wind.Fixed_OM_Cost_per_MWyr, # fixed_om_cost (from new technology)
            50000.0,                       # max_capacity (not used as constraint)
            0.0,                           # min_stable_gen (not used)
            1.0,                           # ramp_rate (not used)
            1.0,                           # efficiency (not used)
            0.0,                           # startup_cost (not used)
            total_wind_capacity            # existing_capacity
        )
        push!(generators, wind_gen)
    end
    
    # Process Solar generators (use new for costs, existing for capacity reference)
    solar_existing = filter(row -> occursin("Solar", row.Resource) && !occursin("_New", row.Resource), vre_df)
    solar_new = filter(row -> occursin("Solar", row.Resource) && occursin("_New", row.Resource), vre_df)
    if nrow(solar_existing) > 0 && nrow(solar_new) > 0
        existing_solar = solar_existing[1, :]
        new_solar = solar_new[1, :]  # Use new solar for costs
        total_solar_capacity = sum(solar_existing.Existing_Cap_MW)
        
        solar_gen = Generator(
            "Solar",                        # name
            0.0,                            # fuel_cost (no fuel cost for renewables)
            new_solar.Var_OM_Cost_per_MWh, # var_om_cost (from new technology)
            new_solar.Inv_Cost_per_MWyr,   # inv_cost (from new technology)
            new_solar.Fixed_OM_Cost_per_MWyr, # fixed_om_cost (from new technology)
            50000.0,                       # max_capacity (not used as constraint)
            0.0,                           # min_stable_gen (not used)
            1.0,                           # ramp_rate (not used)
            1.0,                           # efficiency (not used)
            0.0,                           # startup_cost (not used)
            total_solar_capacity           # existing_capacity
        )
        push!(generators, solar_gen)
    end
    
    # Process Hydro generators using PumpedHydro_NY from storage_df for costs
    # Find pumped hydro in storage data
    pumped_hydro = filter(row -> occursin("PumpedHydro", row.Resource), storage_df)
    regular_hydro = nrow(hydro_df) > 0 ? hydro_df[1, :] : nothing
    
    if nrow(pumped_hydro) > 0
        pumped_row = pumped_hydro[1, :]  # Use PumpedHydro_NY
        existing_capacity = regular_hydro !== nothing ? regular_hydro.Existing_Cap_MW : pumped_row.Existing_Cap_MW
        
        # Use pumped hydro investment cost if available, otherwise default
        hydro_inv_cost = pumped_row.Inv_Cost_per_MWyr > 0 ? pumped_row.Inv_Cost_per_MWyr : 200000.0
        
        hydro_gen = Generator(
            "Hydro",                        # name
            0.0,                            # fuel_cost
            pumped_row.Var_OM_Cost_per_MWh, # var_om_cost (from pumped hydro)
            hydro_inv_cost,                 # inv_cost (from pumped hydro)
            pumped_row.Fixed_OM_Cost_per_MWyr, # fixed_om_cost (from pumped hydro)
            50000.0,                       # max_capacity (not used as constraint)
            0.0,                           # min_stable_gen (not used)
            1.0,                           # ramp_rate (not used)
            1.0,                           # efficiency (not used)
            0.0,                           # startup_cost (not used)
            existing_capacity              # existing_capacity (use regular hydro capacity if available)
        )
        push!(generators, hydro_gen)
    elseif regular_hydro !== nothing
        # Fallback to regular hydro if no pumped hydro found
        hydro_inv_cost = regular_hydro.Inv_Cost_per_MWyr > 0 ? regular_hydro.Inv_Cost_per_MWyr : 200000.0
        
        hydro_gen = Generator(
            "Hydro",                        # name
            0.0,                            # fuel_cost
            regular_hydro.Var_OM_Cost_per_MWh, # var_om_cost
            hydro_inv_cost,                 # inv_cost
            regular_hydro.Fixed_OM_Cost_per_MWyr, # fixed_om_cost
            50000.0,                       # max_capacity (not used as constraint)
            0.0,                           # min_stable_gen (not used)
            1.0,                           # ramp_rate (not used)
            1.0,                           # efficiency (not used)
            0.0,                           # startup_cost (not used)
            regular_hydro.Existing_Cap_MW   # existing_capacity
        )
        push!(generators, hydro_gen)
    end
    
    return generators
end

"""
    process_nyiso_storage(storage_df::DataFrame)

Process NYISO storage units and return Battery struct.
For simplicity, aggregate all battery storage into single representative unit.
"""
function process_nyiso_storage(storage_df::DataFrame, Battery)
    # Find battery storage rows (existing and new)
    battery_existing = filter(row -> occursin("Battery", row.Resource) && !occursin("_New", row.Resource), storage_df)
    battery_new = filter(row -> occursin("Battery", row.Resource) && occursin("_New", row.Resource), storage_df)
    
    # Use new battery for costs, existing battery for capacity reference
    existing_power_cap = 0.0
    
    # Get existing battery capacity if available
    if nrow(battery_existing) > 0
        existing_row = battery_existing[1, :]
        existing_power_cap = existing_row.Existing_Cap_MW
    end
    
    # Prefer new battery for costs if available, otherwise use existing, otherwise default
    if nrow(battery_new) > 0
        cost_row = battery_new[1, :]
        power_inv_cost = cost_row.Inv_Cost_per_MWyr
        energy_inv_cost = cost_row.Inv_Cost_per_MWhyr
    elseif nrow(battery_existing) > 0
        cost_row = battery_existing[1, :]
        # Use defaults if existing has zero costs
        power_inv_cost = cost_row.Inv_Cost_per_MWyr > 0 ? cost_row.Inv_Cost_per_MWyr : 100000.0
        energy_inv_cost = cost_row.Inv_Cost_per_MWhyr > 0 ? cost_row.Inv_Cost_per_MWhyr : 20000.0
    else
        # Fallback: create default battery
        return Battery(
            "Battery_Default",
            100000.0,  # inv_cost_power ($/MW/year)
            20000.0,   # inv_cost_energy ($/MWh/year)
            5000.0,    # fixed_om_cost
            1.0,       # var_om_cost
            10000.0,   # max_power_capacity
            40000.0,   # max_energy_capacity (4-hour duration)
            0.85,      # efficiency_charge
            0.85,      # efficiency_discharge
            4.0,       # duration
            0.0        # existing_power_capacity
        )
    end
    
    # Enforce energy cost = power cost / duration relationship
    corrected_energy_inv_cost = power_inv_cost / cost_row.Max_Duration
    
    battery = Battery(
        cost_row.Resource,
        power_inv_cost,                   # inv_cost_power
        corrected_energy_inv_cost,        # inv_cost_energy (corrected)
        cost_row.Fixed_OM_Cost_per_MWyr,  # fixed_om_cost
        cost_row.Var_OM_Cost_per_MWh,     # var_om_cost
        10000.0,                          # max_power_capacity (set reasonable upper limit)
        cost_row.Max_Duration * 10000.0,  # max_energy_capacity (duration * max power)
        cost_row.Eff_Up,                  # efficiency_charge
        cost_row.Eff_Down,                # efficiency_discharge
        cost_row.Max_Duration,            # duration
        existing_power_cap                # existing_power_capacity
    )
    
    return battery
end

"""
    create_nyiso_system_profiles(demand_df::DataFrame, variability_df::DataFrame, 
                                 generators::Vector{Generator}, params::SystemParameters)

Create SystemProfiles struct from NYISO demand and variability data.
Note: Current SystemProfiles struct expects wind as the renewable profile - we'll use wind but could extend later for solar.
"""
function create_nyiso_system_profiles(demand_df::DataFrame, variability_df::DataFrame, 
                                     generators, params, SystemProfiles)
    
    # Extract demand profile (use first zone for simplicity)
    demand_column = names(demand_df)[end]  # Last column should be demand
    actual_demand = Vector{Float64}(demand_df[1:params.hours, demand_column])
    
    # Fill zero values by linear interpolation
    actual_demand = interpolate_zero_values(actual_demand)
    
    # Create generator-specific availability profiles
    G = length(generators)
    generator_availabilities = Vector{Vector{Float64}}(undef, G)
    
    # Map each generator to its corresponding availability column in NYISO data
    for g in 1:G
        gen = generators[g]
        availability_profile = nothing
        
        if gen.name == "Nuclear"
            # Find nuclear column
            nuclear_cols = filter(col -> occursin("Nuclear", col) && occursin("Existing", col), names(variability_df))
            if !isempty(nuclear_cols)
                availability_profile = Vector{Float64}(variability_df[1:params.hours, nuclear_cols[1]])
            end
        elseif gen.name == "CC"
            # Find Combined Cycle column
            cc_cols = filter(col -> occursin("CombinedCycle", col) && occursin("NaturalGas", col) && occursin("Existing", col), names(variability_df))
            if !isempty(cc_cols)
                availability_profile = Vector{Float64}(variability_df[1:params.hours, cc_cols[1]])
            end
        elseif gen.name == "CT"
            # Find Combustion Turbine column
            ct_cols = filter(col -> occursin("CombustionTurbine", col) && occursin("NaturalGas", col) && occursin("Existing", col), names(variability_df))
            if !isempty(ct_cols)
                availability_profile = Vector{Float64}(variability_df[1:params.hours, ct_cols[1]])
            end
        elseif gen.name == "ST"
            # Find Steam Turbine column
            st_cols = filter(col -> occursin("SteamTurbine", col) && occursin("NaturalGas", col) && occursin("Existing", col), names(variability_df))
            if !isempty(st_cols)
                availability_profile = Vector{Float64}(variability_df[1:params.hours, st_cols[1]])
            end
        elseif gen.name == "Wind"
            # Find Wind column
            wind_cols = filter(col -> occursin("Wind", col) && occursin("Existing", col), names(variability_df))
            if !isempty(wind_cols)
                availability_profile = Vector{Float64}(variability_df[1:params.hours, wind_cols[1]])
            end
        elseif gen.name == "Solar"
            # Find Solar column
            solar_cols = filter(col -> occursin("Solar", col) && occursin("Existing", col), names(variability_df))
            if !isempty(solar_cols)
                availability_profile = Vector{Float64}(variability_df[1:params.hours, solar_cols[1]])
            end
        elseif gen.name == "Hydro"
            # Find Hydro column
            hydro_cols = filter(col -> occursin("Hydro", col), names(variability_df))
            if !isempty(hydro_cols)
                availability_profile = Vector{Float64}(variability_df[1:params.hours, hydro_cols[1]])
            end
        end
        
        # Set availability profile or default to full availability
        if availability_profile !== nothing
            generator_availabilities[g] = availability_profile
        else
            println("Warning: No availability data found for generator $(gen.name), using 100% availability")
            generator_availabilities[g] = fill(1.0, params.hours)
        end
    end
    
    # Create scenario forecasts (indexed by scenario ω, then by generator g)
    n_scenarios = 5
    demand_scenarios = Vector{Vector{Float64}}()
    generator_availability_scenarios = Vector{Vector{Vector{Float64}}}(undef, n_scenarios)
    
    for ω in 1:n_scenarios
        # Initialize generator availability scenarios for this scenario
        generator_availability_scenarios[ω] = Vector{Vector{Float64}}(undef, G)
        
        # Create demand scenario with small random variation
        push!(demand_scenarios, actual_demand .* (1.0 .+ 0.05 * randn(length(actual_demand))))
        
        # Create generator-specific availability scenarios
        for g in 1:G
            base_availability = generator_availabilities[g]
            # Add small noise based on generator type
            noise_level = generators[g].name in ["Wind", "Solar"] ? 0.1 : 0.02
            scenario_availability = clamp.(base_availability .+ noise_level * randn(length(base_availability)), 0.0, 1.0)
            generator_availability_scenarios[ω][g] = scenario_availability
        end
    end
    
    return SystemProfiles(
        actual_demand,
        generator_availabilities,
        demand_scenarios,
        generator_availability_scenarios,
        n_scenarios,
        params
    )
end

end # module NYISODataLoader