"""
ProfileGeneration.jl

Profile generation module for ToySystemQuad.jl
Handles demand, wind, and outage profile generation with scenarios.
"""

module ProfileGeneration

using Random, Statistics

# Define SystemParameters locally to avoid circular dependency
struct SystemParameters
    hours::Int              # Total simulation hours
    days::Int 
    N::Int              # Number of days
    random_seed::Int        # For reproducibility
    load_shed_penalty::Float64  # $/MWh penalty for unserved energy
    load_shed_quad::Float64     # Quadratic load shed penalty coefficient
end

export get_base_demand_profile, get_base_wind_profile
export generate_demand_profile, generate_wind_profile, generate_wind_forecast
export generate_nuclear_availability, generate_gas_availability
export generate_fleet_availability, generate_single_nuclear_availability, generate_single_gas_availability
export generate_scenarios, create_actual_and_scenarios
export validate_profiles

# =============================================================================
# BASE PROFILE DEFINITIONS
# =============================================================================

"""
    get_base_demand_profile()

Returns the base 24-hour demand profile in MW.
Profile designed with distinct periods for different technologies.
"""
function get_base_demand_profile()
    return [
        # Hours 1-6: Night valley (baseload territory) 
        500, 480, 460, 450, 470, 510,
        # Hours 7-12: Morning ramp (nuclear + wind)
        580, 720, 850, 920, 980, 1020,
        # Hours 13-18: Afternoon (mixed resources)
        1050, 1080, 1120, 1180, 1220, 1250,
        # Hours 19-24: Evening peak then decline (gas + battery discharge)
        1400, 1350, 1200, 1000, 800, 650
    ]
end

"""
    get_base_wind_profile()

Returns the base 24-hour wind capacity factor profile.
Profile designed to be anticorrelated with demand.
"""
function get_base_wind_profile()
    return [
        # Hours 1-6: High wind at night (good for battery charging)
        0.80, 0.85, 0.90, 0.88, 0.82, 0.75,
        # Hours 7-12: Moderate morning wind  
        0.65, 0.50, 0.35, 0.25, 0.20, 0.18,
        # Hours 13-18: Low afternoon wind (gas/battery needed)
        0.15, 0.12, 0.10, 0.15, 0.25, 0.40,
        # Hours 19-24: Evening wind pickup
        0.55, 0.70, 0.75, 0.78, 0.80, 0.82
    ]
end

# =============================================================================
# PROFILE GENERATION FUNCTIONS
# =============================================================================

"""
    generate_demand_profile(params::SystemParameters)

Generate demand profile over the full time horizon with seasonal and weekend effects.
"""
function generate_demand_profile(params::SystemParameters)
    Random.seed!(params.random_seed)
    base_demand = get_base_demand_profile()
    actual_demand = Float64[]
    
    for day in 1:params.days
        # Weekend effect (favor baseload economics)
        is_weekend = (day % 7 in [6, 0])
        weekend_factor = is_weekend ? 0.7 : 1.0
        
        # Seasonal variation (winter peak, summer valley)
        seasonal_factor = 1.0 + 0.25 * cos(2π * day / 365) + 0.1 * sin(4π * day / 365)
        
        # Day-to-day variation
        daily_variation = 1.0 + 0.05 * sin(2π * day / 7)
        
        daily_demand = base_demand .* weekend_factor .* seasonal_factor .* daily_variation
        append!(actual_demand, daily_demand)
    end
    
    return actual_demand
end

"""
    generate_wind_profile(params::SystemParameters)

Generate wind capacity factor profile over the full time horizon with weather patterns.
"""
function generate_wind_profile(params::SystemParameters)
    Random.seed!(params.random_seed)
    base_wind = get_base_wind_profile()
    actual_wind = Float64[]
    
    for day in 1:params.days
        # Wind weather patterns (5-day cycles)
        wind_weather_cycle = 0.3 + 0.9 * (sin(2π * day / 5) + 1) / 2
        
        # Seasonal wind patterns
        seasonal_wind = 1.0 + 0.3 * sin(2π * day / 120)
        
        daily_wind = [max(min(cf * wind_weather_cycle * seasonal_wind, 0.95), 0.05) 
                     for cf in base_wind]
        append!(actual_wind, daily_wind)
    end
    
    return actual_wind
end

"""
    generate_wind_forecast(actual_wind, params::SystemParameters, scenario_id::Int=1)

Generate wind forecast profile with realistic forecast error patterns.
Includes bias, temporal errors, persistence effects, and diurnal accuracy variations.
"""
function generate_wind_forecast(actual_wind, params::SystemParameters, scenario_id::Int=1)
    Random.seed!(params.random_seed + scenario_id * 5000)
    forecast_wind = Float64[]
    
    for t in 1:params.hours
        hour_of_day = ((t-1) % 24) + 1
        day_of_year = div(t-1, 24) + 1
        
        # Base forecast starts with actual
        base_forecast = actual_wind[t]
        
        # Forecast bias - systematic tendency to over/under-predict
        # More bias in extreme conditions
        if base_forecast > 0.7
            bias_factor = 0.95 + 0.1 * (scenario_id - 1) / 4  # High wind under-prediction bias
        elseif base_forecast < 0.3
            bias_factor = 1.05 - 0.1 * (scenario_id - 1) / 4  # Low wind over-prediction bias
        else
            bias_factor = 1.0
        end
        
        # Temporal/ramp forecast errors - worse during transitions
        prev_wind = t > 1 ? actual_wind[t-1] : actual_wind[t]
        next_wind = t < params.hours ? actual_wind[t+1] : actual_wind[t]
        ramp_magnitude = abs(next_wind - prev_wind)
        ramp_error_factor = 1.0 + 0.15 * ramp_magnitude * randn()
        
        # Diurnal accuracy patterns - worse during certain hours
        diurnal_accuracy = if hour_of_day in 6:9 || hour_of_day in 17:20
            0.85  # Worse during transition periods
        else
            0.95  # Better during stable periods
        end
        
        # Persistence bias - forecasts are "stickier" than reality
        if t > 1
            persistence_factor = 0.1
            forecast_wind_raw = (1 - persistence_factor) * base_forecast + 
                               persistence_factor * forecast_wind[t-1]
        else
            forecast_wind_raw = base_forecast
        end
        
        # Weather pattern forecast skill variation
        weather_cycle_day = day_of_year % 7
        weather_skill = if weather_cycle_day in [2, 5]  # Worse skill on certain weather patterns
            0.8
        else
            0.9
        end
        
        # Combine all error sources
        forecast_error = (bias_factor * ramp_error_factor - 1.0) * weather_skill * diurnal_accuracy
        additive_noise = 0.08 * randn()  # Random noise
        
        forecast_value = forecast_wind_raw + forecast_error + additive_noise
        
        # Constrain to [0, 1] range
        forecast_value = max(min(forecast_value, 0.98), 0.02)
        
        push!(forecast_wind, forecast_value)
    end
    
    return forecast_wind
end

"""
    generate_nuclear_availability(params::SystemParameters)

Generate nuclear availability profile for a 2-generator fleet.
Returns the mean availability across the fleet.
"""
function generate_nuclear_availability(params::SystemParameters)
    fleet_mean, _ = generate_fleet_availability(params, :nuclear)
    return fleet_mean
end

"""
    generate_gas_availability(params::SystemParameters)

Generate gas availability profile for a 2-generator fleet.
Returns the mean availability across the fleet.
"""
function generate_gas_availability(params::SystemParameters)
    fleet_mean, _ = generate_fleet_availability(params, :gas)
    return fleet_mean
end

"""
    ensure_24_hours(profile)

Utility function to ensure a daily profile has exactly 24 hours.
"""
function ensure_24_hours(profile)
    if length(profile) < 24
        # Pad with last value
        while length(profile) < 24
            push!(profile, profile[end])
        end
    elseif length(profile) > 24
        # Trim to 24 hours
        profile = profile[1:24]
    end
    return profile
end

# =============================================================================
# FLEET GENERATION HELPER FUNCTIONS
# =============================================================================

"""
    generate_single_nuclear_availability(params::SystemParameters, generator_id::Int)

Generate nuclear availability profile for a single generator with independent outage patterns.
"""
function generate_single_nuclear_availability(params::SystemParameters, generator_id::Int)
    Random.seed!(params.random_seed + generator_id * 1000)
    nuclear_availability = Float64[]
    
    # Planned outage periods vary by generator (staggered maintenance)
    base_outage_days = [12:15, 58:62, 104:107]
    planned_outage_periods = [period .+ (generator_id - 1) * 7 for period in base_outage_days]
    
    for day in 1:params.days
        nuclear_daily_avail = Float64[]
        
        for hour in 1:24
            hour_of_year = (day-1)*24 + hour
            
            # Check for planned maintenance
            planned_outage = any(day in period for period in planned_outage_periods)
            
            # Random forced outages (0.2% per hour probability per generator)
            Random.seed!(hour_of_year + generator_id * 10000)
            forced_outage = rand() < 0.002
            
            # Handle forced outage duration
            if forced_outage && hour <= 18
                outage_duration = 6 + rand(1:18)  # 6-24 hours
                for out_hour in hour:min(24, hour + outage_duration - 1)
                    if out_hour <= 24
                        push!(nuclear_daily_avail, 0.0)
                    end
                end
                hour = min(24, hour + outage_duration)
                continue
            end
            
            availability = (planned_outage || forced_outage) ? 0.0 : 1.0
            push!(nuclear_daily_avail, availability)
        end
        
        # Ensure exactly 24 hours
        nuclear_daily_avail = ensure_24_hours(nuclear_daily_avail)
        append!(nuclear_availability, nuclear_daily_avail)
    end
    
    return nuclear_availability
end

"""
    generate_single_gas_availability(params::SystemParameters, generator_id::Int)

Generate gas availability profile for a single generator with independent outage patterns.
"""
function generate_single_gas_availability(params::SystemParameters, generator_id::Int)
    Random.seed!(params.random_seed + generator_id * 2000)
    gas_availability = Float64[]
    
    for day in 1:params.days
        gas_daily_avail = Float64[]
        
        hour = 1
        while hour <= 24
            hour_of_year = (day-1)*24 + hour
            
            # Gas units have more frequent but shorter outages (2% per hour per generator)
            Random.seed!(hour_of_year + generator_id * 20000)
            forced_outage = rand() < 0.02
            
            if forced_outage && hour <= 22
                outage_duration = 2 + rand(1:6)  # 2-8 hours typical for gas
                for out_hour in hour:min(24, hour + outage_duration - 1)
                    if out_hour <= 24
                        push!(gas_daily_avail, 0.0)
                    end
                end
                hour = min(24, hour + outage_duration)
            else
                push!(gas_daily_avail, 1.0)
                hour += 1
            end
        end
        
        # Ensure exactly 24 hours
        gas_daily_avail = ensure_24_hours(gas_daily_avail)
        append!(gas_availability, gas_daily_avail)
    end
    
    return gas_availability
end

"""
    generate_fleet_availability(params::SystemParameters, unit_type::Symbol)

Generate availability profiles for a fleet of generators and return the mean availability.
Returns both individual generator paths and the fleet mean.
"""
function generate_fleet_availability(params::SystemParameters, unit_type::Symbol)
    n_generators = params.N  #
    generator_paths = []
    
    for gen_id in 1:n_generators
        if unit_type == :nuclear
            path = generate_single_nuclear_availability(params, gen_id)
        elseif unit_type == :gas
            path = generate_single_gas_availability(params, gen_id)
        else
            error("Unknown unit type: $unit_type")
        end
        push!(generator_paths, path)
    end
    
    # Calculate fleet mean availability
    fleet_mean = [mean([generator_paths[g][t] for g in 1:n_generators]) for t in 1:params.hours]
    
    return fleet_mean, generator_paths
end

# =============================================================================
# SCENARIO GENERATION
# =============================================================================

"""
    generate_scenarios(actual_demand, actual_wind, nuclear_availability, gas_availability, 
                      params::SystemParameters; n_scenarios=5)

Generate 5 stochastic scenarios with fleet-based thermal generation for DLAC-i operations.
Each scenario uses independent fleet paths and realistic wind forecasts.
"""
function generate_scenarios(actual_demand, actual_wind, nuclear_availability, gas_availability, 
                           params::SystemParameters; n_scenarios=5)
    Random.seed!(params.random_seed)
    
    demand_scenarios = []
    wind_scenarios = []
    nuclear_availability_scenarios = []
    gas_availability_scenarios = []
    
    # Demand scenario factors (5 scenarios with diverse range)
    demand_factors = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for scenario in 1:n_scenarios
        # Demand scenarios with systematic bias and noise
        scenario_demand = Float64[]
        for t in 1:params.hours
            demand_factor = demand_factors[scenario]
            demand_noise = 1.0 + 0.06 * randn()
            demand_value = max(actual_demand[t] * demand_factor * demand_noise, 
                              0.3 * actual_demand[t])
            push!(scenario_demand, demand_value)
        end
        
        # Wind scenarios using forecast error patterns
        scenario_wind = generate_wind_forecast(actual_wind, params, scenario)
        
        # Nuclear availability scenarios - generate new fleet with different seed
        nuclear_scenario_params = SystemParameters(
            params.hours, params.days, 
            params.N,
            params.random_seed + scenario * 100000,
            params.load_shed_penalty, params.load_shed_quad
        )
        nuclear_fleet_mean, _ = generate_fleet_availability(nuclear_scenario_params, :nuclear)
        
        # Gas availability scenarios - generate new fleet with different seed  
        gas_scenario_params = SystemParameters(
            params.hours, params.days,
            params.N,
            params.random_seed + scenario * 200000, 
            params.load_shed_penalty, params.load_shed_quad
        )
        gas_fleet_mean, _ = generate_fleet_availability(gas_scenario_params, :gas)
        
        push!(demand_scenarios, scenario_demand)
        push!(wind_scenarios, scenario_wind)
        push!(nuclear_availability_scenarios, nuclear_fleet_mean)
        push!(gas_availability_scenarios, gas_fleet_mean)
    end
    
    return demand_scenarios, wind_scenarios, nuclear_availability_scenarios, gas_availability_scenarios
end

"""
    create_actual_and_scenarios(params=nothing)

Main function to create all actual profiles and scenarios. 
Maintains compatibility with existing code.
"""
function create_actual_and_scenarios(params=nothing)
    if params === nothing
        # Default system parameters
        params = SystemParameters(
            720,    # hours (30 days)
            30,     # days
            42,     # random_seed
            10000.0, # load_shed_penalty
            0.001   # load_shed_quad
        )
    end
    
    # Generate actual profiles
    actual_demand = generate_demand_profile(params)
    actual_wind = generate_wind_profile(params)
    nuclear_availability = generate_nuclear_availability(params)
    gas_availability = generate_gas_availability(params)
    
    # Generate scenarios
    demand_scenarios, wind_scenarios, nuclear_avail_scenarios, gas_avail_scenarios = 
        generate_scenarios(actual_demand, actual_wind, nuclear_availability, gas_availability, params)
    
    return actual_demand, actual_wind, nuclear_availability, gas_availability,
           demand_scenarios, wind_scenarios, nuclear_avail_scenarios, gas_avail_scenarios
end

# =============================================================================
# VALIDATION
# =============================================================================

"""
    validate_profiles(actual_demand, actual_wind, nuclear_availability, gas_availability, params)

Validate that generated profiles are reasonable and consistent.
"""
function validate_profiles(actual_demand, actual_wind, nuclear_availability, gas_availability, params)
    @assert length(actual_demand) == params.hours "Demand profile length mismatch"
    @assert length(actual_wind) == params.hours "Wind profile length mismatch"
    @assert length(nuclear_availability) == params.hours "Nuclear availability length mismatch"
    @assert length(gas_availability) == params.hours "Gas availability length mismatch"
    
    @assert all(actual_demand .> 0) "Demand must be positive"
    @assert all(0 .<= actual_wind .<= 1) "Wind capacity factors must be in [0,1]"
    @assert all(0 .<= nuclear_availability .<= 1) "Nuclear availability must be in [0,1]"
    @assert all(0 .<= gas_availability .<= 1) "Gas availability must be in [0,1]"
    
    # Check availability statistics
    nuclear_avail_pct = mean(nuclear_availability) * 100
    gas_avail_pct = mean(gas_availability) * 100

    println("✓ Profile validation passed:")
    println("  - Nuclear availability: $(round(nuclear_avail_pct, digits=1))%")
    println("  - Gas availability: $(round(gas_avail_pct, digits=1))%")
    println("  - Mean demand: $(round(mean(actual_demand), digits=1)) MW")
    println("  - Mean wind CF: $(round(mean(actual_wind), digits=1))")
    
    return true
end

end # module ProfileGeneration