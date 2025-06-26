"""
ProfileGeneration.jl

Profile generation module for ToySystemQuad.jl
Handles demand, wind, and outage profile generation with scenarios.
"""

module ProfileGeneration

using Random, Statistics
using ..SystemConfig: SystemParameters, get_default_system_parameters

export get_base_demand_profile, get_base_wind_profile
export generate_demand_profile, generate_wind_profile
export generate_nuclear_availability, generate_gas_availability
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
    generate_nuclear_availability(params::SystemParameters)

Generate nuclear availability profile including planned and forced outages.
"""
function generate_nuclear_availability(params::SystemParameters)
    Random.seed!(params.random_seed)
    nuclear_availability = Float64[]
    
    # Planned outage periods (maintenance every ~45 days for 3-5 days)
    planned_outage_periods = [12:15, 58:62, 104:107]
    
    for day in 1:params.days
        nuclear_daily_avail = Float64[]
        
        for hour in 1:24
            hour_of_year = (day-1)*24 + hour
            
            # Check for planned maintenance
            planned_outage = any(day in period for period in planned_outage_periods)
            
            # Random forced outages (0.2% per hour probability)
            Random.seed!(hour_of_year + 1000)  # Deterministic but pseudo-random
            forced_outage = rand() < 0.002
            
            # Handle forced outage duration
            if forced_outage && hour <= 18  # Don't start outages too late
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
    generate_gas_availability(params::SystemParameters)

Generate gas availability profile with frequent but shorter outages.
"""
function generate_gas_availability(params::SystemParameters)
    Random.seed!(params.random_seed)
    gas_availability = Float64[]
    
    for day in 1:params.days
        gas_daily_avail = Float64[]
        
        hour = 1
        while hour <= 24
            hour_of_year = (day-1)*24 + hour
            
            # Gas units have more frequent but shorter outages (2% per hour)
            Random.seed!(hour_of_year + 2000)
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
# SCENARIO GENERATION
# =============================================================================

"""
    generate_scenarios(actual_demand, actual_wind, nuclear_availability, gas_availability, 
                      params::SystemParameters; n_scenarios=3)

Generate stochastic scenarios around the actual profiles for DLAC-i operations.
"""
function generate_scenarios(actual_demand, actual_wind, nuclear_availability, gas_availability, 
                           params::SystemParameters; n_scenarios=3)
    Random.seed!(params.random_seed)
    
    demand_scenarios = []
    wind_scenarios = []
    nuclear_availability_scenarios = []
    gas_availability_scenarios = []
    
    # Scenario factors for systematic variations
    demand_factors = n_scenarios == 3 ? [0.85, 1.0, 1.15] : range(0.8, 1.2, length=n_scenarios)
    wind_factors = n_scenarios == 3 ? [1.2, 1.0, 0.8] : range(1.3, 0.7, length=n_scenarios)
    
    for scenario in 1:n_scenarios
        scenario_demand = Float64[]
        scenario_wind = Float64[]
        scenario_nuclear_avail = Float64[]
        scenario_gas_avail = Float64[]
        
        for t in 1:params.hours
            # Demand scenarios with systematic bias and noise
            demand_factor = demand_factors[scenario]
            demand_noise = 1.0 + 0.06 * randn()
            demand_value = max(actual_demand[t] * demand_factor * demand_noise, 
                              0.3 * actual_demand[t])
            
            # Wind scenarios with systematic bias and noise
            wind_factor = wind_factors[scenario]
            wind_noise = 1.0 + 0.12 * randn()
            wind_value = max(min(actual_wind[t] * wind_factor * wind_noise, 0.98), 0.02)
            
            # Outage scenarios with different patterns per scenario
            Random.seed!(t + scenario * 10000)
            
            # Nuclear outages vary by scenario (more outages in higher scenarios)
            nuclear_factor = nuclear_availability[t]
            if nuclear_availability[t] > 0 && rand() < 0.002 * scenario
                nuclear_factor = 0.0
            end
            
            # Gas outages vary by scenario
            gas_factor = gas_availability[t]
            if gas_availability[t] > 0 && rand() < 0.003 * scenario
                gas_factor = max(0.3, gas_availability[t] - 0.4)
            end
            
            push!(scenario_demand, demand_value)
            push!(scenario_wind, wind_value)
            push!(scenario_nuclear_avail, nuclear_factor)
            push!(scenario_gas_avail, gas_factor)
        end
        
        push!(demand_scenarios, scenario_demand)
        push!(wind_scenarios, scenario_wind)
        push!(nuclear_availability_scenarios, scenario_nuclear_avail)
        push!(gas_availability_scenarios, scenario_gas_avail)
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
        # Import get_default_system_parameters from SystemConfig
        params = get_default_system_parameters()
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
    @assert all(nuclear_availability .∈ Ref([0.0, 1.0])) "Nuclear availability must be 0 or 1"
    @assert all(gas_availability .∈ Ref([0.0, 1.0])) "Gas availability must be 0 or 1"
    
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