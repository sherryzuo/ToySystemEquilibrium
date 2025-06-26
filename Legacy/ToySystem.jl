using JuMP, Gurobi, LinearAlgebra, Plots, Statistics, Random, CSV, DataFrames
using Printf
# =============================================================================
# 1. SYSTEM DATA DEFINITION
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
Random.seed!(42)  # For reproducibility
# Define the 4-technology system: Nuclear, Battery, Wind, Gas
function create_toy_system()
    # 3 generators: Nuclear (baseload), Wind (renewable), Gas (peaker)
    generators = [
        # Nuclear: Low fuel cost, high investment, baseload operation
        Generator("Nuclear", 12.0, 2.0, 120000.0, 35000.0, 1200.0, 0.9, 0.1, 0.33, 2000.0),
        
        # Wind: Zero fuel cost, moderate investment, variable output
        Generator("Wind", 0.0, 3.0, 85000.0, 22000.0, 1500.0, 0.0, 1.0, 1.0, 0.0),
        
        # Gas: High fuel cost, low investment, flexible peaker
        Generator("Gas", 90.0, 4.0, 55000.0, 12000.0, 1000.0, 0.2, 1.0, 0.45, 80.0)
    ]
    
    # Battery storage: Medium investment, provides flexibility and arbitrage
    battery = Battery("Battery", 95000.0, 100.0, 6000.0, 1.5, 800.0, 3200.0, 0.90, 0.90, 4.0)
    
    return generators, battery
end

# Create demand and wind profiles with thermal outages that encourage diverse investment
function create_actual_and_scenarios()
    hours = 720  # 30 days
    days = 30
    
    Random.seed!(42)  # For reproducible results
    
    # Realistic demand profile with strong daily cycles and seasonal variation
    base_daily_demand = [
        # Hours 1-6: Night valley (baseload territory) 
        500, 480, 460, 450, 470, 510,
        # Hours 7-12: Morning ramp (nuclear + wind)
        580, 720, 850, 920, 980, 1020,
        # Hours 13-18: Afternoon (mixed resources)
        1050, 1080, 1120, 1180, 1220, 1250,
        # Hours 19-24: Evening peak then decline (gas + battery discharge)
        1400, 1350, 1200, 1000, 800, 650
    ]
    
    # Create ACTUAL deterministic profiles
    actual_demand = Float64[]
    actual_wind = Float64[]
    nuclear_availability = Float64[]
    gas_availability = Float64[]
    
    for day in 1:days
        # Strong weekend effect (favor baseload economics)
        is_weekend = (day % 7 in [6, 0])
        weekend_factor = is_weekend ? 0.7 : 1.0
        
        # Seasonal variation (winter peak, summer valley)
        seasonal_factor = 1.0 + 0.25 * cos(2π * day / 365) + 0.1 * sin(4π * day / 365)
        
        # Add some day-to-day variation
        daily_variation = 1.0 + 0.05 * sin(2π * day / 7)
        
        daily_demand = base_daily_demand .* weekend_factor .* seasonal_factor .* daily_variation
        append!(actual_demand, daily_demand)
        
        # Wind profile designed to complement other resources
        base_wind_pattern = [
            # Hours 1-6: High wind at night (good for battery charging)
            0.80, 0.85, 0.90, 0.88, 0.82, 0.75,
            # Hours 7-12: Moderate morning wind  
            0.65, 0.50, 0.35, 0.25, 0.20, 0.18,
            # Hours 13-18: Low afternoon wind (gas/battery needed)
            0.15, 0.12, 0.10, 0.15, 0.25, 0.40,
            # Hours 19-24: Evening wind pickup
            0.55, 0.70, 0.75, 0.78, 0.80, 0.82
        ]
        
        # Wind weather patterns
        wind_weather_cycle = 0.3 + 0.9 * (sin(2π * day / 5) + 1) / 2
        seasonal_wind = 1.0 + 0.3 * sin(2π * day / 120)
        
        daily_wind = [max(min(cf * wind_weather_cycle * seasonal_wind, 0.95), 0.05) 
                     for cf in base_wind_pattern]
        append!(actual_wind, daily_wind)
        
        # NUCLEAR OUTAGES: Planned and unplanned
        nuclear_daily_avail = Float64[]
        for hour in 1:24
            hour_of_year = (day-1)*24 + hour
            
            # Planned maintenance outages (every ~45 days for 3-5 days)
            planned_outage = false
            if day in [12:15; 58:62; 104:107]  # 3 planned outage periods
                planned_outage = true
            end
            
            # Random forced outages (2% hourly probability, last 6-24 hours)
            Random.seed!(hour_of_year + 1000)  # Deterministic but pseudo-random
            forced_outage = rand() < 0.002  # 0.2% per hour = ~1.44 forced outages/month
            
            # If forced outage starts, it lasts 6-24 hours
            if forced_outage && hour <= 18  # Don't start outages too late in day
                outage_duration = 6 + rand(1:18)  # 6-24 hours
                for out_hour in hour:min(24, hour + outage_duration - 1)
                    if out_hour <= 24
                        push!(nuclear_daily_avail, 0.0)
                    end
                end
                hour = min(24, hour + outage_duration)
                continue
            end
            
            if planned_outage
                push!(nuclear_daily_avail, 0.0)
            else
                push!(nuclear_daily_avail, 1.0)  # 100% available when not in outage
            end
        end
        
        # Pad to 24 hours if needed
        while length(nuclear_daily_avail) < 24
            push!(nuclear_daily_avail, nuclear_daily_avail[end])
        end
        nuclear_daily_avail = nuclear_daily_avail[1:24]  # Trim if over 24
        
        append!(nuclear_availability, nuclear_daily_avail)
        
        # GAS OUTAGES: More frequent but shorter
        gas_daily_avail = Float64[]
        for hour in 1:24
            hour_of_year = (day-1)*24 + hour
            
            # Gas units have more frequent but shorter outages
            Random.seed!(hour_of_year + 2000)
            forced_outage = rand() < 0.02  # 2% per hour = ~14.4 forced outages/month
            
            if forced_outage && hour <= 22
                outage_duration = 2 + rand(1:10)  # 2-8 hours typical for gas
                for out_hour in hour:min(24, hour + outage_duration - 1)
                    if out_hour <= 24
                        push!(gas_daily_avail, 0)  
                    end
                end
                hour = min(24, hour + outage_duration)
                continue
            end
            
            push!(gas_daily_avail, 1.0)
        end
        
        # Pad/trim to 24 hours
        while length(gas_daily_avail) < 24
            push!(gas_daily_avail, gas_daily_avail[end])
        end
        gas_daily_avail = gas_daily_avail[1:24]
        
        append!(gas_availability, gas_daily_avail)
    end
    
    # Create 3 scenarios with actual as the mean
    demand_scenarios = []
    wind_scenarios = []
    nuclear_availability_scenarios = []
    gas_availability_scenarios = []
    
    for scenario in 1:3
        scenario_demand = Float64[]
        scenario_wind = Float64[]
        scenario_nuclear_avail = Float64[]
        scenario_gas_avail = Float64[]
        
        for t in 1:hours
            # Demand scenarios
            demand_scenario_factor = [0.85, 1.0, 1.15][scenario]
            demand_noise = 1.0 + 0.06 * randn()
            
            # Wind scenarios
            wind_scenario_factor = [1.2, 1.0, 0.8][scenario]
            wind_noise = 1.0 + 0.12 * randn()
            
            # Outage scenarios - different outage patterns per scenario
            Random.seed!(t + scenario * 10000)
            
            # Nuclear outages vary by scenario (different maintenance schedules)
            nuclear_scenario_factor = nuclear_availability[t]
            if nuclear_availability[t] > 0 && rand() < 0.002 * scenario  # More outages in higher scenarios
                nuclear_scenario_factor = 0.0
            end
            
            # Gas outages vary by scenario
            gas_scenario_factor = gas_availability[t]
            if gas_availability[t] > 0 && rand() < 0.003 * scenario
                gas_scenario_factor = max(0.3, gas_availability[t] - 0.4)
            end
            
            push!(scenario_demand, max(actual_demand[t] * demand_scenario_factor * demand_noise, 
                                     0.3 * actual_demand[t]))
            push!(scenario_wind, max(min(actual_wind[t] * wind_scenario_factor * wind_noise, 0.98), 0.02))
            push!(scenario_nuclear_avail, nuclear_scenario_factor)
            push!(scenario_gas_avail, gas_scenario_factor)
        end
        
        push!(demand_scenarios, scenario_demand)
        push!(wind_scenarios, scenario_wind)
        push!(nuclear_availability_scenarios, scenario_nuclear_avail)
        push!(gas_availability_scenarios, scenario_gas_avail)
    end
    
    return actual_demand, actual_wind, nuclear_availability, gas_availability,
           demand_scenarios, wind_scenarios, nuclear_availability_scenarios, gas_availability_scenarios
end

# =============================================================================
# 2. CAPACITY EXPANSION MODEL (optimizes for actual deterministic)
# =============================================================================

function solve_capacity_expansion(generators, battery, actual_demand, actual_wind; output_dir="results")
    """
    Solve capacity expansion optimizing for the ACTUAL deterministic profiles over ALL hours
    Now includes thermal availability factors (outages)
    """
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    # Get profiles including thermal availability
    actual_demand, actual_wind, nuclear_availability, gas_availability, 
    demand_scenarios, wind_scenarios, nuclear_avail_scenarios, gas_avail_scenarios = create_actual_and_scenarios()
    
    T = length(actual_demand)
    G = length(generators)
    
    println("Capacity expansion using ALL $T hours with thermal outages")
    println("  Nuclear availability: $(round(mean(nuclear_availability)*100, digits=1))%")
    println("  Gas availability: $(round(mean(gas_availability)*100, digits=1))%")
    
    # Decision variables
    @variable(model, capacity[1:G] >= 0)  
    @variable(model, battery_power_cap >= 0)  
    @variable(model, battery_energy_cap >= 0)
    
    # Operational variables for ALL time periods
    @variable(model, generation[1:G, 1:T] >= 0)
    @variable(model, battery_charge[1:T] >= 0)
    @variable(model, battery_discharge[1:T] >= 0)
    @variable(model, battery_soc[1:T] >= 0)
    @variable(model, load_shed[1:T] >= 0)
    
    # Objective: Minimize annualized investment + operational costs
    investment_cost = sum(generators[g].inv_cost * capacity[g] for g in 1:G) + 
                     battery.inv_cost_power * battery_power_cap + 
                     battery.inv_cost_energy * battery_energy_cap
                     
    fixed_cost = sum(generators[g].fixed_om_cost * capacity[g] for g in 1:G) + 
                 battery.fixed_om_cost * battery_power_cap
                 
    operational_cost = sum(
        sum((generators[g].fuel_cost + generators[g].var_om_cost) * generation[g,t] for g in 1:G) +
        battery.var_om_cost * (battery_charge[t] + battery_discharge[t]) +
        10000 * load_shed[t]  # High penalty for unserved demand
        for t in 1:T)
    
    @objective(model, Min, investment_cost + fixed_cost + operational_cost)
    
    # Power balance for ALL hours
    @constraint(model, power_balance[t=1:T],
        sum(generation[g,t] for g in 1:G) + battery_discharge[t] - 
        battery_charge[t] + load_shed[t] == actual_demand[t])
    
    # Generation limits with availability factors
    for g in 1:G
        if generators[g].name == "Nuclear"
            @constraint(model, [t=1:T], 
                generation[g,t] <= capacity[g] * nuclear_availability[t])
        elseif generators[g].name == "Wind"
            @constraint(model, [t=1:T], 
                generation[g,t] <= capacity[g] * actual_wind[t])
        elseif generators[g].name == "Gas"
            @constraint(model, [t=1:T], 
                generation[g,t] <= capacity[g] * gas_availability[t])
        else
            # Default case
            @constraint(model, [t=1:T], 
                generation[g,t] <= capacity[g])
        end
    end
    
    # Battery constraints
    @constraint(model, [t=1:T], battery_charge[t] <= battery_power_cap)
    @constraint(model, [t=1:T], battery_discharge[t] <= battery_power_cap)
    @constraint(model, [t=1:T], battery_soc[t] <= battery_energy_cap)
    
    # Battery energy balance with proper time coupling
    @constraint(model, battery_soc[1] == battery_energy_cap * 0.5 + 
        battery.efficiency_charge * battery_charge[1] - battery_discharge[1]/battery.efficiency_discharge)
    @constraint(model, [t=2:T], battery_soc[t] == battery_soc[t-1] + 
        battery.efficiency_charge * battery_charge[t] - battery_discharge[t]/battery.efficiency_discharge)
    
    # End with same SOC as start (periodic constraint)
    @constraint(model, battery_soc[T] >= battery_energy_cap * 0.4)
    @constraint(model, battery_soc[T] <= battery_energy_cap * 0.6)
    
    # Battery energy/power ratio
    @constraint(model, battery_energy_cap <= battery_power_cap * battery.duration)
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        result = Dict(
            "status" => "optimal",
            "capacity" => value.(capacity),
            "battery_power_cap" => value(battery_power_cap),
            "battery_energy_cap" => value(battery_energy_cap),
            "generation" => value.(generation),
            "battery_charge" => value.(battery_charge),
            "battery_discharge" => value.(battery_discharge),
            "battery_soc" => value.(battery_soc),
            "load_shed" => value.(load_shed),
            "commitment" => ones(G, T),  # Dummy commitment
            "startup" => zeros(G, T),    # Dummy startup
            "total_cost" => objective_value(model),
            "investment_cost" => value(investment_cost),
            "fixed_cost" => value(fixed_cost),
            "operational_cost" => value(operational_cost),
            "prices" => dual.(power_balance),
            "nuclear_availability" => nuclear_availability,
            "gas_availability" => gas_availability
        )
        
        # Save capacity results
        mkpath(output_dir)
        capacity_df = DataFrame(
            Technology = [gen.name for gen in generators],
            Capacity_MW = value.(capacity),
            Investment_Cost = [generators[g].inv_cost * value(capacity[g]) for g in 1:G],
            Fixed_OM_Cost = [generators[g].fixed_om_cost * value(capacity[g]) for g in 1:G],
            Availability_Factor = [
                generators[g].name == "Nuclear" ? mean(nuclear_availability) :
                generators[g].name == "Gas" ? mean(gas_availability) :
                generators[g].name == "Wind" ? mean(actual_wind) : 1.0
                for g in 1:G
            ]
        )
        
        # Add battery row
        push!(capacity_df, ("Battery_Power", value(battery_power_cap), 
                           battery.inv_cost_power * value(battery_power_cap),
                           battery.fixed_om_cost * value(battery_power_cap), 1.0))
        push!(capacity_df, ("Battery_Energy", value(battery_energy_cap),
                           battery.inv_cost_energy * value(battery_energy_cap), 0.0, 1.0))
        
        CSV.write(joinpath(output_dir, "capacity_expansion_results.csv"), capacity_df)
        
        # Save detailed capacity expansion operational results
        save_operational_results(result, generators, battery, "capacity_expansion", output_dir)
        
        # Save availability profiles
        avail_df = DataFrame(
            Hour = 1:T,
            Nuclear_Availability = nuclear_availability,
            Gas_Availability = gas_availability,
            Wind_CF = actual_wind,
            Demand = actual_demand
        )
        CSV.write(joinpath(output_dir, "availability_profiles.csv"), avail_df)
        
        return result
    else
        return Dict("status" => "infeasible", "termination_status" => termination_status(model))
    end
end

# =============================================================================
# 3. PERFECT FORESIGHT OPERATIONAL MODEL
# =============================================================================

function solve_perfect_foresight_operations(generators, battery, capacities, battery_power_cap, 
                                           battery_energy_cap, demand, wind_cf; output_dir="results")
    """
    Solve operations with perfect foresight for the entire time horizon
    Now includes thermal availability factors (outages) to match capacity expansion
    Uses FIXED capacities from capacity expansion to ensure zero-profit consistency
    """
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    # Get the same profiles used in capacity expansion (including outages)
    actual_demand, actual_wind, nuclear_availability, gas_availability, 
    demand_scenarios, wind_scenarios, nuclear_avail_scenarios, gas_avail_scenarios = create_actual_and_scenarios()
    
    T = length(actual_demand)
    G = length(generators)
    
    println("Solving perfect foresight operations for $T hours with thermal outages...")
    println("  Nuclear availability: $(round(mean(nuclear_availability)*100, digits=1))%")
    println("  Gas availability: $(round(mean(gas_availability)*100, digits=1))%")
    
    # Decision variables - ONLY operational variables (capacities are FIXED)
    @variable(model, generation[1:G, 1:T] >= 0)
    @variable(model, battery_charge[1:T] >= 0)
    @variable(model, battery_discharge[1:T] >= 0)
    @variable(model, battery_soc[1:T] >= 0)
    @variable(model, load_shed[1:T] >= 0)
            
    # investment_cost = sum(generators[g].inv_cost * capacities[g] for g in 1:G) + 
    #                  battery.inv_cost_power * battery_power_cap + 
    #                  battery.inv_cost_energy * battery_energy_cap
                     
    # fixed_cost = sum(generators[g].fixed_om_cost * capacities[g] for g in 1:G) + 
    #              battery.fixed_om_cost * battery_power_cap
    operations_cost = sum(sum((generators[g].fuel_cost + generators[g].var_om_cost) * generation[g,t] for g in 1:G) +
            battery.var_om_cost * (battery_charge[t] + battery_discharge[t]) +
            10000 * load_shed[t] for t in 1:T)
    # Objective: Minimize operational costs ONLY (no investment costs)
    @objective(model, Min, operations_cost)

    
    # Power balance
    @constraint(model, power_balance[t=1:T],
        sum(generation[g,t] for g in 1:G) + battery_discharge[t] - 
        battery_charge[t] + load_shed[t] == actual_demand[t])
    
    # Generation constraints with FIXED capacities and availability factors
    for g in 1:G
        if generators[g].name == "Nuclear"
            @constraint(model, [t=1:T], 
                generation[g,t] <= capacities[g] * nuclear_availability[t])
        elseif generators[g].name == "Wind"
            @constraint(model, [t=1:T], 
                generation[g,t] <= capacities[g] * actual_wind[t])
        elseif generators[g].name == "Gas"
            @constraint(model, [t=1:T], 
                generation[g,t] <= capacities[g] * gas_availability[t])
        else
            # Default case for any other generator types
            @constraint(model, [t=1:T], 
                generation[g,t] <= capacities[g])
        end
    end
    
    # Battery constraints with FIXED capacities
    @constraint(model, [t=1:T], battery_charge[t] <= battery_power_cap)
    @constraint(model, [t=1:T], battery_discharge[t] <= battery_power_cap)
    @constraint(model, [t=1:T], battery_soc[t] <= battery_energy_cap)
    
    # Battery energy balance (same as capacity expansion)
    @constraint(model, battery_soc[1] == battery_energy_cap * 0.5 +
        battery.efficiency_charge * battery_charge[1] - battery_discharge[1]/battery.efficiency_discharge)
    @constraint(model, [t=2:T], battery_soc[t] == battery_soc[t-1] + 
        battery.efficiency_charge * battery_charge[t] - battery_discharge[t]/battery.efficiency_discharge)
    
    # End with same SOC as start (periodic constraint) - same as capacity expansion
    @constraint(model, battery_soc[T] >= battery_energy_cap * 0.4)
    @constraint(model, battery_soc[T] <= battery_energy_cap * 0.6)
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        result = Dict(
            "status" => "optimal",
            "generation" => value.(generation),
            "battery_charge" => value.(battery_charge),
            "battery_discharge" => value.(battery_discharge),
            "battery_soc" => value.(battery_soc),
            "load_shed" => value.(load_shed),
            "commitment" => ones(G, T),  # Dummy commitment for compatibility
            "startup" => zeros(G, T),    # Dummy startup for compatibility
            "total_cost" => objective_value(model),  # This is ONLY operational cost
            "prices" => dual.(power_balance),
            "nuclear_availability" => nuclear_availability,
            "gas_availability" => gas_availability
        )
        
        # Verify that this gives the same operational cost as capacity expansion
        cap_result_check = solve_capacity_expansion(generators, battery, actual_demand, actual_wind, output_dir=output_dir)
        expected_operational_cost = cap_result_check["operational_cost"]
        actual_operational_cost = objective_value(model)
        
        println("Operational Cost Verification:")
        println("  Capacity Expansion Operational Cost: \$(round(expected_operational_cost, digits=0))")
        println("  Perfect Foresight Operational Cost: \$(round(actual_operational_cost, digits=0))")
        println("  Difference: \$(round(abs(expected_operational_cost - actual_operational_cost), digits=0))")
        
        if abs(expected_operational_cost - actual_operational_cost) > 1000
            println("  WARNING: Large operational cost difference!")
        else
            println("  ✓ Operational costs match")
        end
        
        # Save detailed operational results
        save_operational_results(result, generators, battery, "perfect_foresight", output_dir)
        
        # Save outage impact analysis
        outage_analysis_df = DataFrame(
            Hour = 1:T,
            Demand = actual_demand,
            Nuclear_Available_MW = [capacities[1] * nuclear_availability[t] for t in 1:T],
            Gas_Available_MW = [capacities[length(generators)] * gas_availability[t] for t in 1:T],  # Gas is last generator
            Wind_Available_MW = [capacities[2] * actual_wind[t] for t in 1:T],     # Wind is typically 2nd generator
            Total_Thermal_Available = [capacities[1] * nuclear_availability[t] + 
                                     capacities[length(generators)] * gas_availability[t] for t in 1:T],
            Load_Shed = value.(load_shed),
            Battery_Net = value.(battery_discharge) .- value.(battery_charge),
            Price = dual.(power_balance)
        )
        CSV.write(joinpath(output_dir, "perfect_foresight_outage_analysis.csv"), outage_analysis_df)
        
        # Print summary of outage impacts
        total_nuclear_outage_hours = sum(nuclear_availability .< 1.0)
        total_gas_outage_hours = sum(gas_availability .< 1.0)
        max_price = maximum(dual.(power_balance))
        total_load_shed = sum(value.(load_shed))
        
        println("Perfect Foresight Outage Impact Summary:")
        println("  Nuclear outage hours: $total_nuclear_outage_hours ($(round(total_nuclear_outage_hours/T*100, digits=1))%)")
        println("  Gas outage hours: $total_gas_outage_hours ($(round(total_gas_outage_hours/T*100, digits=1))%)")
        println("  Total load shed: $(round(total_load_shed, digits=1)) MWh")
        println("  Maximum price: \$(round(max_price, digits=2))/MWh")
        
        return result
    else
        return Dict("status" => "infeasible", "termination_status" => termination_status(model))
    end
end

# =============================================================================
# 4. DLAC OPERATIONAL MODEL
# =============================================================================

# Updated DLAC-i operations to also use thermal outages
function solve_dlac_i_operations(generators, battery, capacities, battery_power_cap, 
                                 battery_energy_cap, actual_demand, actual_wind, 
                                 demand_scenarios, wind_scenarios; lookahead_hours=24, output_dir="results")
    """
    Solve operations using DLAC-i with thermal outages
    - Operates on ACTUAL demand/wind/outages (realized values)
    - Uses MEAN of scenarios for forecasting in lookahead horizon
    """
    
    # Get the same outage profiles
    actual_demand, actual_wind, nuclear_availability, gas_availability, 
    demand_scenarios, wind_scenarios, nuclear_avail_scenarios, gas_avail_scenarios = create_actual_and_scenarios()
    
    T = length(actual_demand)
    G = length(generators)
    S = length(demand_scenarios)
    
    # Compute mean forecasts from scenarios (including outages)
    mean_demand_forecast = zeros(T)
    mean_wind_forecast = zeros(T)
    mean_nuclear_avail_forecast = zeros(T)
    mean_gas_avail_forecast = zeros(T)
    
    for t in 1:T
        mean_demand_forecast[t] = mean([demand_scenarios[s][t] for s in 1:S])
        mean_wind_forecast[t] = mean([wind_scenarios[s][t] for s in 1:S])
        mean_nuclear_avail_forecast[t] = mean([nuclear_avail_scenarios[s][t] for s in 1:S])
        mean_gas_avail_forecast[t] = mean([gas_avail_scenarios[s][t] for s in 1:S])
    end
    
    println("Solving DLAC-i operations with $lookahead_hours hour lookahead for $T hours...")
    println("  Using actual outages for operations, mean forecast outages for lookahead")
    
    # Results storage
    generation_schedule = zeros(G, T)
    battery_charge_schedule = zeros(T)
    battery_discharge_schedule = zeros(T)
    battery_soc_schedule = zeros(T)
    load_shed_schedule = zeros(T)
    commitment_schedule = zeros(G, T)
    startup_schedule = zeros(G, T)
    prices = zeros(T)
    
    # State tracking
    current_soc = battery_energy_cap * 0.5
    previous_commitment = zeros(G)
    
    for t in 1:T
        if t % 100 == 0
            println("  Processing hour $t/$T")
        end
        
        model = Model(Gurobi.Optimizer)
        set_silent(model)
        
        # Determine lookahead horizon
        horizon_end = min(t + lookahead_hours - 1, T)
        horizon = t:horizon_end
        H = length(horizon)
        
        # Decision variables for the lookahead horizon
        @variable(model, gen[1:G, 1:H] >= 0)
        @variable(model, bat_charge[1:H] >= 0)
        @variable(model, bat_discharge[1:H] >= 0)
        @variable(model, bat_soc[1:H] >= 0)
        @variable(model, load_shed[1:H] >= 0)
        
        # Objective: minimize cost over lookahead horizon
        @objective(model, Min, 
            sum(sum((generators[g].fuel_cost + generators[g].var_om_cost) * gen[g,τ] for g in 1:G) +
                battery.var_om_cost * (bat_charge[τ] + bat_discharge[τ]) +
                10000 * load_shed[τ] for τ in 1:H))
        
        # Power balance constraints
        @constraint(model, power_balance_constraint[τ=1:H],
            sum(gen[g,τ] for g in 1:G) + bat_discharge[τ] - 
            bat_charge[τ] + load_shed[τ] == 
            (τ == 1 ? actual_demand[t] : mean_demand_forecast[horizon[τ]]))
        
        # Generation constraints with availability factors
        for g in 1:G
            if generators[g].name == "Nuclear"
                # Use actual availability for current period, forecast for future
                @constraint(model, [τ=1:H], gen[g,τ] <= capacities[g] * 
                    (τ == 1 ? nuclear_availability[t] : mean_nuclear_avail_forecast[horizon[τ]]))
            elseif generators[g].name == "Wind"
                # Use actual wind for current period, mean forecast for future periods
                @constraint(model, [τ=1:H], gen[g,τ] <= capacities[g] * 
                    (τ == 1 ? actual_wind[t] : mean_wind_forecast[horizon[τ]]))
            elseif generators[g].name == "Gas"
                # Use actual gas availability for current period, forecast for future
                @constraint(model, [τ=1:H], gen[g,τ] <= capacities[g] * 
                    (τ == 1 ? gas_availability[t] : mean_gas_avail_forecast[horizon[τ]]))
            else
                # Default case
                @constraint(model, [τ=1:H], gen[g,τ] <= capacities[g])
            end
        end
        
        # Battery constraints
        @constraint(model, [τ=1:H], bat_charge[τ] <= battery_power_cap)
        @constraint(model, [τ=1:H], bat_discharge[τ] <= battery_power_cap)
        @constraint(model, [τ=1:H], bat_soc[τ] <= battery_energy_cap)
        
        # Battery energy balance
        @constraint(model, bat_soc[1] == current_soc + 
            battery.efficiency_charge * bat_charge[1] - bat_discharge[1]/battery.efficiency_discharge)
        @constraint(model, [τ=2:H], bat_soc[τ] == bat_soc[τ-1] + 
            battery.efficiency_charge * bat_charge[τ] - bat_discharge[τ]/battery.efficiency_discharge)
        
        optimize!(model)
        
        if termination_status(model) == MOI.OPTIMAL
            # Store only first-period decisions (what actually happens)
            generation_schedule[:, t] = value.(gen[:, 1])
            battery_charge_schedule[t] = value(bat_charge[1])
            battery_discharge_schedule[t] = value(bat_discharge[1])
            battery_soc_schedule[t] = value(bat_soc[1])
            load_shed_schedule[t] = value(load_shed[1])
            commitment_schedule[:, t] = ones(G)  # Dummy commitment
            startup_schedule[:, t] = zeros(G)    # Dummy startup
            
            # Extract price
            prices[t] = dual(power_balance_constraint[1])
            
            # Update state
            current_soc = value(bat_soc[1])
        else
            println("Warning: DLAC-i optimization failed at hour $t")
            load_shed_schedule[t] = actual_demand[t]
            prices[t] = 10000
        end
    end
    
    result = Dict(
        "status" => "optimal",
        "generation" => generation_schedule,
        "battery_charge" => battery_charge_schedule,
        "battery_discharge" => battery_discharge_schedule,
        "battery_soc" => battery_soc_schedule,
        "load_shed" => load_shed_schedule,
        "commitment" => commitment_schedule,
        "startup" => startup_schedule,
        "prices" => prices,
        "total_cost" => sum(sum((generators[g].fuel_cost + generators[g].var_om_cost) * 
                               generation_schedule[g,t] for g in 1:G) + battery.var_om_cost * 
                               (battery_charge_schedule[t] + battery_discharge_schedule[t]) +
                               10000 * load_shed_schedule[t] for t in 1:T),
        "nuclear_availability" => nuclear_availability,
        "gas_availability" => gas_availability
    )
    
    # Save detailed operational results
    save_operational_results(result, generators, battery, "dlac_i", output_dir)
    
    return result
end

# =============================================================================
# 5. RESULTS SAVING AND ANALYSIS
# =============================================================================

function save_operational_results(results, generators, battery, model_name, output_dir)
    """Save detailed operational results to CSV files"""
    
    mkpath(output_dir)
    T = length(results["prices"])
    G = length(generators)
    
    # Create main results DataFrame
    df = DataFrame(
        Hour = 1:T,
        Price = results["prices"],
        Load_Shed = results["load_shed"],
        Battery_Charge = results["battery_charge"],
        Battery_Discharge = results["battery_discharge"],
        Battery_SOC = results["battery_soc"]
    )
    
    # Add generation columns
    for g in 1:G
        df[!, "$(generators[g].name)_Generation"] = results["generation"][g, :]
        df[!, "$(generators[g].name)_Commitment"] = results["commitment"][g, :]
    end
    
    # Save main results
    CSV.write(joinpath(output_dir, "$(model_name)_operations.csv"), df)
    
    # Create summary statistics
    summary_df = DataFrame(
        Metric = String[],
        Value = Float64[]
    )
    
    push!(summary_df, ("Total_Operational_Cost", results["total_cost"]))
    push!(summary_df, ("Total_Load_Shed_MWh", sum(results["load_shed"])))
    push!(summary_df, ("Average_Price", mean(results["prices"])))
    push!(summary_df, ("Max_Price", maximum(results["prices"])))
    push!(summary_df, ("Battery_Total_Charge_MWh", sum(results["battery_charge"])))
    push!(summary_df, ("Battery_Total_Discharge_MWh", sum(results["battery_discharge"])))
    
    for g in 1:G
        total_gen = sum(results["generation"][g, :])
        push!(summary_df, ("$(generators[g].name)_Total_Generation_MWh", total_gen))
        push!(summary_df, ("$(generators[g].name)_Total_Startups", sum(results["startup"][g, :])))
    end
    
    CSV.write(joinpath(output_dir, "$(model_name)_summary.csv"), summary_df)
end

function calculate_profits_and_save(generators, battery, operational_results, capacities, 
                                   battery_power_cap, battery_energy_cap, model_name, output_dir)
    """Calculate and save detailed profit analysis"""
    
    G = length(generators)
    T = size(operational_results["generation"], 2)
    
    # Create profit DataFrame
    profit_df = DataFrame(
        Technology = String[],
        Capacity_MW = Float64[],
        Total_Generation_MWh = Float64[],
        Capacity_Factor = Float64[],
        Energy_Revenue = Float64[],
        Fuel_Costs = Float64[],
        VOM_Costs = Float64[],
        Startup_Costs = Float64[],
        Fixed_OM_Costs = Float64[],
        Investment_Costs = Float64[],
        Operating_Profit = Float64[],
        Net_Profit = Float64[],
        Profit_Margin_per_MW = Float64[]
    )
    
    # Generator profits
    for g in 1:G
        gen_name = generators[g].name
        capacity = capacities[g]
        total_gen = sum(operational_results["generation"][g, :])
        capacity_factor = capacity > 0 ? total_gen / (capacity * T) : 0.0
        
        # Revenues and costs
        energy_revenue = sum(operational_results["prices"][t] * operational_results["generation"][g,t] for t in 1:T)
        fuel_costs = sum(generators[g].fuel_cost * operational_results["generation"][g,t] for t in 1:T)
        vom_costs = sum(generators[g].var_om_cost * operational_results["generation"][g,t] for t in 1:T)
        startup_costs = sum(generators[g].startup_cost * operational_results["startup"][g,t] for t in 1:T)
        fixed_om_costs = generators[g].fixed_om_cost * capacity
        investment_costs = generators[g].inv_cost * capacity
        
        operating_profit = energy_revenue - fuel_costs - vom_costs - startup_costs - fixed_om_costs
        net_profit = operating_profit - investment_costs
        profit_margin = capacity > 0 ? net_profit / capacity : 0.0
        
        push!(profit_df, (gen_name, capacity, total_gen, capacity_factor, energy_revenue,
                         fuel_costs, vom_costs, startup_costs, fixed_om_costs, investment_costs,
                         operating_profit, net_profit, profit_margin))
    end
    
    # Battery profit
    battery_energy_revenue = sum(operational_results["prices"][t] * operational_results["battery_discharge"][t] for t in 1:T)
    battery_energy_costs = sum(operational_results["prices"][t] * operational_results["battery_charge"][t] for t in 1:T)
    battery_net_energy_revenue = battery_energy_revenue - battery_energy_costs
    
    battery_vom_costs = sum(battery.var_om_cost * (operational_results["battery_charge"][t] + 
                           operational_results["battery_discharge"][t]) for t in 1:T)
    battery_fixed_costs = battery.fixed_om_cost * battery_power_cap
    battery_investment_costs = battery.inv_cost_power * battery_power_cap + battery.inv_cost_energy * battery_energy_cap
    
    battery_operating_profit = battery_net_energy_revenue - battery_vom_costs - battery_fixed_costs
    battery_net_profit = battery_operating_profit - battery_investment_costs
    battery_profit_margin = battery_power_cap > 0 ? battery_net_profit / battery_power_cap : 0.0
    
    total_discharge = sum(operational_results["battery_discharge"])
    battery_capacity_factor = battery_power_cap > 0 ? total_discharge / (battery_power_cap * T) : 0.0
    
    push!(profit_df, ("Battery", battery_power_cap, total_discharge, battery_capacity_factor,
                     battery_energy_revenue, battery_energy_costs, battery_vom_costs, 0.0,
                     battery_fixed_costs, battery_investment_costs, battery_operating_profit,
                     battery_net_profit, battery_profit_margin))
    
    CSV.write(joinpath(output_dir, "$(model_name)_profits.csv"), profit_df)
    
    return profit_df
end

# =============================================================================
# 6. COMPLETE SOLVE FUNCTION
# =============================================================================

function solve_complete_system(output_dir="results")
    """
    Complete function that solves everything with thermal outages and generates output files
    Returns all results needed for fixed point iteration
    """
    
    println("="^80)
    println("COMPLETE 4-TECHNOLOGY SYSTEM WITH THERMAL OUTAGES")
    println("="^80)
    
    # Create system
    generators, battery = create_toy_system()
    
    # Generate actual deterministic profiles and scenarios with outages
    println("\nGenerating actual deterministic profiles and 3 scenarios with thermal outages...")
    actual_demand, actual_wind, nuclear_availability, gas_availability,
    demand_scenarios, wind_scenarios, nuclear_avail_scenarios, gas_avail_scenarios = create_actual_and_scenarios()
    
    println("  - Actual demand: $(length(actual_demand)) hours")
    println("  - Generated 3 scenarios around actual as mean")
    println("  - Mean demand across scenarios: $(round(mean([mean(s) for s in demand_scenarios]), digits=1)) MW")
    println("  - Actual demand mean: $(round(mean(actual_demand), digits=1)) MW")
    println("  - Nuclear availability: $(round(mean(nuclear_availability)*100, digits=1))%")
    println("  - Gas availability: $(round(mean(gas_availability)*100, digits=1))%")
    println("  - Wind capacity factor: $(round(mean(actual_wind)*100, digits=1))%")
    
    # Save profiles including outages
    mkpath(output_dir)
    profiles_df = DataFrame(
        Hour = 1:length(actual_demand),
        Actual_Demand = actual_demand,
        Actual_Wind_CF = actual_wind,
        Nuclear_Availability = nuclear_availability,
        Gas_Availability = gas_availability,
        Scenario1_Demand = demand_scenarios[1],
        Scenario1_Wind = wind_scenarios[1],
        Scenario1_Nuclear_Avail = nuclear_avail_scenarios[1],
        Scenario1_Gas_Avail = gas_avail_scenarios[1],
        Scenario2_Demand = demand_scenarios[2],
        Scenario2_Wind = wind_scenarios[2],
        Scenario2_Nuclear_Avail = nuclear_avail_scenarios[2],
        Scenario2_Gas_Avail = gas_avail_scenarios[2],
        Scenario3_Demand = demand_scenarios[3],
        Scenario3_Wind = wind_scenarios[3],
        Scenario3_Nuclear_Avail = nuclear_avail_scenarios[3],
        Scenario3_Gas_Avail = gas_avail_scenarios[3]
    )
    CSV.write(joinpath(output_dir, "demand_wind_outage_profiles.csv"), profiles_df)
    
    # Step 1: Capacity Expansion with outages
    println("\n" * "="^60)
    println("STEP 1: CAPACITY EXPANSION WITH THERMAL OUTAGES")
    println("="^60)
    
    cap_result = solve_capacity_expansion(generators, battery, actual_demand, actual_wind, output_dir=output_dir)
    
    if cap_result["status"] != "optimal"
        error("Capacity expansion failed!")
    end
    
    capacities = cap_result["capacity"]
    battery_power_cap = cap_result["battery_power_cap"] 
    battery_energy_cap = cap_result["battery_energy_cap"]
    
    println("Optimal Capacities (accounting for thermal outages):")
    for (i, gen) in enumerate(generators)
        println("  $(gen.name): $(round(capacities[i], digits=1)) MW")
    end
    println("  Battery Power: $(round(battery_power_cap, digits=1)) MW")
    println("  Battery Energy: $(round(battery_energy_cap, digits=1)) MWh")
    println("  Total Investment Cost: $(round(cap_result["investment_cost"], digits=0))")
    println("  Total Operational Cost: $(round(cap_result["operational_cost"], digits=0))")
    println("  Total System Cost: $(round(cap_result["total_cost"], digits=0))")
    
    # Calculate and save Capacity Expansion profits (should be zero!)
    cap_profits = calculate_profits_and_save(generators, battery, cap_result, capacities,
                                           battery_power_cap, battery_energy_cap, 
                                           "capacity_expansion", output_dir)
    
    println("\nCapacity Expansion Profit Check (should be ~zero):")
    println("Technology | Net Profit | Profit Margin")
    println("-"^40)
    for i in 1:length(generators)
        tech = generators[i].name
        net_profit = cap_profits[i, :Net_Profit]
        profit_margin = cap_profits[i, :Profit_Margin_per_MW]
        @printf("%-10s | %10.0f | %12.0f\n", tech, net_profit, profit_margin)
    end
    cap_bat_profit = cap_profits[end, :Net_Profit]
    cap_bat_margin = cap_profits[end, :Profit_Margin_per_MW]
    @printf("%-10s | %10.0f | %12.0f\n", "Battery", cap_bat_profit, cap_bat_margin)
    
    # Analyze capacity adequacy
    max_demand = maximum(actual_demand)
    total_firm_capacity = sum(capacities[1:end-1])  # Exclude wind from firm capacity
    nuclear_firm = capacities[1] * mean(nuclear_availability)
    gas_firm = capacities[end] * mean(gas_availability)  # Assuming gas is last thermal
    wind_contribution = capacities[2] * mean(actual_wind) * 0.3  # 30% wind capacity credit
    
    println("\nCapacity Adequacy Analysis:")
    println("  Peak Demand: $(round(max_demand, digits=1)) MW")
    println("  Nuclear Firm Capacity: $(round(nuclear_firm, digits=1)) MW")
    println("  Gas Firm Capacity: $(round(gas_firm, digits=1)) MW") 
    println("  Wind Capacity Credit: $(round(wind_contribution, digits=1)) MW")
    println("  Battery Power: $(round(battery_power_cap, digits=1)) MW")
    println("  Total Effective Capacity: $(round(nuclear_firm + gas_firm + wind_contribution + battery_power_cap, digits=1)) MW")
    
    # Step 2: Perfect Foresight Operations with outages
    println("\n" * "="^60)
    println("STEP 2: PERFECT FORESIGHT OPERATIONS WITH OUTAGES")
    println("="^60)
    
    pf_result = solve_perfect_foresight_operations(generators, battery, capacities, 
                                                  battery_power_cap, battery_energy_cap,
                                                  actual_demand, actual_wind, output_dir=output_dir)
    
    if pf_result["status"] != "optimal"
        error("Perfect foresight operations failed!")
    end
    
    println("Perfect Foresight Results:")
    println("  Total Operational Cost: $(round(pf_result["total_cost"], digits=0))")
    println("  Total Load Shed: $(round(sum(pf_result["load_shed"]), digits=1)) MWh")
    println("  Average Price: $(round(mean(pf_result["prices"]), digits=2))/MWh")
    println("  Max Price: $(round(maximum(pf_result["prices"]), digits=2))/MWh")
    
    # Calculate and save PF profits (should also be zero!)
    pf_profits = calculate_profits_and_save(generators, battery, pf_result, capacities,
                                           battery_power_cap, battery_energy_cap, 
                                           "perfect_foresight", output_dir)
    
    println("\nPerfect Foresight Profit Check (should be ~zero):")
    println("Technology | Net Profit | Profit Margin")
    println("-"^40)
    for i in 1:length(generators)
        tech = generators[i].name
        net_profit = pf_profits[i, :Net_Profit]
        profit_margin = pf_profits[i, :Profit_Margin_per_MW]
        @printf("%-10s | %10.0f | %12.0f\n", tech, net_profit, profit_margin)
    end
    pf_bat_profit = pf_profits[end, :Net_Profit]
    pf_bat_margin = pf_profits[end, :Profit_Margin_per_MW]
    @printf("%-10s | %10.0f | %12.0f\n", "Battery", pf_bat_profit, pf_bat_margin)
    
    # Verify consistency between capacity expansion and perfect foresight
    cap_total_cost = cap_result["investment_cost"] + cap_result["operational_cost"]
    pf_total_cost = sum([generators[g].inv_cost * capacities[g] for g in 1:length(generators)]) + 
                    battery.inv_cost_power * battery_power_cap + 
                    battery.inv_cost_energy * battery_energy_cap + 
                    pf_result["total_cost"]
    
    println("\nConsistency Check:")
    println("  Capacity Expansion Total Cost: $(round(cap_total_cost, digits=0))")
    println("  Perfect Foresight Total Cost: $(round(pf_total_cost, digits=0))")
    println("  Difference: $(round(abs(cap_total_cost - pf_total_cost), digits=0)) (should be ~0)")
    
    if abs(cap_total_cost - pf_total_cost) > 1000
        println("  WARNING: Large difference suggests inconsistency!")
    else
        println("  ✓ Models are consistent")
    end
    
    # Step 3: DLAC-i Operations with outages
    println("\n" * "="^60)
    println("STEP 3: DLAC-i OPERATIONS WITH OUTAGE UNCERTAINTY")
    println("="^60)
    
    dlac_result = solve_dlac_i_operations(generators, battery, capacities,
                                         battery_power_cap, battery_energy_cap,
                                         actual_demand, actual_wind,
                                         demand_scenarios, wind_scenarios,
                                         lookahead_hours=24, output_dir=output_dir)
    
    if dlac_result["status"] != "optimal"
        error("DLAC-i operations failed!")
    end
    
    println("DLAC-i Results:")
    println("  Total Operational Cost: $(round(dlac_result["total_cost"], digits=0))")
    println("  Total Load Shed: $(round(sum(dlac_result["load_shed"]), digits=1)) MWh")
    println("  Average Price: $(round(mean(dlac_result["prices"]), digits=2))/MWh")
    println("  Max Price: $(round(maximum(dlac_result["prices"]), digits=2))/MWh")
    
    # Calculate and save DLAC-i profits
    dlac_profits = calculate_profits_and_save(generators, battery, dlac_result, capacities,
                                             battery_power_cap, battery_energy_cap,
                                             "dlac_i", output_dir)
    
    # Step 4: Detailed Comparison Analysis
    println("\n" * "="^60)
    println("STEP 4: COMPREHENSIVE COMPARISON ANALYSIS")
    println("="^60)
    
    cost_diff = dlac_result["total_cost"] - pf_result["total_cost"]
    cost_pct = (cost_diff / pf_result["total_cost"]) * 100
    
    println("Perfect Foresight vs DLAC-i (both with thermal outages):")
    println("  PF Cost: $(round(pf_result["total_cost"], digits=0))")
    println("  DLAC-i Cost: $(round(dlac_result["total_cost"], digits=0))")
    println("  Difference: $(round(cost_diff, digits=0)) ($(round(cost_pct, digits=2))% increase)")
    
    println("\nLoad Shedding Comparison:")
    pf_shed = sum(pf_result["load_shed"])
    dlac_shed = sum(dlac_result["load_shed"])
    println("  PF Load Shed: $(round(pf_shed, digits=1)) MWh")
    println("  DLAC-i Load Shed: $(round(dlac_shed, digits=1)) MWh")
    println("  Additional Shed: $(round(dlac_shed - pf_shed, digits=1)) MWh")
    
    println("\nProfit Margin Comparison (Net Profit / Capacity):")
    println("Technology | Cap_Exp | PF_Margin | DLAC_Margin | PF_vs_Cap | DLAC_vs_Cap")
    println("-"^75)
    for i in 1:length(generators)
        tech = generators[i].name
        cap_margin = cap_profits[i, :Profit_Margin_per_MW]
        pf_margin = pf_profits[i, :Profit_Margin_per_MW]
        dlac_margin = dlac_profits[i, :Profit_Margin_per_MW]
        pf_diff = pf_margin - cap_margin
        dlac_diff = dlac_margin - cap_margin
        @printf("%-10s | %7.0f | %9.0f | %11.0f | %9.0f | %11.0f\n", 
                tech, cap_margin, pf_margin, dlac_margin, pf_diff, dlac_diff)
    end
    
    # Battery comparison
    cap_bat_margin = cap_profits[end, :Profit_Margin_per_MW]
    pf_bat_margin = pf_profits[end, :Profit_Margin_per_MW]
    dlac_bat_margin = dlac_profits[end, :Profit_Margin_per_MW]
    pf_bat_diff = pf_bat_margin - cap_bat_margin
    dlac_bat_diff = dlac_bat_margin - cap_bat_margin
    @printf("%-10s | %7.0f | %9.0f | %11.0f | %9.0f | %11.0f\n", 
            "Battery", cap_bat_margin, pf_bat_margin, dlac_bat_margin, pf_bat_diff, dlac_bat_diff)
    
    println("\nInterpretation:")
    println("  Cap_Exp & PF_Margin should be ~0 (zero profit equilibrium)")
    println("  DLAC_Margin shows profit deviation due to operational uncertainty")
    println("  PF_vs_Cap shows any inconsistency (should be ~0)")
    println("  DLAC_vs_Cap shows impact of limited foresight on profitability")
    
    # Technology utilization analysis
    println("\nTechnology Utilization Analysis:")
    println("Technology | Capacity | CF_PF | CF_DLAC | Avail_Factor")
    println("-"^60)
    
    for i in 1:length(generators)
        tech = generators[i].name
        capacity = capacities[i]
        
        if capacity > 0
            pf_cf = sum(pf_result["generation"][i, :]) / (capacity * length(actual_demand))
            dlac_cf = sum(dlac_result["generation"][i, :]) / (capacity * length(actual_demand))
            
            if tech == "Nuclear"
                avail_factor = mean(nuclear_availability)
            elseif tech == "Gas"
                avail_factor = mean(gas_availability)
            elseif tech == "Wind"
                avail_factor = mean(actual_wind)
            else
                avail_factor = 1.0
            end
            
            @printf("%-10s | %8.1f | %5.3f | %7.3f | %12.3f\n", 
                    tech, capacity, pf_cf, dlac_cf, avail_factor)
        else
            @printf("%-10s | %8.1f | %5s | %7s | %12s\n", tech, capacity, "N/A", "N/A", "N/A")
        end
    end
    
    # Battery utilization
    if battery_power_cap > 0
        battery_pf_util = maximum([maximum(pf_result["battery_charge"]), maximum(pf_result["battery_discharge"])]) / battery_power_cap
        battery_dlac_util = maximum([maximum(dlac_result["battery_charge"]), maximum(dlac_result["battery_discharge"])]) / battery_power_cap
        @printf("%-10s | %8.1f | %5.3f | %7.3f | %12.3f\n", "Battery", battery_power_cap, battery_pf_util, battery_dlac_util, 1.0)
    end
    
    # Create comprehensive comparison file including capacity expansion
    comparison_df = DataFrame(
        Technology = vcat([gen.name for gen in generators], ["Battery"]),
        Capacity_MW = vcat(capacities, [battery_power_cap]),
        Cap_Exp_Profit_Margin = vcat([cap_profits[i, :Profit_Margin_per_MW] for i in 1:length(generators)+1]),
        PF_Profit_Margin = vcat([pf_profits[i, :Profit_Margin_per_MW] for i in 1:length(generators)+1]),
        DLAC_i_Profit_Margin = vcat([dlac_profits[i, :Profit_Margin_per_MW] for i in 1:length(generators)+1]),
        PF_vs_CapExp_Difference = vcat([pf_profits[i, :Profit_Margin_per_MW] - cap_profits[i, :Profit_Margin_per_MW] for i in 1:length(generators)+1]),
        DLAC_vs_CapExp_Difference = vcat([dlac_profits[i, :Profit_Margin_per_MW] - cap_profits[i, :Profit_Margin_per_MW] for i in 1:length(generators)+1]),
        PF_Capacity_Factor = vcat([capacities[i] > 0 ? sum(pf_result["generation"][i, :]) / (capacities[i] * length(actual_demand)) : 0.0 for i in 1:length(generators)], 
                                 [battery_power_cap > 0 ? maximum([maximum(pf_result["battery_charge"]), maximum(pf_result["battery_discharge"])]) / battery_power_cap : 0.0]),
        DLAC_i_Capacity_Factor = vcat([capacities[i] > 0 ? sum(dlac_result["generation"][i, :]) / (capacities[i] * length(actual_demand)) : 0.0 for i in 1:length(generators)],
                                     [battery_power_cap > 0 ? maximum([maximum(dlac_result["battery_charge"]), maximum(dlac_result["battery_discharge"])]) / battery_power_cap : 0.0])
    )
    CSV.write(joinpath(output_dir, "three_model_comprehensive_comparison.csv"), comparison_df)
    
    # Save forecast quality analysis including outages
    forecast_df = DataFrame(
        Hour = 1:length(actual_demand),
        Actual_Demand = actual_demand,
        Mean_Forecast_Demand = [mean([demand_scenarios[s][t] for s in 1:3]) for t in 1:length(actual_demand)],
        Demand_Forecast_Error = [actual_demand[t] - mean([demand_scenarios[s][t] for s in 1:3]) for t in 1:length(actual_demand)],
        Actual_Wind = actual_wind,
        Mean_Forecast_Wind = [mean([wind_scenarios[s][t] for s in 1:3]) for t in 1:length(actual_wind)],
        Wind_Forecast_Error = [actual_wind[t] - mean([wind_scenarios[s][t] for s in 1:3]) for t in 1:length(actual_wind)],
        Actual_Nuclear_Avail = nuclear_availability,
        Mean_Forecast_Nuclear_Avail = [mean([nuclear_avail_scenarios[s][t] for s in 1:3]) for t in 1:length(nuclear_availability)],
        Nuclear_Avail_Forecast_Error = [nuclear_availability[t] - mean([nuclear_avail_scenarios[s][t] for s in 1:3]) for t in 1:length(nuclear_availability)],
        Actual_Gas_Avail = gas_availability,
        Mean_Forecast_Gas_Avail = [mean([gas_avail_scenarios[s][t] for s in 1:3]) for t in 1:length(gas_availability)],
        Gas_Avail_Forecast_Error = [gas_availability[t] - mean([gas_avail_scenarios[s][t] for s in 1:3]) for t in 1:length(gas_availability)]
    )
    CSV.write(joinpath(output_dir, "comprehensive_forecast_quality_analysis.csv"), forecast_df)
    
    # Summary of forecast quality including outages
    println("\nForecast Quality Analysis:")
    demand_mae = mean(abs.(forecast_df.Demand_Forecast_Error))
    wind_mae = mean(abs.(forecast_df.Wind_Forecast_Error))
    nuclear_mae = mean(abs.(forecast_df.Nuclear_Avail_Forecast_Error))
    gas_mae = mean(abs.(forecast_df.Gas_Avail_Forecast_Error))
    
    println("  Mean Absolute Demand Forecast Error: $(round(demand_mae, digits=1)) MW")
    println("  Mean Absolute Wind Forecast Error: $(round(wind_mae, digits=3))")
    println("  Mean Absolute Nuclear Avail Forecast Error: $(round(nuclear_mae, digits=3))")
    println("  Mean Absolute Gas Avail Forecast Error: $(round(gas_mae, digits=3))")
    
    # Step 5: Economic insights from outages
    println("\n" * "="^60)
    println("STEP 5: OUTAGE IMPACT ANALYSIS")
    println("="^60)
    
    # Analyze high-price periods
    high_price_threshold = 200  # $/MWh
    pf_high_price_hours = sum(pf_result["prices"] .> high_price_threshold)
    dlac_high_price_hours = sum(dlac_result["prices"] .> high_price_threshold)
    
    println("High Price Analysis (>$(high_price_threshold)/MWh):")
    println("  PF High Price Hours: $pf_high_price_hours")
    println("  DLAC-i High Price Hours: $dlac_high_price_hours")
    
    # Coincident outage analysis
    simultaneous_outages = sum((nuclear_availability .< 1.0) .& (gas_availability .< 1.0))
    nuclear_only_outages = sum((nuclear_availability .< 1.0) .& (gas_availability .>= 1.0))
    gas_only_outages = sum((nuclear_availability .>= 1.0) .& (gas_availability .< 1.0))
    
    println("\nOutage Pattern Analysis:")
    println("  Simultaneous Nuclear+Gas Outages: $simultaneous_outages hours")
    println("  Nuclear-only Outages: $nuclear_only_outages hours") 
    println("  Gas-only Outages: $gas_only_outages hours")
    
    println("\nAll results saved to: $output_dir/")
    
    return Dict(
        "generators" => generators,
        "battery" => battery,
        "actual_demand" => actual_demand,
        "actual_wind" => actual_wind,
        "nuclear_availability" => nuclear_availability,
        "gas_availability" => gas_availability,
        "demand_scenarios" => demand_scenarios,
        "wind_scenarios" => wind_scenarios,
        "nuclear_avail_scenarios" => nuclear_avail_scenarios,
        "gas_avail_scenarios" => gas_avail_scenarios,
        "capacity_expansion" => cap_result,
        "capacities" => capacities,
        "battery_power_cap" => battery_power_cap,
        "battery_energy_cap" => battery_energy_cap,
        "perfect_foresight" => pf_result,
        "dlac_i" => dlac_result,
        "cap_profits" => cap_profits,
        "pf_profits" => pf_profits,
        "dlac_profits" => dlac_profits
    )
end
# =============================================================================
# 7. FIXED POINT ITERATION
# =============================================================================

function calculate_profit_margins(generators, battery, operational_results, capacities, battery_power_cap, battery_energy_cap)
    """Calculate profit margins (profit per MW - investment cost per MW) for fixed point iteration"""
    
    G = length(generators)
    T = size(operational_results["generation"], 2)
    
    profit_margins = zeros(G + 1)  # G generators + 1 battery
    
    # Generator profit margins
    for g in 1:G
        if capacities[g] > 0
            # Calculate profit per MW
            energy_revenue = sum(operational_results["prices"][t] * operational_results["generation"][g,t] for t in 1:T)
            fuel_costs = sum(generators[g].fuel_cost * operational_results["generation"][g,t] for t in 1:T)
            vom_costs = sum(generators[g].var_om_cost * operational_results["generation"][g,t] for t in 1:T)
            startup_costs = sum(generators[g].startup_cost * operational_results["startup"][g,t] for t in 1:T)
            fixed_om_costs = generators[g].fixed_om_cost * capacities[g]
            
            operating_profit = energy_revenue - fuel_costs - vom_costs - startup_costs - fixed_om_costs
            profit_per_mw = operating_profit / capacities[g]
            
            # Profit margin = profit per MW - investment cost per MW
            profit_margins[g] = profit_per_mw - generators[g].inv_cost
        else
            # For zero capacity, estimate profit margin with small test capacity
            profit_margins[g] = -generators[g].inv_cost  # Conservative estimate
        end
    end
    
    # Battery profit margin
    if battery_power_cap > 0
        battery_energy_revenue = sum(operational_results["prices"][t] * operational_results["battery_discharge"][t] for t in 1:T)
        battery_energy_costs = sum(operational_results["prices"][t] * operational_results["battery_charge"][t] for t in 1:T)
        battery_net_energy_revenue = battery_energy_revenue - battery_energy_costs
        
        battery_vom_costs = sum(battery.var_om_cost * (operational_results["battery_charge"][t] + 
                               operational_results["battery_discharge"][t]) for t in 1:T)
        battery_fixed_costs = battery.fixed_om_cost * battery_power_cap
        
        battery_operating_profit = battery_net_energy_revenue - battery_vom_costs - battery_fixed_costs
        battery_profit_per_mw = battery_operating_profit / battery_power_cap
        
        # For battery, use power investment cost
        profit_margins[G + 1] = battery_profit_per_mw - battery.inv_cost_power
    else
        profit_margins[G + 1] = -battery.inv_cost_power
    end
    
    return profit_margins
end

function fixed_point_iteration_pf_vs_dlac(system_data; 
                                          max_iterations=20, 
                                          tolerance=1e-3, 
                                          step_size=0.05,
                                          smoothing_beta=10.0,
                                          output_dir="results")
    """
    Fixed point iteration comparing Perfect Foresight vs DLAC policies
    """
    
    println("\n" * "="^80)
    println("FIXED POINT ITERATION: PERFECT FORESIGHT vs DLAC")
    println("="^80)
    
    generators = system_data["generators"]
    battery = system_data["battery"]
    actual_demand = system_data["actual_demand"]
    actual_wind = system_data["actual_wind"]
    
    G = length(generators)
    
    # Initialize with capacity expansion results
    current_capacities = copy(system_data["capacities"])
    current_battery_power = system_data["battery_power_cap"]
    current_battery_energy = current_battery_power * battery.duration
    
    println("Starting Fixed Point Iteration:")
    println("Initial capacities from capacity expansion:")
    for (i, gen) in enumerate(generators)
        println("  $(gen.name): $(round(current_capacities[i], digits=1)) MW")
    end
    println("  Battery: $(round(current_battery_power, digits=1)) MW")
    
    # Storage for tracking convergence
    iteration_results = []
    
    for iteration in 1:max_iterations
        println("\n--- Iteration $iteration ---")
        
        # Solve Perfect Foresight operations
        println("Solving Perfect Foresight operations...")
        pf_results = solve_perfect_foresight_operations(generators, battery, current_capacities,
                                                       current_battery_power, current_battery_energy,
                                                       actual_demand, actual_wind, output_dir=output_dir)
        
        if pf_results["status"] != "optimal"
            println("ERROR: Perfect foresight failed at iteration $iteration")
            break
        end
        
        # Solve DLAC operations
        println("Solving DLAC operations...")
        dlac_results = solve_dlac_operations(generators, battery, current_capacities,
                                            current_battery_power, current_battery_energy,
                                            actual_demand, actual_wind, 
                                            lookahead_hours=24, output_dir=output_dir)
        
        if dlac_results["status"] != "optimal"
            println("ERROR: DLAC failed at iteration $iteration")
            break
        end
        
        # Calculate profit margins for both policies
        pf_margins = calculate_profit_margins(generators, battery, pf_results, 
                                             current_capacities, current_battery_power, current_battery_energy)
        dlac_margins = calculate_profit_margins(generators, battery, dlac_results,
                                               current_capacities, current_battery_power, current_battery_energy)
        
        println("Current capacities: ", [round(c, digits=1) for c in current_capacities], ", Battery: $(round(current_battery_power, digits=1))")
        println("PF profit margins: ", [round(m, digits=0) for m in pf_margins])
        println("DLAC profit margins: ", [round(m, digits=0) for m in dlac_margins])
        
        # Check convergence for both policies
        max_pf_margin = maximum(abs.(pf_margins))
        max_dlac_margin = maximum(abs.(dlac_margins))
        max_overall_margin = max(max_pf_margin, max_dlac_margin)
        
        println("Max PF margin: $(round(max_pf_margin, digits=0))")
        println("Max DLAC margin: $(round(max_dlac_margin, digits=0))")
        println("Max overall margin: $(round(max_overall_margin, digits=0))")
        
        # Store iteration results
        iter_result = Dict(
            "iteration" => iteration,
            "capacities" => copy(current_capacities),
            "battery_power" => current_battery_power,
            "pf_margins" => copy(pf_margins),
            "dlac_margins" => copy(dlac_margins),
            "pf_cost" => pf_results["total_cost"],
            "dlac_cost" => dlac_results["total_cost"],
            "max_pf_margin" => max_pf_margin,
            "max_dlac_margin" => max_dlac_margin
        )
        push!(iteration_results, iter_result)
        
        # Check convergence (for demonstration, use DLAC margins as the equilibrium condition)
        if max_dlac_margin < tolerance
            println("\nFixed point converged for DLAC policy!")
            println("Final DLAC equilibrium capacities:")
            for (i, gen) in enumerate(generators)
                println("  $(gen.name): $(round(current_capacities[i], digits=1)) MW")
            end
            println("  Battery: $(round(current_battery_power, digits=1)) MW")
            
            # Save convergence results
            save_fixed_point_results(iteration_results, generators, output_dir)
            
            return Dict(
                "converged" => true,
                "converged_policy" => "DLAC",
                "final_capacities" => current_capacities,
                "final_battery_power" => current_battery_power,
                "final_pf_margins" => pf_margins,
                "final_dlac_margins" => dlac_margins,
                "iteration_results" => iteration_results,
                "iterations" => iteration
            )
        end
        
        # Update capacities using DLAC profit margins (since we want DLAC equilibrium)
        # Use softplus smoothing for stability
        println("Updating capacities based on DLAC profit margins...")
        
        for g in 1:G
            if current_capacities[g] > 0
                capacity_update = current_capacities[g] + step_size * current_capacities[g] * dlac_margins[g] / 1000.0  # Scale down margins
                # Apply softplus smoothing
                current_capacities[g] = log(1 + exp(smoothing_beta * capacity_update)) / smoothing_beta
                current_capacities[g] = min(current_capacities[g], generators[g].max_capacity)
            else
                # For zero capacity technologies
                if dlac_margins[g] > 0
                    current_capacities[g] = step_size * 10.0  # Start with small capacity
                end
            end
        end
        
        # Update battery capacity
        if current_battery_power > 0
            battery_update = current_battery_power + step_size * current_battery_power * dlac_margins[G + 1] / 1000.0
            current_battery_power = log(1 + exp(smoothing_beta * battery_update)) / smoothing_beta
            current_battery_power = min(current_battery_power, battery.max_power_capacity)
            current_battery_energy = current_battery_power * battery.duration
        else
            if dlac_margins[G + 1] > 0
                current_battery_power = step_size * 10.0
                current_battery_energy = current_battery_power * battery.duration
            end
        end
        
        println("Updated capacities: ", [round(c, digits=1) for c in current_capacities], ", Battery: $(round(current_battery_power, digits=1))")
    end
    
    println("\nFixed point iteration completed without convergence")
    save_fixed_point_results(iteration_results, generators, output_dir)
    
    return Dict(
        "converged" => false,
        "final_capacities" => current_capacities,
        "final_battery_power" => current_battery_power,
        "iteration_results" => iteration_results,
        "iterations" => max_iterations
    )
end

function save_fixed_point_results(iteration_results, generators, output_dir)
    """Save fixed point iteration results"""
    
    mkpath(output_dir)
    
    # Create iteration summary DataFrame
    iter_df = DataFrame(
        Iteration = [r["iteration"] for r in iteration_results],
        Max_PF_Margin = [r["max_pf_margin"] for r in iteration_results],
        Max_DLAC_Margin = [r["max_dlac_margin"] for r in iteration_results],
        PF_Cost = [r["pf_cost"] for r in iteration_results],
        DLAC_Cost = [r["dlac_cost"] for r in iteration_results],
        Cost_Difference = [r["dlac_cost"] - r["pf_cost"] for r in iteration_results]
    )
    
    # Add capacity columns
    G = length(generators)
    for g in 1:G
        iter_df[!, "$(generators[g].name)_Capacity"] = [r["capacities"][g] for r in iteration_results]
    end
    iter_df[!, "Battery_Power_Capacity"] = [r["battery_power"] for r in iteration_results]
    
    # Add margin columns
    for g in 1:G
        iter_df[!, "$(generators[g].name)_PF_Margin"] = [r["pf_margins"][g] for r in iteration_results]
        iter_df[!, "$(generators[g].name)_DLAC_Margin"] = [r["dlac_margins"][g] for r in iteration_results]
    end
    iter_df[!, "Battery_PF_Margin"] = [r["pf_margins"][G+1] for r in iteration_results]
    iter_df[!, "Battery_DLAC_Margin"] = [r["dlac_margins"][G+1] for r in iteration_results]
    
    CSV.write(joinpath(output_dir, "fixed_point_iteration_results.csv"), iter_df)
    
    println("Fixed point iteration results saved to: $(joinpath(output_dir, "fixed_point_iteration_results.csv"))")
end


function save_fixed_point_results_dlac_i(iteration_results, generators, output_dir)
    """Save fixed point iteration results for DLAC-i comparison"""
    
    mkpath(output_dir)
    
    # Create iteration summary DataFrame
    iter_df = DataFrame(
        Iteration = [r["iteration"] for r in iteration_results],
        Max_PF_Margin = [r["max_pf_margin"] for r in iteration_results],
        Max_DLAC_i_Margin = [r["max_dlac_i_margin"] for r in iteration_results],
        PF_Cost = [r["pf_cost"] for r in iteration_results],
        DLAC_i_Cost = [r["dlac_i_cost"] for r in iteration_results],
        Cost_Difference = [r["dlac_i_cost"] - r["pf_cost"] for r in iteration_results],
        Cost_Increase_Pct = [(r["dlac_i_cost"] - r["pf_cost"]) / r["pf_cost"] * 100 for r in iteration_results]
    )
    
    # Add capacity columns
    G = length(generators)
    for g in 1:G
        iter_df[!, "$(generators[g].name)_Capacity"] = [r["capacities"][g] for r in iteration_results]
    end
    iter_df[!, "Battery_Power_Capacity"] = [r["battery_power"] for r in iteration_results]
    
    # Add margin columns
    for g in 1:G
        iter_df[!, "$(generators[g].name)_PF_Margin"] = [r["pf_margins"][g] for r in iteration_results]
        iter_df[!, "$(generators[g].name)_DLAC_i_Margin"] = [r["dlac_i_margins"][g] for r in iteration_results]
    end
    iter_df[!, "Battery_PF_Margin"] = [r["pf_margins"][G+1] for r in iteration_results]
    iter_df[!, "Battery_DLAC_i_Margin"] = [r["dlac_i_margins"][G+1] for r in iteration_results]
    
    CSV.write(joinpath(output_dir, "fixed_point_iteration_pf_vs_dlac_i.csv"), iter_df)
    
    println("Fixed point iteration results saved to: $(joinpath(output_dir, "fixed_point_iteration_pf_vs_dlac_i.csv"))")
end

# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================

function main()
    println("="^80)
    println("3-GENERATOR TOY SYSTEM: PF vs DLAC-i ANALYSIS WITH FIXED POINT ITERATION")
    println("="^80)
    
    # Step 1: Solve complete system and generate all output files
    println("Step 1: Solving complete system and generating output files...")
    system_results = solve_complete_system("results")
    
    # Step 2: Run fixed point iteration comparing PF and DLAC-i
    println("\nStep 2: Running fixed point iteration (PF vs DLAC-i)...")
    fixed_point_results = fixed_point_iteration_pf_vs_dlac_i(system_results, 
                                                             max_iterations=15,
                                                             tolerance=500.0,  # $500/MW tolerance
                                                             step_size=0.02,
                                                             output_dir="results")
    
    println("\n" * "="^80)
    println("ANALYSIS COMPLETE")
    println("="^80)
    println("All results saved to: results/")
    println("\nKey files:")
    println("  - capacity_expansion_results.csv (capacities optimized for actuals)")
    println("  - perfect_foresight_operations.csv (PF operating on actuals)")
    println("  - dlac_i_operations.csv (DLAC-i: actual operations, mean forecasts)")
    println("  - pf_vs_dlac_i_comparison.csv (profit margin comparison)")
    println("  - forecast_quality_analysis.csv (forecast errors)")
    println("  - fixed_point_iteration_pf_vs_dlac_i.csv (convergence tracking)")
    
    println("\nModel Structure:")
    println("  - Capacity Expansion: Optimized for actual deterministic profiles")
    println("  - Perfect Foresight: Operates on actual deterministic profiles")
    println("  - DLAC-i: Operates on actuals, forecasts using mean of 3 scenarios")
    println("  - Fixed Point: Finds equilibrium where DLAC-i policy breaks even")
    
    if fixed_point_results["converged"]
        println("\nFixed point iteration CONVERGED!")
        println("This shows the capacity equilibrium under DLAC-i operational policy")
        println("(accounting for forecast uncertainty in operational decisions)")
    else
        println("\nFixed point iteration did not converge within iteration limit")
        println("Check fixed_point_iteration_pf_vs_dlac_i.csv to analyze convergence behavior")
    end
    
    # Summary of key differences
    println("\n" * "="^60)
    println("KEY INSIGHTS:")
    println("="^60)
    
    # Show the impact of forecast uncertainty
    pf_cost = system_results["perfect_foresight"]["total_cost"]
    dlac_i_cost = system_results["dlac_i"]["total_cost"]
    cost_increase = ((dlac_i_cost - pf_cost) / pf_cost) * 100
    
    println("Cost of Forecast Uncertainty:")
    println("  Perfect Foresight Cost: $(round(pf_cost, digits=0))")
    println("  DLAC-i Cost: $(round(dlac_i_cost, digits=0))")
    println("  Cost Increase: $(round(cost_increase, digits=2))%")
    
    # Show forecast quality
    forecast_demand_error = mean(abs.([system_results["actual_demand"][t] - 
        mean([system_results["demand_scenarios"][s][t] for s in 1:3]) for t in 1:length(system_results["actual_demand"])]))
    forecast_wind_error = mean(abs.([system_results["actual_wind"][t] - 
        mean([system_results["wind_scenarios"][s][t] for s in 1:3]) for t in 1:length(system_results["actual_wind"])]))
    
    println("\nForecast Quality (Mean Absolute Error):")
    println("  Demand Forecast Error: $(round(forecast_demand_error, digits=1)) MW")
    println("  Wind Forecast Error: $(round(forecast_wind_error, digits=3))")
    
    return system_results, fixed_point_results
end

system_results = solve_complete_system("results")
using Plots, Statistics

function create_price_analysis_plots(cap_result, pf_result, actual_demand, output_dir="results")
    """
    Create comprehensive price analysis plots comparing capacity expansion vs perfect foresight prices
    """
    
    println("Creating price analysis plots...")
    
    # Extract prices
    cap_prices = cap_result["prices"]
    pf_prices = pf_result["prices"]
    T = length(cap_prices)
    
    # Basic statistics
    println("\nPrice Statistics:")
    println("  Capacity Expansion:")
    println("    Mean: \$$(round(mean(cap_prices), digits=2))/MWh")
    println("    Min: \$$(round(minimum(cap_prices), digits=2))/MWh") 
    println("    Max: \$$(round(maximum(cap_prices), digits=2))/MWh")
    println("    Std: \$$(round(std(cap_prices), digits=2))/MWh")
    
    println("  Perfect Foresight:")
    println("    Mean: \$$(round(mean(pf_prices), digits=2))/MWh")
    println("    Min: \$$(round(minimum(pf_prices), digits=2))/MWh")
    println("    Max: \$$(round(maximum(pf_prices), digits=2))/MWh") 
    println("    Std: \$$(round(std(pf_prices), digits=2))/MWh")
    
    # Create multiple plots
    
    # Plot 1: Time series comparison (first week)
    sample_hours = 1:min(168, T)  # First week
    p1 = plot(sample_hours, cap_prices[sample_hours], 
              label="Capacity Expansion", linewidth=2, color=:blue,
              title="Price Comparison - First Week",
              xlabel="Hour", ylabel="Price (\$/MWh)",
              legend=:topright)
    plot!(p1, sample_hours, pf_prices[sample_hours], 
          label="Perfect Foresight", linewidth=2, color=:red, linestyle=:dash)
    
    # Plot 2: Full time series (every 6th hour for visibility)
    sample_full = 1:6:T
    p2 = plot(sample_full, cap_prices[sample_full], 
              label="Capacity Expansion", linewidth=1, color=:blue, alpha=0.8,
              title="Price Comparison - Full Period (Every 6th Hour)",
              xlabel="Hour", ylabel="Price (\$/MWh)",
              legend=:topright)
    plot!(p2, sample_full, pf_prices[sample_full], 
          label="Perfect Foresight", linewidth=1, color=:red, alpha=0.8, linestyle=:dash)
    
    # Plot 3: Scatter plot correlation
    p3 = scatter(cap_prices, pf_prices, 
                alpha=0.5, markersize=2, color=:darkgreen,
                title="Price Correlation",
                xlabel="Capacity Expansion Price (\$/MWh)", 
                ylabel="Perfect Foresight Price (\$/MWh)",
                legend=false)
    
    # Add 45-degree line
    max_price = max(maximum(cap_prices), maximum(pf_prices))
    plot!(p3, [0, max_price], [0, max_price], 
          color=:red, linestyle=:dash, linewidth=2, label="45° line")
    
    # Calculate correlation
    correlation = cor(cap_prices, pf_prices)
    annotate!(p3, max_price*0.1, max_price*0.9, text("r = $(round(correlation, digits=3))", 10))
    
    # Plot 4: Price difference over time
    price_diff = pf_prices .- cap_prices
    p4 = plot(1:T, price_diff, 
              linewidth=1, color=:purple,
              title="Price Difference (PF - CapExp) Over Time",
              xlabel="Hour", ylabel="Price Difference (\$/MWh)",
              legend=false)
    hline!(p4, [0], color=:black, linestyle=:dot, alpha=0.5)
    
    # Plot 5: Price histograms
    p5 = histogram(cap_prices, bins=50, alpha=0.6, color=:blue, 
                   label="Capacity Expansion", normalize=:probability,
                   title="Price Distribution Comparison",
                   xlabel="Price (\$/MWh)", ylabel="Probability")
    histogram!(p5, pf_prices, bins=50, alpha=0.6, color=:red, 
               label="Perfect Foresight", normalize=:probability)
    
    # Plot 6: Price vs Demand relationship
    p6 = scatter(actual_demand, cap_prices, alpha=0.5, markersize=2, color=:blue,
                title="Price vs Demand Relationship", 
                xlabel="Demand (MW)", ylabel="Price (\$/MWh)",
                label="Capacity Expansion")
    scatter!(p6, actual_demand, pf_prices, alpha=0.5, markersize=2, color=:red,
             label="Perfect Foresight")
    
    # Combine all plots
    combined_plot = plot(p1, p2, p3, p4, p5, p6, 
                        layout=(3, 2), size=(1200, 900),
                        plot_title="Comprehensive Price Analysis: Capacity Expansion vs Perfect Foresight")
    
    # Save the combined plot
    mkpath(output_dir)
    savefig(combined_plot, joinpath(output_dir, "comprehensive_price_analysis.png"))
    
    # Create detailed analysis plots
    
    # High price periods analysis
    high_price_threshold = 200  # $/MWh
    cap_high_hours = findall(cap_prices .> high_price_threshold)
    pf_high_hours = findall(pf_prices .> high_price_threshold)
    
    println("\nHigh Price Analysis (>$(high_price_threshold)/MWh):")
    println("  Capacity Expansion: $(length(cap_high_hours)) hours")
    println("  Perfect Foresight: $(length(pf_high_hours)) hours")
    
    # Price duration curves
    cap_sorted = sort(cap_prices, rev=true)
    pf_sorted = sort(pf_prices, rev=true)
    
    p_duration = plot(1:T, cap_sorted, 
                     label="Capacity Expansion", linewidth=2, color=:blue,
                     title="Price Duration Curves",
                     xlabel="Hours (sorted by price)", ylabel="Price (\$/MWh)",
                     yscale=:log10)
    plot!(p_duration, 1:T, pf_sorted,
          label="Perfect Foresight", linewidth=2, color=:red, linestyle=:dash)
    
    # Save duration curve separately
    savefig(p_duration, joinpath(output_dir, "price_duration_curves.png"))
    
    # Scarcity analysis - price vs capacity margin
    total_available_capacity = zeros(T)
    nuclear_availability = get(cap_result, "nuclear_availability", ones(T))
    gas_availability = get(cap_result, "gas_availability", ones(T))
    
    # Assuming generator order: Nuclear, Wind, Gas
    capacities = cap_result["capacity"]
    actual_wind = get(cap_result, "actual_wind", ones(T) * 0.3)  # fallback
    battery_power_cap = cap_result["battery_power_cap"]
    
    for t in 1:T
        nuclear_avail = capacities[1] * nuclear_availability[t]
        wind_avail = capacities[2] * actual_wind[t]
        gas_avail = capacities[3] * gas_availability[t] 
        total_available_capacity[t] = nuclear_avail + wind_avail + gas_avail + battery_power_cap
    end
    
    capacity_margin = total_available_capacity .- actual_demand
    
    p_scarcity = scatter(capacity_margin, cap_prices, alpha=0.6, markersize=2, color=:blue,
                        title="Price vs Capacity Margin",
                        xlabel="Capacity Margin (MW)", ylabel="Price (\$/MWh)",
                        label="Capacity Expansion", yscale=:log10)
    scatter!(p_scarcity, capacity_margin, pf_prices, alpha=0.6, markersize=2, color=:red,
             label="Perfect Foresight")
    
    # Save scarcity plot
    savefig(p_scarcity, joinpath(output_dir, "price_vs_capacity_margin.png"))
    
end

# Example usage function
function analyze_price_differences(system_results, output_dir="results")
    """
    Wrapper function to analyze price differences using system results
    """
    
    cap_result = system_results["capacity_expansion"]
    pf_result = system_results["perfect_foresight"] 
    actual_demand = system_results["actual_demand"]
    
    return create_price_analysis_plots(cap_result, pf_result, actual_demand, output_dir)
end
analyze_price_differences(system_results, "results")
# Run the complete analysis
# system_results, fixed_point_results = main()
