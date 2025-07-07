"""
OptimizationModels.jl

Three core optimization models for ToySystemQuad.jl:
1. Capacity Expansion Model (CEM) - Joint investment and operations optimization
2. Perfect Foresight Operations (DLAC-p) - One-shot perfect foresight operations with fixed capacities
3. DLAC-i Operations - Rolling horizon with imperfect information using mean forecasts
"""

module OptimizationModels

using JuMP, Gurobi, LinearAlgebra, CSV, DataFrames, Statistics
using ..SystemConfig: Generator, Battery, SystemParameters, SystemProfiles, get_default_system_parameters

export solve_capacity_expansion_model, solve_perfect_foresight_operations, solve_dlac_i_operations, solve_slac_operations
export save_operational_results, calculate_profits_and_save, compute_pmr

# =============================================================================
# 1. CAPACITY EXPANSION MODEL (CEM)
# =============================================================================

"""
    solve_capacity_expansion_model(generators, battery; params=nothing, output_dir="results")

Solve joint capacity expansion and operations optimization with perfect foresight.
Minimizes total annualized costs: investment + fixed O&M + operational costs.

Mathematical formulation:
min: Σ(c^inv_n * y_n) + Σ(c^fix_n * y_n) + Σ_t Σ_n (c^op_n * p_n,t) + penalty * δ^d_t

Subject to:
- Power balance: Σ_n p_n,t - Σ_s (p^ch_s,t - p^dis_s,t) + δ^d_t = d_t  ∀t
- Generation limits: p_n,t ≤ y_n * a_n,t  ∀n,t  (with availability factors)
- Storage constraints: SOC dynamics, power limits
- Capacity bounds: y_n ≥ 0
"""
function solve_capacity_expansion_model(generators, battery, profiles::SystemProfiles; output_dir="results")
    params = profiles.params
    
    # Get actual profiles (deterministic for CEM)
    actual_demand = profiles.actual_demand
    actual_wind = profiles.actual_wind
    nuclear_availability = profiles.actual_nuclear_availability
    gas_availability = profiles.actual_gas_availability
    
    T = params.hours
    G = length(generators)
    
    println("Solving Capacity Expansion Model (CEM) for $T hours")
    println("  Nuclear availability: $(round(mean(nuclear_availability)*100, digits=1))%")
    println("  Gas availability: $(round(mean(gas_availability)*100, digits=1))%")
    
    # Create optimization model
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    # Decision variables
    @variable(model, y[1:G] >= 0)  # Generator capacities
    @variable(model, y_bat_power >= 0)  # Battery power capacity
    @variable(model, y_bat_energy >= 0)  # Battery energy capacity
    
    # Operational variables
    @variable(model, p[1:G, 1:T] >= 0)  # Generation for fixed demand
    @variable(model, p_flex[1:G, 1:T] >= 0)  # Generation for flexible demand
    @variable(model, p_ch[1:T] >= 0)  # Battery charging
    @variable(model, p_dis[1:T] >= 0)  # Battery discharging
    @variable(model, soc[1:T] >= 0)  # Battery state of charge
    @variable(model, δ_d_fixed[1:T] >= 0)  # Fixed demand load shedding
    @variable(model, δ_d_flex[1:T] >= 0)  # Flexible demand load shedding
    
    # Objective: Total annualized costs
    investment_cost = sum(generators[g].inv_cost * y[g] for g in 1:G) + 
                     battery.inv_cost_power * y_bat_power + 
                     battery.inv_cost_energy * y_bat_energy
                     
    fixed_cost = sum(generators[g].fixed_om_cost * y[g] for g in 1:G) + 
                 battery.fixed_om_cost * y_bat_power
                 
    operational_cost = sum(
        sum((generators[g].fuel_cost + generators[g].var_om_cost) * (p[g,t] + p_flex[g,t]) for g in 1:G) +
        battery.var_om_cost * (p_ch[t] + p_dis[t]) +
        params.load_shed_penalty * (δ_d_fixed[t] + 0.5 * δ_d_flex[t]^2 / params.flex_demand_mw)
        for t in 1:T)
    
    @objective(model, Min, investment_cost + fixed_cost + operational_cost)
    
    # Power balance constraints
    @constraint(model, power_balance[t=1:T],
        sum(p[g,t] for g in 1:G) + p_dis[t] - p_ch[t] + sum(p_flex[g,t] for g in 1:G) + δ_d_flex[t] + δ_d_fixed[t] == actual_demand[t] + params.flex_demand_mw)
    
    # Flexible demand constraint: total flexible generation + shedding = available flexible demand
    @constraint(model, [t=1:T], sum(p_flex[g,t] for g in 1:G) + δ_d_flex[t] == params.flex_demand_mw)
    
    # Generation limits with availability factors
    for g in 1:G
        if generators[g].name == "Nuclear"
            @constraint(model, [t=1:T], p[g,t] + p_flex[g,t] <= y[g] * nuclear_availability[t])
        elseif generators[g].name == "Wind"
            @constraint(model, [t=1:T], p[g,t] + p_flex[g,t] <= y[g] * actual_wind[t])
        elseif generators[g].name == "Gas"
            @constraint(model, [t=1:T], p[g,t] + p_flex[g,t] <= y[g] * gas_availability[t])
        else
            @constraint(model, [t=1:T], p[g,t] + p_flex[g,t] <= y[g])
        end
    end
    
    # Battery constraints
    @constraint(model, [t=1:T], p_ch[t] <= y_bat_power)
    @constraint(model, [t=1:T], p_dis[t] <= y_bat_power)
    @constraint(model, [t=1:T], soc[t] <= y_bat_energy)
    
    # Battery SOC dynamics
    @constraint(model, soc[1] == y_bat_energy * 0.5 + 
        battery.efficiency_charge * p_ch[1] - p_dis[1]/battery.efficiency_discharge)
    @constraint(model, [t=2:T], soc[t] == soc[t-1] + 
        battery.efficiency_charge * p_ch[t] - p_dis[t]/battery.efficiency_discharge)
    
    # Battery energy/power ratio and boundary conditions
    @constraint(model, y_bat_energy <= y_bat_power * battery.duration)
    @constraint(model, soc[T] >= y_bat_energy * 0.4)
    @constraint(model, soc[T] <= y_bat_energy * 0.6)
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        result = Dict(
            "status" => "optimal",
            "model_type" => "CEM",
            "capacity" => value.(y),
            "battery_power_cap" => value(y_bat_power),
            "battery_energy_cap" => value(y_bat_energy),
            "generation" => value.(p),
            "generation_flex" => value.(p_flex),
            "battery_charge" => value.(p_ch),
            "battery_discharge" => value.(p_dis),
            "battery_soc" => value.(soc),
            "load_shed" => value.(δ_d_fixed) + value.(δ_d_flex),
            "load_shed_fixed" => value.(δ_d_fixed),
            "load_shed_flex" => value.(δ_d_flex),
            "commitment" => ones(G, T),  # No UC constraints for simplicity
            "startup" => zeros(G, T),
            "total_cost" => objective_value(model),
            "investment_cost" => value(investment_cost),
            "fixed_cost" => value(fixed_cost),
            "operational_cost" => value(operational_cost),
            "prices" => dual.(power_balance),
            "nuclear_availability" => nuclear_availability,
            "gas_availability" => gas_availability,
            "demand_used" => actual_demand,
            "wind_used" => actual_wind
        )
        
        # Save results
        save_capacity_results(result, generators, battery, output_dir)
        save_operational_results(result, generators, battery, "capacity_expansion", output_dir)
        
        return result
    else
        return Dict("status" => "infeasible", "termination_status" => termination_status(model))
    end
end

# =============================================================================
# 2. PERFECT FORESIGHT OPERATIONS (DLAC-p)
# =============================================================================

"""
    solve_perfect_foresight_operations(generators, battery, capacities, battery_power_cap, battery_energy_cap;
                                       params=nothing, output_dir="results")

Solve operations with perfect foresight using FIXED capacities from capacity expansion.
This is a single optimization over the entire time horizon with perfect information.

Mathematical formulation:
min: Σ_t Σ_n (c^op_n * p_n,t) + penalty * δ^d_t

Subject to:
- Power balance: Σ_n p_n,t - Σ_s (p^ch_s,t - p^dis_s,t) + δ^d_t = d_t  ∀t
- Generation limits: p_n,t ≤ y_n * a_n,t  ∀n,t  (capacities y_n are FIXED)
- Storage constraints: SOC dynamics, power limits
"""
function solve_perfect_foresight_operations(generators, battery, capacities, battery_power_cap, battery_energy_cap,
                                           profiles::SystemProfiles; output_dir="results")
    params = profiles.params
    
    # Get the same profiles as used in capacity expansion
    actual_demand = profiles.actual_demand
    actual_wind = profiles.actual_wind
    nuclear_availability = profiles.actual_nuclear_availability
    gas_availability = profiles.actual_gas_availability
    
    T = params.hours
    G = length(generators)
    
    println("Solving Perfect Foresight Operations (DLAC-p) for $T hours")
    println("  Using FIXED capacities from capacity expansion")
    println("  Nuclear availability: $(round(mean(nuclear_availability)*100, digits=1))%")
    println("  Gas availability: $(round(mean(gas_availability)*100, digits=1))%")
    
    # Create optimization model
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    # Operational variables only (capacities are FIXED parameters)
    @variable(model, p[1:G, 1:T] >= 0)  # Generation for fixed demand
    @variable(model, p_flex[1:G, 1:T] >= 0)  # Generation for flexible demand
    @variable(model, p_ch[1:T] >= 0)  # Battery charging
    @variable(model, p_dis[1:T] >= 0)  # Battery discharging
    @variable(model, soc[1:T] >= 0)  # Battery state of charge
    @variable(model, δ_d_fixed[1:T] >= 0)  # Fixed demand load shedding
    @variable(model, δ_d_flex[1:T] >= 0)  # Flexible demand load shedding
    
    # Objective: Operational costs only
    @objective(model, Min, 
        sum(sum((generators[g].fuel_cost + generators[g].var_om_cost) * (p[g,t] + p_flex[g,t]) for g in 1:G) +
            battery.var_om_cost * (p_ch[t] + p_dis[t]) +
            params.load_shed_penalty * (δ_d_fixed[t] + 0.5 * δ_d_flex[t]^2 / params.flex_demand_mw)
            for t in 1:T))
    
    # Power balance constraints
    @constraint(model, power_balance[t=1:T],
        sum(p[g,t] for g in 1:G) + p_dis[t] - p_ch[t] + sum(p_flex[g,t] for g in 1:G) + δ_d_flex[t] + δ_d_fixed[t] == actual_demand[t] + params.flex_demand_mw)
    
    # Flexible demand constraint: total flexible generation + shedding = available flexible demand
    @constraint(model, [t=1:T], sum(p_flex[g,t] for g in 1:G) + δ_d_flex[t] == params.flex_demand_mw)
    
    # Generation limits with FIXED capacities and availability factors
    for g in 1:G
        if generators[g].name == "Nuclear"
            @constraint(model, [t=1:T], p[g,t] + p_flex[g,t] <= capacities[g] * nuclear_availability[t])
        elseif generators[g].name == "Wind"
            @constraint(model, [t=1:T], p[g,t] + p_flex[g,t] <= capacities[g] * actual_wind[t])
        elseif generators[g].name == "Gas"
            @constraint(model, [t=1:T], p[g,t] + p_flex[g,t] <= capacities[g] * gas_availability[t])
        else
            @constraint(model, [t=1:T], p[g,t] + p_flex[g,t] <= capacities[g])
        end
    end
    
    # Battery constraints with FIXED capacities
    @constraint(model, [t=1:T], p_ch[t] <= battery_power_cap)
    @constraint(model, [t=1:T], p_dis[t] <= battery_power_cap)
    @constraint(model, [t=1:T], soc[t] <= battery_energy_cap)
    
    # Battery SOC dynamics
    @constraint(model, soc[1] == battery_energy_cap * 0.5 + 
        battery.efficiency_charge * p_ch[1] - p_dis[1]/battery.efficiency_discharge)
    @constraint(model, [t=2:T], soc[t] == soc[t-1] + 
        battery.efficiency_charge * p_ch[t] - p_dis[t]/battery.efficiency_discharge)
    
    # Boundary conditions
    @constraint(model, soc[T] >= battery_energy_cap * 0.4)
    @constraint(model, soc[T] <= battery_energy_cap * 0.6)
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        result = Dict(
            "status" => "optimal",
            "model_type" => "DLAC-p",
            "generation" => value.(p),
            "generation_flex" => value.(p_flex),
            "battery_charge" => value.(p_ch),
            "battery_discharge" => value.(p_dis),
            "battery_soc" => value.(soc),
            "load_shed" => value.(δ_d_fixed) + value.(δ_d_flex),
            "load_shed_fixed" => value.(δ_d_fixed),
            "load_shed_flex" => value.(δ_d_flex),
            "commitment" => ones(G, T),
            "startup" => zeros(G, T),
            "total_cost" => objective_value(model),
            "prices" => dual.(power_balance),
            "nuclear_availability" => nuclear_availability,
            "gas_availability" => gas_availability
        )
        
        # Save operational results
        save_operational_results(result, generators, battery, "perfect_foresight", output_dir)
        
        # Print comparison with CEM if available
        println("Perfect Foresight Operations Summary:")
        println("  Total operational cost: \$$(round(objective_value(model), digits=0))")
        println("  Total load shed: $(round(sum(value.(δ_d_fixed) + value.(δ_d_flex)), digits=1)) MWh")
        println("  Maximum price: \$$(round(maximum(dual.(power_balance)), digits=2))/MWh")
        
        return result
    else
        return Dict("status" => "infeasible", "termination_status" => termination_status(model))
    end
end

# =============================================================================
# 3. DLAC-I OPERATIONS (Deterministic Look-Ahead with Imperfect Information)
# =============================================================================

"""
    solve_dlac_i_operations(generators, battery, capacities, battery_power_cap, battery_energy_cap;
                            lookahead_hours=24, params=nothing, output_dir="results")

Solve operations using DLAC-i policy with rolling horizon optimization.
At each time t, solves a lookahead problem using:
- ACTUAL demand/wind/outages for period t (current)
- MEAN forecast from scenarios for periods t+1 to t+H

Mathematical formulation at time t:
min: Σ_{t'∈T^H_t} Σ_n (c^op_n * p̃_n,t,t') + penalty * δ̃^d_t,t'

Subject to:
- Power balance: Σ_n p̃_n,t,t' - Σ_s (p̃^ch_s,t,t' - p̃^dis_s,t,t') + δ̃^d_t,t' = d_t,t'  ∀t'∈T^H_t
- Generation limits: p̃_n,t,t' ≤ y_n * ã_n,t,t'  ∀n,t'  (capacities FIXED)
- Storage constraints: SOC dynamics, power limits
- Non-anticipativity: p̃_n,t,t = p_n,t (first period binding)

Where d_t,t' and ã_n,t,t' are actual values for t'=t, mean forecasts for t'>t.
"""
function solve_dlac_i_operations(generators, battery, capacities, battery_power_cap, battery_energy_cap,
                                 profiles::SystemProfiles; lookahead_hours=24, output_dir="results")
    params = profiles.params
    
    # Get actual profiles and scenarios for forecasting
    actual_demand = profiles.actual_demand
    actual_wind = profiles.actual_wind
    nuclear_availability = profiles.actual_nuclear_availability
    gas_availability = profiles.actual_gas_availability
    demand_scenarios = profiles.demand_scenarios
    wind_scenarios = profiles.wind_scenarios
    nuclear_avail_scenarios = profiles.nuclear_availability_scenarios
    gas_avail_scenarios = profiles.gas_availability_scenarios
    
    T = params.hours
    G = length(generators)
    S = length(demand_scenarios)
    
    # Compute mean forecasts from scenarios
    mean_demand_forecast = [mean([demand_scenarios[s][t] for s in 1:S]) for t in 1:T]
    mean_wind_forecast = [mean([wind_scenarios[s][t] for s in 1:S]) for t in 1:T]
    mean_nuclear_avail_forecast = [mean([nuclear_avail_scenarios[s][t] for s in 1:S]) for t in 1:T]
    mean_gas_avail_forecast = [mean([gas_avail_scenarios[s][t] for s in 1:S]) for t in 1:T]
    
    println("Solving DLAC-i Operations with $(lookahead_hours)-hour lookahead for $T hours")
    println("  Using actual values for current period, mean forecasts for lookahead")
    println("  Forecast accuracy: Demand $(round(cor(actual_demand, mean_demand_forecast), digits=3)), Wind $(round(cor(actual_wind, mean_wind_forecast), digits=3))")
    
    # Initialize result storage
    generation_schedule = zeros(G, T)
    generation_flex_schedule = zeros(G, T)
    battery_charge_schedule = zeros(T)
    battery_discharge_schedule = zeros(T)
    battery_soc_schedule = zeros(T)
    load_shed_schedule = zeros(T)
    load_shed_fixed_schedule = zeros(T)
    load_shed_flex_schedule = zeros(T)
    prices = zeros(T)
    
    # State tracking
    current_soc = battery_energy_cap * 0.5
    
    # Rolling horizon optimization
    for t in 1:T
        if t % 100 == 0
            println("  Processing hour $t/$T")
        end
        
        # Determine lookahead horizon
        horizon_end = min(t + lookahead_hours - 1, T)
        horizon = t:horizon_end
        H = length(horizon)
        
        # Create optimization model for current horizon
        model = Model(Gurobi.Optimizer)
        set_silent(model)
        
        # Here-and-now variables for current period decisions (t=1 only)
        @variable(model, x_p[1:G] >= 0)                 # Generation for fixed demand (current period)
        @variable(model, x_p_flex[1:G] >= 0)            # Generation for flexible demand (current period)
        @variable(model, x_p_ch >= 0)                   # Battery charging (current period)
        @variable(model, x_p_dis >= 0)                  # Battery discharging (current period)
        @variable(model, x_soc >= 0)                    # Battery SOC (current period)
        @variable(model, x_δ_d_fixed >= 0)              # Fixed demand load shedding (current period)
        @variable(model, x_δ_d_flex >= 0)               # Flexible demand load shedding (current period)
        
        # Decision variables for lookahead horizon
        @variable(model, p̃[1:G, 1:H] >= 0)  # Generation for fixed demand
        @variable(model, p̃_flex[1:G, 1:H] >= 0)  # Generation for flexible demand
        @variable(model, p̃_ch[1:H] >= 0)    # Battery charging
        @variable(model, p̃_dis[1:H] >= 0)   # Battery discharging
        @variable(model, s̃oc[1:H] >= 0)     # Battery SOC
        @variable(model, δ̃_d_fixed[1:H] >= 0)  # Fixed demand load shedding
        @variable(model, δ̃_d_flex[1:H] >= 0)   # Flexible demand load shedding
        
        # Non-anticipativity constraints linking here-and-now to first period variables
        @constraint(model, [g=1:G], p̃[g, 1] == x_p[g])
        @constraint(model, [g=1:G], p̃_flex[g, 1] == x_p_flex[g])
        @constraint(model, p̃_ch[1] == x_p_ch)
        @constraint(model, p̃_dis[1] == x_p_dis)
        @constraint(model, s̃oc[1] == x_soc)
        @constraint(model, δ̃_d_fixed[1] == x_δ_d_fixed)
        @constraint(model, δ̃_d_flex[1] == x_δ_d_flex)
        
        # Objective: Minimize cost over lookahead horizon (all periods)
        @objective(model, Min, 
            sum(sum((generators[g].fuel_cost + generators[g].var_om_cost) * (p̃[g,τ] + p̃_flex[g,τ]) for g in 1:G) +
                battery.var_om_cost * (p̃_ch[τ] + p̃_dis[τ]) +
                params.load_shed_penalty * (δ̃_d_fixed[τ] + 0.5 * δ̃_d_flex[τ]^2 / params.flex_demand_mw)
                for τ in 1:H))
        
        # Power balance constraints
        @constraint(model, power_balance_lookahead[τ=1:H],
            sum(p̃[g,τ] for g in 1:G) + p̃_dis[τ] - p̃_ch[τ] + sum(p̃_flex[g,τ] for g in 1:G) + δ̃_d_fixed[τ] + δ̃_d_flex[τ] == 
            (τ == 1 ? actual_demand[t] : mean_demand_forecast[horizon[τ]]) + params.flex_demand_mw)
        
        # Flexible demand constraint: total flexible generation + shedding = available flexible demand
        @constraint(model, [τ=1:H], sum(p̃_flex[g,τ] for g in 1:G) + δ̃_d_flex[τ] == params.flex_demand_mw)
        
        # Generation constraints with availability factors
        for g in 1:G
            if generators[g].name == "Nuclear"
                @constraint(model, [τ=1:H], p̃[g,τ] + p̃_flex[g,τ] <= capacities[g] * 
                    (τ == 1 ? nuclear_availability[t] : mean_nuclear_avail_forecast[horizon[τ]]))
            elseif generators[g].name == "Wind"
                @constraint(model, [τ=1:H], p̃[g,τ] + p̃_flex[g,τ] <= capacities[g] * 
                    (τ == 1 ? actual_wind[t] : mean_wind_forecast[horizon[τ]]))
            elseif generators[g].name == "Gas"
                @constraint(model, [τ=1:H], p̃[g,τ] + p̃_flex[g,τ] <= capacities[g] * 
                    (τ == 1 ? gas_availability[t] : mean_gas_avail_forecast[horizon[τ]]))
            else
                @constraint(model, [τ=1:H], p̃[g,τ] + p̃_flex[g,τ] <= capacities[g])
            end
        end
        
        # Battery constraints
        @constraint(model, [τ=1:H], p̃_ch[τ] <= battery_power_cap)
        @constraint(model, [τ=1:H], p̃_dis[τ] <= battery_power_cap)
        @constraint(model, [τ=1:H], s̃oc[τ] <= battery_energy_cap)
        
        # Battery SOC dynamics
        @constraint(model, s̃oc[1] == current_soc + 
            battery.efficiency_charge * p̃_ch[1] - p̃_dis[1]/battery.efficiency_discharge)
        @constraint(model, [τ=2:H], s̃oc[τ] == s̃oc[τ-1] + 
            battery.efficiency_charge * p̃_ch[τ] - p̃_dis[τ]/battery.efficiency_discharge)
        
        optimize!(model)
        
        if termination_status(model) == MOI.OPTIMAL
            # Store first-period decisions (use here-and-now variables)
            generation_schedule[:, t] = value.(x_p)
            generation_flex_schedule[:, t] = value.(x_p_flex)
            battery_charge_schedule[t] = value(x_p_ch)
            battery_discharge_schedule[t] = value(x_p_dis)
            battery_soc_schedule[t] = value(x_soc)
            load_shed_schedule[t] = value(x_δ_d_fixed) + value(x_δ_d_flex)
            load_shed_fixed_schedule[t] = value(x_δ_d_fixed)
            load_shed_flex_schedule[t] = value(x_δ_d_flex)
            prices[t] = dual(power_balance_lookahead[1])
            
            # Update SOC state for next iteration
            current_soc = value(x_soc)
        else
            println("Warning: DLAC-i optimization failed at hour $t")
            load_shed_schedule[t] = actual_demand[t]
            prices[t] = params.load_shed_penalty
        end
    end
    
    result = Dict(
        "status" => "optimal",
        "model_type" => "DLAC-i",
        "generation" => generation_schedule,
        "generation_flex" => generation_flex_schedule,
        "battery_charge" => battery_charge_schedule,
        "battery_discharge" => battery_discharge_schedule,
        "battery_soc" => battery_soc_schedule,
        "load_shed" => load_shed_schedule,
        "load_shed_fixed" => load_shed_fixed_schedule,
        "load_shed_flex" => load_shed_flex_schedule,
        "commitment" => ones(G, T),
        "startup" => zeros(G, T),
        "prices" => prices,
        "total_cost" => sum(sum((generators[g].fuel_cost + generators[g].var_om_cost) * 
                               (generation_schedule[g,t] + generation_flex_schedule[g,t]) for g in 1:G) + 
                               battery.var_om_cost * (battery_charge_schedule[t] + battery_discharge_schedule[t]) +
                               params.load_shed_penalty * (load_shed_fixed_schedule[t] + 0.5 * load_shed_flex_schedule[t]^2 / params.flex_demand_mw) for t in 1:T),
        "nuclear_availability" => nuclear_availability,
        "gas_availability" => gas_availability
    )
    
    # Save operational results
    save_operational_results(result, generators, battery, "dlac_i", output_dir)
    
    println("DLAC-i Operations Summary:")
    println("  Total operational cost: \$$(round(result["total_cost"], digits=0))")
    println("  Total load shed: $(round(sum(load_shed_schedule), digits=1)) MWh")
    println("  Maximum price: \$$(round(maximum(prices), digits=2))/MWh")
    
    return result
end

# =============================================================================
# RESULTS SAVING AND ANALYSIS
# =============================================================================

"""
    save_capacity_results(result, generators, battery, output_dir)

Save capacity expansion results to CSV files.
"""
function save_capacity_results(result, generators, battery, output_dir)
    mkpath(output_dir)
    G = length(generators)
    
    # Main capacity results
    capacity_df = DataFrame(
        Technology = [gen.name for gen in generators],
        Capacity_MW = result["capacity"],
        Investment_Cost = [generators[g].inv_cost * result["capacity"][g] for g in 1:G],
        Fixed_OM_Cost = [generators[g].fixed_om_cost * result["capacity"][g] for g in 1:G]
    )
    
    # Add battery rows
    push!(capacity_df, ("Battery_Power", result["battery_power_cap"], 
                       battery.inv_cost_power * result["battery_power_cap"],
                       battery.fixed_om_cost * result["battery_power_cap"]))
    push!(capacity_df, ("Battery_Energy", result["battery_energy_cap"],
                       battery.inv_cost_energy * result["battery_energy_cap"], 0.0))
    
    CSV.write(joinpath(output_dir, "capacity_expansion_results.csv"), capacity_df)
    
    return capacity_df
end

"""
    save_operational_results(results, generators, battery, model_name, output_dir)

Save detailed operational results to CSV files.
"""
function save_operational_results(results, generators, battery, model_name, output_dir)
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
    
    # Add load shedding breakdown if available
    if haskey(results, "load_shed_fixed")
        df[!, "Load_Shed_Fixed"] = results["load_shed_fixed"]
        df[!, "Load_Shed_Flex"] = results["load_shed_flex"]
    end
    
    # Add generation columns for fixed demand
    for g in 1:G
        df[!, "$(generators[g].name)_Generation"] = results["generation"][g, :]
    end
    
    # Add flexible generation columns if available
    if haskey(results, "generation_flex")
        for g in 1:G
            df[!, "$(generators[g].name)_Generation_Flex"] = results["generation_flex"][g, :]
        end
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
        
        # Add flexible generation if available
        if haskey(results, "generation_flex")
            total_gen_flex = sum(results["generation_flex"][g, :])
            push!(summary_df, ("$(generators[g].name)_Total_Generation_Flex_MWh", total_gen_flex))
        end
    end
    
    CSV.write(joinpath(output_dir, "$(model_name)_summary.csv"), summary_df)
end

"""
    calculate_profits_and_save(generators, battery, operational_results, capacities, 
                              battery_power_cap, battery_energy_cap, model_name, output_dir)

Calculate and save detailed profit analysis following π_n(y) formulation.
"""
function calculate_profits_and_save(generators, battery, operational_results, capacities, 
                                   battery_power_cap, battery_energy_cap, model_name, output_dir)
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
        Fixed_OM_Costs = Float64[],
        Investment_Costs = Float64[],
        Operating_Profit = Float64[],
        Net_Profit = Float64[],
        PMR_Percent = Float64[]
    )
    
    # Generator profits: π_n = Σ_t (λ_t * (p_n,t + p_flex_n,t) - c^op_n * (p_n,t + p_flex_n,t))
    for g in 1:G
        gen_name = generators[g].name
        capacity = capacities[g]
        total_gen = sum(operational_results["generation"][g, :])
        
        # Add flexible generation if available
        if haskey(operational_results, "generation_flex")
            total_gen += sum(operational_results["generation_flex"][g, :])
            total_gen_flex = sum(operational_results["generation_flex"][g, :])
        else
            total_gen_flex = 0.0
        end
        
        capacity_factor = capacity > 0 ? total_gen / (capacity * T) : 0.0
        
        # Revenue: Σ_t λ_t * (p_n,t + p_flex_n,t)
        energy_revenue = sum(operational_results["prices"][t] * operational_results["generation"][g,t] for t in 1:T)
        if haskey(operational_results, "generation_flex")
            energy_revenue += sum(operational_results["prices"][t] * operational_results["generation_flex"][g,t] for t in 1:T)
        end
        
        # Operating costs: Σ_t c^op_n * (p_n,t + p_flex_n,t)
        fuel_costs = sum(generators[g].fuel_cost * operational_results["generation"][g,t] for t in 1:T)
        vom_costs = sum(generators[g].var_om_cost * operational_results["generation"][g,t] for t in 1:T)
        if haskey(operational_results, "generation_flex")
            fuel_costs += sum(generators[g].fuel_cost * operational_results["generation_flex"][g,t] for t in 1:T)
            vom_costs += sum(generators[g].var_om_cost * operational_results["generation_flex"][g,t] for t in 1:T)
        end
        
        # Fixed costs
        fixed_om_costs = generators[g].fixed_om_cost * capacity
        investment_costs = generators[g].inv_cost * capacity
        
        # Profit calculation
        operating_profit = energy_revenue - fuel_costs - vom_costs
        net_profit = operating_profit - fixed_om_costs - investment_costs
        
        # PMR = (π_n / y_n) / c^inv_n * 100 = net_profit per unit capacity as % of investment cost
        pmr_percent = capacity > 0 ? (net_profit / capacity) / generators[g].inv_cost * 100 : 0.0
        
        push!(profit_df, (gen_name, capacity, total_gen, capacity_factor, energy_revenue,
                         fuel_costs, vom_costs, fixed_om_costs, investment_costs,
                         operating_profit, net_profit, pmr_percent))
    end
    
    # Battery profit calculation
    battery_energy_revenue = sum(operational_results["prices"][t] * operational_results["battery_discharge"][t] for t in 1:T)
    battery_energy_costs = sum(operational_results["prices"][t] * operational_results["battery_charge"][t] for t in 1:T)
    battery_net_energy_revenue = battery_energy_revenue - battery_energy_costs
    
    battery_vom_costs = sum(battery.var_om_cost * (operational_results["battery_charge"][t] + 
                           operational_results["battery_discharge"][t]) for t in 1:T)
    battery_fixed_costs = battery.fixed_om_cost * battery_power_cap
    battery_investment_costs = battery.inv_cost_power * battery_power_cap + battery.inv_cost_energy * battery_energy_cap
    
    battery_operating_profit = battery_net_energy_revenue - battery_vom_costs
    battery_net_profit = battery_operating_profit - battery_fixed_costs - battery_investment_costs
    
    total_discharge = sum(operational_results["battery_discharge"])
    battery_capacity_factor = battery_power_cap > 0 ? total_discharge / (battery_power_cap * T) : 0.0
    
    # Battery PMR as % of power investment cost
    battery_pmr_percent = battery_power_cap > 0 ? (battery_net_profit / battery_power_cap) / battery.inv_cost_power * 100 : 0.0
    
    push!(profit_df, ("Battery", battery_power_cap, total_discharge, battery_capacity_factor,
                     battery_energy_revenue, battery_energy_costs, battery_vom_costs,
                     battery_fixed_costs, battery_investment_costs, battery_operating_profit,
                     battery_net_profit, battery_pmr_percent))
    
    CSV.write(joinpath(output_dir, "$(model_name)_profits.csv"), profit_df)
    
    return profit_df
end

"""
    compute_pmr(operational_results, generators, battery, capacities, battery_power_cap, battery_energy_cap)

Compute Profit-to-Market-Rate (PMR) for all technologies.
PMR = (net_profit / total_fixed_costs) * 100

Returns array of PMR percentages: [gen1_pmr, gen2_pmr, gen3_pmr, battery_pmr]
"""
function compute_pmr(operational_results, generators, battery, capacities, battery_power_cap, battery_energy_cap)
    G = length(generators)
    T = length(operational_results["prices"])
    pmr = zeros(G + 1)  # +1 for battery
    
    # Generator PMRs
    for g in 1:G
        if capacities[g] > 1e-6  # Only compute for non-zero capacities
            # Revenue including flexible generation
            energy_revenue = sum(operational_results["prices"][t] * operational_results["generation"][g,t] for t in 1:T)
            if haskey(operational_results, "generation_flex")
                energy_revenue += sum(operational_results["prices"][t] * operational_results["generation_flex"][g,t] for t in 1:T)
            end
            
            # Costs including flexible generation
            fuel_costs = sum(generators[g].fuel_cost * operational_results["generation"][g,t] for t in 1:T)
            vom_costs = sum(generators[g].var_om_cost * operational_results["generation"][g,t] for t in 1:T)
            if haskey(operational_results, "generation_flex")
                fuel_costs += sum(generators[g].fuel_cost * operational_results["generation_flex"][g,t] for t in 1:T)
                vom_costs += sum(generators[g].var_om_cost * operational_results["generation_flex"][g,t] for t in 1:T)
            end
            # Note: startup costs not tracked in current implementation, assuming zero
            startup_costs = 0.0
            fixed_om_costs = generators[g].fixed_om_cost * capacities[g]
            investment_costs = generators[g].inv_cost * capacities[g]
            
            operating_profit = energy_revenue - fuel_costs - vom_costs - startup_costs
            net_profit = operating_profit - (investment_costs + fixed_om_costs)
            
            # PMR calculation
            total_fixed_costs = investment_costs + fixed_om_costs
            if total_fixed_costs > 1e-6
                pmr[g] = (net_profit / total_fixed_costs) * 100
            end
        end
    end
    
    # Battery PMR
    if battery_power_cap > 1e-6
        # Revenue from arbitrage
        battery_energy_revenue = sum(operational_results["prices"][t] * operational_results["battery_discharge"][t] for t in 1:T)
        battery_energy_costs = sum(operational_results["prices"][t] * operational_results["battery_charge"][t] for t in 1:T)
        battery_net_energy_revenue = battery_energy_revenue - battery_energy_costs
        
        # Costs
        battery_vom_costs = sum(battery.var_om_cost * (operational_results["battery_charge"][t] + 
                               operational_results["battery_discharge"][t]) for t in 1:T)
        battery_fixed_costs = battery.fixed_om_cost * battery_power_cap
        battery_investment_costs = battery.inv_cost_power * battery_power_cap + battery.inv_cost_energy * battery_energy_cap
        
        battery_operating_profit = battery_net_energy_revenue - battery_vom_costs
        battery_net_profit = battery_operating_profit - (battery_investment_costs + battery_fixed_costs)
        
        # Battery PMR
        battery_total_fixed_costs = battery_investment_costs + battery_fixed_costs
        if battery_total_fixed_costs > 1e-6
            pmr[G + 1] = (battery_net_profit / battery_total_fixed_costs) * 100
        end
    end
    
    return pmr
end

# =============================================================================
# 4. SLAC OPERATIONS (Stochastic LAC)
# =============================================================================

"""
    solve_slac_operations(generators, battery, capacities, battery_power_cap, battery_energy_cap;
                         lookahead_hours=24, output_dir="results", scenario_weights=nothing)

Solve SLAC operations with stochastic scenario-based optimization instead of mean forecasts.
SLAC optimizes over scenarios with probabilistic weights rather than taking means like DLAC-i.

Args:
- scenario_weights: Vector of weights for each scenario (default: equal weights)
- Other args same as DLAC-i
"""
function solve_slac_operations(generators, battery, capacities, battery_power_cap, battery_energy_cap,
                              profiles::SystemProfiles; lookahead_hours=24, output_dir="results", 
                              scenario_weights=nothing)
    params = profiles.params
    
    # Get actual profiles and scenarios
    actual_demand = profiles.actual_demand
    actual_wind = profiles.actual_wind
    nuclear_availability = profiles.actual_nuclear_availability
    gas_availability = profiles.actual_gas_availability
    demand_scenarios = profiles.demand_scenarios
    wind_scenarios = profiles.wind_scenarios
    nuclear_avail_scenarios = profiles.nuclear_availability_scenarios
    gas_avail_scenarios = profiles.gas_availability_scenarios
    
    T = params.hours
    G = length(generators)
    S = length(demand_scenarios)
    
    # Set default equal weights if not provided
    if scenario_weights === nothing
        scenario_weights = fill(1.0/S, S)
    else
        # Normalize weights to sum to 1
        scenario_weights = scenario_weights ./ sum(scenario_weights)
    end
    
    println("Solving SLAC Operations with $(lookahead_hours)-hour lookahead for $T hours")
    println("  Using actual values for current period, scenario optimization for lookahead")
    println("  Number of scenarios: $S, weights: $(round.(scenario_weights, digits=3))")
    
    # Initialize result storage
    generation_schedule = zeros(G, T)
    generation_flex_schedule = zeros(G, T)
    battery_charge_schedule = zeros(T)
    battery_discharge_schedule = zeros(T)
    battery_soc_schedule = zeros(T)
    load_shed_schedule = zeros(T)
    load_shed_fixed_schedule = zeros(T)
    load_shed_flex_schedule = zeros(T)
    prices = zeros(T)
    
    # State tracking
    current_soc = battery_energy_cap * 0.5
    
    # Rolling horizon optimization
    for t in 1:T
        if t % 100 == 0
            println("  Processing hour $t/$T")
        end
        
        # Determine lookahead horizon
        horizon_end = min(t + lookahead_hours - 1, T)
        horizon = t:horizon_end
        H = length(horizon)
        
        # Create optimization model for current horizon with scenario constraints
        model = Model(Gurobi.Optimizer)
        set_silent(model)
        
        # Here-and-now variables for current period decisions (t=1 only)
        @variable(model, x_p[1:G] >= 0)                 # Generation for fixed demand (current period)
        @variable(model, x_p_flex[1:G] >= 0)            # Generation for flexible demand (current period)
        @variable(model, x_p_ch >= 0)                   # Battery charging (current period)
        @variable(model, x_p_dis >= 0)                  # Battery discharging (current period)
        @variable(model, x_soc >= 0)                    # Battery SOC (current period)
        @variable(model, x_δ_d_fixed >= 0)              # Fixed demand load shedding (current period)
        @variable(model, x_δ_d_flex >= 0)               # Flexible demand load shedding (current period)
        
        # Scenario-specific variables for ALL time periods (1:H)
        @variable(model, p̃[1:G, 1:H, 1:S] >= 0)        # Generation for fixed demand by scenario
        @variable(model, p̃_flex[1:G, 1:H, 1:S] >= 0)   # Generation for flexible demand by scenario
        @variable(model, p̃_ch[1:H, 1:S] >= 0)           # Battery charging by scenario
        @variable(model, p̃_dis[1:H, 1:S] >= 0)          # Battery discharging by scenario
        @variable(model, s̃oc[1:H, 1:S] >= 0)            # Battery SOC by scenario
        @variable(model, δ̃_d_fixed[1:H, 1:S] >= 0)       # Fixed demand load shedding by scenario
        @variable(model, δ̃_d_flex[1:H, 1:S] >= 0)        # Flexible demand load shedding by scenario
        
        # Non-anticipativity constraints linking here-and-now to scenario variables
        @constraint(model, [g=1:G, s=1:S], p̃[g, 1, s] == x_p[g])
        @constraint(model, [g=1:G, s=1:S], p̃_flex[g, 1, s] == x_p_flex[g])
        @constraint(model, [s=1:S], p̃_ch[1, s] == x_p_ch)
        @constraint(model, [s=1:S], p̃_dis[1, s] == x_p_dis)
        @constraint(model, [s=1:S], s̃oc[1, s] == x_soc)
        @constraint(model, [s=1:S], δ̃_d_fixed[1, s] == x_δ_d_fixed)
        @constraint(model, [s=1:S], δ̃_d_flex[1, s] == x_δ_d_flex)
        
        # Objective: Minimize expected cost over scenarios (all periods)
        scenario_costs = []
        for s in 1:S
            scenario_cost = sum(sum((generators[g].fuel_cost + generators[g].var_om_cost) * (p̃[g,τ,s] + p̃_flex[g,τ,s]) for g in 1:G) +
                              battery.var_om_cost * (p̃_ch[τ,s] + p̃_dis[τ,s]) +
                              params.load_shed_penalty * (δ̃_d_fixed[τ,s] + 0.5 * δ̃_d_flex[τ,s]^2 / params.flex_demand_mw)
                              for τ in 1:H)
            push!(scenario_costs, scenario_cost)
        end
        
        @objective(model, Min, sum(scenario_weights[s] * scenario_costs[s] for s in 1:S))
        
        # Power balance constraints (based on perfect_foresight model structure)
        @constraint(model, power_balance_lookahead[τ=1:H, s=1:S],
            sum(p̃[g,τ,s] for g in 1:G) + p̃_dis[τ,s] - p̃_ch[τ,s] + 
            sum(p̃_flex[g,τ,s] for g in 1:G) + δ̃_d_flex[τ,s] + δ̃_d_fixed[τ,s] == 
            (τ == 1 ? actual_demand[t] : demand_scenarios[s][horizon[τ]]) + params.flex_demand_mw)
        
        # Flexible demand constraint: total flexible generation + shedding = available flexible demand
        @constraint(model, [τ=1:H, s=1:S], sum(p̃_flex[g,τ,s] for g in 1:G) + δ̃_d_flex[τ,s] == params.flex_demand_mw)
        
        # Generation limits with availability factors (based on perfect_foresight model)
        for g in 1:G
            if generators[g].name == "Nuclear"
                @constraint(model, [τ=1:H, s=1:S], p̃[g,τ,s] + p̃_flex[g,τ,s] <= capacities[g] * 
                    (τ == 1 ? nuclear_availability[t] : nuclear_avail_scenarios[s][horizon[τ]]))
            elseif generators[g].name == "Wind"
                @constraint(model, [τ=1:H, s=1:S], p̃[g,τ,s] + p̃_flex[g,τ,s] <= capacities[g] * 
                    (τ == 1 ? actual_wind[t] : wind_scenarios[s][horizon[τ]]))
            elseif generators[g].name == "Gas"
                @constraint(model, [τ=1:H, s=1:S], p̃[g,τ,s] + p̃_flex[g,τ,s] <= capacities[g] * 
                    (τ == 1 ? gas_availability[t] : gas_avail_scenarios[s][horizon[τ]]))
            else
                @constraint(model, [τ=1:H, s=1:S], p̃[g,τ,s] + p̃_flex[g,τ,s] <= capacities[g])
            end
        end
        
        # Battery constraints (based on perfect_foresight model)
        @constraint(model, [τ=1:H, s=1:S], p̃_ch[τ,s] <= battery_power_cap)
        @constraint(model, [τ=1:H, s=1:S], p̃_dis[τ,s] <= battery_power_cap)
        @constraint(model, [τ=1:H, s=1:S], s̃oc[τ,s] <= battery_energy_cap)
        
        # Battery SOC dynamics (based on perfect_foresight model)
        @constraint(model, [s=1:S], s̃oc[1,s] == current_soc + 
            battery.efficiency_charge * p̃_ch[1,s] - p̃_dis[1,s]/battery.efficiency_discharge)
        @constraint(model, [τ=2:H, s=1:S], s̃oc[τ,s] == s̃oc[τ-1,s] + 
            battery.efficiency_charge * p̃_ch[τ,s] - p̃_dis[τ,s]/battery.efficiency_discharge)
        
        # Battery boundary conditions (based on perfect_foresight model)
        @constraint(model, [s=1:S], s̃oc[H,s] >= battery_energy_cap * 0.4)
        @constraint(model, [s=1:S], s̃oc[H,s] <= battery_energy_cap * 0.6)
        
        optimize!(model)
        
        if termination_status(model) == MOI.OPTIMAL
            # Store first-period decisions (use here-and-now variables)
            generation_schedule[:, t] = value.(x_p)
            generation_flex_schedule[:, t] = value.(x_p_flex)
            battery_charge_schedule[t] = value(x_p_ch)
            battery_discharge_schedule[t] = value(x_p_dis)
            battery_soc_schedule[t] = value(x_soc)
            load_shed_schedule[t] = value(x_δ_d_fixed) + value(x_δ_d_flex)
            load_shed_fixed_schedule[t] = value(x_δ_d_fixed)
            load_shed_flex_schedule[t] = value(x_δ_d_flex)
            
            # Calculate probability-weighted dual prices from power balance constraints
            scenario_prices = []
            for s in 1:S
                try
                    # Extract dual price from power balance constraint for current period (τ=1) and scenario s
                    scenario_price = dual(power_balance_lookahead[1, s])./scenario_weights[s]
                    push!(scenario_prices, scenario_price)
                catch e
                    println("Warning: Failed to retrieve dual price for scenario $s at hour $t: $e")
                    push!(scenario_prices, 50.0)
                end
            end
            
            # Expected price across scenarios (probability-weighted)
            prices[t] = sum(scenario_weights[s] * scenario_prices[s] for s in 1:S)
            
            # Verify consistency (scenarios may have different dual prices but decisions should be identical)
            if length(scenario_prices) > 1
                max_price = maximum(scenario_prices)
                min_price = minimum(scenario_prices)
                if max_price - min_price > 0.01
                    println("  Hour $t: Dual prices vary across scenarios - Range: $(round(min_price, digits=2))-$(round(max_price, digits=2)), Weighted: $(round(prices[t], digits=2))")
                end
            end
            
            # Update SOC state for next iteration
            current_soc = value(x_soc)
        else
            println("Warning: SLAC optimization failed at hour $t")
            load_shed_schedule[t] = actual_demand[t]
            prices[t] = params.load_shed_penalty
        end
    end

    
    result = Dict(
        "status" => "optimal",
        "model_type" => "SLAC",
        "generation" => generation_schedule,
        "generation_flex" => generation_flex_schedule,
        "battery_charge" => battery_charge_schedule,
        "battery_discharge" => battery_discharge_schedule,
        "battery_soc" => battery_soc_schedule,
        "load_shed" => load_shed_schedule,
        "load_shed_fixed" => load_shed_fixed_schedule,
        "load_shed_flex" => load_shed_flex_schedule,
        "commitment" => ones(G, T),
        "startup" => zeros(G, T),
        "prices" => prices,
        "total_cost" => sum(sum((generators[g].fuel_cost + generators[g].var_om_cost) * 
                               (generation_schedule[g,t] + generation_flex_schedule[g,t]) for g in 1:G) + 
                               battery.var_om_cost * (battery_charge_schedule[t] + battery_discharge_schedule[t]) +
                               params.load_shed_penalty * (load_shed_fixed_schedule[t] + 
                               0.5 * load_shed_flex_schedule[t]^2 / params.flex_demand_mw) for t in 1:T),
        "scenario_weights" => scenario_weights
    )
    
    save_operational_results(result, generators, battery, "slac", output_dir)
    
    println("SLAC Operations Summary:")
    println("  Total operational cost: \$$(round(result["total_cost"], digits=0))")
    println("  Total load shed: $(round(sum(load_shed_schedule), digits=1)) MWh")
    println("  Maximum price: \$$(round(maximum(prices), digits=2))/MWh")
    println("  Scenario weights used: $(round.(scenario_weights, digits=3))")
    
    return result
end

end # module OptimizationModels