"""
OptimizationModels.jl

Three core optimization models for ToySystemQuad.jl:
1. Capacity Expansion Model (CEM) - Joint investment and operations optimization
2. Perfect Foresight Operations (DLAC-p) - One-shot perfect foresight operations with fixed capacities
3. DLAC-i Operations - Rolling horizon with imperfect information using mean forecasts
"""

module OptimizationModels

using JuMP, Gurobi, LinearAlgebra, CSV, DataFrames, Statistics
using ..SystemConfig: Generator, Battery, SystemParameters, SystemProfiles

export solve_capacity_expansion_model, solve_perfect_foresight_operations
export save_operational_results, calculate_profits_and_save, compute_pmr
export build_dlac_i_model, update_and_solve_dlac_i, build_slac_model, update_and_solve_slac
export ModelCache, solve_dlac_i_operations_cached, solve_slac_operations_cached

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
    
    T = params.hours
    G = length(generators)
    
    # Use generator-specific availability profiles
    availabilities = profiles.generator_availabilities
    
    println("Solving Capacity Expansion Model (CEM) for $T hours")
    for g in 1:G
        avg_availability = mean(availabilities[g])
        println("  $(generators[g].name) availability: $(round(avg_availability*100, digits=1))%")
    end
    
    # Create optimization model
    model = Model(Gurobi.Optimizer)
    # set_silent(model)
    
    # Decision variables
    # Upper bounds for capacities (needed for McCormick relaxation)
    # Use very large upper bounds for non-fixed technologies; fixed ones (Hydro, Nuclear) use existing capacity
    y_max = [ (generators[g].name == "Hydro" || generators[g].name == "Nuclear") ?
              generators[g].existing_capacity : 50000.0 for g in 1:G ]
    @variable(model, y[1:G] >= 0)  # Generator capacities
    @constraint(model, [g=1:G], y[g] <= y_max[g])
    @variable(model, y_bat_power >= 0)  # Battery power capacity
    @variable(model, y_bat_energy >= 0)  # Battery energy capacity
    
    # Operational variables
    @variable(model, p[1:G, 1:T] >= 0)  # Generation for fixed demand
    @variable(model, p_flex[1:G, 1:T] >= 0)  # Generation for flexible demand
    @variable(model, p_ch[1:T] >= 0)  # Battery charging
    @variable(model, p_dis[1:T] >= 0)  # Battery discharging
    @variable(model, soc[1:T] >= 0)  # Battery state of charge (consistent with PF model)
    @variable(model, δ_d_fixed[1:T] >= 0)  # Fixed demand load shedding
    @variable(model, δ_d_flex[1:T] >= 0)  # Flexible demand load shedding
    
    # Linearized unit commitment variables
    # Startup and shutdown are bounded to [0,1] for valid McCormick relaxation
    @variable(model, 0 <= u[1:G, 1:T] <= 1)  # Commitment level (continuous, allows partial commitment)
    @variable(model, 0 <= v_start[1:G, 1:T] <= 1)  # Startup variables
    @variable(model, 0 <= v_shut[1:G, 1:T] <= 1)   # Shutdown variables

    # McCormick relaxation for u[g,t] * y[g] → z[g,t]
    @variable(model, z[1:G, 1:T] >= 0)
    for g in 1:G
        for t in 1:T
            @constraint(model, z[g,t] <= y[g])
            @constraint(model, z[g,t] <= y_max[g] * u[g,t])
            @constraint(model, z[g,t] >= y[g] - y_max[g] * (1 - u[g,t]))
        end
    end

    # McCormick relaxation for y[g] * v_start[g,t] → w_start[g,t]
    @variable(model, w_start[1:G, 1:T] >= 0)
    for g in 1:G
        for t in 1:T
            @constraint(model, w_start[g,t] <= y[g])
            @constraint(model, w_start[g,t] <= y_max[g] * v_start[g,t])
            @constraint(model, w_start[g,t] >= y[g] - y_max[g] * (1 - v_start[g,t]))
        end
    end
    
    # Objective: Total annualized costs
    investment_cost = sum(generators[g].inv_cost * y[g] for g in 1:G) + 
                     battery.inv_cost_power * y_bat_power + 
                     battery.inv_cost_energy * y_bat_energy
                     
    fixed_cost = sum(generators[g].fixed_om_cost * y[g] for g in 1:G) + 
                 battery.fixed_om_cost * y_bat_power
                 
    operational_cost = sum(
        sum((generators[g].fuel_cost + generators[g].var_om_cost) * (p[g,t] + p_flex[g,t]) for g in 1:G) +
        sum(generators[g].startup_cost * w_start[g,t] for g in 1:G) +  # Startup costs via McCormick variable w_start
        battery.var_om_cost * (p_ch[t] + p_dis[t]) +
        params.load_shed_penalty * (δ_d_fixed[t] + 0.5 * δ_d_flex[t]^2 / params.flex_demand_mw)
        for t in 1:T)
    
    @objective(model, Min, investment_cost + fixed_cost + operational_cost)
    
    # Power balance constraints
    @constraint(model, power_balance[t=1:T],
        sum(p[g,t] for g in 1:G) + p_dis[t] - p_ch[t] + sum(p_flex[g,t] for g in 1:G) + δ_d_flex[t] + δ_d_fixed[t] == actual_demand[t] + params.flex_demand_mw)
    
    # Flexible demand constraint: total flexible generation + shedding = available flexible demand
    @constraint(model, [t=1:T], sum(p_flex[g,t] for g in 1:G) + δ_d_flex[t] == params.flex_demand_mw)
    
    # Generation limits with availability factors and unit commitment (convex relaxation via z)
    for g in 1:G
        @constraint(model, [t=1:T], p[g,t] + p_flex[g,t] <= availabilities[g][t] * z[g,t])
        # Minimum generation when committed (only for thermal generators with min power > 0)
        if generators[g].min_stable_gen > 0.0
            @constraint(model, [t=1:T], p[g,t] + p_flex[g,t] >= generators[g].min_stable_gen * availabilities[g][t] * z[g,t])
        end
    end
    
    # Commitment transition constraints using startup and shutdown variables
    for g in 1:G
        # First hour: assume initial commitment is 0
        @constraint(model, u[g,1] - v_start[g,1] + v_shut[g,1] == 0)
        # Subsequent hours enforce u[t] - u[t-1] = v_start[t] - v_shut[t]
        @constraint(model, [t=2:T], u[g,t] - u[g,t-1] == v_start[g,t] - v_shut[g,t])
    end
    
    # Fix hydro and nuclear capacities to existing levels
    for g in 1:G
        if generators[g].name == "Hydro"
            @constraint(model, y[g] == generators[g].existing_capacity)
            println("  Fixed Hydro capacity to $(round(generators[g].existing_capacity, digits=1)) MW")
        elseif generators[g].name == "Nuclear"
            @constraint(model, y[g] == generators[g].existing_capacity)
            println("  Fixed Nuclear capacity to $(round(generators[g].existing_capacity, digits=1)) MW")
        end
    end
    
    # Battery constraints (no SOC bounds as requested)
    @constraint(model, [t=1:T], p_ch[t] <= y_bat_power)
    @constraint(model, [t=1:T], p_dis[t] <= y_bat_power)
    
    # Battery SOC dynamics 
    @constraint(model, soc[1] == y_bat_energy * 0.5 + 
        battery.efficiency_charge * p_ch[1] - p_dis[1]/battery.efficiency_discharge)
    @constraint(model, [t=2:T], soc[t] == soc[t-1] + 
        battery.efficiency_charge * p_ch[t] - p_dis[t]/battery.efficiency_discharge)
    
    # Battery SOC bounds (energy capacity constraint)
    @constraint(model, [t=1:T], soc[t] <= y_bat_energy)
    
    # Battery energy/power ratio
    @constraint(model, y_bat_energy <= y_bat_power * battery.duration)
    
    # End-of-horizon boundary conditions (consistent with PF model)
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
            "commitment" => value.(u),  # Actual commitment levels
            "startup" => value.(v_start),     # Actual startup variables
            "shutdown" => value.(v_shut),    # Actual shutdown variables
            "total_cost" => objective_value(model),
            "investment_cost" => value(investment_cost),
            "fixed_cost" => value(fixed_cost),
            "operational_cost" => value(operational_cost),
            "prices" => dual.(power_balance),
            "generator_availabilities" => availabilities,
            "demand_used" => actual_demand
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
    
    T = params.hours
    G = length(generators)
    
    # Use generator-specific availability profiles
    availabilities = profiles.generator_availabilities
    
    println("Solving Perfect Foresight Operations (DLAC-p) for $T hours")
    println("  Using FIXED capacities from capacity expansion")
    for g in 1:G
        avg_availability = mean(availabilities[g])
        println("  $(generators[g].name) availability: $(round(avg_availability*100, digits=1))%")
    end
    
    # Create optimization model
    model = Model(Gurobi.Optimizer)
    # set_silent(model)
    
    # Operational variables only (capacities are FIXED parameters)
    @variable(model, p[1:G, 1:T] >= 0)  # Generation for fixed demand
    @variable(model, p_flex[1:G, 1:T] >= 0)  # Generation for flexible demand
    @variable(model, p_ch[1:T] >= 0)  # Battery charging
    @variable(model, p_dis[1:T] >= 0)  # Battery discharging
    @variable(model, soc[1:T] >= 0)  # Battery state of charge
    @variable(model, δ_d_fixed[1:T] >= 0)  # Fixed demand load shedding
    @variable(model, δ_d_flex[1:T] >= 0)  # Flexible demand load shedding
    
    # Linearized unit commitment variables
    @variable(model, 0 <= u[1:G, 1:T] <= 1)  # Commitment level (continuous, allows partial commitment)
    @variable(model, 0 <= v_start[1:G, 1:T] <= 1)  # Startup variables
    @variable(model, 0 <= v_shut[1:G, 1:T] <= 1)   # Shutdown variables
    
    # Objective: Operational costs only
    @objective(model, Min, 
        sum(sum((generators[g].fuel_cost + generators[g].var_om_cost) * (p[g,t] + p_flex[g,t]) for g in 1:G) +
            sum(generators[g].startup_cost * capacities[g] * v_start[g,t] for g in 1:G) +  # Startup costs scaled by fixed capacity
            battery.var_om_cost * (p_ch[t] + p_dis[t]) +
            params.load_shed_penalty * (δ_d_fixed[t] + 0.5 * δ_d_flex[t]^2 / params.flex_demand_mw)
            for t in 1:T))
    
    # Power balance constraints
    @constraint(model, power_balance[t=1:T],
        sum(p[g,t] for g in 1:G) + p_dis[t] - p_ch[t] + sum(p_flex[g,t] for g in 1:G) + δ_d_flex[t] + δ_d_fixed[t] == actual_demand[t] + params.flex_demand_mw)
    
    # Flexible demand constraint: total flexible generation + shedding = available flexible demand
    @constraint(model, [t=1:T], sum(p_flex[g,t] for g in 1:G) + δ_d_flex[t] == params.flex_demand_mw)
    
    # Generation limits with FIXED capacities, availability factors, and unit commitment
    for g in 1:G
        @constraint(model, [t=1:T], p[g,t] + p_flex[g,t] <= u[g,t] * capacities[g] * availabilities[g][t])
        # Minimum generation when committed (only for thermal generators with min power > 0)
        if generators[g].min_stable_gen > 0.0
            @constraint(model, [t=1:T], p[g,t] + p_flex[g,t] >= u[g,t] * generators[g].min_stable_gen * capacities[g] * availabilities[g][t])
        end
    end
    
    # Commitment transition constraints using startup and shutdown variables
    for g in 1:G
        # First hour: assume initial commitment is 0
        @constraint(model, u[g,1] - v_start[g,1] + v_shut[g,1] == 0)
        # Subsequent hours enforce u[t] - u[t-1] = v_start[t] - v_shut[t]
        @constraint(model, [t=2:T], u[g,t] - u[g,t-1] == v_start[g,t] - v_shut[g,t])
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
    
    # Battery energy/power ratio (for exact consistency with CEM)
    @constraint(model, battery_energy_cap <= battery_power_cap * battery.duration)
    
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
            "commitment" => value.(u),  # Actual commitment levels
            "startup" => value.(v_start),     # Actual startup variables
            "shutdown" => value.(v_shut),    # Actual shutdown variables
            "total_cost" => objective_value(model),
            "prices" => dual.(power_balance),
            "generator_availabilities" => availabilities
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
    build_dlac_i_model(generators, battery, capacities, battery_power_cap, battery_energy_cap, 
                       lookahead_hours, params)

Build the DLAC-i optimization model structure once for reuse across rolling horizon iterations.
Returns the model, variables, and constraint references for RHS updates.

This function creates all the model structure (variables, constraints) with dummy RHS values
that will be updated in each rolling horizon iteration by update_and_solve_dlac_i().
"""
function build_dlac_i_model(generators, battery, capacities, battery_power_cap, battery_energy_cap, 
                            lookahead_hours::Int, params)
    G = length(generators)
    H = lookahead_hours
    
    # Create optimization model
    model = Model(Gurobi.Optimizer)
    # set_silent(model)
    
    # Here-and-now variables for current period decisions (t=1 only)
    @variable(model, x_p[1:G] >= 0)                 # Generation for fixed demand (current period)
    @variable(model, x_p_flex[1:G] >= 0)            # Generation for flexible demand (current period)
    @variable(model, x_p_ch >= 0)                   # Battery charging (current period)
    @variable(model, x_p_dis >= 0)                  # Battery discharging (current period)
    @variable(model, x_soc >= 0)                    # Battery SOC (current period)
    @variable(model, x_δ_d_fixed >= 0)              # Fixed demand load shedding (current period)
    @variable(model, x_δ_d_flex >= 0)               # Flexible demand load shedding (current period)
    @variable(model, 0 <= x_u[1:G] <= 1)           # Commitment level (current period)
    @variable(model, 0 <= x_v_start[1:G] <= 1)     # Startup variables (current period)
    @variable(model, 0 <= x_v_shut[1:G] <= 1)      # Shutdown variables (current period)
    
    # Decision variables for lookahead horizon
    @variable(model, p̃[1:G, 1:H] >= 0)  # Generation for fixed demand
    @variable(model, p̃_flex[1:G, 1:H] >= 0)  # Generation for flexible demand
    @variable(model, p̃_ch[1:H] >= 0)    # Battery charging
    @variable(model, p̃_dis[1:H] >= 0)   # Battery discharging
    @variable(model, s̃oc[1:H] >= 0)     # Battery SOC
    @variable(model, δ̃_d_fixed[1:H] >= 0)  # Fixed demand load shedding
    @variable(model, δ̃_d_flex[1:H] >= 0)   # Flexible demand load shedding
    @variable(model, 0 <= ũ[1:G, 1:H] <= 1)        # Commitment level (lookahead)
    @variable(model, 0 <= ṽ_start[1:G, 1:H] <= 1)  # Startup variables (lookahead)
    @variable(model, 0 <= ṽ_shut[1:G, 1:H] <= 1)   # Shutdown variables (lookahead)
    
    # Store variables for easy access
    variables = Dict(
        "x_p" => x_p, "x_p_flex" => x_p_flex, "x_p_ch" => x_p_ch, "x_p_dis" => x_p_dis,
        "x_soc" => x_soc, "x_δ_d_fixed" => x_δ_d_fixed, "x_δ_d_flex" => x_δ_d_flex,
        "x_u" => x_u, "x_v_start" => x_v_start, "x_v_shut" => x_v_shut,
        "p̃" => p̃, "p̃_flex" => p̃_flex, "p̃_ch" => p̃_ch, "p̃_dis" => p̃_dis,
        "s̃oc" => s̃oc, "δ̃_d_fixed" => δ̃_d_fixed, "δ̃_d_flex" => δ̃_d_flex,
        "ũ" => ũ, "ṽ_start" => ṽ_start, "ṽ_shut" => ṽ_shut
    )
    
    # Non-anticipativity constraints linking here-and-now to first period variables
    @constraint(model, [g=1:G], p̃[g, 1] == x_p[g])
    @constraint(model, [g=1:G], p̃_flex[g, 1] == x_p_flex[g])
    @constraint(model, p̃_ch[1] == x_p_ch)
    @constraint(model, p̃_dis[1] == x_p_dis)
    @constraint(model, s̃oc[1] == x_soc)
    @constraint(model, δ̃_d_fixed[1] == x_δ_d_fixed)
    @constraint(model, δ̃_d_flex[1] == x_δ_d_flex)
    @constraint(model, [g=1:G], ũ[g, 1] == x_u[g])
    @constraint(model, [g=1:G], ṽ_start[g, 1] == x_v_start[g])
    @constraint(model, [g=1:G], ṽ_shut[g, 1] == x_v_shut[g])
    
    # Power balance constraints (RHS will be updated)
    power_balance_constrs = @constraint(model, power_balance_lookahead[τ=1:H],
        sum(p̃[g,τ] for g in 1:G) + p̃_dis[τ] - p̃_ch[τ] + sum(p̃_flex[g,τ] for g in 1:G) + δ̃_d_fixed[τ] + δ̃_d_flex[τ] == 0.0)
    
    # Flexible demand constraint: total flexible generation + shedding = available flexible demand
    @constraint(model, [τ=1:H], sum(p̃_flex[g,τ] for g in 1:G) + δ̃_d_flex[τ] == params.flex_demand_mw)
    
    # Generation constraints with availability factors and UC (capacity limits will be updated)
    generation_constrs = Dict{String, Any}()
    min_generation_constrs = Dict{String, Any}()
    for g in 1:G
        generation_constrs[generators[g].name] = @constraint(model, [τ=1:H], 
            p̃[g,τ] + p̃_flex[g,τ] <= ũ[g,τ] * 0.0)  # RHS will be updated with capacities[g] * availability
        # Minimum generation constraints (only for thermal generators with min power > 0)
        if generators[g].min_stable_gen > 0.0
            min_generation_constrs[generators[g].name] = @constraint(model, [τ=1:H],
                p̃[g,τ] + p̃_flex[g,τ] >= ũ[g,τ] * 0.0)  # RHS will be updated with min_power * capacities[g] * availability
        end
    end
    
    # Commitment transition constraints for lookahead horizon
    startup_constrs = @constraint(model, [g=1:G, τ=2:H],
        ũ[g,τ] - ũ[g,τ-1] == ṽ_start[g,τ] - ṽ_shut[g,τ])
    # Transition constraints for the current period relative to previous commitment
    startup_initial_constrs = @constraint(model, [g=1:G],
        variables["x_u"][g] - variables["x_v_start"][g] + variables["x_v_shut"][g] == 0.0)
    
    # Battery constraints (capacity limits - will be updated when capacities change)
    battery_charge_constrs = @constraint(model, [τ=1:H], p̃_ch[τ] <= battery_power_cap)
    battery_discharge_constrs = @constraint(model, [τ=1:H], p̃_dis[τ] <= battery_power_cap)
    battery_energy_constrs = @constraint(model, [τ=1:H], s̃oc[τ] <= battery_energy_cap)
    
    # Battery SOC dynamics (initial condition RHS will be updated)
    soc_initial_constr = @constraint(model, s̃oc[1] == 0.0 + 
        battery.efficiency_charge * p̃_ch[1] - p̃_dis[1]/battery.efficiency_discharge)
    soc_evolution_constrs = @constraint(model, [τ=2:H], s̃oc[τ] == s̃oc[τ-1] + 
        battery.efficiency_charge * p̃_ch[τ] - p̃_dis[τ]/battery.efficiency_discharge)
    
    # DLAC-i Objective: Minimize total operational cost over lookahead horizon
    # This is the deterministic equivalent of SLAC's stochastic objective
    operational_cost = sum(
        sum((generators[g].fuel_cost + generators[g].var_om_cost) * (p̃[g,τ] + p̃_flex[g,τ]) for g in 1:G) +
        sum(generators[g].startup_cost * capacities[g] * ṽ_start[g,τ] for g in 1:G) +  # Startup costs scaled by capacity
        battery.var_om_cost * (p̃_ch[τ] + p̃_dis[τ]) +
        params.load_shed_penalty * (δ̃_d_fixed[τ] + 0.5 * δ̃_d_flex[τ]^2 / params.flex_demand_mw)
        for τ in 1:H
    )
    
    @objective(model, Min, operational_cost)
    
    # Store constraint references for RHS updates
    constraint_refs = Dict(
        "power_balance" => power_balance_constrs,
        "generation" => generation_constrs,
        "min_generation" => min_generation_constrs,
        "startup" => startup_constrs,
        "startup_initial" => startup_initial_constrs,
        "soc_initial" => soc_initial_constr,
        "soc_evolution" => soc_evolution_constrs,
        "battery_charge" => battery_charge_constrs,
        "battery_discharge" => battery_discharge_constrs,
        "battery_energy" => battery_energy_constrs
    )
    
    return model, variables, constraint_refs
end

"""
    update_and_solve_dlac_i(model, variables, constraint_refs, generators, battery, capacities,
                            battery_power_cap, battery_energy_cap, current_soc, t, 
                            actual_demand, availabilities, mean_demand_forecast, mean_generator_forecasts,
                            horizon, params)

Update the DLAC-i model with current time-step data and solve with warm start.
Uses generator-indexed availability profiles.
"""
function update_and_solve_dlac_i(model, variables, constraint_refs, generators, battery,
                                 capacities, battery_power_cap, battery_energy_cap,
                                 prev_commitment::Vector{Float64},
                                 current_soc::Float64, t::Int,
                                 actual_demand, availabilities, mean_demand_forecast, mean_generator_forecasts,
                                 horizon, params)
    
    G = length(generators)
    H = length(horizon)
    
    # OPTIMIZATION: Store warm start values BEFORE constraint modifications
    warm_start_values = Dict()
    if termination_status(model) == MOI.OPTIMAL
        try
            # Store current solution values for warm starting
            warm_start_values["x_p"] = [value(variables["x_p"][g]) for g in 1:G]
            warm_start_values["x_p_flex"] = [value(variables["x_p_flex"][g]) for g in 1:G]
            warm_start_values["x_p_ch"] = value(variables["x_p_ch"])
            warm_start_values["x_p_dis"] = value(variables["x_p_dis"])
            warm_start_values["x_soc"] = value(variables["x_soc"])
            warm_start_values["x_δ_d_fixed"] = value(variables["x_δ_d_fixed"])
            warm_start_values["x_δ_d_flex"] = value(variables["x_δ_d_flex"])
            # Store UC variable warm starts
            warm_start_values["x_u"] = [value(variables["x_u"][g]) for g in 1:G]
            warm_start_values["x_v_start"] = [value(variables["x_v_start"][g]) for g in 1:G]
            warm_start_values["x_v_shut"] = [value(variables["x_v_shut"][g]) for g in 1:G]
        catch
            # No previous solution available - warm_start_values remains empty
        end
    end
    
    # Update power balance constraint RHS
    for (τ_idx, τ) in enumerate(1:H)
        actual_horizon_idx = horizon[τ_idx]
        demand_value = (τ == 1) ? actual_demand[t] : mean_demand_forecast[actual_horizon_idx]
        set_normalized_rhs(constraint_refs["power_balance"][τ], demand_value + params.flex_demand_mw)
    end
    
    # Update generation constraints RHS with availability factors
    for g in 1:G
        gen_name = generators[g].name
        for (τ_idx, τ) in enumerate(1:H)
            actual_horizon_idx = horizon[τ_idx]
            
            # Use generator-indexed availability profiles
            availability = (τ == 1) ? availabilities[g][t] : mean_generator_forecasts[g][actual_horizon_idx]
            
            # Update maximum generation constraint (including UC)
            set_normalized_rhs(constraint_refs["generation"][gen_name][τ], capacities[g] * availability)
            
            # Update minimum generation constraint if it exists
            if haskey(constraint_refs["min_generation"], gen_name)
                min_gen_limit = generators[g].min_stable_gen * capacities[g] * availability
                set_normalized_rhs(constraint_refs["min_generation"][gen_name][τ], min_gen_limit)
            end
        end
    end

    # Update transition constraints for current period using previous commitments
    for g in 1:G
        set_normalized_rhs(constraint_refs["startup_initial"][g], prev_commitment[g])
    end

    # Update battery SOC initial condition (constraint is: s̃oc[1] == current_soc + charge - discharge)
    # We need to modify the constraint to be: s̃oc[1] - p̃_ch[1] * efficiency + p̃_dis[1] / efficiency == current_soc
    # But since constraint is already built as s̃oc[1] == RHS, we set RHS to current_soc
    # (the charge/discharge terms are handled by the model variables automatically)
    set_normalized_rhs(constraint_refs["soc_initial"], current_soc)
    
    # DLAC-i objective is static - operational costs over deterministic forecast horizon
    # Unlike SLAC, no need to rebuild objective since we use deterministic mean forecasts
    
    # Apply warm start values AFTER constraint updates but BEFORE optimize!
    if !isempty(warm_start_values)
        try
            for g in 1:G
                set_start_value(variables["x_p"][g], warm_start_values["x_p"][g])
                set_start_value(variables["x_p_flex"][g], warm_start_values["x_p_flex"][g])
                # Apply UC variable warm starts
                set_start_value(variables["x_u"][g], warm_start_values["x_u"][g])
                set_start_value(variables["x_v_start"][g], warm_start_values["x_v_start"][g])
                set_start_value(variables["x_v_shut"][g], warm_start_values["x_v_shut"][g])
            end
            set_start_value(variables["x_p_ch"], warm_start_values["x_p_ch"])
            set_start_value(variables["x_p_dis"], warm_start_values["x_p_dis"])
            set_start_value(variables["x_soc"], warm_start_values["x_soc"])
            set_start_value(variables["x_δ_d_fixed"], warm_start_values["x_δ_d_fixed"])
            set_start_value(variables["x_δ_d_flex"], warm_start_values["x_δ_d_flex"])
        catch
            # Skip if warm start fails
        end
    end
    
    # Optimize
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        # Return first-period decisions (use here-and-now variables)
        return (
            value.(variables["x_p"]),
            value.(variables["x_p_flex"]),
            value(variables["x_p_ch"]),
            value(variables["x_p_dis"]),
            value(variables["x_soc"]),
            value(variables["x_δ_d_fixed"]),
            value(variables["x_δ_d_flex"]),
            value.(variables["x_u"]),
            value.(variables["x_v_start"]),
            value.(variables["x_v_shut"]),
            dual(constraint_refs["power_balance"][1])
        )
    else
        # Return failure indicators
        return (
            fill(NaN, G),
            fill(NaN, G),
            NaN, NaN, NaN, NaN, NaN,
            fill(NaN, G), fill(NaN, G), fill(NaN, G),  # UC variables
            params.load_shed_penalty
        )
    end
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
            # Include startup costs in PMR calculation
            startup_costs = 0.0
            if haskey(operational_results, "startup")
                startup_costs = sum(generators[g].startup_cost * capacities[g] * operational_results["startup"][g,t] for t in 1:T)
            end
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
    build_slac_model(generators, battery, capacities, battery_power_cap, battery_energy_cap, 
                     lookahead_hours, params, num_scenarios)

Build the SLAC optimization model structure once for reuse across rolling horizon iterations.
Returns the model, variables, and constraint references for RHS updates.

SLAC is more complex than DLAC-i due to scenario-based optimization with scenario-specific variables.
"""
function build_slac_model(generators, battery, capacities, battery_power_cap, battery_energy_cap, 
                          lookahead_hours::Int, params, num_scenarios::Int)
    G = length(generators)
    H = lookahead_hours
    S = num_scenarios
    
    # Create optimization model
    model = Model(Gurobi.Optimizer)
    # set_silent(model)
    
    # Here-and-now variables for current period decisions (t=1 only)
    @variable(model, x_p[1:G] >= 0)                 # Generation for fixed demand (current period)
    @variable(model, x_p_flex[1:G] >= 0)            # Generation for flexible demand (current period)
    @variable(model, x_p_ch >= 0)                   # Battery charging (current period)
    @variable(model, x_p_dis >= 0)                  # Battery discharging (current period)
    @variable(model, x_soc >= 0)                    # Battery SOC (current period)
    @variable(model, x_δ_d_fixed >= 0)              # Fixed demand load shedding (current period)
    @variable(model, x_δ_d_flex >= 0)               # Flexible demand load shedding (current period)
    @variable(model, 0 <= x_u[1:G] <= 1)           # Commitment level (current period)
    @variable(model, 0 <= x_v_start[1:G] <= 1)     # Startup variables (current period)
    @variable(model, 0 <= x_v_shut[1:G] <= 1)      # Shutdown variables (current period)
    
    # Scenario-specific variables for ALL time periods (1:H)
    @variable(model, p̃[1:G, 1:H, 1:S] >= 0)        # Generation for fixed demand by scenario
    @variable(model, p̃_flex[1:G, 1:H, 1:S] >= 0)   # Generation for flexible demand by scenario
    @variable(model, p̃_ch[1:H, 1:S] >= 0)           # Battery charging by scenario
    @variable(model, p̃_dis[1:H, 1:S] >= 0)          # Battery discharging by scenario
    @variable(model, s̃oc[1:H, 1:S] >= 0)            # Battery SOC by scenario
    @variable(model, δ̃_d_fixed[1:H, 1:S] >= 0)       # Fixed demand load shedding by scenario
    @variable(model, δ̃_d_flex[1:H, 1:S] >= 0)        # Flexible demand load shedding by scenario
    @variable(model, 0 <= ũ[1:G, 1:H, 1:S] <= 1)        # Commitment level by scenario
    @variable(model, 0 <= ṽ_start[1:G, 1:H, 1:S] <= 1)  # Startup variables by scenario
    @variable(model, 0 <= ṽ_shut[1:G, 1:H, 1:S] <= 1)   # Shutdown variables by scenario
    
    # Store variables for easy access
    variables = Dict(
        "x_p" => x_p, "x_p_flex" => x_p_flex, "x_p_ch" => x_p_ch, "x_p_dis" => x_p_dis,
        "x_soc" => x_soc, "x_δ_d_fixed" => x_δ_d_fixed, "x_δ_d_flex" => x_δ_d_flex,
        "x_u" => x_u, "x_v_start" => x_v_start, "x_v_shut" => x_v_shut,
        "p̃" => p̃, "p̃_flex" => p̃_flex, "p̃_ch" => p̃_ch, "p̃_dis" => p̃_dis,
        "s̃oc" => s̃oc, "δ̃_d_fixed" => δ̃_d_fixed, "δ̃_d_flex" => δ̃_d_flex,
        "ũ" => ũ, "ṽ_start" => ṽ_start, "ṽ_shut" => ṽ_shut
    )
    
    # Non-anticipativity constraints linking here-and-now to scenario variables
    @constraint(model, [g=1:G, s=1:S], p̃[g, 1, s] == x_p[g])
    @constraint(model, [g=1:G, s=1:S], p̃_flex[g, 1, s] == x_p_flex[g])
    @constraint(model, [s=1:S], p̃_ch[1, s] == x_p_ch)
    @constraint(model, [s=1:S], p̃_dis[1, s] == x_p_dis)
    @constraint(model, [s=1:S], s̃oc[1, s] == x_soc)
    @constraint(model, [s=1:S], δ̃_d_fixed[1, s] == x_δ_d_fixed)
    @constraint(model, [s=1:S], δ̃_d_flex[1, s] == x_δ_d_flex)
    @constraint(model, [g=1:G, s=1:S], ũ[g, 1, s] == x_u[g])
    @constraint(model, [g=1:G, s=1:S], ṽ_start[g, 1, s] == x_v_start[g])
    @constraint(model, [g=1:G, s=1:S], ṽ_shut[g, 1, s] == x_v_shut[g])
    
    # Power balance constraints (RHS will be updated for each scenario and time period)
    power_balance_constrs = @constraint(model, power_balance_lookahead[τ=1:H, s=1:S],
        sum(p̃[g,τ,s] for g in 1:G) + p̃_dis[τ,s] - p̃_ch[τ,s] + 
        sum(p̃_flex[g,τ,s] for g in 1:G) + δ̃_d_flex[τ,s] + δ̃_d_fixed[τ,s] == 0.0)
    
    # Flexible demand constraint: total flexible generation + shedding = available flexible demand
    @constraint(model, [τ=1:H, s=1:S], sum(p̃_flex[g,τ,s] for g in 1:G) + δ̃_d_flex[τ,s] == params.flex_demand_mw)
    
    # Generation constraints with availability factors and UC (capacity limits will be updated)
    generation_constrs = Dict{String, Any}()
    min_generation_constrs = Dict{String, Any}()
    for g in 1:G
        generation_constrs[generators[g].name] = @constraint(model, [τ=1:H, s=1:S], 
            p̃[g,τ,s] + p̃_flex[g,τ,s] <= ũ[g,τ,s] * 0.0)  # RHS will be updated with capacities[g] * availability
        # Minimum generation constraints (only for thermal generators with min power > 0)
        if generators[g].min_stable_gen > 0.0
            min_generation_constrs[generators[g].name] = @constraint(model, [τ=1:H, s=1:S],
                p̃[g,τ,s] + p̃_flex[g,τ,s] >= ũ[g,τ,s] * 0.0)  # RHS will be updated with min_power * capacities[g] * availability
        end
    end
    
    # Commitment transition constraints for scenario-dependent lookahead horizon
    startup_constrs = @constraint(model, [g=1:G, τ=2:H, s=1:S],
        ũ[g,τ,s] - ũ[g,τ-1,s] == ṽ_start[g,τ,s] - ṽ_shut[g,τ,s])
    # Transition constraints for the current period relative to previous commitment
    startup_initial_constrs = @constraint(model, [g=1:G],
        variables["x_u"][g] - variables["x_v_start"][g] + variables["x_v_shut"][g] == 0.0)
    
    # Battery constraints (capacity limits - will be updated when capacities change)
    battery_charge_constrs = @constraint(model, [τ=1:H, s=1:S], p̃_ch[τ,s] <= battery_power_cap)
    battery_discharge_constrs = @constraint(model, [τ=1:H, s=1:S], p̃_dis[τ,s] <= battery_power_cap)
    battery_energy_constrs = @constraint(model, [τ=1:H, s=1:S], s̃oc[τ,s] <= battery_energy_cap)
    
    # Battery SOC dynamics (initial condition will be updated via RHS)
    # Create as s̃oc[1,s] - battery.efficiency_charge * p̃_ch[1,s] + p̃_dis[1,s]/battery.efficiency_discharge == current_soc
    soc_initial_constrs = @constraint(model, [s=1:S], s̃oc[1,s] - 
        battery.efficiency_charge * p̃_ch[1,s] + p̃_dis[1,s]/battery.efficiency_discharge == 0.0)
    soc_evolution_constrs = @constraint(model, [τ=2:H, s=1:S], s̃oc[τ,s] == s̃oc[τ-1,s] + 
        battery.efficiency_charge * p̃_ch[τ,s] - p̃_dis[τ,s]/battery.efficiency_discharge)
    
    # Battery boundary conditions (end-of-horizon constraints - depend on energy capacity)
    battery_boundary_min_constrs = @constraint(model, [s=1:S], s̃oc[H,s] >= battery_energy_cap * 0.4)
    battery_boundary_max_constrs = @constraint(model, [s=1:S], s̃oc[H,s] <= battery_energy_cap * 0.6)
    
    # Set dummy objective (will be updated)
    @objective(model, Min, 0)
    
    # Store constraint references for RHS updates
    constraint_refs = Dict(
        "power_balance" => power_balance_constrs,
        "generation" => generation_constrs,
        "min_generation" => min_generation_constrs,
        "startup" => startup_constrs,
        "startup_initial" => startup_initial_constrs,
        "soc_initial" => soc_initial_constrs,
        "soc_evolution" => soc_evolution_constrs,
        "battery_charge" => battery_charge_constrs,
        "battery_discharge" => battery_discharge_constrs,
        "battery_energy" => battery_energy_constrs,
        "battery_boundary_min" => battery_boundary_min_constrs,
        "battery_boundary_max" => battery_boundary_max_constrs
    )
    
    return model, variables, constraint_refs
end

"""
    update_and_solve_slac(model, variables, constraint_refs, generators, battery,
                          capacities, battery_power_cap, battery_energy_cap, prev_commitment, current_soc, t,
                          actual_demand, availabilities, demand_scenarios, generator_availability_scenarios,
                          horizon, params, scenario_weights)

Update the SLAC model with current time-step data and scenario forecasts, then solve with warm start.
Uses generator-indexed availability profiles.
"""
function update_and_solve_slac(model, variables, constraint_refs, generators, battery,
                               capacities, battery_power_cap, battery_energy_cap,
                               prev_commitment::Vector{Float64},
                               current_soc::Float64, t::Int,
                               actual_demand, availabilities, demand_scenarios, generator_availability_scenarios,
                               horizon, params, scenario_weights; model_cache=nothing)
    
    # Track current time step for debugging
    constraint_refs["debug_t"] = t
    
    G = length(generators)
    H = length(horizon)
    S = length(scenario_weights)
    
    # OPTIMIZATION: Store warm start values BEFORE constraint modifications
    debug_mode = haskey(ENV, "DEBUG_SLAC_CACHE") && ENV["DEBUG_SLAC_CACHE"] == "1"
    warm_start_values = Dict()
    warm_start_applied = false
    
    if termination_status(model) == MOI.OPTIMAL
        try
            # Store current solution values for warm starting
            warm_start_values["x_p"] = [value(variables["x_p"][g]) for g in 1:G]
            warm_start_values["x_p_flex"] = [value(variables["x_p_flex"][g]) for g in 1:G]
            warm_start_values["x_p_ch"] = value(variables["x_p_ch"])
            warm_start_values["x_p_dis"] = value(variables["x_p_dis"])
            warm_start_values["x_soc"] = value(variables["x_soc"])
            warm_start_values["x_δ_d_fixed"] = value(variables["x_δ_d_fixed"])
            warm_start_values["x_δ_d_flex"] = value(variables["x_δ_d_flex"])
            # Store UC variable warm starts for SLAC
            warm_start_values["x_u"] = [value(variables["x_u"][g]) for g in 1:G]
            warm_start_values["x_v_start"] = [value(variables["x_v_start"][g]) for g in 1:G]
            warm_start_values["x_v_shut"] = [value(variables["x_v_shut"][g]) for g in 1:G]
            warm_start_applied = true
        catch e
            # No previous solution available
            warm_start_applied = false
        end
    end
    
    # OPTIMIZATION: Update constraints that change in rolling horizon
    # Balance between updating only necessary constraints vs correctness
    
    # OPTIMIZATION: Efficient power balance constraint updates
    # Only update first period with actual data; later periods may not change much
    constraint_updates = 0
    
    for (τ_idx, τ) in enumerate(1:H)
        actual_horizon_idx = horizon[τ_idx]
        for s in 1:S
            demand_value = (τ == 1) ? actual_demand[t] : demand_scenarios[s][actual_horizon_idx]
            set_normalized_rhs(constraint_refs["power_balance"][τ, s], demand_value + params.flex_demand_mw)
            constraint_updates += 1
        end
    end
    
    # OPTIMIZATION: Streamlined generation constraint updates
    gen_constraint_updates = 0
    for g in 1:G
        gen_name = generators[g].name
        for (τ_idx, τ) in enumerate(1:H)
            actual_horizon_idx = horizon[τ_idx]
            for s in 1:S
                # Use generator-indexed availability profiles
                availability = (τ == 1) ? availabilities[g][t] : generator_availability_scenarios[s][g][actual_horizon_idx]
                
                # Update maximum generation constraint (with UC)
                set_normalized_rhs(constraint_refs["generation"][gen_name][τ, s], capacities[g] * availability)
                
                # Update minimum generation constraint if it exists
                if haskey(constraint_refs["min_generation"], gen_name)
                    min_gen_limit = generators[g].min_stable_gen * capacities[g] * availability
                    set_normalized_rhs(constraint_refs["min_generation"][gen_name][τ, s], min_gen_limit)
                end
                
                gen_constraint_updates += 1
            end
        end
    end
    
    # Update transition constraints for current period using previous commitments
    for g in 1:G
        set_normalized_rhs(constraint_refs["startup_initial"][g], prev_commitment[g])
    end

    # Update battery SOC initial condition
    for s in 1:S
        set_normalized_rhs(constraint_refs["soc_initial"][s], current_soc)
    end
    
    # CRITICAL: Always rebuild SLAC objective because rolling horizon changes structure
    # Period τ=1 uses actual realizations (deterministic) while τ>1 uses scenarios (stochastic)
    # As we roll forward, the mix of deterministic vs stochastic periods changes
    # This fundamental structure change requires objective rebuilding every iteration
    should_rebuild = true  # Always rebuild for correctness
    
    if should_rebuild
        scenario_costs = []
        for s in 1:S
            scenario_cost = sum(sum((generators[g].fuel_cost + generators[g].var_om_cost) * (variables["p̃"][g,τ,s] + variables["p̃_flex"][g,τ,s]) for g in 1:G) +
                              sum(generators[g].startup_cost * capacities[g] * variables["ṽ_start"][g,τ,s] for g in 1:G) +  # Startup costs scaled by capacity
                              battery.var_om_cost * (variables["p̃_ch"][τ,s] + variables["p̃_dis"][τ,s]) +
                              params.load_shed_penalty * (variables["δ̃_d_fixed"][τ,s] + 0.5 * variables["δ̃_d_flex"][τ,s]^2 / params.flex_demand_mw)
                              for τ in 1:H)
            push!(scenario_costs, scenario_cost)
        end
        
        @objective(model, Min, sum(scenario_weights[s] * scenario_costs[s] for s in 1:S))
        
        # Update cache
        if model_cache !== nothing
            model_cache.last_scenario_weights = copy(scenario_weights)
            model_cache.objective_needs_rebuild = false
        end
    end
    
    # Apply warm start values AFTER constraint updates but BEFORE optimize!
    if warm_start_applied && !isempty(warm_start_values)
        try
            for g in 1:G
                set_start_value(variables["x_p"][g], warm_start_values["x_p"][g])
                set_start_value(variables["x_p_flex"][g], warm_start_values["x_p_flex"][g])
                # Apply UC variable warm starts for SLAC
                set_start_value(variables["x_u"][g], warm_start_values["x_u"][g])
                set_start_value(variables["x_v_start"][g], warm_start_values["x_v_start"][g])
                set_start_value(variables["x_v_shut"][g], warm_start_values["x_v_shut"][g])
            end
            set_start_value(variables["x_p_ch"], warm_start_values["x_p_ch"])
            set_start_value(variables["x_p_dis"], warm_start_values["x_p_dis"])
            set_start_value(variables["x_soc"], warm_start_values["x_soc"])
            set_start_value(variables["x_δ_d_fixed"], warm_start_values["x_δ_d_fixed"])
            set_start_value(variables["x_δ_d_flex"], warm_start_values["x_δ_d_flex"])
        catch
            # Skip if warm start fails
        end
    end
    
    # OPTIMIZATION: Streamlined optimization call
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        # Calculate probability-weighted dual prices from power balance constraints
        scenario_prices = []
        for s in 1:S
            try
                scenario_price = dual(constraint_refs["power_balance"][1, s]) / scenario_weights[s]
                push!(scenario_prices, scenario_price)
            catch e
                push!(scenario_prices, params.load_shed_penalty)
            end
        end
        
        # Expected price across scenarios (probability-weighted)
        expected_price = sum(scenario_weights[s] * scenario_prices[s] for s in 1:S)
        
        # Return first-period decisions (use here-and-now variables)
        return (
            value.(variables["x_p"]),
            value.(variables["x_p_flex"]),
            value(variables["x_p_ch"]),
            value(variables["x_p_dis"]),
            value(variables["x_soc"]),
            value(variables["x_δ_d_fixed"]),
            value(variables["x_δ_d_flex"]),
            value.(variables["x_u"]),
            value.(variables["x_v_start"]),
            value.(variables["x_v_shut"]),
            expected_price
        )
    else
        # Return failure indicators
        return (
            fill(NaN, G),
            fill(NaN, G),
            NaN, NaN, NaN, NaN, NaN,
            fill(NaN, G), fill(NaN, G), fill(NaN, G),  # UC variables
            params.load_shed_penalty
        )
    end
end

# =============================================================================
# EQUILIBRIUM-AWARE MODEL CACHING FUNCTIONS
# =============================================================================

"""
    ModelCache

Structure to hold cached optimization models for equilibrium iterations.
Stores built models, variables, and constraint references for capacity updates.
"""
mutable struct ModelCache
    dlac_i_model::Union{Nothing, JuMP.Model}
    dlac_i_variables::Union{Nothing, Dict}
    dlac_i_constraint_refs::Union{Nothing, Dict}
    
    slac_model::Union{Nothing, JuMP.Model}
    slac_variables::Union{Nothing, Dict}
    slac_constraint_refs::Union{Nothing, Dict}
    
    # Cache metadata
    lookahead_hours::Int
    num_scenarios::Int
    last_capacities::Vector{Float64}
    last_battery_power_cap::Float64
    last_battery_energy_cap::Float64
    
    # Scenario weight caching for objective optimization
    last_scenario_weights::Vector{Float64}
    objective_needs_rebuild::Bool
    
    function ModelCache(lookahead_hours::Int, num_scenarios::Int)
        new(nothing, nothing, nothing, nothing, nothing, nothing,
            lookahead_hours, num_scenarios, Float64[], 0.0, 0.0,
            Float64[], true)
    end
end

"""
    solve_dlac_i_operations_cached(generators, battery, capacities, battery_power_cap, battery_energy_cap,
                                   profiles::SystemProfiles, model_cache::ModelCache; 
                                   lookahead_hours=24, output_dir="results")

Cached version of DLAC-i operations that reuses models across equilibrium iterations.
Updates capacity constraints when capacities change, uses warm starts between iterations.
"""
function solve_dlac_i_operations_cached(generators, battery, capacities, battery_power_cap, battery_energy_cap,
                                        profiles::SystemProfiles, model_cache::ModelCache; 
                                        lookahead_hours=24, output_dir="results")
    params = profiles.params
    
    # Get forecast data with new generator-indexed structure
    actual_demand = profiles.actual_demand
    availabilities = profiles.generator_availabilities
    demand_scenarios = profiles.demand_scenarios
    generator_availability_scenarios = profiles.generator_availability_scenarios
    
    T = params.hours
    G = length(generators)
    S = length(demand_scenarios)
    
    # Compute mean forecasts from scenarios with generator-indexed structure
    mean_demand_forecast = [mean([demand_scenarios[s][t] for s in 1:S]) for t in 1:T]
    # Mean availability forecasts for each generator: mean_generator_forecasts[g][t]
    mean_generator_forecasts = Vector{Vector{Float64}}(undef, G)
    for g in 1:G
        mean_generator_forecasts[g] = [mean([generator_availability_scenarios[s][g][t] for s in 1:S]) for t in 1:T]
    end
    
    # Check if model needs to be rebuilt or capacity constraints updated
    capacities_changed = (model_cache.last_capacities != capacities || 
                         model_cache.last_battery_power_cap != battery_power_cap ||
                         model_cache.last_battery_energy_cap != battery_energy_cap)
    
    # Build or rebuild model if needed
    if model_cache.dlac_i_model === nothing || model_cache.lookahead_hours != lookahead_hours
        println("  Building DLAC-i model for equilibrium (lookahead: $lookahead_hours hours)")
        model_cache.dlac_i_model, model_cache.dlac_i_variables, model_cache.dlac_i_constraint_refs = 
            build_dlac_i_model(generators, battery, capacities, battery_power_cap, battery_energy_cap, 
                              lookahead_hours, params)
        model_cache.lookahead_hours = lookahead_hours
    elseif capacities_changed
        # Update capacity constraints for existing model
        println("  Updating DLAC-i capacity constraints for new equilibrium iteration")
        update_capacity_constraints_dlac_i!(model_cache.dlac_i_model, model_cache.dlac_i_constraint_refs, 
                                           generators, capacities, battery_power_cap, battery_energy_cap)
    end
    
    # Update cache metadata
    model_cache.last_capacities = copy(capacities)
    model_cache.last_battery_power_cap = battery_power_cap
    model_cache.last_battery_energy_cap = battery_energy_cap
    
    # Use the regular DLAC-i rolling horizon logic but with cached model
    return solve_dlac_i_with_cached_model(generators, battery, capacities, battery_power_cap, battery_energy_cap,
                                         profiles, model_cache.dlac_i_model, model_cache.dlac_i_variables, 
                                         model_cache.dlac_i_constraint_refs, lookahead_hours, output_dir)
end

"""
    solve_slac_operations_cached(generators, battery, capacities, battery_power_cap, battery_energy_cap,
                                 profiles::SystemProfiles, model_cache::ModelCache; 
                                 lookahead_hours=24, output_dir="results", scenario_weights=nothing)

Cached version of SLAC operations that reuses models across equilibrium iterations.
"""
function solve_slac_operations_cached(generators, battery, capacities, battery_power_cap, battery_energy_cap,
                                     profiles::SystemProfiles, model_cache::ModelCache; 
                                     lookahead_hours=24, output_dir="results", scenario_weights=nothing)
    params = profiles.params
    
    # Get scenario data with new generator-indexed structure
    actual_demand = profiles.actual_demand
    availabilities = profiles.generator_availabilities
    demand_scenarios = profiles.demand_scenarios
    generator_availability_scenarios = profiles.generator_availability_scenarios
    
    T = params.hours
    G = length(generators)
    S = length(demand_scenarios)
    
    # Set default equal weights if not provided
    if scenario_weights === nothing
        scenario_weights = fill(1.0/S, S)
    else
        scenario_weights = scenario_weights ./ sum(scenario_weights)
    end
    
    # Check if model needs to be rebuilt or capacity constraints updated
    capacities_changed = (model_cache.last_capacities != capacities || 
                         model_cache.last_battery_power_cap != battery_power_cap ||
                         model_cache.last_battery_energy_cap != battery_energy_cap)
    
    # Build or rebuild model if needed
    if model_cache.slac_model === nothing || model_cache.lookahead_hours != lookahead_hours || model_cache.num_scenarios != S
        println("  Building SLAC model for equilibrium (lookahead: $lookahead_hours hours, scenarios: $S)")
        model_cache.slac_model, model_cache.slac_variables, model_cache.slac_constraint_refs = 
            build_slac_model(generators, battery, capacities, battery_power_cap, battery_energy_cap, 
                           lookahead_hours, params, S)
        model_cache.lookahead_hours = lookahead_hours
        model_cache.num_scenarios = S
    elseif capacities_changed
        # Update capacity constraints for existing model
        println("  Updating SLAC capacity constraints for new equilibrium iteration")
        update_capacity_constraints_slac!(model_cache.slac_model, model_cache.slac_constraint_refs, 
                                        generators, capacities, battery_power_cap, battery_energy_cap, S)
    end
    
    # Update cache metadata
    model_cache.last_capacities = copy(capacities)
    model_cache.last_battery_power_cap = battery_power_cap
    model_cache.last_battery_energy_cap = battery_energy_cap
    
    # Use the regular SLAC rolling horizon logic but with cached model
    return solve_slac_with_cached_model(generators, battery, capacities, battery_power_cap, battery_energy_cap,
                                       profiles, model_cache.slac_model, model_cache.slac_variables, 
                                       model_cache.slac_constraint_refs, lookahead_hours, output_dir, scenario_weights, model_cache)
end

"""
    update_capacity_constraints_dlac_i!(model, constraint_refs, generators, capacities, 
                                        battery_power_cap, battery_energy_cap)

Update capacity constraints in a DLAC-i model when capacities change during equilibrium iterations.
Updates battery capacity constraints; generation and UC constraints are updated dynamically 
in update_and_solve_dlac_i() based on availability factors.

Note: UC constraints (generation limits with commitment, minimum generation, startup logic) 
are handled automatically through the rolling horizon constraint updates.
"""
function update_capacity_constraints_dlac_i!(model, constraint_refs, generators, capacities, 
                                             battery_power_cap, battery_energy_cap)
    G = length(generators)
    
    # Update battery power capacity constraints (charge and discharge limits)
    if haskey(constraint_refs, "battery_charge")
        for τ in 1:length(constraint_refs["battery_charge"])
            set_normalized_rhs(constraint_refs["battery_charge"][τ], battery_power_cap)
        end
    end
    
    if haskey(constraint_refs, "battery_discharge") 
        for τ in 1:length(constraint_refs["battery_discharge"])
            set_normalized_rhs(constraint_refs["battery_discharge"][τ], battery_power_cap)
        end
    end
    
    # Update battery energy capacity constraints (SOC limits)
    if haskey(constraint_refs, "battery_energy")
        for τ in 1:length(constraint_refs["battery_energy"])
            set_normalized_rhs(constraint_refs["battery_energy"][τ], battery_energy_cap)
        end
    end
    
    # Note: Generation capacity constraints (including UC) are updated dynamically in each rolling 
    # horizon iteration through update_and_solve_dlac_i() based on availability factors.
    # UC constraints are automatically included: p[g,t] <= u[g,t] * capacity[g] * availability[g,t]
    
    return nothing
end

"""
    update_capacity_constraints_slac!(model, constraint_refs, generators, capacities, 
                                      battery_power_cap, battery_energy_cap, num_scenarios)

Update capacity constraints in a SLAC model when capacities change during equilibrium iterations.
Updates battery capacity constraints; generation and UC constraints are updated dynamically 
in update_and_solve_slac() based on availability factors and scenarios.

Note: UC constraints (generation limits with commitment, minimum generation, startup logic) 
are handled automatically through the rolling horizon constraint updates with scenario indexing.
"""
function update_capacity_constraints_slac!(model, constraint_refs, generators, capacities, 
                                           battery_power_cap, battery_energy_cap, num_scenarios)
    G = length(generators)
    S = num_scenarios
    
    # Update battery power capacity constraints (charge and discharge limits)
    # SLAC has constraints indexed by [τ, s] (time and scenario)
    if haskey(constraint_refs, "battery_charge")
        for τ in 1:size(constraint_refs["battery_charge"], 1)
            for s in 1:S
                set_normalized_rhs(constraint_refs["battery_charge"][τ, s], battery_power_cap)
            end
        end
    end
    
    if haskey(constraint_refs, "battery_discharge")
        for τ in 1:size(constraint_refs["battery_discharge"], 1)
            for s in 1:S
                set_normalized_rhs(constraint_refs["battery_discharge"][τ, s], battery_power_cap)
            end
        end
    end
    
    # Update battery energy capacity constraints (SOC limits)
    if haskey(constraint_refs, "battery_energy")
        for τ in 1:size(constraint_refs["battery_energy"], 1)
            for s in 1:S
                set_normalized_rhs(constraint_refs["battery_energy"][τ, s], battery_energy_cap)
            end
        end
    end
    
    # Update battery boundary conditions (end-of-horizon constraints)
    if haskey(constraint_refs, "battery_boundary_min")
        for s in 1:S
            set_normalized_rhs(constraint_refs["battery_boundary_min"][s], battery_energy_cap * 0.4)
        end
    end
    
    if haskey(constraint_refs, "battery_boundary_max")
        for s in 1:S
            set_normalized_rhs(constraint_refs["battery_boundary_max"][s], battery_energy_cap * 0.6)
        end
    end
    
    # Note: Generation capacity constraints (including UC) are updated dynamically in each rolling 
    # horizon iteration through update_and_solve_slac() based on availability factors and scenarios.
    # UC constraints are automatically included: p[g,t,s] <= u[g,t,s] * capacity[g] * availability[g,t,s]
    
    return nothing
end

"""
    solve_dlac_i_with_cached_model(generators, battery, capacities, battery_power_cap, battery_energy_cap,
                                   profiles, cached_model, cached_variables, cached_constraint_refs, 
                                   lookahead_hours, output_dir)

Execute DLAC-i operations using a pre-built cached model structure.
This is essentially the rolling horizon loop from the original solve_dlac_i_operations but using cached models.
"""
function solve_dlac_i_with_cached_model(generators, battery, capacities, battery_power_cap, battery_energy_cap,
                                        profiles, cached_model, cached_variables, cached_constraint_refs, 
                                        lookahead_hours, output_dir)
    params = profiles.params
    
    # Get forecast data with new generator-indexed structure
    actual_demand = profiles.actual_demand
    availabilities = profiles.generator_availabilities
    demand_scenarios = profiles.demand_scenarios
    generator_availability_scenarios = profiles.generator_availability_scenarios
    
    T = params.hours
    G = length(generators)
    S = length(demand_scenarios)
    
    # Compute mean forecasts from scenarios with generator-indexed structure
    mean_demand_forecast = [mean([demand_scenarios[s][t] for s in 1:S]) for t in 1:T]
    # Mean availability forecasts for each generator: mean_generator_forecasts[g][t]
    mean_generator_forecasts = Vector{Vector{Float64}}(undef, G)
    for g in 1:G
        mean_generator_forecasts[g] = [mean([generator_availability_scenarios[s][g][t] for s in 1:S]) for t in 1:T]
    end
    
    println("Solving DLAC-i Operations with cached model ($(lookahead_hours)-hour lookahead for $T hours)")
    
    # Initialize result storage
    generation_schedule = zeros(G, T)
    generation_flex_schedule = zeros(G, T)
    battery_charge_schedule = zeros(T)
    battery_discharge_schedule = zeros(T)
    battery_soc_schedule = zeros(T)
    load_shed_schedule = zeros(T)
    load_shed_fixed_schedule = zeros(T)
    load_shed_flex_schedule = zeros(T)
    commitment_schedule = zeros(G, T)
    startup_schedule = zeros(G, T)
    shutdown_schedule = zeros(G, T)
    shutdown_schedule = zeros(G, T)
    prices = zeros(T)
    
    # State tracking
    current_soc = battery_energy_cap * 0.5
    
    # Rolling horizon optimization using cached model
    for t in 1:T
        # Determine lookahead horizon
        horizon_end = min(t + lookahead_hours - 1, T)
        horizon = t:horizon_end
        
        # Update and solve the cached model
        prev_commitment = t > 1 ? commitment_schedule[:, t-1] : zeros(G)
        result = update_and_solve_dlac_i(
            cached_model, cached_variables, cached_constraint_refs, generators, battery,
            capacities, battery_power_cap, battery_energy_cap,
            prev_commitment,
            current_soc, t,
            actual_demand, availabilities, mean_demand_forecast, mean_generator_forecasts,
            horizon, params
        )
        
        # Extract results
        x_p_val, x_p_flex_val, x_p_ch_val, x_p_dis_val, x_soc_val, x_δ_d_fixed_val, x_δ_d_flex_val, x_u_val, x_v_start_val, x_v_shut_val, price_val = result
        
        if !any(isnan, x_p_val)
            # Store first-period decisions
            generation_schedule[:, t] = x_p_val
            generation_flex_schedule[:, t] = x_p_flex_val
            battery_charge_schedule[t] = x_p_ch_val
            battery_discharge_schedule[t] = x_p_dis_val
            battery_soc_schedule[t] = x_soc_val
            load_shed_schedule[t] = x_δ_d_fixed_val + x_δ_d_flex_val
            load_shed_fixed_schedule[t] = x_δ_d_fixed_val
            load_shed_flex_schedule[t] = x_δ_d_flex_val
            commitment_schedule[:, t] = x_u_val
            startup_schedule[:, t] = x_v_start_val
            shutdown_schedule[:, t] = x_v_shut_val
            prices[t] = price_val
            
            # Update SOC state for next iteration
            current_soc = x_soc_val
        else
            println("Warning: DLAC-i cached optimization failed at hour $t")
            load_shed_schedule[t] = actual_demand[t]
            prices[t] = params.load_shed_penalty
        end
    end
    
    # Create and return result dictionary (same structure as original)
    result = Dict(
        "status" => "optimal",
        "model_type" => "DLAC-i-cached",
        "generation" => generation_schedule,
        "generation_flex" => generation_flex_schedule,
        "battery_charge" => battery_charge_schedule,
        "battery_discharge" => battery_discharge_schedule,
        "battery_soc" => battery_soc_schedule,
        "load_shed" => load_shed_schedule,
        "load_shed_fixed" => load_shed_fixed_schedule,
        "load_shed_flex" => load_shed_flex_schedule,
        "commitment" => commitment_schedule,  # Actual UC variables
        "startup" => startup_schedule,        # Actual startup variables
        "shutdown" => shutdown_schedule,      # Actual shutdown variables
        "prices" => prices,
        "total_cost" => sum(sum((generators[g].fuel_cost + generators[g].var_om_cost) * 
                               (generation_schedule[g,t] + generation_flex_schedule[g,t]) for g in 1:G) + 
                               sum(generators[g].startup_cost * capacities[g] * startup_schedule[g,t] for g in 1:G) +  # Add startup costs
                               battery.var_om_cost * (battery_charge_schedule[t] + battery_discharge_schedule[t]) +
                               params.load_shed_penalty * (load_shed_fixed_schedule[t] + 0.5 * load_shed_flex_schedule[t]^2 / params.flex_demand_mw) for t in 1:T),
        "generator_availabilities" => availabilities
    )
    
    # Save operational results
    save_operational_results(result, generators, battery, "dlac_i_cached", output_dir)
    
    println("DLAC-i Cached Operations Summary:")
    println("  Total operational cost: \$$(round(result["total_cost"], digits=0))")
    println("  Total load shed: $(round(sum(load_shed_schedule), digits=1)) MWh")
    println("  Maximum price: \$$(round(maximum(prices), digits=2))/MWh")
    
    return result
end

"""
    solve_slac_with_cached_model(generators, battery, capacities, battery_power_cap, battery_energy_cap,
                                 profiles, cached_model, cached_variables, cached_constraint_refs, 
                                 lookahead_hours, output_dir, scenario_weights)

Execute SLAC operations using a pre-built cached model structure.
"""
function solve_slac_with_cached_model(generators, battery, capacities, battery_power_cap, battery_energy_cap,
                                      profiles, cached_model, cached_variables, cached_constraint_refs, 
                                      lookahead_hours, output_dir, scenario_weights, model_cache=nothing, save_results=true)
    params = profiles.params
    
    # Get scenario data with new generator-indexed structure
    actual_demand = profiles.actual_demand
    availabilities = profiles.generator_availabilities
    demand_scenarios = profiles.demand_scenarios
    generator_availability_scenarios = profiles.generator_availability_scenarios
    
    T = params.hours
    G = length(generators)
    S = length(scenario_weights)
    
    println("Solving SLAC Operations with cached model ($(lookahead_hours)-hour lookahead, $S scenarios)")
    
    # Initialize result storage
    generation_schedule = zeros(G, T)
    generation_flex_schedule = zeros(G, T)
    battery_charge_schedule = zeros(T)
    battery_discharge_schedule = zeros(T)
    battery_soc_schedule = zeros(T)
    load_shed_schedule = zeros(T)
    load_shed_fixed_schedule = zeros(T)
    load_shed_flex_schedule = zeros(T)
    commitment_schedule = zeros(G, T)
    startup_schedule = zeros(G, T)
    prices = zeros(T)
    
    # State tracking
    current_soc = battery_energy_cap * 0.5
    
    # Rolling horizon optimization using cached model
    for t in 1:T
        # if t % 100 == 0
        #     println("  Processing hour $t/$T")
        # end
        
        # Determine lookahead horizon
        horizon_end = min(t + lookahead_hours - 1, T)
        horizon = t:horizon_end
        
        # Update and solve the cached model
        prev_commitment = t > 1 ? commitment_schedule[:, t-1] : zeros(G)
        result = update_and_solve_slac(
            cached_model, cached_variables, cached_constraint_refs, generators, battery,
            capacities, battery_power_cap, battery_energy_cap, prev_commitment,
            current_soc, t,
            actual_demand, availabilities, demand_scenarios, generator_availability_scenarios,
            horizon, params, scenario_weights; model_cache=model_cache
        )
        
        # Extract results
        x_p_val, x_p_flex_val, x_p_ch_val, x_p_dis_val, x_soc_val, x_δ_d_fixed_val, x_δ_d_flex_val, x_u_val, x_v_start_val, x_v_shut_val, price_val = result
        
        if !any(isnan, x_p_val)
            # Store first-period decisions
            generation_schedule[:, t] = x_p_val
            generation_flex_schedule[:, t] = x_p_flex_val
            battery_charge_schedule[t] = x_p_ch_val
            battery_discharge_schedule[t] = x_p_dis_val
            battery_soc_schedule[t] = x_soc_val
            load_shed_schedule[t] = x_δ_d_fixed_val + x_δ_d_flex_val
            load_shed_fixed_schedule[t] = x_δ_d_fixed_val
            load_shed_flex_schedule[t] = x_δ_d_flex_val
            commitment_schedule[:, t] = x_u_val
            startup_schedule[:, t] = x_v_start_val
            shutdown_schedule[:, t] = x_v_shut_val
            prices[t] = price_val
            
            # Update SOC state for next iteration
            current_soc = x_soc_val
        else
            println("Warning: SLAC cached optimization failed at hour $t")
            load_shed_schedule[t] = actual_demand[t]
            prices[t] = params.load_shed_penalty
        end
    end
    
    # Create and return result dictionary (same structure as original)
    result = Dict(
        "status" => "optimal",
        "model_type" => "SLAC-cached",
        "generation" => generation_schedule,
        "generation_flex" => generation_flex_schedule,
        "battery_charge" => battery_charge_schedule,
        "battery_discharge" => battery_discharge_schedule,
        "battery_soc" => battery_soc_schedule,
        "load_shed" => load_shed_schedule,
        "load_shed_fixed" => load_shed_fixed_schedule,
        "load_shed_flex" => load_shed_flex_schedule,
        "commitment" => commitment_schedule,  # Actual UC variables
        "startup" => startup_schedule,        # Actual startup variables
        "shutdown" => shutdown_schedule,      # Actual shutdown variables
        "prices" => prices,
        "total_cost" => sum(sum((generators[g].fuel_cost + generators[g].var_om_cost) * 
                               (generation_schedule[g,t] + generation_flex_schedule[g,t]) for g in 1:G) + 
                               battery.var_om_cost * (battery_charge_schedule[t] + battery_discharge_schedule[t]) +
                               params.load_shed_penalty * (load_shed_fixed_schedule[t] + 
                               0.5 * load_shed_flex_schedule[t]^2 / params.flex_demand_mw) for t in 1:T),
        "scenario_weights" => scenario_weights
    )
    if save_results
        # Save operational results
        save_operational_results(result, generators, battery, "slac_cached", output_dir)
    end
    println("SLAC Cached Operations Summary:")
    println("  Total operational cost: \$$(round(result["total_cost"], digits=0))")
    println("  Total load shed: $(round(sum(load_shed_schedule), digits=1)) MWh")
    println("  Maximum price: \$$(round(maximum(prices), digits=2))/MWh")
    println("  Scenario weights used: $(round.(scenario_weights, digits=3))")
    
    return result
end

end # module OptimizationModels