#!/usr/bin/env python3

import pandas as pd
import os

def calculate_metrics_from_csvs():
    """Calculate system metrics directly from equilibrium CSV files"""
    
    base_dir = "results/equilibrium/anderson"
    
    # Policy directories and display names
    policies = {
        "perfectforesight": "Perfect Foresight",
        "slac": "SLAC", 
        "dlac_i": "DLAC-i"
    }
    
    # Investment costs ($/MW/year) - from SystemConfig.jl with multipliers
    inv_costs = {
        'Nuclear': 120000.0,
        'CC': 70000.0 * 1.5,      # Gas * 1.5 multiplier
        'CT': 70000.0 * 1.5,      # Gas * 1.5 multiplier  
        'ST': 70000.0 * 1.5,      # Gas * 1.5 multiplier
        'Wind': 85000.0 * 0.9,    # Wind * 0.9 multiplier
        'Solar': 85000.0 * 0.7,   # Solar * 0.7 multiplier
        'Hydro': 120000.0,
        'Battery_power': 95000.0 * 0.8,   # Battery power * 0.8 multiplier
        'Battery_energy': 100.0 * 0.8     # Battery energy * 0.8 multiplier ($/MWh/yr)
    }
    
    # Fixed O&M costs ($/MW/year)
    fixed_om_costs = {
        'Nuclear': 35000.0, 'CC': 12000.0, 'CT': 12000.0, 'ST': 12000.0,
        'Wind': 12000.0, 'Solar': 12000.0, 'Hydro': 35000.0, 'Battery': 6000.0
    }
    
    battery_duration = 4.0  # hours
    
    print("="*80)
    print("EQUILIBRIUM SYSTEM METRICS SUMMARY")
    print("="*80)
    
    results = {}
    
    for policy_dir, policy_name in policies.items():
        print(f"\n{policy_name}:")
        
        # File paths
        policy_path = os.path.join(base_dir, policy_dir)
        eq_log_file = os.path.join(policy_path, "equilibrium_log.csv")
        
        # Find operations file
        ops_files = [
            "perfect_foresight_operations.csv",
            "slac_cached_operations.csv", 
            "dlac_i_cached_operations.csv"
        ]
        ops_file = None
        for filename in ops_files:
            potential_path = os.path.join(policy_path, filename)
            if os.path.exists(potential_path):
                ops_file = potential_path
                break
        
        if not os.path.exists(eq_log_file):
            print(f"  Error: {eq_log_file} not found")
            continue
            
        if not ops_file:
            print(f"  Error: Operations file not found in {policy_path}")
            continue
        
        try:
            # Read equilibrium log (final capacities and operational cost)
            eq_df = pd.read_csv(eq_log_file)
            final_row = eq_df.iloc[-1]
            
            # Extract final capacities (MW)
            capacities = {
                'Nuclear': final_row['Nuclear_capacity_MW'],
                'CC': final_row['CC_capacity_MW'],
                'CT': final_row['CT_capacity_MW'],
                'ST': final_row['ST_capacity_MW'],
                'Wind': final_row['Wind_capacity_MW'],
                'Solar': final_row['Solar_capacity_MW'], 
                'Hydro': final_row['Hydro_capacity_MW'],
                'Battery': final_row['Battery_capacity_MW']
            }
            
            operational_cost = final_row['total_cost']
            
            # Calculate investment costs
            investment_cost = 0.0
            for tech, capacity in capacities.items():
                if tech == 'Battery':
                    battery_energy_cap = capacity * battery_duration
                    investment_cost += inv_costs['Battery_power'] * capacity
                    investment_cost += inv_costs['Battery_energy'] * battery_energy_cap
                else:
                    investment_cost += inv_costs[tech] * capacity
            
            # Calculate fixed O&M costs
            fixed_om_cost = 0.0
            for tech, capacity in capacities.items():
                if tech == 'Battery':
                    fixed_om_cost += fixed_om_costs['Battery'] * capacity
                else:
                    fixed_om_cost += fixed_om_costs[tech] * capacity
            
            # Total system cost
            total_system_cost = operational_cost + investment_cost + fixed_om_cost
            
            # Read operations file
            ops_df = pd.read_csv(ops_file)
            
            # Calculate unmet demand
            total_unmet_demand = ops_df['Load_Shed'].sum()
            
            # Calculate total demand served
            generation_cols = [col for col in ops_df.columns if 'Generation' in col]
            total_generation = ops_df[generation_cols].sum().sum()
            battery_discharge = ops_df.get('Battery_Discharge', pd.Series(0)).sum()
            total_served = total_generation + battery_discharge
            total_demand = total_served + total_unmet_demand
            
            unmet_demand_rate = (total_unmet_demand / total_demand) * 100 if total_demand > 0 else 0
            
            # Calculate demand-weighted average price
            if total_demand > 0:
                # Weight prices by total demand (served + unmet)
                demand_weights = total_served + ops_df['Load_Shed']
                weighted_avg_price = (ops_df['Price'] * demand_weights).sum() / demand_weights.sum()
            else:
                weighted_avg_price = ops_df['Price'].mean()
            
            max_price = ops_df['Price'].max()
            min_price = ops_df['Price'].min()
            price_volatility = ops_df['Price'].std() / ops_df['Price'].mean()
            
            # Store results
            results[policy_name] = {
                'total_system_cost': total_system_cost / 1e6,  # Convert to millions
                'operational_cost': operational_cost / 1e6,
                'investment_cost': investment_cost / 1e6,
                'fixed_om_cost': fixed_om_cost / 1e6,
                'unmet_demand_MWh': total_unmet_demand,
                'unmet_demand_rate_pct': unmet_demand_rate,
                'weighted_avg_price': weighted_avg_price,
                'max_price': max_price,
                'min_price': min_price,
                'price_volatility': price_volatility
            }
            
            # Print results
            print(f"  Total System Cost: ${total_system_cost/1e6:.1f} Million")
            print(f"    - Operational: ${operational_cost/1e6:.1f}M")
            print(f"    - Investment: ${investment_cost/1e6:.1f}M")
            print(f"    - Fixed O&M: ${fixed_om_cost/1e6:.1f}M")
            print(f"  Unmet Demand: {total_unmet_demand:.1f} MWh ({unmet_demand_rate:.3f}%)")
            print(f"  Demand-Weighted Avg Price: ${weighted_avg_price:.2f}/MWh")
            print(f"  Price Range: ${min_price:.2f} - ${max_price:.2f}/MWh")
            print(f"  Price Volatility: {price_volatility:.3f}")
            
        except Exception as e:
            print(f"  Error processing {policy_name}: {e}")
    
    print("\n" + "="*80)
    return results

if __name__ == "__main__":
    results = calculate_metrics_from_csvs()