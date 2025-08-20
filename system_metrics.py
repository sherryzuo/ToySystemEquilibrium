#!/usr/bin/env python3

import pandas as pd
import os
import numpy as np

def calculate_system_metrics(base_dir="results/equilibrium/anderson"):
    """Calculate total system cost, unmet demand, and weighted average price for all equilibrium policies"""
    
    # Policy mapping
    policies = {
        "perfectforesight": "Perfect Foresight",
        "slac": "SLAC",
        "dlac_i": "DLAC-i"
    }
    
    # Cost parameters (from SystemConfig.jl)
    cost_params = {
        'Nuclear': {'inv_cost': 120000.0, 'fixed_om': 35000.0},
        'CC': {'inv_cost': 105000.0, 'fixed_om': 12000.0},  # Gas * 1.5 multiplier
        'CT': {'inv_cost': 105000.0, 'fixed_om': 12000.0},  # Gas * 1.5 multiplier
        'ST': {'inv_cost': 105000.0, 'fixed_om': 12000.0},  # Gas * 1.5 multiplier
        'Wind': {'inv_cost': 76500.0, 'fixed_om': 12000.0},  # Wind * 0.9 multiplier
        'Solar': {'inv_cost': 59500.0, 'fixed_om': 12000.0}, # Solar * 0.7 multiplier
        'Hydro': {'inv_cost': 120000.0, 'fixed_om': 35000.0},
        'Battery_power': {'inv_cost': 76000.0, 'fixed_om': 6000.0},  # Battery * 0.8 multiplier
        'Battery_energy': {'inv_cost': 80.0}  # Battery energy * 0.8 multiplier
    }
    
    battery_duration = 4.0  # hours
    
    results = {}
    
    print("="*80)
    print("EQUILIBRIUM SYSTEM METRICS SUMMARY")
    print("="*80)
    
    for policy_dir, display_name in policies.items():
        policy_path = os.path.join(base_dir, policy_dir)
        
        # Read equilibrium log for capacities and operational cost
        equilibrium_log = os.path.join(policy_path, "equilibrium_log.csv")
        operations_file = None
        
        # Find operations file
        for file_name in ["perfect_foresight_operations.csv", "slac_cached_operations.csv", "dlac_i_cached_operations.csv"]:
            ops_path = os.path.join(policy_path, file_name)
            if os.path.exists(ops_path):
                operations_file = ops_path
                break
        
        if not os.path.exists(equilibrium_log):
            print(f"Warning: {equilibrium_log} not found")
            continue
            
        if not operations_file:
            print(f"Warning: Operations file not found for {display_name}")
            continue
            
        try:
            # Read final capacities from equilibrium log
            eq_df = pd.read_csv(equilibrium_log)
            final_row = eq_df.iloc[-1]
            
            # Extract capacities
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
            for tech, cap in capacities.items():
                if tech == 'Battery':
                    battery_energy_cap = cap * battery_duration
                    investment_cost += cost_params['Battery_power']['inv_cost'] * cap
                    investment_cost += cost_params['Battery_energy']['inv_cost'] * battery_energy_cap
                else:
                    investment_cost += cost_params[tech]['inv_cost'] * cap
            
            # Calculate fixed O&M costs
            fixed_om_cost = 0.0
            for tech, cap in capacities.items():
                if tech == 'Battery':
                    fixed_om_cost += cost_params['Battery_power']['fixed_om'] * cap
                else:
                    fixed_om_cost += cost_params[tech]['fixed_om'] * cap
            
            # Total system cost
            total_system_cost = operational_cost + investment_cost + fixed_om_cost
            
            # Read operations data
            ops_df = pd.read_csv(operations_file)
            
            # Calculate metrics
            total_unmet_demand = ops_df['Load_Shed'].sum()
            unmet_demand_rate = (total_unmet_demand / (ops_df['Load_Shed'].sum() + ops_df.get('Nuclear_Generation', 0).sum() + 
                                                      ops_df.get('CC_Generation', 0).sum() + ops_df.get('CT_Generation', 0).sum() + 
                                                      ops_df.get('Wind_Generation', 0).sum() + ops_df.get('Solar_Generation', 0).sum() +
                                                      ops_df.get('Battery_Discharge', 0).sum())) * 100
            
            # Calculate demand-weighted average price
            total_demand = ops_df['Load_Shed'].sum()
            for col in ops_df.columns:
                if 'Generation' in col or col == 'Battery_Discharge':
                    total_demand += ops_df[col].sum()
            
            if total_demand > 0:
                weighted_avg_price = (ops_df['Price'] * ops_df['Load_Shed'].sum()).sum() / total_demand
            else:
                weighted_avg_price = ops_df['Price'].mean()
            
            max_price = ops_df['Price'].max()
            price_volatility = ops_df['Price'].std() / ops_df['Price'].mean()
            
            # Store and display results
            results[display_name] = {
                'total_system_cost_M': total_system_cost / 1e6,
                'operational_cost_M': operational_cost / 1e6,
                'investment_cost_M': investment_cost / 1e6,
                'fixed_om_cost_M': fixed_om_cost / 1e6,
                'unmet_demand_MWh': total_unmet_demand,
                'unmet_demand_rate_pct': unmet_demand_rate,
                'weighted_avg_price': weighted_avg_price,
                'max_price': max_price,
                'price_volatility': price_volatility
            }
            
            print(f"\n{display_name}:")
            print(f"  Total System Cost: ${total_system_cost/1e6:.1f} Million")
            print(f"    - Operational: ${operational_cost/1e6:.1f}M")
            print(f"    - Investment: ${investment_cost/1e6:.1f}M") 
            print(f"    - Fixed O&M: ${fixed_om_cost/1e6:.1f}M")
            print(f"  Unmet Demand: {total_unmet_demand:.1f} MWh")
            print(f"  Unmet Demand Rate: {unmet_demand_rate:.3f}%")
            print(f"  Demand-Weighted Avg Price: ${weighted_avg_price:.2f}/MWh")
            print(f"  Max Price: ${max_price:.2f}/MWh")
            print(f"  Price Volatility: {price_volatility:.3f}")
            
        except Exception as e:
            print(f"Error processing {display_name}: {e}")
    
    print("\n" + "="*80)
    return results

if __name__ == "__main__":
    results = calculate_system_metrics()