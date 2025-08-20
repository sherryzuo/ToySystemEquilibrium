import matplotlib.pyplot as plt
import pandas as pd
import os

def read_equilibrium_capacities(base_dir="results/equilibrium/anderson"):
    """Read final equilibrium capacities from log files"""
    
    # Policy mapping: (directory_name, display_name)
    policies = [
        ("perfectforesight", "Perfect Foresight"),
        ("slac", "SLAC"), 
        ("dlac_i", "DLAC-i")
    ]
    
    scenarios = []
    data = {}
    
    for policy_dir, display_name in policies:
        log_file = os.path.join(base_dir, policy_dir, "equilibrium_log.csv")
        
        if os.path.exists(log_file):
            try:
                # Read the last row of the equilibrium log
                df = pd.read_csv(log_file)
                final_row = df.iloc[-1]
                
                # Extract capacities (MW) for all NYISO classes
                nuclear = final_row["Nuclear_capacity_MW"]
                cc = final_row["CC_capacity_MW"]
                ct = final_row["CT_capacity_MW"] 
                st = final_row["ST_capacity_MW"]
                wind = final_row["Wind_capacity_MW"]
                solar = final_row["Solar_capacity_MW"]
                hydro = final_row["Hydro_capacity_MW"]
                battery = final_row["Battery_capacity_MW"]
                
                # Store data: [Nuclear, CC, CT, ST, Wind, Solar, Hydro, Battery]
                scenarios.append(display_name)
                data[display_name] = [nuclear, cc, ct, st, wind, solar, hydro, battery]
                
                print(f"{display_name} capacities:")
                print(f"  Nuclear: {nuclear:.1f} MW")
                print(f"  CC (Combined Cycle): {cc:.1f} MW")
                print(f"  CT (Combustion Turbine): {ct:.1f} MW") 
                print(f"  ST (Steam Turbine): {st:.1f} MW")
                print(f"  Wind: {wind:.1f} MW")
                print(f"  Solar: {solar:.1f} MW")
                print(f"  Hydro: {hydro:.1f} MW")
                print(f"  Battery: {battery:.1f} MW")
                print()
                
            except Exception as e:
                print(f"Error reading {log_file}: {e}")
        else:
            print(f"Log file not found: {log_file}")
    
    return scenarios, data

# Read actual equilibrium results
scenarios, data = read_equilibrium_capacities()

# Fallback to hardcoded data if reading fails
if not scenarios or not data:
    print("Warning: Could not read equilibrium results, using fallback data")
    scenarios = ['Perfect Foresight', 'SLAC', 'DLAC-i']
    categories = ['Nuclear', 'CC', 'CT', 'ST', 'Wind', 'Solar', 'Hydro', 'Battery']
    data = {
        'Perfect Foresight': [3536.8, 13658.9, 5386.5, 0.0, 20026.6, 1184.8, 4924.0, 2935.5],
        'SLAC': [3536.8, 13849.2, 5652.3, 0.0, 20104.9, 864.1, 4924.0, 2726.6],
        'DLAC-i': [3536.8, 14254.8, 6239.2, 0.0, 20077.8, 611.8, 4924.0, 2093.4]
    }
else:
    # Define categories for all NYISO classes
    categories = ['Nuclear', 'CC', 'CT', 'ST', 'Wind', 'Solar', 'Hydro', 'Battery']

totals = [sum(data[scenario]) for scenario in scenarios]

# Define colors for all 8 technology classes (colorblind-friendly palette)
colors = [
    '#1f77b4',  # Nuclear - Blue
    '#ff7f0e',  # CC - Orange  
    '#2ca02c',  # CT - Green
    '#d62728',  # ST - Red
    '#9467bd',  # Wind - Purple
    '#8c564b',  # Solar - Brown
    '#e377c2',  # Hydro - Pink
    '#7f7f7f'   # Battery - Gray
]

# Set font to Times New Roman and further increase all font sizes for slide visibility
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.titlesize'] = 22      # Much larger title font size
plt.rcParams['axes.labelsize'] = 16      # Much larger axes label font size
plt.rcParams['xtick.labelsize'] = 18     # Much larger x-tick font size
plt.rcParams['ytick.labelsize'] = 18     # Much larger y-tick font size
plt.rcParams['legend.fontsize'] = 16    # Larger legend font size
plt.rcParams['legend.title_fontsize'] = 16  # Larger legend title font size

# Create plot
fig, ax = plt.subplots(figsize=(10, 8))
bottom = [0] * len(scenarios)

for i, category in enumerate(categories):
    values = [data[scenario][i] for scenario in scenarios]
    bars = ax.bar(scenarios, values, bottom=bottom, label=category, color=colors[i])
    
    # Add label inside each segment
    for j, bar in enumerate(bars):
        height = bar.get_height()
        if height > 40:  # only label if space allows
            ax.text(bar.get_x() + bar.get_width() / 2, bottom[j] + height / 2,
                    f'{height:.1f}', ha='center', va='center', fontsize=16, color='white')  # Larger segment label font
        bottom[j] += height

# Add total value on top of each stack
for i, total in enumerate(totals):
    ax.text(i, bottom[i] + 30, f'Total: {total:.0f} MW', ha='center', va='bottom', fontsize=18)  # Larger total label font

# Customize plot
ax.set_ylabel('Capacity (MW)')
ax.set_title('Equilibrium Capacity Mix by Policy - All NYISO Classes')
ax.legend(title='Technology', loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
ax.set_ylim(0, max(totals) * 1.2)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=15)

# Adjust layout to accommodate legend
plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
plt.show()
