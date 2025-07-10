import matplotlib.pyplot as plt

# Define the data
scenarios = ['Perfect Foresight', 'SLAC', 'DLAC-i']
categories = ['Nuclear', 'Wind', 'Gas', 'Battery']
data = {
    'Perfect Foresight': [130.2, 148.0, 1760.3, 419.7],
    'SLAC': [186.5, 81.6, 1922.1, 311.0],
    'DLAC-i': [272.1, 158.7, 1986.2, 152.3]
}
totals = [sum(data[scenario]) for scenario in scenarios]
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

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
ax.set_title('Equilibrium Capacity Mix by Policy')
ax.legend(title='Technology', loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=4)
ax.set_ylim(0, max(totals) * 1.2)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
