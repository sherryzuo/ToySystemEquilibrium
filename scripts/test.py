import matplotlib.pyplot as plt
# Data from the image
labels = ["NUC", "CCGT", "WIND"]
PAH_values = [76.57, 188.16, 9.99]
Optimal_values = [86.46, 114.96, 23.09]
# Bar positions
bar_width = 0.35
x = [0, 0.5]

# Define new distinct colors for each technology
colors = {
    "NUC": "#a6cee3",   # Light blue
    "CCGT": "#b2df8a",  # Light green
    "WIND": "#fb9a99"   # Light red
}
plt.rcParams['font.size'] = 18
# Set font to Times New Roman and further increase all font sizes for slide visibility
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.titlesize'] = 22    # Much larger title font size
plt.rcParams['axes.labelsize'] = 22   # Much larger axes label font size
plt.rcParams['xtick.labelsize'] = 22   # Much larger x-tick font size
plt.rcParams['ytick.labelsize'] = 22   # Much larger y-tick font size
plt.rcParams['legend.fontsize'] = 22  # Larger legend font size
plt.rcParams['legend.title_fontsize'] = 22 # Larger legend title font size

# Create the stacked bar plot with segment labels
fig, ax = plt.subplots(figsize=(8, 7))

# Plot PAH stack
pah_bottom = 0
for i, tech in enumerate(labels):
    value = PAH_values[i]
    ax.bar(x[0], value, bar_width, bottom=pah_bottom, color=colors[tech], label=tech if x[0] == 0 else "")
    ax.text(x[0], pah_bottom + value / 2, f'{value:.2f}', ha='center', va='center')
    pah_bottom += value

# Plot Optimal stack
opt_bottom = 0
for i, tech in enumerate(labels):
    value = Optimal_values[i]
    ax.bar(x[1], value, bar_width, bottom=opt_bottom, color=colors[tech])
    ax.text(x[1], opt_bottom + value / 2, f'{value:.2f}', ha='center', va='center')
    opt_bottom += value

# Add total labels
ax.text(x[0], sum(PAH_values) + 5, f'{sum(PAH_values):.1f} MW', ha='center')
ax.text(x[1], sum(Optimal_values) + 5, f'{sum(Optimal_values):.1f} MW', ha='center')

# Increase space above bars
ax.set_ylim(0, max(sum(PAH_values), sum(Optimal_values)) * 1.25)

# Set x-axis
ax.set_xticks(x)
ax.set_xticklabels(['PAH Capacity Mix', 'Optimal Capacity Mix'])
ax.set_ylabel('Capacity (MW)')
ax.set_title('Capacity Mix Comparison')

# # Add legend
# ax.legend(title='Technology', loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3)


plt.tight_layout()
plt.show()