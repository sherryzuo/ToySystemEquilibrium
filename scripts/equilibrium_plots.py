import pandas as pd
import matplotlib.pyplot as plt
import argparse
import math
import os
import sys
import subprocess

# Try to import seaborn, install if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    print("Seaborn not found. Installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
        import seaborn as sns
        HAS_SEABORN = True
        print("Seaborn installed successfully!")
    except Exception as e:
        print(f"Could not install seaborn: {e}")
        print("Continuing without seaborn (using matplotlib only)")
        HAS_SEABORN = False

def plot_capacity_equilibrium_analysis(log_file, output_dir=None, label=None, start_iter=None, end_iter=None, zoom_only=False):
    """
    Creates four figures from capacity equilibrium log data:
    1. Maximum PMR over iterations
    2. All PMRs over iterations
    3. Individual capacity evolution plots for each generator
    4. Total cost evolution over iterations
    
    Args:
        log_file: Path to the CSV log file
        output_dir: Directory to save output plots (defaults to same directory as log file)
        start_iter: Start iteration for zoomed plots (optional)
        end_iter: End iteration for zoomed plots (optional)
        zoom_only: If True, only create zoomed plots, skip full plots
    
    Returns:
        List of paths to the generated figures
    """
    # Read data
    df = pd.read_csv(log_file)
    
    # Handle zooming
    zoom_suffix = ""
    if start_iter is not None or end_iter is not None:
        # Filter data based on iteration range
        if start_iter is not None:
            df = df[df['Iteration'] >= start_iter]
        if end_iter is not None:
            df = df[df['Iteration'] <= end_iter]
        
        # Update suffix for file names
        start_str = str(start_iter) if start_iter is not None else "start"
        end_str = str(end_iter) if end_iter is not None else "end"
        zoom_suffix = f"_zoom_{start_str}_to_{end_str}"
        
        if len(df) == 0:
            print(f"Warning: No data found in iteration range {start_iter} to {end_iter}")
            return []
        
        print(f"Plotting zoomed view: iterations {df['Iteration'].min()} to {df['Iteration'].max()}")
    
    # Set plot style
    plt.style.use('ggplot')
    if HAS_SEABORN:
        sns.set_palette("colorblind")
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(log_file)
    
    # Extract file name without extension for output files
    base_name = os.path.splitext(os.path.basename(log_file))[0]
    
    # Extract generator capacity columns and names (updated for our format)
    capacity_columns = [col for col in df.columns if '_capacity_MW' in col]
    generator_names = [col.replace('_capacity_MW', '') for col in capacity_columns]
    
    # Extract PMR columns (updated for our format)
    pmr_columns = [gen + "_pmr" for gen in generator_names]
    
    # List to store output file paths
    output_files = []
    
    #-------------------------------------------------------------------------
    # Figure 1: Max PMR over iterations
    #-------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(df['Iteration'], df['max_pmr'], marker='o', linestyle='-', 
             linewidth=2, markersize=6, color='darkblue')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Maximum PMR', fontsize=12)
    plt.title(f'Maximum Absolute Profit Margin Ratio (PMR) Over Iterations \n{label}', fontsize=14)
    plt.grid(True)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add final value annotation
    final_max_pmr = df['max_pmr'].iloc[-1]
    plt.annotate(f'Final: {final_max_pmr:.4f}', 
                 xy=(df['Iteration'].iloc[-1], final_max_pmr),
                 xytext=(5, 5), textcoords='offset points',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
    
    plt.tight_layout()
    
    # Save figure
    max_pmr_file = os.path.join(output_dir, f"{base_name}_max_pmr{zoom_suffix}.png")
    plt.savefig(max_pmr_file, dpi=300, bbox_inches='tight')
    output_files.append(max_pmr_file)
    plt.close()
    
    #-------------------------------------------------------------------------
    # Figure 2: All PMRs over iterations
    #-------------------------------------------------------------------------
    plt.figure(figsize=(12, 8))
    
    # Plot all PMRs together
    for i, (col, name) in enumerate(zip(pmr_columns, generator_names)):
        plt.plot(df['Iteration'], df[col], marker='o', linestyle='-', 
                 label=name, linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel(f'Profit Margin Ratio (PMR) ', fontsize=12)
    plt.title(f'Generator Profit Margin Ratios \n {label}', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True)
    plt.axhline(y=0, color='black', linestyle='--')
    
    # # Add max PMR as a separate line
    # plt.plot(df['Iteration'], df['max_pmr'], marker='s', linestyle=':', 
    #          label='Maximum PMR', color='black', linewidth=2)
    
    # Create a table of final values
    last_row = df.iloc[-1]
    textstr = "Final PMRs:\n"
    for i, name in enumerate(generator_names):
        pmr_col = pmr_columns[i]
        textstr += f"{name}: {last_row[pmr_col]:.4f}\n"
    
    # Add text box with final PMRs
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    plt.text(0.02, 0.02, textstr, fontsize=10,
             verticalalignment='bottom', transform=plt.gca().transAxes,
             bbox=props)
    
    plt.tight_layout()
    
    # Save figure
    all_pmrs_file = os.path.join(output_dir, f"{base_name}_all_pmrs{zoom_suffix}.png")
    plt.savefig(all_pmrs_file, dpi=300, bbox_inches='tight')
    output_files.append(all_pmrs_file)
    plt.close()
    
    #-------------------------------------------------------------------------
    # Figure 3: Individual capacity evolution plots
    #-------------------------------------------------------------------------
    # Calculate grid dimensions for subplots
    n_generators = len(generator_names)
    n_cols = min(3, n_generators)
    n_rows = math.ceil(n_generators / n_cols)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_generators > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot capacity evolution for each generator
    for i, (col, name) in enumerate(zip(capacity_columns, generator_names)):
        ax = axes[i]
        # Choose color palette based on seaborn availability
        if HAS_SEABORN:
            color = sns.color_palette("colorblind")[i % 10]
        else:
            # Fallback to matplotlib default colors
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            color = colors[i % len(colors)]
        
        ax.plot(df['Iteration'], df[col], marker='o', linestyle='-', 
                linewidth=2, markersize=6, color=color)
        
        # Set titles and labels
        ax.set_title(f'{name} Capacity Evolution \n {label}', fontsize=12)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Capacity (MW)')
        ax.grid(True)
        
        # Add final capacity as text
        last_row = df.iloc[-1]
        final_capacity = last_row[col]
        
        # Add PMR information as a secondary y-axis
        pmr_col = pmr_columns[i]
        ax2 = ax.twinx()
        ax2.plot(df['Iteration'], df[pmr_col], marker='x', linestyle='--', 
                 color='darkred', alpha=0.7)
        ax2.set_ylabel('PMR', color='darkred')
        ax2.tick_params(axis='y', labelcolor='darkred')
        ax2.axhline(y=0, color='darkred', linestyle='--', alpha=0.5)
        
        # Add text with final values
        textstr = f"Final Capacity: {final_capacity:.2f} MW\nFinal PMR: {last_row[pmr_col]:.4f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    capacity_file = os.path.join(output_dir, f"{base_name}_capacity_evolution{zoom_suffix}.png")
    plt.savefig(capacity_file, dpi=300, bbox_inches='tight')
    output_files.append(capacity_file)
    plt.close()
    
    #-------------------------------------------------------------------------
    # Figure 4: Total cost evolution
    #-------------------------------------------------------------------------
    if 'total_cost' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['Iteration'], df['total_cost'], marker='o', linestyle='-', 
                 linewidth=2, markersize=4, color='darkgreen')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Total Cost ($)', fontsize=12)
        plt.title(f'Total Cost Evolution \\n{label}', fontsize=14)
        plt.grid(True)
        
        # Add final value annotation
        final_cost = df['total_cost'].iloc[-1]
        plt.annotate(f'Final: ${final_cost:,.0f}', 
                     xy=(df['Iteration'].iloc[-1], final_cost),
                     xytext=(5, 5), textcoords='offset points',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
        
        # Format y-axis with scientific notation if needed
        if final_cost > 1e6:
            plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        
        # Save figure
        cost_file = os.path.join(output_dir, f"{base_name}_cost_evolution{zoom_suffix}.png")
        plt.savefig(cost_file, dpi=300, bbox_inches='tight')
        output_files.append(cost_file)
        plt.close()
    
    print(f"Generated {len(output_files)} plot files:")
    for file in output_files:
        print(f" - {file}")
    
    return output_files

def main():
    """
    Main function with argument parsing for ToySystemEquilibrium repository.
    """
    parser = argparse.ArgumentParser(description='Plot equilibrium analysis from ToySystemEquilibrium')
    parser.add_argument('log_file', nargs='?', 
                       help='Path to equilibrium log CSV file')
    parser.add_argument('--output-dir', '-o', 
                       help='Output directory for plots (default: same as log file)')
    parser.add_argument('--label', '-l', 
                       help='Label for plots (default: auto-generated from filename)')
    parser.add_argument('--policy', '-p', choices=['dlac', 'pf', 'both', 'validation'],
                       help='Policy to plot (will look for log files automatically)')
    parser.add_argument('--validation', action='store_true',
                       help='Plot validation results from results/validation/equilibrium/')
    parser.add_argument('--start-iter', type=int,
                       help='Start iteration for zoomed plot (optional)')
    parser.add_argument('--end-iter', type=int,
                       help='End iteration for zoomed plot (optional)')
    parser.add_argument('--zoom-only', action='store_true',
                       help='Only create zoomed plots, skip full plots')
    
    args = parser.parse_args()
    
    # If no log file specified, try to find them automatically
    if not args.log_file and not args.policy:
        # Look for log files in results/equilibrium/
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        equilibrium_dir = os.path.join(base_dir, "results", "equilibrium")
        
        if os.path.exists(equilibrium_dir):
            print("Available equilibrium results:")
            for policy_dir in os.listdir(equilibrium_dir):
                policy_path = os.path.join(equilibrium_dir, policy_dir)
                if os.path.isdir(policy_path):
                    log_file = os.path.join(policy_path, "equilibrium_log.csv")
                    if os.path.exists(log_file):
                        print(f"  {policy_dir}: {log_file}")
            print("\nUsage: python equilibrium_plots.py <log_file> [options]")
            print("   or: python equilibrium_plots.py --policy <dlac|pf|both|validation>")
            print("   or: python equilibrium_plots.py --validation")
        else:
            print("No equilibrium results found. Run equilibrium solver first.")
        return
    
    # Handle validation flag or validation policy
    if args.validation or args.policy == "validation":
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        validation_dir = os.path.join(base_dir, "results", "validation", "equilibrium")
        
        print(f"\nLooking for validation results in: {validation_dir}")
        
        if os.path.exists(validation_dir):
            # Look for all policy directories in validation
            for policy_dir in os.listdir(validation_dir):
                policy_path = os.path.join(validation_dir, policy_dir)
                if os.path.isdir(policy_path):
                    log_file = os.path.join(policy_path, "equilibrium_log.csv")
                    
                    if os.path.exists(log_file):
                        output_dir = args.output_dir or os.path.join(policy_path, "plots")
                        os.makedirs(output_dir, exist_ok=True)
                        
                        label = args.label or f"Validation Test - {policy_dir.upper()}"
                        
                        print(f"\nPlotting validation {policy_dir} equilibrium analysis...")
                        plot_capacity_equilibrium_analysis(log_file, output_dir, label, 
                                                          args.start_iter, args.end_iter, args.zoom_only)
                    else:
                        print(f"Log file not found for validation {policy_dir}: {log_file}")
        else:
            print(f"Validation directory not found: {validation_dir}")
        return

    # Auto-find log files based on policy
    if args.policy:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        equilibrium_dir = os.path.join(base_dir, "results", "equilibrium")
        
        policies_to_plot = []
        if args.policy == "both":
            policies_to_plot = ["dlac_i", "perfectforesight"]
        elif args.policy == "dlac":
            policies_to_plot = ["dlac_i"]
        elif args.policy == "pf":
            policies_to_plot = ["perfectforesight"]
        
        for policy in policies_to_plot:
            policy_path = os.path.join(equilibrium_dir, policy)
            log_file = os.path.join(policy_path, "equilibrium_log.csv")
            
            if os.path.exists(log_file):
                output_dir = args.output_dir or os.path.join(policy_path, "plots")
                os.makedirs(output_dir, exist_ok=True)
                
                label = args.label or f"ToySystemEquilibrium - {policy.upper()}"
                
                print(f"\nPlotting {policy} equilibrium analysis...")
                plot_capacity_equilibrium_analysis(log_file, output_dir, label, 
                                                  args.start_iter, args.end_iter, args.zoom_only)
            else:
                print(f"Log file not found for {policy}: {log_file}")
        return
    
    # Single log file specified
    log_file = args.log_file
    if not os.path.exists(log_file):
        print(f"Error: Log file not found: {log_file}")
        return
    
    output_dir = args.output_dir or os.path.join(os.path.dirname(log_file), "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Auto-generate label from filename if not provided
    if not args.label:
        filename = os.path.basename(log_file)
        if "dlac" in filename.lower():
            label = "ToySystemEquilibrium - DLAC-i"
        elif "pf" in filename.lower() or "perfect" in filename.lower():
            label = "ToySystemEquilibrium - Perfect Foresight"
        else:
            label = "ToySystemEquilibrium - Equilibrium Analysis"
    else:
        label = args.label
    
    print(f"Plotting equilibrium analysis from: {log_file}")
    print(f"Output directory: {output_dir}")
    print(f"Label: {label}")
    
    plot_capacity_equilibrium_analysis(log_file, output_dir, label, 
                                      args.start_iter, args.end_iter, args.zoom_only)

if __name__ == "__main__":
    main()