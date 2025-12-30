import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

# Set seaborn theme for publication quality (Consistent with plot_calibration_results.py)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

def load_optimization_logs(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Optimization log file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def plot_gp_variance(df, output_path, label):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    
    # Plot Max Sigma (Orange - Critical/Upper Bound)
    sns.lineplot(data=df, x='iteration', y='sigma_max', label='Max Uncertainty', 
                 color='#ff7f0e', linestyle='--', linewidth=1.5, ax=ax)
    
    # Plot Mean Sigma (Blue - Average/Main)
    sns.lineplot(data=df, x='iteration', y='sigma_mean', label='Mean Uncertainty', 
                 color='#1f77b4', marker='o', linewidth=2, ax=ax)
    
    # Plot Min Sigma (Grey - Lower Bound)
    sns.lineplot(data=df, x='iteration', y='sigma_min', label='Min Uncertainty', 
                 color='#7f7f7f', linestyle=':', linewidth=1.5, ax=ax)
    
    # Formatting
    ax.set_title(f'Surrogate Model Uncertainty Reduction ({label})', fontweight='bold')
    ax.set_xlabel('BO Iteration')
    ax.set_ylabel('Predicted Standard Deviation (Sigma)')
    ax.legend(frameon=True)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    sns.despine()
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    
    # Save PDF
    pdf_path = os.path.splitext(output_path)[0] + '.pdf'
    plt.savefig(pdf_path)
    print(f"PDF saved to: {pdf_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot GP Variance Reduction from Optimization Logs")
    parser.add_argument("--json", type=str, required=True, help="Path to optimization_logs.json")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--label", type=str, default="B2", help="Experiment label")
    
    args = parser.parse_args()
    
    try:
        df = load_optimization_logs(args.json)
        
        # Calculate iteration relative to BO start (0, 1, 2...)
        # The logs are already 0-indexed relative to BO start in the loop logic
        # But let's check. In run_calibration_l1_loop.py, bo_start_iter is used.
        # But 'iteration' in json is 'i', which is the global iteration index? 
        # range(bo_start_iter, total_iters) -> i
        # So it is the global index.
        # Maybe we want to shift x-axis to be "BO Iterations"?
        # Let's keep global iteration for context, but maybe annotate.
        
        output_path = args.output
        if not output_path:
             root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(args.json))))
             output_path = os.path.join(root, f"plots/{args.label}_gp_variance.png")

        plot_gp_variance(df, output_path, args.label)
        
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()
