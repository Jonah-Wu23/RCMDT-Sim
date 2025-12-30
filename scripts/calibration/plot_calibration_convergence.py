# -*- coding: utf-8 -*-
"""
Calibration Convergence Visualization
Generates publication-quality convergence plots for L1 calibration results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import warnings

# Suppress FutureWarnings from seaborn
warnings.filterwarnings('ignore', category=FutureWarning)

# Set publication-quality style with Times New Roman (IEEE Style)
# plt.style.use('seaborn-v0_8-whitegrid') # Disabled for IEEE style
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False
})


def plot_convergence_dual(df, output_dir):
    """Generate dual-panel convergence plot (Combined Loss + Per-Route RMSE)"""
    
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))  # IEEE double column width
    
    # Panel 1: Combined Loss with Cumulative Best
    ax1 = axes[0]
    
    # Calculate cumulative best
    cum_best = df['rmse'].cummin()
    
    # Plot all iterations (support warm start types)
    colors = {
        'initial': '#1f77b4',  # Blue for LHS Initial
        'bo': '#ff7f0e',        # Orange for Bayesian Opt.
        'warm_initial': '#aec7e8',  # Light blue for warm start initial
        'warm_bo': '#ffbb78'        # Light orange for warm start BO
    }
    labels = {
        'initial': 'LHS Initial',
        'bo': 'Bayesian Opt.',
        'warm_initial': 'Warm Start (Initial)',
        'warm_bo': 'Warm Start (BO)'
    }
    
    for phase in df['type'].unique():
        mask = df['type'] == phase
        if mask.sum() == 0:
            continue
        color = colors.get(phase, '#95a5a6')  # 默认灰色
        label = labels.get(phase, phase)
        ax1.scatter(df.loc[mask, 'iter'], df.loc[mask, 'rmse'], 
                   c=color, label=label, alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
    
    # Plot cumulative best line
    ax1.plot(df['iter'], cum_best, 'k-', linewidth=2, label='Cumulative Best', alpha=0.8)
    ax1.fill_between(df['iter'], cum_best, cum_best.max(), alpha=0.1, color='green')
    
    # Mark best point
    best_idx = df['rmse'].idxmin()
    best_row = df.loc[best_idx]
    ax1.scatter([best_row['iter']], [best_row['rmse']], 
               marker='*', s=300, c='gold', edgecolors='black', linewidths=1.5, 
               zorder=10, label=f'Best: {best_row["rmse"]:.1f}')
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Combined Loss (RMSE)')
    ax1.set_title('(a) Optimization Convergence')
    ax1.legend(loc='upper right')
    ax1.set_xlim(-0.5, len(df) - 0.5)
    
    # Panel 2: Per-Route RMSE
    ax2 = axes[1]
    
    # Plot 68X and 960 RMSE
    ax2.plot(df['iter'], df['rmse_68x'], 'o-', color='#2ecc71', 
            label='68X (Target)', linewidth=2, markersize=6, alpha=0.8)
    ax2.plot(df['iter'], df['rmse_960'], 's-', color='#9b59b6', 
            label='960 (Anchor)', linewidth=2, markersize=6, alpha=0.8)
    
    # Draw constraint threshold
    ax2.axhline(y=350, color='red', linestyle='--', linewidth=2, 
               label='960 Constraint (350)', alpha=0.7)
    
    # Shade constraint violation region
    ax2.fill_between(df['iter'], 350, df['rmse_960'].max() + 50, 
                    alpha=0.15, color='red', label='Violation Zone')
    
    # Mark constraint violations
    violations = df[df['rmse_960'] > 350]
    if len(violations) > 0:
        ax2.scatter(violations['iter'], violations['rmse_960'], 
                   marker='x', s=100, c='red', linewidths=2, zorder=5)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Route RMSE (seconds)')
    ax2.set_title('(b) Per-Route Performance')
    ax2.legend(loc='upper right', frameon=True, facecolor='white', 
               edgecolor='gray', framealpha=1.0)
    ax2.set_xlim(-0.5, len(df) - 0.5)
    ax2.set_ylim(0, max(df['rmse_960'].max(), df['rmse_68x'].max()) * 1.1)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'l1_calibration_convergence.png'
    plt.savefig(output_path)
    print(f"[INFO] Saved convergence plot: {output_path}")
    
    # Also save as PDF for publication
    pdf_path = output_dir / 'l1_calibration_convergence.pdf'
    plt.savefig(pdf_path, format='pdf')
    print(f"[INFO] Saved PDF version: {pdf_path}")
    
    plt.close()


def plot_parameter_evolution(df, output_dir):
    """Generate parameter evolution heatmap"""
    
    params = ['t_board', 't_fixed', 'tau', 'sigma', 'minGap', 'accel', 'decel']
    
    # Normalize parameters to [0, 1] for visualization
    param_data = df[params].copy()
    param_normalized = (param_data - param_data.min()) / (param_data.max() - param_data.min())
    
    fig, ax = plt.subplots(figsize=(7.16, 2.5))  # IEEE double column width
    
    # Create heatmap
    sns.heatmap(param_normalized.T, cmap='RdYlBu_r', ax=ax,
                xticklabels=df['iter'].astype(int),
                yticklabels=params,
                cbar_kws={'label': 'Normalized Value'})
    
    # Mark best iteration
    best_iter = df['rmse'].idxmin()
    ax.axvline(x=best_iter + 0.5, color='black', linewidth=2, linestyle='--')
    # Position BEST text at the top (y < 0) to avoid x-axis labels at bottom
    ax.text(best_iter + 0.7, -0.2, 'BEST', fontsize=8, color='black', fontweight='bold')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Parameter')
    ax.set_title('Parameter Exploration History', pad=20)
    
    plt.tight_layout()
    
    output_path = output_dir / 'l1_parameter_evolution.png'
    plt.savefig(output_path)
    print(f"[INFO] Saved parameter evolution: {output_path}")
    plt.close()


def plot_phase_comparison(df, output_dir):
    """Generate box plot comparing LHS vs BO phases"""
    
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))  # IEEE double column width
    
    # Prepare data for seaborn (support warm start types)
    phase_map = {
        'initial': 'LHS Initial', 
        'bo': 'Bayesian Opt.',
        'warm_initial': 'Warm Start',
        'warm_bo': 'Warm Start'
    }
    df['Phase'] = df['type'].map(lambda x: phase_map.get(x, x))
    
    metrics = [('rmse', 'Combined Loss'), ('rmse_68x', '68X RMSE'), ('rmse_960', '960 RMSE')]
    palette = {'LHS Initial': '#1f77b4', 'Bayesian Opt.': '#ff7f0e', 'Warm Start': '#2ca02c'}
    
    for ax, (metric, title) in zip(axes, metrics):
        # Filter out constraint-violated entries for clearer visualization
        plot_df = df[df['rmse'] < 1000].copy()
        
        sns.boxplot(x='Phase', y=metric, hue='Phase', data=plot_df, ax=ax, 
                   palette=palette, legend=False)
        sns.stripplot(x='Phase', y=metric, data=plot_df, ax=ax,
                     color='black', alpha=0.5, size=5, zorder=3)

        counts = plot_df['Phase'].value_counts()
        ymax = plot_df[metric].max()
        y_text = ymax * 1.05 if ymax > 0 else 1.0
        for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):
            phase = label.get_text()
            n = counts.get(phase, 0)
            ax.text(tick, y_text, f"n={n}", ha='center', va='bottom', fontsize=7)
        
        ax.set_ylabel('RMSE (seconds)')
        ax.set_xlabel('')
        ax.set_title(title)
    
    plt.tight_layout()
    
    output_path = output_dir / 'l1_phase_comparison.png'
    plt.savefig(output_path)
    print(f"[INFO] Saved phase comparison: {output_path}")
    plt.close()


def generate_summary_report(df, output_dir):
    """Generate text summary report"""
    
    best_idx = df['rmse'].idxmin()
    best = df.loc[best_idx]
    
    valid_runs = df[df['rmse_960'] <= 350]
    
    report = f"""
================================================================================
L1 CALIBRATION SUMMARY REPORT
Generated: 2025-12-21
================================================================================

OPTIMIZATION SETTINGS:
  - Total Iterations: {len(df)}
  - Initial Samples (LHS): {len(df[df['type']=='initial'])}
  - Bayesian Optimization: {len(df[df['type']=='bo'])}
  - Warm Start Points: {len(df[df['type'].str.startswith('warm')])}

CONVERGENCE ANALYSIS:
  - First 10 Avg Loss: {df.head(10)['rmse'].mean():.2f}
  - Last 10 Avg Loss: {df.tail(10)['rmse'].mean():.2f}
  - Improvement: {((df.head(10)['rmse'].mean() - df.tail(10)['rmse'].mean()) / df.head(10)['rmse'].mean() * 100):.1f}%

CONSTRAINT EFFECTIVENESS:
  - Constraint Threshold: 350 (960 RMSE)
  - Violations: {len(df[df['rmse_960'] > 350])} / {len(df)} ({100*len(df[df['rmse_960'] > 350])/len(df):.1f}%)
  - Satisfaction Rate: {100*len(valid_runs)/len(df):.1f}%

BEST RESULT (Iteration {int(best['iter'])}):
  Parameters:
    t_board  = {best['t_board']:.4f} s
    t_fixed  = {best['t_fixed']:.4f} s
    tau      = {best['tau']:.4f} s
    sigma    = {best['sigma']:.4f}
    minGap   = {best['minGap']:.4f} m
    accel    = {best['accel']:.4f} m/s^2
    decel    = {best['decel']:.4f} m/s^2
  
  Performance:
    Combined Loss: {best['rmse']:.2f}
    68X RMSE:      {best['rmse_68x']:.2f} (Target)
    960 RMSE:      {best['rmse_960']:.2f} (Within Constraint)

QUALITY ASSESSMENT:
  68X: {'EXCELLENT' if best['rmse_68x'] < 200 else 'GOOD' if best['rmse_68x'] < 300 else 'NEEDS IMPROVEMENT'}
  960: {'WITHIN CONSTRAINT' if best['rmse_960'] < 350 else 'CONSTRAINT VIOLATED'}

================================================================================
"""
    
    report_path = output_dir / 'l1_calibration_summary.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[INFO] Saved summary report: {report_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Plot L1 Calibration Convergence')
    parser.add_argument('--log', type=str, 
                       default='data/calibration/l1_calibration_log_20251221_005412.csv',
                       help='Path to calibration log CSV')
    parser.add_argument('--output', type=str, default='plots',
                       help='Output directory for plots')
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    log_path = base_dir / args.log
    output_dir = base_dir / args.output
    output_dir.mkdir(exist_ok=True)
    
    print(f"[INFO] Loading calibration log: {log_path}")
    df = pd.read_csv(log_path)
    print(f"[INFO] Loaded {len(df)} iterations")
    
    # Generate all plots
    print("\n[INFO] Generating convergence plot...")
    plot_convergence_dual(df, output_dir)
    
    print("[INFO] Generating parameter evolution heatmap...")
    plot_parameter_evolution(df, output_dir)
    
    print("[INFO] Generating phase comparison...")
    plot_phase_comparison(df, output_dir)
    
    print("\n[INFO] Generating summary report...")
    report = generate_summary_report(df, output_dir)
    print(report)
    
    print("\n[SUCCESS] All visualizations generated!")


if __name__ == '__main__':
    main()
