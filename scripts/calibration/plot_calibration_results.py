import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import argparse
from typing import Optional, Dict, Any
import numpy as np

# Set seaborn theme for publication quality
sns.set_theme(style="whitegrid", context="paper")
# Font settings for better readability in papers
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

def load_log(log_path: str) -> pd.DataFrame:
    """
    Load calibration log and validate required columns.
    
    Args:
        log_path: Absolute path to the CSV log file.
        
    Returns:
        pd.DataFrame: Loaded and preprocessed dataframe.
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")
    
    df = pd.read_csv(log_path)
    if df.empty:
        raise ValueError(f"Log file is empty: {log_path}")
    
    required_cols = ['iter', 'rmse']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: '{col}' in log file.")
    
    # Normalize type column
    if 'type' not in df.columns:
        df['type'] = 'unknown'
    else:
        df['type'] = df['type'].fillna('unknown').astype(str)
        df.loc[df['type'] == '', 'type'] = 'unknown'
        
    return df

def compute_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute statistics and summary info from the log data.
    
    Args:
        df: The calibration dataframe.
        
    Returns:
        Dict: Dictionary containing best_rmse, improvement metrics, etc.
    """
    # Cumulative minimum for convergence curve
    df['min_rmse'] = df['rmse'].cummin()
    
    best_idx = df['rmse'].idxmin()
    best_row = df.loc[best_idx]
    
    # Detect when Bayesian Optimization (BO) starts
    # We look for the first iteration where type contains 'bo' (case-insensitive)
    bo_mask = df['type'].str.contains('bo', case=False)
    bo_starts = df[bo_mask]['iter'].min()
    
    # Calculate improvement percentage
    initial_rmse = df.iloc[0]['rmse']
    best_rmse = best_row['rmse']
    total_improvement = (initial_rmse - best_rmse) / initial_rmse * 100 if initial_rmse != 0 else 0
    
    # BO improvement check
    bo_improvement_desc = "N/A (No BO phase found)"
    bo_improvement_val = 0.0
    if pd.notnull(bo_starts):
        pre_bo = df[df['iter'] < bo_starts]
        post_bo = df[df['iter'] >= bo_starts]
        
        if not pre_bo.empty and not post_bo.empty:
            min_pre = pre_bo['rmse'].min()
            min_post = post_bo['rmse'].min()
            bo_improvement_val = (min_pre - min_post) / min_pre * 100 if min_pre != 0 else 0
            
            if bo_improvement_val > 5: # Threshold for "significant"
                bo_improvement_desc = f"Significant (Improved {bo_improvement_val:.2f}% over pre-BO best)"
            elif bo_improvement_val > 0.01:
                bo_improvement_desc = f"Slight (Improved {bo_improvement_val:.2f}%)"
            else:
                bo_improvement_desc = "None (No lower RMSE found during BO phase)"

    return {
        'best_rmse': best_rmse,
        'best_iter': int(best_row['iter']),
        'best_params': best_row.to_dict(),
        'bo_start_iter': bo_starts if pd.notnull(bo_starts) else None,
        'total_improvement': total_improvement,
        'bo_improvement_desc': bo_improvement_desc,
        'bo_improvement_val': bo_improvement_val
    }

def make_plot(df: pd.DataFrame, summary: Dict[str, Any], output_path: str, args: argparse.Namespace):
    """
    Generate and save the publication-quality plot.
    """
    # Create directory if not exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=args.figsize, dpi=args.dpi)
    
    # 1. Plot all samples with color-coded stages
    # Use distinct symbols and colors for different types
    unique_types = df['type'].unique()
    palette = sns.color_palette("muted", len(unique_types))
    
    sns.scatterplot(
        data=df, 
        x='iter', 
        y='rmse', 
        hue='type', 
        palette=palette,
        alpha=0.6, 
        s=80,
        ax=ax,
        edgecolor='w',
        linewidth=0.5,
        zorder=3
    )
    
    # 2. Add trend line (Moving Average) to show global trend
    if args.trend:
        window = max(3, len(df) // 10)
        df['rolling_rmse'] = df['rmse'].rolling(window=window, min_periods=1, center=True).mean()
        sns.lineplot(
            data=df, 
            x='iter', 
            y='rolling_rmse', 
            color='gray', 
            linestyle='--', 
            alpha=0.3, 
            label=f'Trend ({window}-pt Moving Avg)',
            ax=ax,
            zorder=2
        )

    # 3. Plot convergence curve (cumulative min)
    sns.lineplot(
        data=df, 
        x='iter', 
        y='min_rmse', 
        color='#E74C3C', # Professional red
        linewidth=2.5, 
        label='Convergence (Best So Far)',
        ax=ax,
        zorder=5
    )
    
    # 4. Mark BO start position
    if summary['bo_start_iter'] is not None:
        bo_iter = summary['bo_start_iter']
        ax.axvline(
            x=bo_iter - 0.5, 
            color='#27AE60', # Professional green
            linestyle=':', 
            linewidth=2,
            alpha=0.8,
            zorder=4
        )
        # Calculate text position (top of graph)
        y_max = ax.get_ylim()[1]
        ax.text(
            bo_iter - 0.3, 
            y_max * 0.98, 
            f'BO Stage Starts @ iter={int(bo_iter)}', 
            rotation=90, 
            verticalalignment='top',
            color='#27AE60',
            fontsize=10,
            weight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        )
        
    # 5. Highlight best point with a large star
    best_rmse = summary['best_rmse']
    best_iter = summary['best_iter']
    ax.scatter(
        [best_iter], [best_rmse], 
        marker='*', color='gold', s=400, 
        edgecolor='black', zorder=10, 
        label=f'Optimal (iter={best_iter})'
    )
    
    # Add clear annotation for the best point
    ax.annotate(
        f"Global Optimum\nRMSE={best_rmse:.4f}\nIter={best_iter}",
        xy=(best_iter, best_rmse),
        xytext=(30, 30),
        textcoords="offset points",
        arrowprops=dict(
            arrowstyle="->", 
            connectionstyle="arc3,rad=.2", 
            color='black',
            linewidth=1.5
        ),
        fontsize=11,
        weight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.2)
    )

    # 6. Final Polish
    ax.set_title(args.title, fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('Iteration Index', fontsize=12, weight='bold')
    ax.set_ylabel(args.ylabel, fontsize=12, weight='bold')
    
    if args.logy:
        ax.set_yscale('log')
    
    # Improve legend
    ax.legend(title="Legend", title_fontsize='11', loc='upper right', frameon=True, shadow=True)
    
    # Clean grid
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    sns.despine(trim=True)
    
    plt.tight_layout()
    
    # Save files in multiple formats
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    
    if args.save_pdf:
        pdf_path = os.path.splitext(output_path)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"PDF version saved to: {pdf_path}")
        
    if args.save_svg:
        svg_path = os.path.splitext(output_path)[0] + '.svg'
        plt.savefig(svg_path, bbox_inches='tight')
        print(f"SVG version saved to: {svg_path}")

    print(f"Publication plot saved to: {output_path}")
    
    if args.show:
        plt.show()

def find_latest_log(base_dir: str) -> str:
    """Find the most recent calibration log file."""
    pattern = os.path.join(base_dir, 'data/calibration/l1_calibration_log_*.csv')
    files = glob.glob(pattern)
    if not files:
        # Fallback to the legacy name
        legacy_path = os.path.join(base_dir, 'data/calibration/l1_calibration_log.csv')
        if os.path.exists(legacy_path):
            return legacy_path
        else:
            raise FileNotFoundError(f"No log files found in {os.path.join(base_dir, 'data/calibration/')}")
    # Return newest by modification time
    return max(files, key=os.path.getmtime)

def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality calibration convergence plots.")
    parser.add_argument("--log", type=str, help="Path to the log CSV file (default: finds latest)")
    parser.add_argument("--output", type=str, default="data/calibration/l1_convergence.png", help="Output image path")
    parser.add_argument("--title", type=str, default="L1 Calibration Convergence Analysis", help="Plot title")
    parser.add_argument("--ylabel", type=str, default="RMSE (Calibration Error)", help="Y-axis label")
    parser.add_argument("--logy", action="store_true", help="Use log scale for Y axis (useful for large residuals)")
    parser.add_argument("--trend", action="store_false", default=True, help="Disable trend line")
    parser.add_argument("--save-pdf", action="store_true", default=True, help="Disable PDF saving (default: True)")
    parser.add_argument("--save-svg", action="store_true", help="Also save as SVG")
    parser.add_argument("--show", action="store_true", help="Show plot window after saving")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output image (default 300)")
    parser.add_argument("--figsize", type=float, nargs=2, default=[11.0, 7.0], help="Figure size (width height)")
    
    args = parser.parse_args()

    # Determine project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    try:
        log_file = args.log if args.log else find_latest_log(project_root)
        print(f"\n[INFO] Loading calibration log: {os.path.relpath(log_file, project_root)}")
        
        df = load_log(log_file)
        summary = compute_summary(df)
        
        # Determine output path absolute
        if not os.path.isabs(args.output):
            output_path = os.path.join(project_root, args.output)
        else:
            output_path = args.output
            
        # Summary text output (Enhanced format)
        print("\n" + "="*60)
        print("                CALIBRATION SUMMARY REPORT")
        print("="*60)
        print(f"Optimal RMSE:       {summary['best_rmse']:.6f}")
        print(f"Optimal Iteration:  {summary['best_iter']}")
        print(f"BO Start Iter:      {int(summary['bo_start_iter']) if summary['bo_start_iter'] is not None else 'N/A'}")
        print(f"Total Improvement:  {summary['total_improvement']:.2f}% (from initial sample)")
        print(f"BO Effectiveness:   {summary['bo_improvement_desc']}")
        print("-" * 60)
        print("Best Micro-Parameters (L1):")
        # Filter out meta columns for clarity
        meta_cols = ['iter', 'rmse', 'type', 'min_rmse', 'rolling_rmse']
        for k, v in summary['best_params'].items():
            if k not in meta_cols:
                # Format float for better looking
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                print(f"  > {k:15}: {val_str}")
        print("="*60 + "\n")
        
        make_plot(df, summary, output_path, args)
        
    except Exception as e:
        print(f"\n[ERROR] Failed to generate plot: {str(e)}")
        # Provide more context if it's a known issue
        if "Missing required column" in str(e):
            print("Tip: Make sure the CSV log has 'iter' and 'rmse' columns.")
        # import traceback
        # traceback.print_exc()

if __name__ == "__main__":
    main()
