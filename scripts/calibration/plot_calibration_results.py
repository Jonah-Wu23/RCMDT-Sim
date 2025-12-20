import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_calibration(log_path, output_path):
    df = pd.read_csv(log_path)
    if df.empty:
        print("Log file is empty.")
        return

    plt.figure(figsize=(10, 6))
    
    # 绘制所有点的 RMSE
    plt.scatter(df['iter'], df['rmse'], c='blue', label='All Samples', alpha=0.5)
    
    # 绘制累计最小值 (收敛曲线)
    df['min_rmse'] = df['rmse'].cummin()
    plt.plot(df['iter'], df['min_rmse'], 'r-', linewidth=2, label='Convergence (Min RMSE)')
    
    # 标记 BO 阶段
    bo_starts = df[df['type'] == 'bo']['iter'].min()
    if pd.notnull(bo_starts):
        plt.axvline(x=bo_starts - 0.5, color='green', linestyle='--', label='BO Starts')
    
    plt.title('L1 Calibration Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE (J1)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    
    # 打印最优参数
    best = df.loc[df['rmse'].idxmin()]
    print("\n--- Best Parameters Found ---")
    print(best)

    print(best)

import glob

def find_latest_log(base_dir):
    pattern = os.path.join(base_dir, 'data/calibration/l1_calibration_log_*.csv')
    files = glob.glob(pattern)
    if not files:
        # Fallback to the old name if no timestamped files found
        return os.path.join(base_dir, 'data/calibration/l1_calibration_log.csv')
    return max(files, key=os.path.getctime)

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_file = find_latest_log(project_root)
    print(f"Analyzing log file: {log_file}")
    
    out_img = os.path.join(project_root, 'data/calibration/l1_convergence.png')
    plot_calibration(log_file, out_img)
