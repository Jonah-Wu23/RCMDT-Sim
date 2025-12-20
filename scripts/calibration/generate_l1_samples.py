import json
import os
import pandas as pd
import numpy as np
from scipy.stats import qmc
import argparse

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_samples(config_path, output_path, n_samples_override=None):
    # 加载配置
    config = load_config(config_path)
    params = config['parameters']
    
    # 优先使用命令行参数，其次使用配置文件，最后默认 10
    n_samples = n_samples_override or config['sampling'].get('initial_sample_size', 10)
    seed = config['sampling'].get('seed', None)
    
    # 初始化拉丁超立方采样器 (LHS)
    d = len(params)
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    sample = sampler.random(n=n_samples)
    
    # 将样本缩放到参数范围
    l_bounds = [p['min'] for p in params]
    u_bounds = [p['max'] for p in params]
    
    scaled_sample = qmc.scale(sample, l_bounds, u_bounds)
    
    # 创建 DataFrame
    param_names = [p['name'] for p in params]
    df = pd.DataFrame(scaled_sample, columns=param_names)
    
    # 添加 ID 列
    df.insert(0, 'param_id', [f'p{i:03d}' for i in range(len(df))])
    
    # 保存为 CSV
    ensure_dir(output_path)
    df.to_csv(output_path, index=False)
    print(f"成功生成 {n_samples} 个样本至: {output_path}")
    print("前 5 行样本预览:")
    print(df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 LHS 生成 L1 参数样本")
    parser.add_argument("--config", default="config/calibration/l1_parameter_config.json", help="配置 JSON 路径")
    parser.add_argument("--output", default="data/calibration/l1_initial_samples.csv", help="输出 CSV 路径")
    parser.add_argument("--n_samples", type=int, default=10, help="初始样本数量 (默认: 10)")
    
    args = parser.parse_args()
    
    #同样处理相对路径，假设从项目根目录运行
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, args.config)
    output_path = os.path.join(base_dir, args.output)
    
    print(f"正在加载配置: {config_path}")
    generate_samples(config_path, output_path, n_samples_override=args.n_samples)
