# Figures: Provenance & Generation

本目录包含论文图和诊断图。当前主要图如下（按用途分组）：

论文证据链 / 主图：
- `plots/P14_ghost_audit.png`（Rule C 观测算子审计）
- `plots/P14_robustness_cdf.png`（K-S 分布级稳健性证据）
- `plots/fig1_a1_baselines.png`（A1 baseline 对比）
- `plots/Fig1_threshold_sensitivity_pmpeak.png`（Rule C 阈值敏感性，pmpeak）

诊断/解释性图：
- `plots/ghost_physical_evidence_68X.png`, `plots/ghost_physical_evidence_960.png`（Ghost vs Clean 的物理证据散点）
- `plots/dwell_distribution_68X.png`, `plots/dwell_distribution_960.png`（holding proxy vs sim dwell CDF）
- `plots/trajectory_stepped_68X.png`, `plots/trajectory_stepped_960.png`（轨迹分解诊断）

派生数据：
- `plots/ghost_jam_examples.csv`（Rule C 典型 ghost jam 案例，来自阈值敏感性脚本）

## Regenerate (P14)

使用真实输入（默认见 `README.md`）：

```bash
python scripts/visualization/plot_p14_robustness.py ^
  --raw data2/processed/link_stats_offpeak.csv ^
  --sim sumo/output/offpeak_stopinfo.xml ^
  --dist data/processed/kmb_route_stop_dist.csv ^
  --t_critical 325 --speed_kmh 5 --max_dist_m 1500 ^
  --worst_window_ks 0.3337 ^
  --out-audit plots/P14_ghost_audit.png ^
  --out-cdf plots/P14_robustness_cdf.png
```

## Regenerate (Fig1: A1 baselines)

```bash
python scripts/visualization/plot_a1_baselines.py ^
  --results data/experiments_v4/a1_dapper_baselines/results.csv ^
  --output plots
```

## Regenerate (Fig1: Threshold Sensitivity + Ghost Examples)

```bash
python scripts/visualization/plot_threshold_sensitivity_ieee.py ^
  --sensitivity-csv data/calibration_v3/sensitivity/threshold_sensitivity_results_pmpeak.csv ^
  --raw-csv data/processed/link_stats.csv ^
  --output-dir plots
```

## Regenerate (Ghost Physical Evidence)

```bash
python scripts/visualization/plot_ghost_physical_evidence.py --real_links data2/processed/link_stats_offpeak.csv --real_dist data/processed/kmb_route_stop_dist.csv --out plots/ghost_physical_evidence_68X.png --route 68X --t_critical 325 --speed_kmh 5
python scripts/visualization/plot_ghost_physical_evidence.py --real_links data2/processed/link_stats_offpeak.csv --real_dist data/processed/kmb_route_stop_dist.csv --out plots/ghost_physical_evidence_960.png --route 960 --t_critical 325 --speed_kmh 5
```

## Regenerate (Trajectory Diagnostics)

```bash
python scripts/visualization/plot_trajectory_stepped.py --real_links data2/processed/link_stats_offpeak.csv --real_dist data/processed/kmb_route_stop_dist.csv --sim sumo/output/offpeak_stopinfo.xml --out plots/trajectory_stepped_68X.png --route 68X --t_critical 325 --speed_kmh 5
python scripts/visualization/plot_trajectory_stepped.py --real_links data2/processed/link_stats_offpeak.csv --real_dist data/processed/kmb_route_stop_dist.csv --sim sumo/output/offpeak_stopinfo.xml --out plots/trajectory_stepped_960.png --route 960 --t_critical 325 --speed_kmh 5
```

## Regenerate (Dwell Distribution)

```bash
python scripts/visualization/plot_dwell_distribution.py --real_links data2/processed/link_stats_offpeak.csv --real_dist data/processed/kmb_route_stop_dist.csv --sim sumo/output/offpeak_stopinfo.xml --out plots/dwell_distribution_68X.png --route 68X --v_stop_kmh 5 --min_dwell_s 5
python scripts/visualization/plot_dwell_distribution.py --real_links data2/processed/link_stats_offpeak.csv --real_dist data/processed/kmb_route_stop_dist.csv --sim sumo/output/offpeak_stopinfo.xml --out plots/dwell_distribution_960.png --route 960 --v_stop_kmh 5 --min_dwell_s 5
```

## Notes

- `t_critical=325` 和 `speed_kmh=5` 定义 Rule C。
- 轨迹/停站相关图为机制诊断，不作为主要验证证据。
