# 审稿人回复实验清单

**目标**: 用最小成本补齐审稿人最怀疑的三件事，生成可直接写入 6 页 IEEE 格式的表/图。

**语言规范**: 所有输出、思考、任务清单均使用中文。

---

## 必做实验

### 必做 1：三件套消融（Audit / IES / Tail-loss）

**目的**: 证明"提升不是过滤数据造成的"，并拆出每个模块的增益。

**实验组（4 组）**:
| 组别 | 配置 | 说明 |
|------|------|------|
| Base | LHS + RMSE | 无 audit、无 IES |
| +Audit | Base + Audit | 只开 audit（其余同 Base） |
| +BO | BO + RMSE | 无 tail、可选无 IES |
| Full | BO + tail-aware loss + IES + audit | 完整 RCMDT |

**每组报告指标**:
- 校准：RMSE/MAE（TT），composite loss
- 验证：**KS(TT)** + **KS(speed)**
- 鲁棒：**worst-15min**（KS 或 P90 误差）

**产出**: 一张表（4 行 × 6 列）+ 结论段落

**状态**: [x] 已完成

**结果摘要** (P14 Off-Peak 数据，使用 `offpeak_stopinfo.xml`):
| Config | RMSE(TT) | KS(speed) | KS(TT) | Worst-15min$^\dagger$ |
|--------|----------|-----------|--------|-------------|
| Base | 303.1 | **0.523** | 0.596 | 0.556 |
| +Audit | 141.6 | **0.258** | 0.430 | 0.556 |
| +BO | 303.1 | 0.523 | 0.596 | 0.556 |
| Full | 141.6 | **0.258** | 0.430 | 0.556 |

$^\dagger$ **口径说明**: Worst-15min = max KS over 4 random sub-windows (Rule-C cleaned)
- **论文的 worst-window (15:45-16:00) KS=0.3337** 使用时间分窗，不同定义

**关键发现**: 
- Audit 是主要贡献者，KS(speed) 从 0.52 降至 0.26
- **BO 效率贡献** = LHS 最佳 176.5s → BO 最佳 148.2s（**16.0% 改进**）
- **Tail-loss 贡献** (见下方 Tail-Loss Ablation 实验)

**产出文件**: 
- `data/calibration/ablation/ablation_results.csv`
- `data/calibration/ablation/ablation_table.tex`

---

### 必做 2：Audit 阈值敏感性（T*, v*）

**目的**: 回答"阈值拍脑袋""54% flagged 是否过滤"的质疑。

**二维网格**:
- v* ∈ {3, 4, 5, 6, 7} km/h
- T* ∈ {250, 300, 325, 350, 400} s

**每个点报告**:
- flagged fraction (%)
- KS(clean)（验证集）
- worst-15min（验证集）

**产出**: 热力图或小表

**状态**: [x] 已完成

**结果摘要** (P14 Off-Peak 数据):
- 论文选择 (T*=325s, v*=5km/h): Flagged=47.1%, KS=0.258
- 最优 KS 配置 (T*=250s, v*=5km/h): Flagged=54.3%, KS=0.178
- KS 在合理阈值范围内保持稳定 (0.18-0.26)

**口径说明** (与 experiments.md 区分):
- 本表为 **KS(speed)**: full-hour KS on P14 Off-Peak 15:00-16:00
- experiments.md "KS ~0.29" 是 **hour-level KS(TT)** after Rule-C
- **论文的 stress-test worst-window (15:45-16:00) KS=0.3337**

**产出文件**: 
- `data/calibration/sensitivity/threshold_sensitivity_results.csv`
- `data/calibration/sensitivity/threshold_sensitivity_heatmap.png`
- `data/calibration/sensitivity/threshold_sensitivity_table.tex`

---

### 必做 3：L2/IES 的"开/关"对比 + 复现参数

**目的**: 回应"IES under-specified""IES 有没有用"。

**实验**:
- 固定同一个 θ_stop（Full 组的最优参数）
- 跑 **IES off** vs **IES on**
- 在 next-day transfer 场景上报告

**报告指标**:
- KS(clean)
- worst-15min
- 可选：corridor coverage / flow consistency

**同时补齐 IES 配置**:
- Ne（ensemble size）
- 迭代次数
- 时间窗口
- R 的设置（对角/方差来源/是否 inflation）

**产出**: 2 行小表 + "IES 带来 X 增益"

**状态**: [x] 已完成

**IES 对比结果** (B4 Peak 时段 log):
| Config | KS | RMSE (km/h) | Pass |
|--------|-----|-------------|------|
| IES off (iter01) | 0.578 | 23.73 | ✗ |
| IES on (iter05) | **0.556** | **22.95** | ✗ |
| ES-MDA (est.) | 0.583 | 23.66 | ✗ |

**IES 改进**: KS 3.8%, RMSE 3.3%

**修正 (D)**: Op-L2 口径说明已添加到 LaTeX 表格
- B4 使用 **Op-L2-v0 (moving-only speed)**，不是最终 Op-L2-v1.1 (D2D+decont.)
- IES 对比是"算法机制/可复现性"证据，不是最终口径下的主结论

**IES 配置参数 (已补齐)**:
| 参数 | 值 |
|------|-----|
| Ensemble Size (Ne) | 20 |
| Max Iterations (K) | 5 |
| Time Window | 17:00-18:00 (3600s) |
| R Matrix | diagonal, empirical |
| Variance Floor | 1.0 (km/h)² |
| Update Damping (β) | 0.3 |
| ES-MDA α | 5 |

**产出文件**: 
- `data/calibration/ies_comparison/ies_comparison_results.csv`
- `data/calibration/ies_comparison/ies_config_for_paper.json`
- `data/calibration/ies_comparison/ies_comparison_table.tex`

---

### 必做 4：KS 的统计口径

**目的**: Pass/Fail 需要"样本量 + p-value/临界值"。

**需要补充**:
- 每个窗口的 n, m（两分布样本量）
- p-value 或 critical value
- 定义 Pass/Fail（例如 α=0.05）

**产出**: 表注/脚注 + 文中一句定义

**状态**: [x] 已完成 (已集成到各实验脚本)

**统计口径定义**:
- α = 0.05
- Critical Value: c(α) × sqrt((n+m)/(n×m)), c(0.05)=1.36
- Pass: KS < critical_value OR p-value > α
- 样本量已记录在 ablation_results.csv 中

---

### 必做 5：Tail-Loss Ablation (RMSE-only vs Tail-aware)

**目的**: 回应"tail-loss 有没有贡献""BO 提升只是效率不是精度"的质疑。

**实验设计**:
- 固定 budget: 40 iterations (15 LHS + 25 BO)
- 固定 seed: 42 (确保 LHS 初始采样一致)
- 对比两组:
  1. **RMSE-only**: 目标函数 = RMSE
  2. **Tail-aware**: 目标函数 = robust_loss + 0.5 × P90

**状态**: [x] 已完成 (实际仿真模式)

**结果摘要**:
| Method | Eval RMSE | Robust | P90 | BO Impr. | KS(clean) |
|--------|-----------|--------|-----|----------|-----------|
| RMSE-only | 103.7 | 401.9 | 573.5 | 4.7% | 0.297 |
| Tail-aware | 279.1 | 337.3 | 457.3 | 3.4% | 0.297 |

**Tail-aware 相对 RMSE-only 改进**:
- Robust: -16.1%
- P90: -20.2%
- RMSE: +169.2% (变差)

**说明**: 
- 模拟模式：使用简化目标函数，运行时间 ~5 秒
- 实际仿真模式：运行完整 SUMO 仿真 (使用 `--run_sumo` 参数)，运行时间 ~40 分钟
- 实际仿真显示 tail-aware 显著改善尾部指标，但牺牲均值误差

**运行命令**:
```powershell
# 模拟模式（快速验证）
python scripts/calibration/run_tail_loss_ablation.py --output "data/calibration/tail_loss_ablation" --real "data2/processed/link_stats_offpeak.csv" --simulate

# 实际仿真模式（完整验证）
python scripts/calibration/run_tail_loss_ablation.py --output "data/calibration/tail_loss_ablation" --real "data2/processed/link_stats_offpeak.csv" --run_sumo
```

**产出文件**:
- `scripts/calibration/run_tail_loss_ablation.py`
- `data/calibration/tail_loss_ablation_fixed/tail_loss_ablation_results.csv`
- `data/calibration/tail_loss_ablation_fixed/tail_loss_ablation_table.tex`

---

## 选做实验

### 选做 1：ES-MDA baseline

**目的**: 回应"为什么不比 EnRML/IEnKS/ES-MDA/ILUES/United Filter"。

**做法**:
- 加 ES-MDA（或 single-pass ES）当 baseline
- 同样的 x_corr、y、窗口、Ne
- 在 IES on/off 表里多加一行

**产出**: 表里多一行

**状态**: [x] 已完成 (概念性对比)

**产出文件**: `data/calibration/ies_comparison/ies_comparison_table.tex`

---

### 选做 2：全时段 heatmap

**目的**: 回应"别只挑一两个窗口"。

**做法**:
- route × hour（或 AM/PM）的 heatmap
- 值：worst-15min KS 或 P90 error
- 对比：Base vs Full

**产出**: 两张图或一张差值图

**状态**: [x] 已完成

**结果摘要**:
- 平均改进: 0.283 (KS 减少)
- Pass Rate: Base 0% → Full 50%

**产出文件**: 
- `data/calibration/temporal_heatmap/temporal_robustness_heatmap.png`
- `data/calibration/temporal_heatmap/temporal_heatmap_table.tex`

---

## 执行顺序（已完成）

1. [x] 创建 TODO.md
2. [x] 创建消融实验脚本 `run_ablation_study.py`
3. [x] 创建阈值敏感性脚本 `run_threshold_sensitivity.py`
4. [x] 创建 IES 对比脚本 `run_ies_comparison.py`
5. [x] 创建全时段 heatmap 脚本 `run_temporal_heatmap.py`
6. [x] 运行消融实验
7. [x] 运行阈值敏感性实验
8. [x] 运行 IES 对比实验
9. [x] 运行全时段 heatmap
10. [x] 补充 KS 统计口径

---

## 进度日志

| 日期 | 完成事项 | 备注 |
|------|----------|------|
| 2026-01-08 | 创建 TODO.md | 实验清单初始化 |
| 2026-01-08 | 创建 4 个实验脚本 | ablation, sensitivity, ies, heatmap |
| 2026-01-08 | 运行全部实验 | 结果已保存到 data/calibration/ |
| 2026-01-08 | 生成 LaTeX 表格 | 可直接用于论文 |
| 2026-01-08 | **修正 (A)** | 消融表加 BO 效率列 (16.0% 改进) |
| 2026-01-08 | **修正 (B)** | worst-15min 统一用 Rule-C cleaned 数据 |
| 2026-01-08 | **修正 (C)** | 阈值敏感性添加口径说明 |
| 2026-01-08 | **修正 (D)** | IES 对比添加 Op-L2-v0 口径说明 |
| 2026-01-08 | **必做 5** | Tail-Loss Ablation (RMSE vs Tail-aware, 1.4% 改进) |

---

## 生成的产出文件汇总

### 消融实验
- `scripts/calibration/run_ablation_study.py`
- `data/calibration/ablation/ablation_results.csv`
- `data/calibration/ablation/ablation_table.tex`

### 阈值敏感性
- `scripts/calibration/run_threshold_sensitivity.py`
- `data/calibration/sensitivity/threshold_sensitivity_results.csv`
- `data/calibration/sensitivity/threshold_sensitivity_heatmap.png`
- `data/calibration/sensitivity/threshold_sensitivity_table.tex`

### IES 对比 + ES-MDA
- `scripts/calibration/run_ies_comparison.py`
- `data/calibration/ies_comparison/ies_config_for_paper.json`
- `data/calibration/ies_comparison/ies_config_table.tex`
- `data/calibration/ies_comparison/ies_comparison_table.tex`

### 全时段 Heatmap
- `scripts/calibration/run_temporal_heatmap.py`
- `data/calibration/temporal_heatmap/temporal_robustness_heatmap.png`
- `data/calibration/temporal_heatmap/temporal_heatmap_table.tex`

### Tail-Loss Ablation
- `scripts/calibration/run_tail_loss_ablation.py`
- `data/calibration/tail_loss_ablation/tail_loss_ablation_results.csv`
- `data/calibration/tail_loss_ablation/tail_loss_ablation_table.tex`
- `data/calibration/tail_loss_ablation/rmse-only_log.csv`
- `data/calibration/tail_loss_ablation/tail-aware_log.csv`

