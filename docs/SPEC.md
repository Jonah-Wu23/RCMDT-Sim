# RCMDT Protocol v4 规范说明文档

> **版本**: 4.0  
> **创建日期**: 2026-01-09  
> **目的**: 建立"不可争辩"的统一口径，确保所有实验可复现、可审计  
> **语言规范**: 所有输出、思考、任务清单均使用中文

---

## 1. 概述

本规范定义了 RCMDT（Robust Calibration of Mobility Digital Twins）论文实验体系的完整协议。**所有实验必须严格遵循本规范**，以确保：

1. **可复现性**: 任何研究者可以重现完全相同的结果
2. **可审计性**: 所有参数、阈值、指标定义都有明确记录
3. **不可辩驳性**: 审稿人无法在"你到底算的是什么"上提出质疑

---

## 2. 数据切分协议

### 2.1 可用数据集

> **重要说明**：时间均为香港时间 (HKT = UTC+8)。**没有 AM Peak 数据**。

| 数据集 | 日期 | 时段 | HKT 时间 | UTC 时间 | 用途 |
|--------|------|------|----------|----------|------|
| `data/` | 2025-12-19 | **PM Peak** | 17:00-18:00 | 09:00-10:00 | 校准 |
| `data2/` | 2025-12-30 | **Off-Peak (P14)** | 15:00-16:00 | 07:00-08:00 | 迁移验证 |

### 2.2 窗口单位

- **基本单位**: 1 小时滑动窗口
- **stress test**: 15 分钟子窗口（exhaustive）

### 2.3 注意事项

- ⚠️ **没有 AM Peak 数据**，实验只覆盖 PM Peak 和 Off-Peak
- 文件中的时间戳可能是 UTC，需要 +8 小时转换为香港时间

---

## 3. Audit 观测算子协议 (Op-L2-v1.1)

### 3.1 核心问题

审稿人最核心质疑：**audit 到底影响不影响校准？提升是不是主要来自过滤？**

### 3.2 Rule C 定义

```
flagged = (T >= T*) AND (v_eff <= v*)
```

| 参数 | 符号 | 值 | 单位 | 说明 |
|------|------|-----|------|------|
| 行程时间阈值 | T* | 325 | 秒 | 长时间样本 |
| 速度阈值 | v* | 5 | km/h | 低速样本 |

### 3.3 输出定义

| 输出 | 计算方式 | 说明 |
|------|----------|------|
| `flagged` | 满足 Rule C 的样本 | non-transport regime |
| `clean` | 不满足 Rule C 的样本 | transport regime |
| `flagged_fraction` | n_flagged / n_raw | 被过滤比例 |
| `n_clean` | len(clean) | 清洗后样本数 |
| `n_raw` | 原始样本数 | - |
| `n_sim` | 仿真样本数 | - |

### 3.4 Protocol Ablation 实验组

为回应"decoupling 不清楚""过滤是不是在作弊"的质疑，设计以下消融实验：

| 组别 | ID | Audit in L1 | L2 | 说明 |
|------|-----|-------------|-----|------|
| Zero-shot | A0 | ✗ | ✗ | 默认参数基线 |
| Raw-L1 | A1 | ✗ | ✗ | L1 用 raw D2D |
| Audit-for-Validation-Only | A2 | ✗ | ✗ | L1 用 raw，验证用 clean |
| Audit-in-Calibration | A3 | ✓ | ✗ | L1 只在 clean 集上算 loss |
| Full RCMDT | A4 | ✓ | ✓ | A3 + L2(IES) |

**预期结论**: 定量回答"过滤本身带来多少提升 vs 模型校准带来多少提升"

---

## 4. 指标定义

### 4.1 KS 距离

#### KS(speed)
- **描述**: 基于速度 CDF 的两样本 KS 距离
- **目标**: link_speed (km/h)
- **公式**: $D_{n,m} = \sup_x |F_n(x) - G_m(x)|$
- **用途**: 主要验证指标（与外部 IRN 可比）

#### KS(TT)
- **描述**: 基于 D2D 行程时间 CDF 的 KS 距离
- **目标**: door_to_door_travel_time (seconds)
- **公式**: $D_{n,m} = \sup_x |F_n(x) - G_m(x)|$
- **用途**: 校准目标对应的验证指标

### 4.2 Pass/Fail 判据

```python
alpha = 0.05
c_alpha = 1.36  # for two-sample KS test at alpha=0.05
D_crit = c_alpha * sqrt((n + m) / (n * m))

if KS < D_crit:
    result = "PASS"
else:
    result = "FAIL"
```

### 4.3 Worst-Window 定义

**方法**: `exhaustive`（禁止 random sub-windows）

```python
def compute_worst_window(data, window_minutes=15, step_minutes=1):
    """
    Exhaustive worst-window 计算
    
    1. 将 1 小时切分为所有可能的连续 15 分钟子窗口
    2. 起始点从 :00 到 :45，步长 1 分钟（共 46 个子窗口）
    3. 对每个子窗口计算 KS 距离
    4. 返回最大 KS 值及对应窗口时间
    """
    worst_ks = 0
    worst_start = None
    worst_end = None
    
    for start_minute in range(0, 60 - window_minutes + 1, step_minutes):
        end_minute = start_minute + window_minutes
        subset = filter_by_time(data, start_minute, end_minute)
        
        if len(subset) >= 10:  # minimum samples
            ks = compute_ks(subset)
            if ks > worst_ks:
                worst_ks = ks
                worst_start = start_minute
                worst_end = end_minute
    
    return {
        'worst_window_ks': worst_ks,
        'worst_window_start': f"{hour}:{worst_start:02d}",
        'worst_window_end': f"{hour}:{worst_end:02d}"
    }
```

**输出**:
- `worst_window_ks_speed`
- `worst_window_ks_tt`
- `worst_window_start_time` (必须输出)
- `worst_window_end_time` (必须输出)

### 4.4 Bootstrap 置信区间

```python
def bootstrap_ci(values, n_bootstrap=1000, confidence=0.95):
    """跨窗口汇总的 95% 置信区间"""
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return {
        'mean': np.mean(values),
        'median': np.median(values),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }
```

---

## 5. 统一运行预算

### 5.1 L1 优化器预算

| 阶段 | 迭代次数 | 说明 |
|------|----------|------|
| LHS 初始采样 | 15 | Latin Hypercube Sampling |
| BO 优化 | 25 | Bayesian Optimization |
| **总计** | **40** | - |

### 5.2 SUMO 仿真预算

- **每候选参数复制数**: 1
- **仿真时长**: 3600 秒 (1 小时)
- **约束**: 所有优化器/smoother 方法必须使用相同的 SUMO runs budget

### 5.3 随机种子

```python
MASTER_SEED = 42
SEED_SET = [42, 123, 456, 789, 1024]
```

---

## 6. L2 Smoother 可复现配置

### 6.1 状态向量 x_corr

| 组件 | 描述 | 边界 | 单位 |
|------|------|------|------|
| `capacityFactor` | 路段通行能力因子 | [0.5, 3.0] | - |
| `minGap` | 最小跟车间距 | [0.5, 5.0] | meters |
| `impatience` | 驾驶员不耐烦度 | [0.0, 1.0] | - |

**维度**: d = 3

### 6.2 观测向量 y

| 组件 | 描述 | 单位 |
|------|------|------|
| `link_speed` | 路段平均速度 | km/h |
| `segment_count` | 有效样本数 | - |

**维度**: p = n_links × 2

### 6.3 公共参数

| 参数 | 符号 | 值 | 说明 |
|------|------|-----|------|
| 集合大小 | Ne | 20 | Ensemble size |
| 最大迭代 | K | 5 | Max iterations |
| 时间窗口 | - | 17:00-18:00 | 3600s |
| R 矩阵类型 | - | diagonal | 对角阵 |
| R 来源 | - | empirical | 经验估计 |
| 方差下限 | - | 1.0 | (km/h)² |
| 更新阻尼 | β | 0.3 | Damping |

### 6.4 方法特定参数

| 方法 | 来源 | 正则化/膨胀 | 局部化 |
|------|------|------------|--------|
| **IES** | Custom (LTPP) | - | Patch-wise (L=16) |
| **ES-MDA** | dapper.mods.iEnKS | α_k = K | None |
| **EnRML** | dapper.mods.iEnKS | Levenberg-Marquardt | None |
| **IEnKS** | dapper.mods.iEnKS | Finite-Window | Domain-specific |

---

## 7. L1 Stop 参数配置

### 7.1 参数边界

| 参数 | 描述 | 边界 | 单位 |
|------|------|------|------|
| `t_board` | 单乘客上车耗时 | [0.5, 5.0] | seconds |
| `t_fixed` | 固定停站开销 | [5.0, 20.0] | seconds |
| `tau` | 驾驶员反应时间 | [0.1, 2.0] | seconds |
| `sigma` | 速度偏差系数 | [0.1, 0.8] | - |
| `minGap` | 最小跟车间距 | [0.5, 5.0] | meters |
| `accel` | 最大加速度 | [0.5, 3.0] | m/s² |
| `decel` | 最大减速度 | [1.0, 5.0] | m/s² |

### 7.2 复合损失函数

$$J_{L1} = \text{RMSE} + \alpha \cdot (\text{MAE} + \lambda \cdot \text{std}(|e|)) + \beta \cdot Q_{0.9}(|e|)$$

| 项 | 权重 | 说明 |
|----|------|------|
| RMSE | 1.0 | 均方根误差（基准拟合） |
| MAE + dispersion | α=1.0, λ=0.5 | 鲁棒性惩罚 |
| Tail risk | β=0.5 | 90 分位数尾部惩罚 |

---

## 8. 输出目录结构

```
data/experiments_v4/
├── protocol_ablation/          # Protocol Ablation (A0-A4)
│   ├── results.csv
│   ├── summary.csv
│   ├── tables/ablation_table.tex
│   ├── figures/ablation_comparison.png
│   └── README.md
│
├── smoother_baselines/         # L2 Smoother Baselines
│   ├── results.csv
│   ├── summary.csv
│   ├── tables/smoother_comparison.tex
│   ├── tables/l2_config_table.tex
│   ├── figures/smoother_convergence.png
│   └── README.md
│
├── threshold_sensitivity/      # Audit 阈值敏感性
│   ├── results.csv
│   ├── summary.csv
│   ├── tables/threshold_table.tex
│   ├── figures/threshold_heatmap.png
│   ├── figures/pareto_front.png
│   └── README.md
│
├── loss_sensitivity/           # 损失函数权重敏感性
│   ├── results.csv
│   ├── summary.csv
│   ├── tables/loss_weight_table.tex
│   ├── figures/tradeoff_curve.png
│   └── README.md
│
├── optimizer_baselines/        # 优化器对比
│   ├── results.csv
│   ├── summary.csv
│   ├── tables/optimizer_comparison.tex
│   ├── figures/convergence_curves.png
│   ├── figures/best_objective_boxplot.png
│   └── README.md
│
├── misflag_incident/           # Misflag/Incident 鲁棒性
│   ├── results.csv
│   ├── case_studies/
│   ├── tables/misflag_table.tex
│   ├── figures/incident_analysis.png
│   └── README.md
│
└── global_summary/             # 全研究期统计汇总
    ├── results.csv
    ├── summary.csv
    ├── tables/global_summary.tex
    ├── figures/ks_distribution.png
    ├── figures/heatmap_route_period.png
    ├── figures/pass_rate_bar.png
    └── README.md
```

---

## 9. 输出模板要求

### 9.1 必须输出的列

每个实验结果 CSV 必须包含：

```csv
route,period,day,n_raw,n_clean,n_sim,flagged_fraction,ks_speed,ks_speed_critical,ks_speed_passed,ks_tt,ks_tt_critical,ks_tt_passed,worst_window_ks_speed,worst_window_ks_tt,worst_window_start,worst_window_end,rmse,mae,p90
```

### 9.2 汇总统计列

```csv
mean,median,std,ci_lower_95,ci_upper_95,pass_rate
```

### 9.3 LaTeX 表格 Caption 要求

**每个表格的 caption 必须包含**:
1. 指标类型：speed 或 TT
2. 窗口定义：full-hour 或 worst-window
3. 样本量：n, m 值
4. Pass/Fail 判据：α=0.05, D_crit 公式

---

## 10. Sanity Checks

### 10.1 强制检查项

| ID | 检查项 | 规则 | 失败处理 |
|----|--------|------|----------|
| SC1 | 单位一致性 | speed in km/h, time in seconds | 报错终止 |
| SC2 | 掩码一致性 | real 和 sim 使用相同窗口/路段 | 报错终止 |
| SC3 | n_clean 最小值 | n_clean >= 10 | 标记 NA |
| SC4 | 窗口一致性 | real 和 sim 时间窗口对齐 | 报错终止 |
| SC5 | KS 范围 | 0 <= KS <= 1 | 报错终止 |
| SC6 | 禁止 random worst-window | 必须用 exhaustive | 报错终止 |

### 10.2 检查代码示例

```python
def sanity_check(results):
    """Protocol v4 Sanity Checks"""
    errors = []
    
    # SC1: 单位一致性
    if results['speed_unit'] != 'km/h':
        errors.append("SC1: Speed unit must be km/h")
    
    # SC3: n_clean 最小值
    if results['n_clean'] < 10:
        results['ks_speed'] = np.nan
        results['ks_tt'] = np.nan
        warnings.warn("SC3: n_clean < 10, marking as NA")
    
    # SC5: KS 范围
    if not (0 <= results['ks_speed'] <= 1):
        errors.append("SC5: KS value out of range [0,1]")
    
    # SC6: 禁止 random worst-window
    if results.get('worst_window_method') == 'random':
        errors.append("SC6: Random worst-window is forbidden")
    
    if errors:
        raise ValueError("\n".join(errors))
    
    return results
```

---

## 11. 禁止事项

### 11.1 严格禁止

| ID | 禁止事项 | 详细说明 |
|----|----------|----------|
| C1 | 混用旧口径数据 | random worst-15min、B4(v0) 主线、单位/mask 不一致 |
| C2 | n_clean 小却输出 0.0 | n_clean < 10 必须标 NA |
| C3 | 图表缺少信息 | caption 必须完整 |
| C4 | 直接计算 KS | 必须调用 metrics_v4.py |
| C5 | B4(v0) 作主线证据 | 只能作为 legacy mechanism 放 appendix |

### 11.2 代码约束

```python
# 禁止: 直接计算 KS
# ks = scipy.stats.ks_2samp(real, sim)  # ❌ FORBIDDEN

# 正确: 调用统一评估器
from scripts.eval.metrics_v4 import compute_metrics
results = compute_metrics(real_data, sim_data)  # ✓ CORRECT
```

---

## 12. 路由与场景配置

### 12.1 主要路由

| 路由 | 方向 | 描述 | 角色 |
|------|------|------|------|
| **68X** | Inbound | 元朗 → 旺角（密集城区） | 优化目标 |
| **960** | Inbound | 湾仔 → 屯门（过海线路） | 约束锚点 |

### 12.2 约束处理

```python
# 960 作为约束锚点
CONSTRAINT_THRESHOLD = 350  # seconds RMSE
PENALTY_COEFFICIENT = 10.0

def compute_loss(rmse_68x, rmse_960):
    if rmse_960 <= CONSTRAINT_THRESHOLD:
        return rmse_68x
    else:
        return 2000 + (rmse_960 - CONSTRAINT_THRESHOLD) * PENALTY_COEFFICIENT
```

### 12.3 主要场景

| 场景 | 描述 | 时间窗 |
|------|------|--------|
| **P14** | Off-Peak transfer | 15:00-16:00 |

---

## 13. 与审稿意见的对应关系

| 审稿问题 | 对应模块 | Protocol v4 解决方案 |
|----------|----------|---------------------|
| "decoupling 不清楚" | Module A | Protocol Ablation (A0-A4) |
| "过滤是不是在作弊" | Module A | Audit-for-Validation-Only (A2) 对照组 |
| "IES under-specified" | Module B | L2 Smoother 完整配置表 |
| "没对比 EnRML/IEnKS" | Module B | Smoother Baselines 统一口径对比 |
| "阈值拍脑袋" | Module C | 阈值敏感性 + Pareto 图 |
| "权重没讨论" | Module D | Loss/Weight Sensitivity 扫描 |
| "BO vs LHS 8.35% 不够硬" | Module E | Optimizer Baselines 系统对比 |
| "audit 会误杀真实拥堵" | Module F | Misflag/Incident 案例分析 |
| "只挑窗口" | Module G | 全研究期统计 + CI |
| "KS 没统计基础" | 本规范 | Pass/Fail 判据 + bootstrap CI |

---

## 14. 论文重写要点

### 14.1 新叙事结构

1. **语义污染问题** → 引出 Audit 需求
2. **Protocol/治理** → 建立可审计框架
3. **跨窗口统计** → 全研究期证据
4. **Baselines** → 公平对比
5. **Failure Modes** → 诚实披露局限

### 14.2 必须加入的 6 个"堵嘴点"

1. **Pre-registered protocol**: 所有阈值/预算/窗口/指标固定
2. **Equal-budget baselines**: BO/Random/CMA-ES 同预算
3. **Cross-window aggregation**: 不挑窗口；给 CI、pass rate
4. **Metric alignment**: TT & speed 都报；解释为何 speed 更可比
5. **Failure modes**: misflag/incident 例子 + 处理策略
6. **Reproducibility table**: L2 配置、x_corr/y 维度、R、Ne、K、projection

---

## 15. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 4.0 | 2026-01-09 | 完整重建实验体系；统一指标定义；exhaustive worst-window；bootstrap CI；禁止 random sub-windows |

---

## 附录 A: metrics_v4.py 接口规范

```python
"""
scripts/eval/metrics_v4.py - 唯一权威评估器

所有实验脚本必须调用此模块计算指标，禁止自行计算 KS/worst-window。
"""

def compute_metrics(
    real_data: pd.DataFrame,
    sim_data: pd.DataFrame,
    audit_config: dict = None,
    bootstrap_n: int = 1000
) -> dict:
    """
    计算 Protocol v4 规定的所有指标
    
    Parameters
    ----------
    real_data : pd.DataFrame
        真实数据，必须包含 'speed', 'travel_time', 'timestamp' 列
    sim_data : pd.DataFrame
        仿真数据，必须包含 'speed', 'travel_time', 'timestamp' 列
    audit_config : dict, optional
        Audit 配置，默认使用 Rule C (T*=325s, v*=5km/h)
    bootstrap_n : int
        Bootstrap 采样次数
        
    Returns
    -------
    dict
        包含所有 Protocol v4 规定的指标
    """
    pass


def apply_audit(data: pd.DataFrame, config: dict) -> tuple:
    """
    应用 Rule C 审计规则
    
    Returns
    -------
    (clean_data, flagged_data, stats)
    """
    pass


def compute_worst_window(
    real_data: pd.DataFrame,
    sim_data: pd.DataFrame,
    window_minutes: int = 15,
    step_minutes: int = 1
) -> dict:
    """
    Exhaustive worst-window 计算（禁止 random）
    """
    pass
```

---

## 附录 B: Markdown 表格模板

> **注意**：禁止使用 LaTeX (.tex) 格式，论文使用 Word 排版，所有表格使用 Markdown (.md) 格式。

### Protocol Ablation 结果表

**表格说明**: Protocol Ablation Results: KS(speed) on P14 Off-Peak Transfer.  
Pass/Fail criterion: α=0.05, D_crit=1.36×sqrt((n+m)/(nm)).  
n_clean denotes samples after Rule C audit (T* ≥ 325s, v* ≤ 5 km/h).

| Config | Audit | BO | IES | Tail | KS(speed) | n_clean | Pass |
|--------|-------|-----|-----|------|-----------|---------|------|
| A0 Zero-shot | ✗ | ✗ | ✗ | ✗ | 0.541 | 70 | FAIL |
| A1 Raw-L1 | ✗ | ✗ | ✗ | ✗ | 0.541 | 70 | FAIL |
| A2 Audit-Val-Only | ✓* | ✗ | ✗ | ✗ | 0.262 | 37 | FAIL |
| A3 Audit-in-Cal | ✓ | ✗ | ✗ | ✗ | 0.262 | 37 | FAIL |
| A4 Full RCMDT | ✓ | ✓ | ✓ | ✓ | 0.262 | 37 | FAIL |

*注：✓* 表示 Audit 仅用于验证，不用于 L1 校准。*

---

*本规范文件与 `protocol_v4.yaml` 配套使用，共同定义 RCMDT 实验体系的完整协议。*
