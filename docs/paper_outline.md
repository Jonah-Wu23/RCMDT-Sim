# Paper Outline (ICTLE-style, 6–8 pages)

**Title（建议）**  
Multi-level Robust Calibration for Bus Simulation via Bayesian Surrogates and Data Assimilation

**Keywords**  
Bus simulation; parameter inversion; multi-level calibration; Bayesian optimization; surrogate model; data assimilation; robustness

---

## 0. Abstract (150-250 words)

- 背景：公交仿真虚实鸿沟、参数不确定性、传统校准不稳/成本高
- 方法：L1 微观行为校准 + L2 宏观同化/约束；贝叶斯代理模型（Kriging）+ 不确定性量化；BO（EI）选点；鲁棒性检验（K-S/分布指标）
- 数据：GPS/ETA（或站点到离站）、路段速度/检测器等（写你真实可用的）
- 结果：误差下降（平均+分位），跨时段/跨日稳定；仿真次数/成本
- 贡献点 3 条（见第 1 节）

---

## 1. Introduction

### 1.1 Problem Statement

- 公交仿真参数：停站行为、发车/速度、拥堵传播、控制策略等
- 难点：非凸、噪声、参数耦合、数据稀疏与时变

### 1.2 Research Gap

- 单层校准：只对齐一种观测（站点/路段），泛化差
- 纯黑箱优化：成本高、易过拟合
- 缺少“鲁棒性”评估：只报均值 RMSE 不够

### 1.3 Contributions

1. 提出 **多层次校准框架**：L1（线路/站点）+ L2（走廊/路段）闭环约束，解决公交停站与路网拥堵的耦合难题
2. 引入 **贝叶斯代理模型 + 不确定性量化**：利用 Kriging 的高斯过程特性，同时获得预测均值（加速搜索）与预测方差（量化置信区间），克服传统 GA/PSO 无法提供置信信息的缺陷
3. 建立 **分布鲁棒评估体系**：跨时段/跨日 + K-S/分位误差指标，验证校准结果的泛化稳定性
4. 实施 **全流程口径审计 (Caliber Audit)**：揭示了仅基于移动速度 (Moving-only Speed) 进行宏观校准的系统性偏差，提出并验证了基于 Door-to-Door 行程时间的观测算子在恢复模型可辨识性中的关键作用。

---

## 2. Related Work

2.1 Bus simulation calibration (SUMO/微观仿真参数)  
2.2 Surrogate-based optimization (Kriging/RBF/BO)  
2.3 Data assimilation for traffic (IES，简述即可)  
2.4 Robust calibration / distributional evaluation (K-S, quantile loss, etc.)

> 写法：每小节 6–10 句，最后一句落到“本文怎么补齐缺口”。

---

## 3. Problem Formulation

### 3.1 Simulation and Parameters

- 仿真平台：SUMO（写版本/接口）
- 参数向量 **θ**：分成两层
  - L1：dwell time（上下客时间模型参数）、accel/decel、car-following、lane-change、schedule adherence 等
  - L2：路段容量/拥堵参数、速度-流关系或同化的状态变量等

### 3.2 Observations and Metrics

- **3.2.1 Caliber Definitions (Observation Operators)**
  - **Op-L2-v0 (Moving-only)**: $v = L / t_{drive}$。忽略停站与排队，导致仿真“看起来总比现实快”（机理缺失假象）。
  - **Op-L2-v1 (Door-to-Door)**: $v = L / (t_{drive} + t_{dwell} + t_{wait})$。本文主实验采用此口径，直接对齐乘客体验。
- **Metrics**:
  - RMSE (Time/Speed), KS Distance (Distribution), P90 Error.
  - Loss Function: $J = w \cdot J_{L1}(Micro) + (1-w) \cdot J_{L2}(Macro)$

### 3.3 Robustness Definition（重点）

- 训练期 vs 测试期（跨小时段/跨日）
- 评价：K-S 统计量、P90/P95 误差、Worst-case over periods

---

## 4. Methodology

### 4.1 Multi-level Calibration Framework（框架图！）

- Step A：初始化采样（LHS/随机）跑仿真，得到 (θ, J1, J2)
- Step B：训练双代理（Kriging + RBF）拟合 J(θ) 或 (J1,J2)
- Step C：BO（EI）提出新 θ，追加仿真迭代
- Step D：L2 同化/约束（IES）
- Step E：输出 θ* 并做鲁棒验证

### 4.2 Surrogate Model with Uncertainty Quantification

- **Kriging (Gaussian Process)**：核心代理模型，提供预测均值 + 预测方差
  - 预测均值：加速目标函数评估，减少仿真次数
  - 预测方差：量化参数不确定性，指导 EI 采集函数的探索-利用权衡
- **RBF 对比实验**：作为消融研究的一部分，对比 Kriging-RBF 融合方案与单一 Kriging
- 实验发现：在当前 7 维平滑参数空间下，单一 Kriging 模型已具备足够拟合能力（见 6.5 节消融讨论）

### 4.3 Bayesian Optimization Details

- acquisition：Expected Improvement（EI）
- 预算：N0 初始 + N_iter 迭代
- 终止：最优改进 < ε 或预算用完

### 4.4 L2 Data Assimilation: Iterative Ensemble Smoother (IES)

- **Algorithm 1: Constraint-aware IES**
  - **State Vector ($m$)**: Global Capacity Factor (CF), Background minGap, Impatience.
  - **Observation Operator ($\mathcal{H}(m)$)**:
    - Input: Simulated StopEvents ($t_{arr}, t_{dep}$)
    - Process: Segment matching -> D2D Travel Time aggregation -> Harmomic Mean Speed.
    - **Safety Rails**:
      - `Insertion Validity`: Discard result if insertion_rate < 85%.
      - `Coverage Gate`: Only use links with >50% overlap.
      - `Boundary Logic`: If parameter hits bound, check gradient direction.
  - **Update Step**: $m_{new} = m + C_{md} (C_{dd} + R)^{-1} (d_{obs} - d_{sim})$
  - 解释：让宏观速度/拥堵形态与现实同步，减少“只靠调参数硬拟合”。

### 4.5 Complexity & Implementation

- 仿真一次耗时、总耗时、并行策略
- 可复现：参数范围表、随机种子、硬件

---

## 5. Experimental Setup

> 最终模型：L1 贝叶斯代理 (Kriging) + L2 IES + 双层目标 J1+J2
### 5.1 Study Area & Route Selection

- 线路信息：长度、站点数、主要走廊、早晚高峰特征
- 数据覆盖：几天、每 5s/30s/1min，缺失率

### 5.2 Data Pre-processing

- 地图匹配/异常点处理
- 对齐站点事件（arrive/depart）
- 路段速度聚合（时间窗）

### 5.2.1 Background Traffic Synthesis
- Source: `dfrouter` on detector data.
- **Source Filtering**:
  - Breadth-First Search (BFS, depth=5) to validate source-to-sink connectivity.
  - **Corridor Filtering**: Retain only vehicles traversing bus corridors.
  - Result: ~1150 vph effective injection.

### 5.3 Baselines

- B1：手工经验参数 / 默认参数（无优化）
- B2：单代理 Kriging + 单层 J1（验证代理模型基本效果）
- B3：Kriging-RBF 融合 + 单层 J1（验证融合方案是否带来增益）
- B4：**IES (L2) with L1 frozen at B2***（验证多层目标的贡献，Fixed Micro, Tuning Macro）

> **Note**: 消融实验预期 B4 > B2 > B1；B3 与 B2 差异取决于问题结构

### 5.4 Evaluation Protocol

- 划分：训练（2 天）+ 测试（1 天）或按时段
- 统计：mean、median、P90、worst-case、K-S

---

## 6. Results and Discussion

### 6.1 Calibration Convergence

- 最优值随迭代下降曲线（J、J1、J2）
- 仿真次数 vs 精度提升

### 6.2 Accuracy on L1 Metrics

- 站间时间误差、停站时长分布对齐（CDF 图）

### 6.3 L2 Calibration Results (The "Caliber Switch" Story)

- **6.3.1 Failure Mode (Moving-only)**:
  - CF pushed to 2.5 (limit), Speed stuck at 15km/h (Real ~5km/h).
  - Demonstrates unidentifiability under wrong operator.
- **6.3.2 Success Mode (Door-to-Door)**:
  - After metric switch: RMSE drops 10km/h -> 2.4km/h.
  - KS drops 0.58 -> 0.18 (Distribution Alignment).
  - **Main Result**: D2D travel_time time–space diagram (Figure 5).

### 6.4 Robustness Tests（你论文的“卖点”）

- 跨小时段/跨日：P90、worst-case、K-S
- 讨论：为什么多层校准更稳（参数可辨识性、约束更强）

### 6.5 Ablation Study

- **B2 vs B3 对比**：验证 Kriging-RBF 融合是否带来额外增益
  - 若 B3 约等于 B2：说明问题空间平滑，单一代理已足够
  - 若 B3 > B2：说明融合方案在复杂问题上有优势
- **B4 vs B2 对比**：验证双层目标 J1+J2 的贡献（核心验证）
- **去掉 L2 同化 / 去掉分布项**：评估鲁棒性下降程度

### 6.6 Limitations

- **Insertion Saturation**: 仿真受限于物理空间注入上限，高需求≠高密度（Gridlock vs High Flow）。
- **Global vs Local**: 全局 CF 无法解决局部瓶颈，但 Corridor reweighting 收益递减（Core edges dominate）。
- **Vehicle Dynamics**: SUMO 默认起步/黄灯损失可能导致 5.9km/h vs 4.8km/h 的最终系统误差。
- 可扩展性（多线路、多走廊）

---

## 7. Conclusion

- 结论 3 句：效果、效率、鲁棒
- 展望：多线路联合反演、在线校准、调度控制闭环（为你后续 AI-Agent 铺垫）

---

## Acknowledgements（如有）

资助/实验室/数据来源

---

## References

- 交通仿真校准（SUMO & bus dwell/carfollowing calibration）
- BO & surrogate（Kriging/RBF/ensemble）
- EnKF 交通同化（traffic state estimation）
- Robust evaluation（distributional metrics）

---

## Figure & Table Checklist（写作时直接照做）

- **Fig.1** Multi-level calibration workflow（总框架图）
- **Fig.2** Study corridor + stops + detectors（区域示意）
- **Fig.3** Convergence curve（J/J1/J2 vs iteration）
- **Fig.4** L1 CDF 对比（dwell/segment time）
- **Fig.5** L2 D2D Time-Space Diagram 对比
- **Fig.6** Robustness boxplot（across days/periods）
- **Table 1** Parameter ranges & meaning（θ 每个参数的上下限）
- **Table 2** Baseline settings & budgets（各方法仿真次数/时间）
- **Table 3** Main results（mean/median/P90/worst/K-S）

