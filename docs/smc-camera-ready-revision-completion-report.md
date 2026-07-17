# SMC Camera-Ready 源码、实验、图片与 IEEE Word 定稿完成报告

更新日期：2026-07-17（Asia/Shanghai）

状态：已完成源码修正、真实实验补跑、生产评估、Fig. 1–Fig. 5 与 Table I 重生成、IEEE Word 定稿及逐项验证。用户于 2026-07-16 授权最多 8 个并发 CPU worker，并授权在 ETA 差值重建与定点补齐 68X `2→3` 之间采用经审计的方案。L1 完成 325 个物理仿真；L2 名义 450 格中实际调度 421 格，420 成功、1 格经三次确定性尝试后真实失败，其余 29 格按设计降级规则未调度；最终消融使用四个共同种子完成 40 次真实仿真，另有 10 个 seed 4 计划项明确记录为 blocked。evaluate、160 行指标、Table I、六个 300-DPI PNG、来源 sidecar、独立重算及重复绘图均已完成。随后以 `论文模板SMC.doc` 为格式权威生成 IEEE 审阅稿、干净 DOCX 和 Word 原生 PDF。2026-07-17 09:58 用户再次修改审阅稿；当前 tracked 文件已完成语言修订和 OOXML 结构核验，Word 视觉渲染、干净 DOCX 与 PDF 的同步更新因本机 Word 调用额度限制暂未完成。原始主稿与模板保持原哈希，Figure 1 流程图未更换或编辑。

## 任务范围与验收边界

验收依据为 `docs/superpowers/specs/2026-07-16-smc-camera-ready-revision-design.md`。本任务只处理工作目录 `D:\Documents\Bus Project\SMC\Sorce code` 内的源码、测试、实验配置、实验输出、绘图脚本、生成图片和本报告。

设计文档同时包含 Word 主稿编辑与六页排版要求。源码/实验阶段遵守了当时“不得修改 Word 主稿”的边界；用户随后明确启动 Word 阶段，并指定 `D:\Documents\Bus Project\SMC\论文模板SMC.doc` 为格式权威且要求 IEEE 格式。Word 阶段只读取原稿和模板，另行生成带修订痕迹稿、干净终稿及 PDF，没有覆盖原始主稿或模板。审稿文件未要求修改 Figure 1 流程图内容，仅要求其上方文本块对齐，因此流程图嵌入字节保持不变，只调整周围正文使两栏齐底。

## 设计要求核对表

| 编号 | 设计要求 | 实现或证据位置 | 状态 |
|---|---|---|---|
| BASE-1 | 完整读取设计文档并建立逐项清单 | 本报告 | 已完成 |
| BASE-2 | 修改前记录源码、结果、图片和工作区基线 | 本报告“修改前基线” | 已完成 |
| SRC-1 | 建立单一 `paper-manifest/v1`，固定数据、日期、线路方向、窗口、参数边界、15+25 预算、Ne=10、K=3、阻尼 0.3、五个种子和输出 | `config/paper_camera_ready_manifest.json`、`contracts.py` | 已实现并验证 |
| SRC-2 | L1 参数采用 `theta_bus` 语义并区分 `minGap_bus` 与 `minGap_background` | manifest、`objective.py`、L2 配置 | 已实现并验证 |
| SRC-3 | 按第 6.3 节实现累计到达时间误差、至少三个下游站点、68X 复合目标、960 可行性约束和基础设施失败语义 | `src/calibration/objective.py`、12 项测试 | 已实现并验证 |
| SRC-4 | BO 与 continued LHS 共用 15 点初始设计、40 次成功评估预算和相同仿真种子日程 | `search.py`、`l1_stage.py`、L1 冻结 CSV 及 325 个 run 状态 | 已完成并验证 |
| SRC-5 | 统一 Rule C 为 `T > 325 s`、`v_eff < 5 km/h`、`distance <= 1500 m` | `audit.py`、`sumo_data.py`、manifest、测试 | 已实现并验证 |
| SRC-6 | 实现完整配置哈希、仿真有效哈希、组件哈希、机制字段和 A0-A4 相等性校验 | `contracts.py`、`simulation.py`、`pipeline.py` | 已实现并验证 |
| SRC-7 | 实现 `run-status/v1`、确定性重试、尝试隔离和成功预算计数 | `simulation.py`、L1/L2/final 状态；`blocked-disposition/v1` | 已验证；L1 全部首尝试成功；L2 唯一失败格三次同 manifest/seed 重试均被输出验证器拒绝；blocked 计划不伪装成 run status |
| SRC-8 | 实现 `paper-metrics/v1` 长表、样本下限、半开子窗口和空指标失败 | `metrics.py`、指标测试 | 已实现并验证 |
| SRC-9 | 将新结果字段统一为 `cross_day_*`，记录 2025-12-19 与 2025-12-30 | manifest、`paper_metrics.csv`、Table I/reporting 合同 | 已完成并验证 |
| E1-1 | 在共同资格集合上比较固定 Rule C、MAD 和 Isolation Forest，拟合仅使用开发集 | `audit.py`、`audit/audit_manifest.json` | 已完成；IF 只在开发集 76 键拟合，跨日冻结应用，未触发 fallback |
| E1-2 | 保存拟合统计、模型哈希、版本、保留/标记键和 3×3 Rule C 敏感性网格 | `audit_manifest.json`、`fig3_sensitivity.csv` | 已完成；IF 全状态哈希和 9 格敏感性均独立重算一致 |
| E1-3 | 报告 retention、全窗 K-S、worst-15-min K-S、IRN contradiction 的分子/分母/未匹配数 | `audit/audit_metrics.csv` | 已完成；IRN 小分母及未匹配数在本报告明确披露 |
| E2-1 | 按机制矩阵运行 A0、A1、A2、A3、A4，使用共同种子、窗口和清洁评估总体 | `ablation/ablation_runs.csv`、40 个 final manifest | 已完成并验证；共同种子为 `[0,1,2,3]` |
| E2-2 | A1/A3/A4 共用每个种子的 L1 参数；A2/A3/A4 共用 L2 先验和系综种子日程 | L2/final manifests 与独立审计 | 40 个 final run、421 个 L2 成员的冻结与 seed 公式检查全部通过 |
| E2-3 | 每个配置和种子生成开发集与跨日仿真输出，至少三个共同有效种子 | `ablation/stage-summary.json` | 四个共同种子、A0–A4 两 split 共 40 次成功；seed 4 的 10 项明确 blocked |
| E3-1 | 五个正常种子执行等预算 BO-LHS；无可行初始点的种子按协议失败 | `l1_stage.py`、`l1/bo_lhs_evaluations.csv`、五个 `selected.json` | 五种子真实补跑完成并验证 |
| E3-2 | 报告 cumulative-best、最终最佳值和预声明目标到达次数 | `l1_stage.py`、`figures.py`、`l1/bo_lhs_summary.csv`、Fig. 5 | 已完成并验证 |
| E4-1 | 使用 2025-12-19 开发集和 2025-12-30 跨日集，禁止静默省略跨日失败 | final outputs、`paper_metrics.csv` | 已完成；160 行指标均来自四个共同种子的两 split，无 seed 4 混入 |
| FIG-1 | 重新生成含 `theta_bus`、`x_corr`、审计路径和冻结协议的架构图 | `plots/camera_ready_revision_20260716/Fig1_camera_ready_architecture.png` | 已生成并通过 bbox、视觉和规格核验 |
| FIG-2 | 重新生成原始/清洁分布、Rule C 几何和轨迹累积证据 | `Fig2_camera_ready_contamination.png` 及两份 reporting CSV | 已生成；200 行 contamination、9 行可追溯轨迹，键级 Rule C 已验证；子图 (b) 规则说明已上移并通过散点遮挡回归测试 |
| FIG-3 | 重新生成 Rule C 敏感性及固定/统计/自适应审计比较 | `Fig3_camera_ready_audit.png`、审计/敏感性 CSV | 已生成；完成图例/IRN 注释防重叠测试；子图 (d) 柱顶留白已通过轴上界回归测试与视觉复核 |
| FIG-4 | 从共同协议长表重新生成最终分布验证 CDF | `Fig4_camera_ready_cdf.png`、758 行 CDF 输入 | 已生成；仅含 A4 与共同种子 0–3 |
| FIG-5 | 从等预算真实实验重新生成 BO-LHS cumulative-best 曲线 | `Fig5_camera_ready_bo_lhs.png`、400 行 L1 输入 | 已生成；只累计可行点 |
| TAB-1 | 从验证后的长表生成 A0-A4 Table I 源文件，含均值、样本标准差、worst-window、跨日和样本数 | `tables/table_i.csv`、`Table_I_camera_ready_ablation.png` | 已生成；5 行均值与 `ddof=1` 样本标准差独立重算一致 |
| REP-1 | 每个图表写入 `artifact-sidecar/v1`，记录输入哈希、脚本版本和输出哈希 | `figures.py`、绘图 CLI、绘图测试及生产/复现 sidecar 核验 | 已实现并验证 |
| VER-1 | 运行 P14 fixture smoke | `python scripts/smoke/p14_smoke.py` | 已通过 |
| VER-2 | 运行聚焦单元测试、语法检查及项目实际存在的相关验证命令 | 最终 123 项、绘图 17 项、compileall、P14、`git diff --check` | 全部相关检查通过；环境级 `pip check` 既存冲突另行披露 |
| VER-3 | 验证全部图片格式、像素、DPI、字体、图例、坐标轴、颜色和来源 | `verification/figure-specification-check.log`、人工视觉复核 | 六个 PNG 均通过；Times New Roman 解析至 `C:\Windows\Fonts\times.ttf` |
| VER-4 | 从冻结输出复现指标、表格和图片并比较哈希或内容 | 独立 evaluate 重算；`verification/figure-reproducibility-check.log` | 160 指标/表/审计/绘图输入零差异；六图字节和像素完全一致 |
| VER-5 | 审计最终工作区差异并确认 Word 主稿零变更 | `verification/final-workspace-boundary-user-layout-fix-v2.log`（SHA-256 `f0e21da0d86cbaa64e110e2afc031ab99df407a0620e6cda03a9178de03eeabf`） | 已完成；递归 Word 数量 0，用户 README 改动保留，pycache 状态数量 0 |
| WORD-1 | 按 IEEE 模板修改、跟踪并渲染 Word 主稿，保持 6 页 | 当前审阅稿及 `data/camera_ready_revision_20260716/manuscript/camera-ready-humanized-*` | 文字与结构已完成；当前审阅稿含 32 组段落级替换修订，接受后为 3197 个空格分词；Word 视觉分页和正式 PDF 待同步 |
| WORD-3 | 图表应跟随首次引用，不集中为两页图版；Table I 必须是原生表格 | 最终稿第 3–5 页 | 已完成；Fig. 2 位于第 3 页、Fig. 3 与原生 Table I 位于第 4 页、Fig. 4–5 位于第 5 页；Table I 为 6×7 Word 表格 |
| WORD-2 | 不改 Figure 1 流程图；仅处理审稿意见要求的上方文本对齐 | 最终稿第 2 页；嵌入 `word/media/image2.png` | 已完成；SHA-256 与原稿均为 `6d65e98734cb94851b545f11d744c32535fd00efb6318a1de203aa28333e5e84` |
| WORD-4 | 在用户手动调整后的最新版上补充论述并尽量利用六页面积 | `scripts/manuscript/expand_user_latest_manuscript.py`、最终稿第 5–6 页 | 已完成；增加可复现性、指标解释、阶段对照和未来验证边界，共 3 段 539 词；图表分页不变，第 6 页正文与参考文献约占四分之三页面 |
| REV-1 | 将 AE/R1/R2/R3 意见映射到主稿位置和证据 | `smc-camera-ready-reviewer-change-log.md` | 已完成 |

可选的 DAPPER、ES-MDA、IQR 基线不属于最低交付要求。旧结果未通过独立更新与指标差异检查，当前不纳入相机就绪证据。

## 修改前基线

### 工作区和版本

- Git 分支：`main`
- 基线提交：`ac0dae67400b8e425862c1eb9b9a0c029bd249b3`
- 用户已有改动：`README.md` 增加 2 行，最终 SHA-256 `2a93a9cd1ec4cd4ccb0d7a5f8ea544fef8550d054d5430d70c10f94aa9b05453`；`README_research_zh.md` 为未跟踪文件，最终 SHA-256 `89592ea0bfd513ffe3f51fee8b422e178392bf29138cc34dd95af2b7962d3658`。两者时间戳仍为 2026-03-31，均不属于本任务修改目标且已原样保留；最终 `git diff --numstat README.md` 仍为 `2 0`。
- 工作目录内没有 `.doc` 或 `.docx` 文件。本任务未访问设计文档所列的工作目录外权威主稿。

### 关键源码和配置哈希

| 文件 | 修改前 SHA-256 | 发现 |
|---|---|---|
| `src/calibration/objective.py` | `C1E04DAACAC5A8432616BE290F8E29BBD983FC7CEB38BEDFE805FA616D2E5D90` | 未强制每条线路至少三个下游站点；失败返回数值惩罚；L1 runner 未使用完整复合目标 |
| `config/calibration/l1_parameter_config.json` | `DD259D67F01AFD8538176115E48520D9940BE98133276FE7B9BD488FF32E37A5` | 使用含糊的 `minGap`；初始样本数为 20 |
| `config/calibration/l2_priors.json` | `74E500DB4815C12613CB81165AB1B666BC4ACACE1C3E1B198CE11ADFFCA9765C` | Ne=20、K=5，与论文协议冲突 |
| `config/calibration/l2_protocol_config.py` | `CBFB82DDC58684D649E1DD824CF6C1210C2905D57407FBAF3DFF004A10F01D8E` | Ne=10、K=3 正确，但背景参数仍命名为 `minGap` |
| `scripts/calibration/run_calibration_l1_loop.py` | `63B6C3F5B3D11F1CF5B214BEDBB168AAAA7063F1997FF55BC21CD4A45DB26BB9` | 只实现 BO 后续点；未向 SUMO 传入确定性种子；缺少重试；失败写成 `1e6` |
| `scripts/calibration/run_ies_loop.py` | `0A3378728233289427CCBC95E2D819487DAB9A086BB64736CC4AF15E88988A79` | 默认 Ne=20、K=5；不同标签共用输出目录，存在覆盖风险 |
| `scripts/experiments/run_unified_l2_experiments.py` | `762E5CABBB50FC7BB91BF400B9B950A3B6A67D6ABF6C5FECECE890EAF791715E` | 仅三个种子；缺 A1；A2/A4 使用硬编码终值且未运行 IES |
| `scripts/experiments/compute_pipeline_metrics.py` | `D40911C634715E12F5E45C99CD7F06FF991EE67BC136B3F5DF4E08D54259AE40` | 全窗样本下限为 3；Rule C 边界错误；仍使用 `next_day_*` |

### 输入数据基线

| 数据 | 日期与窗口 | 行数 | SHA-256 |
|---|---|---:|---|
| `data/processed/link_speeds.csv` | 2025-12-19 17:00-18:00 HKT | 226 | `ABA7EA65ED75B1AF88C2324906B102B4202B6FA0E67F5D30F4D3C30E32B522F6` |
| `data/processed/link_stats.csv` | 2025-12-19 17:00-18:00 HKT | 86 | `FDE3DA7D37FF4983C41EA630466D126BE912D804D8D8A17B42D70DE282D530AE` |
| `data2/processed/link_speeds_offpeak.csv` | 2025-12-30 15:00-16:00 HKT | 137 | `1D8925C67E32EA666349023556169C76C006F2C72AB2850B14D6D50FC6F7D2B3` |
| `data2/processed/link_stats_offpeak.csv` | 2025-12-30 15:00-16:00 HKT | 70 | `D5708A6B059BE32C735906D9D17C24C86A810F95D70F5E3A3A0F6E1195CF8A58` |
| `data/processed/kmb_route_stop_dist.csv` | 路线与站序映射 | 110 | `83CD9FB2F66BF36A9BC9F4A00CC08CD4CD4FBAB7D8F2B4D9FD525CA795FE7569` |

共同资格过滤后的基线规模为：开发集 76 个 link key，Rule C 标记 33 个，保留 43 个，对应 126 个事件；跨日集 63 个 link key，标记 33 个，保留 30 个，对应 57 个事件。该统计仅用于基线核查，最终结果必须由新实现重新生成。

### 旧实验结果状态

- `data/calibration_v3/ablation/ablation_results_v3.csv` 中 Base 与 `+BO` 完全相同，`+Audit` 与 Full 完全相同，无法证明对应机制实际生效。
- `data/experiments_v4/unified_l2/protocol_ablation/full_metrics.csv` 只有 A0、A2、A3、A4，每项三个种子；12 行的 `next_day_ks_speed` 和 `next_day_ks_tt` 全为空。
- `data/experiments_v4/a1_dapper_baselines/summary.csv` 和 `a1_smoother_baselines/summary.csv` 的各方法 K-S 均为 `0.5454545`，不满足可选基线进入论文的条件。
- `data/calibration/B2_jl1_recalculated.csv` 的 40 行复合目标与各分项全部为空，不能将旧 RMSE 记录重命名为复合目标。
- `data/calibration/B2_log.csv` 的最佳初始 LHS 为 `176.5333`，最佳 BO 为 `184.5333`；旧 BO 图所示的 `148.2` 与日志不一致，因此该图被拒绝。
- `scripts/calibration/build_moving_obs_from_irn.py` 的 link-to-IRN 映射函数返回空字典；现有 `moving_irn` 文件由清洁样本全局中位数按比例缩放生成，未使用 IRN 速度，不能作为相机就绪输入。

### 旧图片基线

项目中没有一组同时满足设计文档 Fig. 1–Fig. 5 新定义、且具有本次实验来源记录的正式图片。与新图内容最接近的旧候选如下，全部只作为修改前基线，不作为相机就绪证据：

| 文件 | 像素 | DPI | 修改前 SHA-256 |
|---|---:|---:|---|
| `plots/Fig1_threshold_sensitivity_pmpeak.png` | 978×1173 | 300 | `FCCC0CE2B1402AC34AFC4D5EAFA4FE34ED627F0E85B3CAB6B10DB91A167019E0` |
| `plots/P14_ghost_audit.png` | 2148×840 | 300 | `04F97FC7947022FB1D9208EEA7F37BE2E295E6D1FAFA097A561D0ECC5531922D` |
| `plots/P14_robustness_cdf.png` | 1050×900 | 300 | `98018F1D1B5D1C143E184E7CDD2EE688D43FDD71A914CEAF5251CC2CBDBB6A00` |
| `plots/trajectory_stepped_68X.png` | 1036×882 | 300 | `7D3464A75ADCAE795AAAB8EFBB0B830AB2C307CC84E359AFB4D8CC79174D2A71` |
| `plots/trajectory_stepped_960.png` | 1036×882 | 300 | `A9D0E3988CAF4BDCD61DA545FBEA0586F6283A55ABD698A2F53BE5A70B27E1FA` |

旧 BO 图与日志的数值冲突，旧 CDF/审计图也没有 `artifact-sidecar/v1` 和完整输入哈希；本任务不会覆盖这些文件，生产图将写入 `plots/camera_ready_revision_20260716`。

### 现有真实命令

仓库唯一 CI 工作流为 `.github/workflows/smoke.yml`：Python 3.11，安装 `requirements.txt`，执行：

```powershell
python scripts/smoke/p14_smoke.py
```

`reproduce.ps1` 的 fixture 默认入口为：

```powershell
.\reproduce.ps1
```

真实 P14 评估和绘图入口为：

```powershell
.\reproduce.ps1 -UseFixtures:$false
```

仓库没有 `pyproject.toml`、`setup.cfg`、`tox.ini`、`pytest.ini` 或 Makefile，也没有既有 Python 单元测试。根目录直接执行 pytest 会进入本地 `DAPPER-master` 第三方目录并在其 `conftest.py` 收集阶段失败；本任务新增的聚焦测试将显式限定为 `tests` 目录，并在报告中同时保留该基线事实。

## 实验环境与命令

### 环境基线

- Windows PowerShell 7.5.8
- Python 3.11.4，项目 `.venv`
- SUMO 1.20.0
- NumPy 1.26.4
- pandas 2.2.3
- SciPy 1.14.1
- Matplotlib 3.10.8
- scikit-learn 1.5.2
- pyesmda 0.4.3
- 16 个逻辑处理器

最终冻结环境与输入合同证据：

| 文件 | SHA-256 |
|---|---|
| `data/camera_ready_revision_20260716/environment/environment.json` | `0dc13f9cdea7e97a77436d8e065eb086be39419a65ce641fba23fa26d2396b2a` |
| `data/camera_ready_revision_20260716/manifests/effective_manifest.json` | `c5c898edf974ed5fcada3444d2b50f28fcacd3768d9998364f4dbec49e592080` |
| `data/camera_ready_revision_20260716/manifests/input-verification.json` | `b17f393078baf5faf9898e6c680b186d450040c55a0b5717d380d0f2543fbe61` |

### 已执行命令

以下命令均从项目根目录、项目 `.venv` 执行。未安装或升级依赖。

```powershell
.\.venv\Scripts\python.exe scripts\smoke\p14_smoke.py
.\.venv\Scripts\python.exe scripts\experiments\run_camera_ready_revision.py pilot --workers-for-estimate 4
.\.venv\Scripts\python.exe scripts\data_processing\build_l1_eta_observations.py
.\.venv\Scripts\python.exe scripts\experiments\run_camera_ready_revision.py l1 --workers 8
.\.venv\Scripts\python.exe scripts\experiments\run_camera_ready_revision.py ablation --workers 8
.\.venv\Scripts\python.exe scripts\experiments\run_camera_ready_revision.py evaluate
.\.venv\Scripts\python.exe scripts\plots\generate_camera_ready_revision_figures.py --input-dir data\camera_ready_revision_20260716 --output-dir plots\camera_ready_revision_20260716
.\.venv\Scripts\python.exe scripts\plots\generate_camera_ready_revision_figures.py --input-dir data\camera_ready_revision_20260716 --output-dir plots\camera_ready_revision_20260716 --overwrite
.\.venv\Scripts\python.exe scripts\plots\generate_camera_ready_revision_figures.py --input-dir data\camera_ready_revision_20260716 --output-dir data\camera_ready_revision_20260716\verification\figure_reproduction
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -ra tests\test_l1_objective.py tests\test_paper_*.py
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -ra tests\test_paper_figures.py
.\.venv\Scripts\python.exe -m compileall src\paper_experiments scripts\data_processing\build_l1_eta_observations.py scripts\experiments\run_camera_ready_revision.py scripts\plots
git diff --check
.\.venv\Scripts\python.exe -m pip check
```

测试进程通过 `PYTHONDONTWRITEBYTECODE=1` 或临时 `PYTHONPYCACHEPREFIX` 禁止污染仓库字节码缓存；Codex 执行器首次未提供 `WINDIR`，导致 Matplotlib 三个模块在收集阶段退出，失败日志已保留。将进程环境中的 `SystemRoot=C:\WINDOWS` 映射回 `WINDIR` 后，原命令不改测试标准地重跑通过。正式阶段统一使用 `run_camera_ready_revision.py`，每个物理 run 都有不可变 manifest、状态、日志和输出；没有调用旧 RMSE-only 或旧 unified L2 结果替代真实实验。运行中最多观察到 8 个 SUMO，未超过授权。最终数据目录与生产图合计约 `4.292 GiB`，未超过约 4.5 GiB 授权。

## 源码修改说明

### 协议与配置

- 新增 `config/paper_camera_ready_manifest.json`，固定开发/跨日数据、HKT 半开窗口、四个线路方向、L1/L2 边界、15+25 等预算、五个种子、Rule C、A0-A4 机制矩阵、SUMO 输入和输出 schema。
- manifest 中 10 个数据输入和 10 个基础模拟/映射输入均与实际 SHA-256 一致，共核验 20 项；新增的三项是 ETA 源、targeted hybrid L1 事件及其派生 sidecar。两个 IRN 目录使用按相对路径、字节数和文件哈希组成的 `sha256-tree-v1`。
- `config/calibration/l2_priors.json` 改为 `Ne=10`、`K=3`、初始阻尼 `0.3`；`config/calibration/l2_protocol_config.py` 将背景参数明确命名为 `minGap_background`。
- `requirements.txt` 补入新协议实际使用且环境中已安装的 `scikit-learn>=1.5,<2`；没有执行依赖安装。

### L1、搜索和仿真合同

- `src/calibration/objective.py` 实现逐线路累计到达时间契约、至少三个下游站点、`JL1_68X` 四分项、`RMSE_960<=350 s` 约束和规定惩罚；缺失/畸形输出抛出基础设施错误，不再作为数值输入代理模型。
- 每车到站时间必须随站序严格递增；重复或倒序时间被拒绝，首个按时间成功匹配的站作为相对时间零点。
- `search.py` 固定共享 15 点 LHS、独立 continued LHS 25 点、BO EI 候选池和逐种子预声明目标。
- L1 汇总和 Fig. 5 的累计最佳值只接纳 `feasible=True` 的评估；首个可行点之前保留 `NaN`，不可行惩罚值不会被提升为最佳值或目标到达记录。
- `contracts.py` 对参数别名采用“恰好一个”规则，防止 `selected_parameters`/`final_parameters` 被静默忽略；A0-A4 校验固定 split、窗口、路线、审计、SUMO seed/输入、精确 baseline 常数及冻结关系。
- `simulation.py` 只接收由完整 per-run manifest 预计算的 provenance/effective/component hashes；`run-status/v1` 的成功、失败和复用路径使用同一身份，输出内容哈希独立记录。
- `pipeline.py` 和 `run_camera_ready_revision.py` 实现真实输入哈希核验、不可变 per-run manifest、环境记录、真实 pilot 与工作量估算。旧脚本仍保留用于历史复核，不纳入新证据链。

### 审计、L2 和指标

- `audit.py` 实现共同资格、link-hour 聚合、严格 Rule C、MAD、Isolation Forest、预声明 quantile fallback、retention 和 3×3 敏感性。Isolation Forest 固定 `n_estimators=200`、`max_samples=auto`、`contamination=auto`、`random_state=42`；模型哈希覆盖标准化统计、参数及拟合后的树、阈值、offset 等语义状态，并排除 scikit-learn 1.5 对有限特征无语义的未初始化 `missing_go_to_left` 字节。独立重复拟合产生相同哈希，改动树阈值、offset 或冻结统计会改变哈希。
- quantile fallback 只在 Isolation Forest 的拟合或应用抛出异常时启用；失败记录包含操作阶段、异常类型、消息和摘要。fallback 在开发集以线性插值冻结 `Q95(travel time)` 与 `Q05(speed)`，同一模型直接应用跨日集；Rule C 或 MAD 的失败不会触发该 fallback。本次生产 evaluate 中 Isolation Forest 成功，fallback 未启用。
- `evaluation_stage.py` 生成 `audit-manifest/v2`：在完整 raw eligible records 上拟合审计方法，并冻结每个 split 的 A0-supported 评估全集；记录 `cross_day_model_refit=false`、精确 raw/supported/unsupported/flagged/retained 键、模型状态哈希、包版本和 IRN 证据。IRN 证据包括映射、观测索引、link-edge 映射、目录树、入选文件清单及清单哈希、排除数和 segment median 数量。
- `irn.py` 建立 45 个 link 到 460 个唯一 IRN segment 的确定性映射，映射哈希为 `73c8b6d0742ac303f46ac9b8897810c56260d8fe052538e558ecf3b55e180667`。IRN 选择依据 XML 内部 `<date>/<time>`，不依据抓取文件名：开发集排除内部时间 16:55 的 1 个文件，使用 31 个文件、形成 4351 个 segment median，入选清单哈希为 `b7fdfac92f867df28b782ec105b1c722bc3211df8092bcb44b09a66f541d4f1d`；跨日排除 14:55、14:59 的 2 个文件，使用 24 个文件、形成 4383 个 segment median，入选清单哈希为 `99fc1a994e15a42b226334231809d9919ff1b7ed4e0caabf16b3a57c88d91723`。
- `sumo_data.py` 将模拟事件时间改为起点离站 `current.ended`，与真实 `departure_ts` 对齐；旧终点到站口径在一个现有 A0 输出的 136 条链路中会令 27 条落入不同 15 分钟桶，并令 6 条跨越整窗边界。
- L2 Rule C 改为先按 link-hour 聚合再删键，不再逐事件删样本。开发 M11 冻结索引真实产生 raw 11 键 `[1,2,3,4,5,6,7,8,9,10,22]`，moving-only 保留 5 键 `[1,2,5,7,22]`；两个语义的键集和哈希已在校准 manifest 中分别冻结。
- `ies.py` 固定 `Ne=10`、`K=3`、阻尼 `0.3` 和 `10000*seed+100*iteration` 扰动日程。
- L2 入口增加 L1 来源拒绝门：五个种子必须各有完整 40 次 BO 成功预算；`selected_for_l2` 必须等于 `methods.BO`、可行且是该种子的最佳可行 BO 行；参数边界、候选哈希、目标值、run ID/目录、manifest/provenance/effective/component/output 哈希、`run-status.json` 和 `stopinfo.xml` 必须逐项一致。任一缺失、篡改、越界或不可行值都会在启动 IES 前失败。本次五种子已通过该门，L2 才进入执行阶段。
- 最终阶段把未执行的降级计划写成 `blocked-run-disposition/v1`，不再伪装为 `run-status/v1`。本次运行进程载入旧实现后产生的 10 个 legacy blocked `run-status.json` 已逐文件核验并移入 `quarantine/legacy-blocked-run-status-v1`，原 SHA-256 全部保留；活跃目录重新生成 10 个 `blocked-disposition.json`，与 `ablation_runs.csv` 的 blocked 行逐字段一致。活跃 final 目录只剩 80 个成功 `run-status.json`（40 parent + 40 attempt）。
- `metrics.py` 实现事件级 `D`、20/20 全窗下限、5/5 子窗下限、900 秒半开窗每 60 秒滑动和 `paper-metrics/v1`；成功行若样本或窗长不符会被 schema 拒绝。

### 绘图与报告接口

- `figures.py` 与 `generate_camera_ready_revision_figures.py` 固定 Fig. 1–Fig. 5/Table I 的生产输入合同；缺目录、缺列、非有限值、少于共同三种子或来源键不一致时直接失败。Fig. 2 的 Rule C 先对开发集固定单小时的 link-hour 键取旅行时间、速度和距离中位数，再把一个键级判定传播到该键的全部事件；验证器拒绝同键多判定或逐事件阈值结果。Fig. 5 从 `bo_lhs_evaluations.csv` 直接重建每种子目标和可行 cumulative-best，不读取 `bo_lhs_summary.csv` 作为绘图数据。
- Fig. 1 渲染测试检查 7 个 `FancyBboxPatch`：逐框读取文字与框的像素 bbox，要求文字四边均位于对应框内。生产图已通过该检查。首次生产视觉复核发现 Fig. 2(b) 规则说明压住散点、Fig. 3(d) IRN 注释与图例重叠；仅修改绘图布局，增加白底说明和防重叠测试，未改任何实验数据，随后从相同 CSV 重生成并复核通过。用户后续复核要求 Fig. 2(b) 说明再上移且 Fig. 3(d) 柱顶增加余量；修改前 PNG SHA-256 分别为 `32caef411ef39c44d8683a504acb69da22d19d7a87237f1253fae85280889d13` 和 `e0599d67411daee7722efab4a0ee8a90e9e3e8601003a9433caf52850f74be1d`。本轮将 Fig. 2(b) 说明的轴高从 0.04 调至 0.18，将 Fig. 3(d) y 上限从 1.00 调至 1.08，新增散点遮挡与柱顶留白测试；数据、阈值、图例、颜色和尺寸均未改变。
- 所有生产 PNG 固定 300 DPI，并写 `artifact-sidecar/v1`，记录 manifest、全部 CSV 的路径/字节数/SHA-256、脚本哈希、命令、PNG 哈希、像素、DPI、字体、调色板和 provenance hash。
- 绘图测试仅在临时目录使用合成 fixture，未向生产 `plots` 目录写入假数据或假图片。原六图在 `verification/figure_reproduction` 完整复现；用户布局修正后，受影响的 Fig. 2/3 在 `verification/figure_reproduction_user_layout_fix` 再次独立生成并与当前生产 PNG 逐字节一致。

## 补跑结果

### 单次真实 SUMO pilot

命令：

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_camera_ready_revision.py pilot --workers-for-estimate 4
```

结果：首轮成功，SUMO seed `900000`，实际运行 `31.4870 s`，生成 160 条有效模拟链路事件。`stopinfo.xml` SHA-256 为 `fb346d24021deea2d7b661ed3046262915862cc529ecffb89e888feca6b14092`。

| Pilot 证据 | SHA-256 |
|---|---|
| `data/camera_ready_revision_20260716/pilot/pilot-summary.json` | `78E6E8253E0BD72A119133E8729AA34953CE8D53B84E02A9DFCD3725C065A0BF` |
| `data/camera_ready_revision_20260716/pilot/environment.json` | `AA4C655BEE9D56846CD28520CB6EBE9C70FE6F30ADE99712B120010894068C9F` |
| `pilot/A0/development/seed-0/run-manifest.json` | `0877B27DC2146AFEEED686182BAEB10330A625A1A86A0A253410F98408D706FA` |
| `pilot/A0/development/seed-0/run-status.json` | `730C7123C2F2A4B4482FBDF2219887D9690EDB79D5DB0E2359F8FC5E739B5CE6` |

该 run 的 manifest hash 为 `ea3aa8e76f5618fdc6e5ccc1b086099f288b8b3f315f2e3d992ae10f17501690`，provenance hash 为 `7c9a13b1cd55afe91270bc19bcdf316fe15fc179752169556c512312d42b0ef0`，simulation-effective hash 为 `8b2f0f599b7babc47dbda8346e79f6a5529b5b79edb1b30457dabced74336ef1`。

### 主实验工作量估算

设计的名义物理仿真次数为：L1 325、L2 IES 450、最终 A0-A4×五种子×两 split 50，共 825。以 pilot 中位数估算，串行约 `7.216 h`；4 worker 理想下界约 `1.804 h`，线性磁盘投影约 `4.47 GiB`；pilot 后超时按设计取 `max(1800 s, 3×median)=1800 s`。用户随后授权最多 8 个并发 CPU worker。

实际主实验调度 786 个逻辑格：L1 325、L2 421、final 40。被接受的逻辑主运行为 785 个（325+420+40）；L2 唯一失败格实际以同一 manifest/seed 调用 SUMO 三次，所以主实验 SUMO 进程调用总数为 788（325+420+3+40），pilot 另计。该失败 outcome 的其余 29 个成员未调度，10 个 seed 4 final 计划项 blocked。最终数据目录约 `4.291 GiB`，连同生产图约 `4.292 GiB`，没有超过约 4.5 GiB 授权。

### L1 观测链前置校验

在启动 325 次 L1 仿真前，使用 pilot `stopinfo.xml` 和 manifest 指定的 `data/processed/link_speeds.csv` 调用正式累计时间目标。目标函数按设计从 sequence 1 构造可达链时失败：`route=68X, bound=inbound has 1 matched downstream stops; at least 3 are required`。开发集各方向从 sequence 1 连续可达的相邻链路数为：68X inbound 1、68X outbound 1、960 inbound 22、960 outbound 0；manifest 选择的 68X inbound 因缺少 `2→3` 无法计算 `JL1_68X`。

为排除既有 CSV 偶然损坏，执行了仓库现有清洗命令，将结果写入隔离诊断目录，未覆盖原始数据：

```powershell
.\.venv\Scripts\python.exe scripts\data_processing\clean_kmb_links_offpeak.py --eta data\processed\station_eta.csv --dist data\processed\kmb_route_stop_dist.csv --out-times data\camera_ready_revision_20260716\diagnostics\rebuilt_peak_link_times.csv --out-speeds data\camera_ready_revision_20260716\diagnostics\rebuilt_peak_link_speeds.csv --out-stats data\camera_ready_revision_20260716\diagnostics\rebuilt_peak_link_stats.csv
```

首次诊断因目标目录不存在而以 `OSError: Cannot save file into a non-existent directory` 退出；创建本次运行的 `diagnostics` 目录后按同一参数重跑成功，完整 stdout/stderr 保存为 `diagnostics/rebuild_peak_links.log`（SHA-256 `FD48B7849A73C2315B8CD8A04411E5E5CFF4400A643D0406843A6BB96E434A0F`）。重建的 226 行事件文件 SHA-256 为 `ABA7EA65ED75B1AF88C2324906B102B4202B6FA0E67F5D30F4D3C30E32B522F6`，聚合文件 SHA-256 为 `FDE3DA7D37FF4983C41EA630466D126BE912D804D8D8A17B42D70DE282D530AE`，均与 manifest 原始输入逐字节一致，68X 链缺口仍存在。

进一步核查了项目内其他处理后链路文件：开发日 `link_times.csv` 与上述事件文件相同；`link_speed_stats.csv`、`enriched_link_stats.csv`、跨日 link 文件也没有同时满足 68X/960 inbound 从 sequence 1 至少 3 个下游站的观测链。`station_eta.csv` 含相邻站 ETA 状态，但现有清洗器在 68X `2→3` 上把同一抓取时刻的状态排除后匹配到约 9–11 分钟后的下一班车，随后被既定 2 km/h 下限过滤。改用 ETA 值推导、容许同抓取时刻或填补缺链都会改变观测构造方法及实验结论，设计文档没有预声明，故未自行采用。

用户随后明确授权在 ETA 重建或补齐 `2→3` 中采用经判断的方案。独立只读审计确认 `eta_seq` 仅表示各站局部预测排名，缺少跨站车辆身份；全链同 rank 重建会产生错配与条件删失，因此该 3292 行尝试被移入 `quarantine/eta-full-rebuild-rejected`，没有进入正式实验。最终采用定点、可复现的 hybrid：完整保留 `link_speeds.csv` 的 226 条 pass-derived 事件，只为缺失的 68X inbound `2→3` 追加 ETA-derived proxy。规则要求 capture、上游 ETA、下游 ETA 全部位于 `[17:00,18:00)` HKT，旅行时间 `>10 s`，速度 `2–100 km/h`，先按 ETA 对去重，再排除任一端标为 `Scheduled Bus` 的记录。

正式命令使用脚本内由 manifest 固定的默认路径：

```powershell
.\.venv\Scripts\python.exe scripts\data_processing\build_l1_eta_observations.py
```

正式 hybrid 文件共 233 行：226 条 base 在原始 10 个测量字段上逐行多重集一致，补充 7 条唯一 link key 均为 68X inbound `2→3`；旅行时间为 81–82 s，均值 `81.714286 s`，Scheduled Bus 计数为 0。派生文件 `observations/l1_hybrid_link_events.csv` SHA-256 为 `EC6EE25EA0C6C359AD372912C47A6E2FCB580A5C42335C523CD520177A9F1BC4`；sidecar SHA-256 为 `5836CE0502491F6B5D9F84C8EE612C8D2479F08F79B432CCE7ED1BEBD900DBCA`。同一无参命令再次执行时只允许既有内容逐字节相同，否则拒绝覆盖；最终不可变复现退出 0，输出/sidecar 哈希不变，日志 SHA-256 `630cf790399c3901f42a0030f014126c55fc9c1d5146e6e639afc937b696f3d83f`。最终 preflight 为 68X inbound 15 个、960 inbound 22 个连续下游链路；pilot 正式目标成功计算 13/8 个匹配站，基线候选真实判定为不可行，未硬编码通过。

### L1 正式补跑与冻结审计

正式命令为：

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_camera_ready_revision.py l1 --workers 8
```

首轮从 `2026-07-16 11:24:11.072 +08:00` 至 `12:22:00.461 +08:00`，墙钟耗时 `3469.389 s`（57 分 49.389 秒）。325 个物理候选全部在 attempt 1、exit code 0 下成功：75 个 shared initial、125 个 BO subsequent、125 个 LHS subsequent；失败和重试均为 0。各物理 run 的 `duration_s` 合计 `17039.714 s`，均值 `52.430 s`。L1 目录冻结后占用 `1,892,801,663` 字节，即 `1.763 GiB`。

聚合评估文件含 400 行报告记录：BO 200 行、LHS 200 行。每种方法和种子均为 1–40 的完整预算；15 个 shared initial 在两种方法中按协议各报告一次，因此 400 个报告行对应 325 个物理 run。报告行中 `feasible=True` 为 297 行，`feasible=False` 为 103 行。汇总文件含 10 行，五个 `selected.json` 均成功；L2 按预声明协议冻结每个种子的最佳可行 BO 候选，即使某种子的 LHS 最终值更低也不改用 LHS。

| 种子 | 预声明目标 | BO 最佳可行值 | BO 到达评估 | LHS 最佳可行值 | LHS 到达评估 |
|---:|---:|---:|---:|---:|---:|
| 0 | 2404.637853 | 2308.226407 | 19 | 2516.870009 | 未到达 |
| 1 | 1635.005424 | 1608.555842 | 16 | 1721.058341 | 未到达 |
| 2 | 2212.689138 | 2328.292971 | 未到达 | 2329.146461 | 未到达 |
| 3 | 2311.867803 | 2433.545056 | 未到达 | 1647.207749 | 39 |
| 4 | 2541.644599 | 1261.747710 | 17 | 2259.359839 | 31 |

五个 `selected_for_l2` 均为对应种子的最佳可行 BO 行。设计要求的 `JL1_68X` 四个组成量、960 约束和 feasibility 如下；`JL1 = RMSE + MAE + 0.5×SD(|e|) + 0.3×Q90(|e|)`：

| 种子 | BO eval | RMSE 68X | MAE 68X | SD(|e|) | Q90(|e|) | JL1 68X | RMSE 960 | feasible |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 19 | 883.872660 | 688.742857 | 553.953208 | 1528.780952 | 2308.226407 | 248.669147 | true |
| 1 | 39 | 628.269115 | 536.029365 | 327.711154 | 934.672619 | 1608.555842 | 204.828372 | true |
| 2 | 33 | 893.283197 | 697.706960 | 557.817056 | 1527.980952 | 2328.292971 | 335.812713 | true |
| 3 | 8 | 929.401507 | 729.495421 | 575.867685 | 1622.380952 | 2433.545056 | 263.175099 | true |
| 4 | 27 | 501.487598 | 462.200433 | 194.577930 | 669.235714 | 1261.747710 | 282.405396 | true |

冻结文件哈希如下：

| 文件 | SHA-256 |
|---|---|
| `l1/bo_lhs_evaluations.csv` | `f09af10c905c287c670e8cc410c04942853aad9d29165a5cb43087120babdc13` |
| `l1/bo_lhs_summary.csv` | `7924dd9ae4dc4d371cb6dc4d1f3a217c58af1e55c99d1e4a608bbd992f59dfca` |
| `l1/stage-result.json` | `725cb1bb6171513297bb9ceda1d014409cbc18831fc86bb1b3558c814b8035df` |
| `l1/l1-stage-console.log` | `afbb86df86391a018b749082fd0753eed97acbab36d6d7e8f3f0fa94c8ccc55c` |

首轮汇总暴露出一个报告层错误：`final_best_objective` 曾对可行与不可行行直接取最小值，而 `selected_*` 已正确选择最佳可行行。该错误没有改变 325 个仿真、400 个评估行、任何 `stopinfo.xml` 或五个冻结候选。修正后，10 个汇总行均令 `final_best_objective == selected_objective == 最佳可行 objective`；其中 6 行数值发生修正：

| 种子/方法 | 修正前 `final_best_objective` | 修正后 |
|---|---:|---:|
| 0/BO | 2028.343495 | 2308.226407 |
| 0/LHS | 2028.234585 | 2516.870009 |
| 2/BO | 2015.729797 | 2328.292971 |
| 2/LHS | 2076.050558 | 2329.146461 |
| 3/BO | 2005.816345 | 2433.545056 |
| 4/LHS | 2081.653988 | 2259.359839 |

修正前汇总保存在 `quarantine/l1-summary-before-feasible-best-fix/bo_lhs_summary.csv`，SHA-256 为 `5f3efe2d60834270ee615fb971682923e6f4b2ba9758166884ee4105152fdc12`。首次尝试从完整 checkpoint 重建汇总时，旧恢复路径仍并发重新生成 Sobol/LHS 候选，SciPy 抛出 `ValueError: got differing extents in dimension 0 (got 30 and 18)`；随后不可变写入保护拒绝覆盖 seed 3 的 `selected.json`。失败日志完整保留为 `l1/l1-stage-summary-rebuild-console.log`，SHA-256 为 `3dfba08e019731344d49eeb78bae7c7cdc1078f27c386527b902ad6941b18998`。

该失败没有启动 SUMO。它只留下一个意外候选的 manifest 与 manifest-hashes，已移至 `quarantine/l1-summary-rebuild-sobol-race/eval-016-a6c2947a03ff444da5a86055144ded37723c091918824f318392efa24e92a596`；两文件 SHA-256 分别为 `e9380d34ce8c5b2cfd92adc978a95db439786e8b24ee1368304981c7ec018b22`、`a30e1bc5b83cf398f4fe41f0312a613499cac9cebde709d95427ab2fa96d8d09`。恢复路径随后改为直接读取完整 checkpoint 中的候选参数，逐项重建合同并核验已有 manifest、状态、输出哈希和目标值，不再调用任何 Sobol/LHS/BO 候选生成器。最终 checkpoint 重建没有新增物理仿真并成功输出当前汇总；日志 `l1/l1-stage-summary-rebuild-checkpoint-console.log` SHA-256 为 `7a1b7f0f68ef7789011fd143b15eaa2677e5c40fee69cc29b364908ad56f0dba`。

独立只读审计确认：325 个父级 `run-status.json` 全部成功、attempt 1、exit 0，run ID 全部唯一；75/125/125 的物理阶段计数正确；325 个 `stopinfo.xml` 实际 SHA-256 与状态文件完全一致；400 个报告行的 SUMO seed 全部满足 `100000 + 1000*optimization_seed + evaluation_index`；五个种子的前 15 个 BO/LHS 候选哈希、目标值和可行性完全相同；10 个汇总行全部等于对应方法的最佳可行行。上述核查的哈希、seed、shared 设计和汇总不一致数均为 0。

### L2 IES 与最终消融补跑

正式命令为：

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_camera_ready_revision.py ablation --workers 8
```

命令退出码为 0，墙钟约 `4829.4 s`（80 分 29.4 秒），阶段状态按设计写为 `partial`，而不是把失败隐藏为成功。L2 名义网格为 `3 配置 × 5 种子 × 3 iterations × 10 members = 450`；member SUMO seed 固定为 `200000 + 10000*seed + 100*iteration + member`。实际调度 421 个成员：420 成功；唯一失败为 `A3/seed4/iteration1/member0`，attempt 1–3 均以相同 SUMO seed `240100`、相同 manifest 和输入哈希完成 SUMO（exit 0），但 post-output validator 每次都拒绝缺少必需的 `68X inbound 12→13` 链。该配置种子的其余 29 个成员没有调度，也没有替换种子。

因此 15 个 L2 config-seed outcomes 中 14 个成功、1 个失败；成功输出含 42 行 iteration、420 行 ensemble 参数和 2820 行 simulation 观测。独立审计重算 56/56 个 outcome 产物哈希、420/420 个 stopinfo 哈希、421/421 个成员 seed 公式，均无差异；raw 观测为 11 维、moving-only 为 5 维，初始 ensemble、先验和 seed schedule 的共享关系全部成立，`reused=true` 为 0。

设计第 11、12、14 节允许在一个种子失效时使用 3–5 个共同种子，并要求四个有效种子必须如实报告为四个。本次共同成功种子精确为 `[0,1,2,3]`。最终阶段据此完成 `5 配置 × 4 种子 × 2 split = 40` 次真实仿真，final SUMO seed 固定为 `300000 + 1000*seed + split_index`，其中 development=0、cross_day=1；全部 attempt 1 成功、无重试。seed 4 的 A0–A4 × 两 split 共 10 个计划项明确记录为 blocked。40 个 stopinfo 共解析出 6969 个有效事件，每个 run 68–281 条，事件时间均在 `[0,3600)`，旅行时间、距离和速度均为正。461 个 L2/final manifest 的 canonical、provenance、simulation-effective 和 component hashes 均独立重算一致。

| L2/final 证据 | SHA-256 |
|---|---|
| `ablation/ablation_runs.csv` | `173a488605a905c7d940c25ad367f1700783d52eceef5fa2cdb303f5eb10c93a` |
| `ablation/stage-summary.json` | `8968687170f12d9cfbc4b8ed2c839f61f1cef5c9481e772cfaaf5600b10a7db7` |
| `ablation/ablation-stage-console.log` | `b20dc6b04a9a481196a40cbf2ba98a963d0e28bcc9a838d0d6832335e56811aa` |
| `l2/A3/seed-4/l2-status.json` | `98527bb8e8751386de46162ab1e047060be352a2cf4fc577b15afc3dd073abf9` |
| `l2/observations/observation-contract.json`（文件） | `66685d963920f639e1fab6863063c0eb2c7ff5500af3e190785ea74aecad1105` |
| raw observation | `fccefe295543fc5bde77ad183e8f1a1bc787b8ba51f46be234cf60e8bdbba42a` |
| moving observation | `d19755a125bf891fe385ffc2540d35531be603f4f27a63b382ce04af355ba9fd` |

`observation-contract.json` 内的规范化语义合同哈希为 `e185653fb994f34c483760ccc62e45c683327107a4d3c12e0ceed9d574ce2b0a`；它与文件字节 SHA-256 用途不同，不混称。

### 生产评估、审计与 Table I

正式命令为：

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_camera_ready_revision.py evaluate
```

命令退出码 0，观察墙钟约 15 秒。`evaluation-stage-console.log` 只包含阶段结果 JSON 加 `stage=evaluate`，自身没有嵌入 command、duration、timestamp 或 exit code；这些运行信息来自本次实际执行记录，未冒充为日志内字段。stage result 为 `succeeded`，SHA-256 `8d16eb6277845d747572e558f005f7354c2e68fae7706b6b326bcbe492026c4b`；console SHA-256 为 `7c634e71e06dfb234a5edec5315d2f05e947df8f8a7214dbcf8665571928885d`。

`paper_metrics.csv` 精确包含 `5 configurations × 4 seeds × 2 splits × 4 metrics = 160` 行，全为 succeeded 且没有 seed 4。独立使用 SciPy 重算所有 full-window 与 worst-15-min K-S、窗口、样本数和哈希，零差异；所有 worst 窗口均为 900 秒、以 60 秒步进，每源至少 5 个样本，full-window 每源至少 20 个样本。固定清洁人口为 development 18 个 link keys/60 个真实事件，cross-day 13 个 link keys/29 个真实事件。

Table I 的均值与 `ddof=1` 样本标准差独立重算一致：

| 配置 | Dev KS-speed | Dev worst KS | Cross-day KS-speed | Cross-day worst KS |
|---|---:|---:|---:|---:|
| A0 | 0.154169 ± 0.002818 | 0.560606 ± 0.000000 | 0.389555 ± 0.009001 | 0.739744 ± 0.007402 |
| A1 | 0.189406 ± 0.043582 | 0.679563 ± 0.046405 | 0.339426 ± 0.034783 | 0.875000 ± 0.050000 |
| A2 | 0.161301 ± 0.016017 | 0.521875 ± 0.031435 | 0.393600 ± 0.007120 | 0.738848 ± 0.024540 |
| A3 | 0.199779 ± 0.039018 | 0.677772 ± 0.079721 | 0.350879 ± 0.027265 | 0.827083 ± 0.092139 |
| A4 | 0.196847 ± 0.043817 | 0.716589 ± 0.094514 | 0.347875 ± 0.034952 | 0.875000 ± 0.050000 |

按较低 K-S 更好、变化率定义为 `(A0−A4)/A0`，A4 相对 A0 的 development full/worst 为 `−27.68%/−27.82%`，负值表示 A4 更差；cross-day full 为 `+10.70%`，cross-day worst 为 `−18.28%`。更直接检验审计作用的 A4 相对 A3 对比也呈混合方向：development full 改善约 `1.47%`、worst 恶化约 `5.73%`；cross-day full 改善约 `0.86%`、worst 恶化约 `5.79%`。因此结果不能表述为 A4 全面优于 A0 或 A3，也没有选择性隐瞒不利结果。

Isolation Forest 只在 development 的 76 个 raw eligible keys 上拟合，并冻结应用于 cross-day 63 keys；scikit-learn 1.5.2，全状态哈希 `a81301668ac9ab1186cfd06e6bf4534fd1864bcfe1be14ef6a00c350a526606d`，独立重复拟合一致，quantile fallback 未触发。A0-supported 集合为 development 34、cross-day 30；unsupported 分别为 42、33。Rule C/MAD/IF 共享同一 raw eligible universe 和同一 A0 reference/support；retention 的分母是 raw eligible，而各方法的 K-S 使用各自 `retained ∩ A0-supported` 的 evaluation keys，因此方法间 evaluation key 数可以不同。

IRN contradiction 必须连同小分母与未匹配数解释：Rule C development 为 `1/1` 且 32 未匹配，cross-day 为 `3/3` 且 30 未匹配；MAD 两 split 均 `0/0` 且 0 未匹配；IF development 为 `0/0` 且 3 未匹配，cross-day 为 `1/1` 且 12 未匹配。这些比率不是分类准确率，正分母时的 100% 也不能据此作强结论。

| 评估产物 | 行数 | SHA-256 |
|---|---:|---|
| `metrics/paper_metrics.csv` | 160 | `daf8fba62b74a93a4eda543d10051acbbc2a0c4b2ad8d6a2b7062dc27e0fa160` |
| `tables/table_i.csv` | 5 | `5f2073a25bb9b718312ce18b6b5718e5cac80e7ac300ac384049674ff6fd65da` |
| `audit/audit_metrics.csv` | 6 | `d4ded27147f30f4ae9bdb432a61cab7cf1044465905eaa5f948b35d7580f73d0` |
| `audit/audit_manifest.json` | — | `2d3bf54714263d929364fe9adb229b6c781763574e990d10ca4083c9d667ab54` |
| `reporting/fig2_contamination.csv` | 200 | `5ca808cc7375f1013325fd246e9025c753674fc3b63a0824dba02005d13812de` |
| `reporting/fig2_trajectory.csv` | 9 | `e249eab28f8d0e885f359ff6dba7f9b4eef44896920f3a49ca5440ca6ec3534c` |
| `reporting/fig3_sensitivity.csv` | 9 | `fa7b4840a72da013d488a4ca94427d0e7e8c7a963c6abdf4ae30463c9d7941a8` |
| `reporting/fig4_cdf_samples.csv` | 758 | `1dcf5aece473c9a049376ec58bb14048147adf0b6b9d72898b8df9b19c16bdc8` |

## 图片与表格清单及数据来源

生产命令为：

```powershell
.\.venv\Scripts\python.exe scripts\plots\generate_camera_ready_revision_figures.py --input-dir data\camera_ready_revision_20260716 --output-dir plots\camera_ready_revision_20260716
```

视觉修正后以 `--overwrite` 从同一生产 CSV 重生成；该选项只替换本次新建的 camera-ready 交付物，没有覆盖旧实验数据或旧图片。最终清单如下：

| 交付物 | 生产输入 | 像素 / DPI | PNG SHA-256 | sidecar SHA-256 |
|---|---|---|---|---|
| `Fig1_camera_ready_architecture.png` | `manifests/effective_manifest.json` | 2148×825 / 300 | `5edfd0cdffb34f47971d744f4bded740795ca907e1892be614f4479d45ca4e9b` | `035bb85aa7754fc8222de8b1e9bde15a639ab5d051614fa33569fd22f709ada9` |
| `Fig2_camera_ready_contamination.png` | manifest、`fig2_contamination.csv`、`fig2_trajectory.csv` | 2148×795 / 300 | `28911248c57dec3ce1815b1d6ca8820d0ad5ab73ba40be2a5edbe3d32e9b3320` | `af8dd05a8732d1607baacc4596a13663f76815933122642925ca7557e057964f` |
| `Fig3_camera_ready_audit.png` | manifest、`fig3_sensitivity.csv`、`audit_metrics.csv` | 2148×1440 / 300 | `15c12949098c863a462818367390121ed78b9a21cb878c567cea190afdac2e8a` | `7d712b67326e067d01b9954cb2bfc3314b21c50e36d86456a62179332f0dd8eb` |
| `Fig4_camera_ready_cdf.png` | manifest、`fig4_cdf_samples.csv` | 2148×795 / 300 | `111ec47f0d060ffd4ac80d343ffd443ec8c7f7a61a639c0ddbc4b7f36c085a2d` | `11aa69e569fe517e597b75ea23827e305958d6bf7615d37b56dccd5986b46afb` |
| `Fig5_camera_ready_bo_lhs.png` | manifest、`l1/bo_lhs_evaluations.csv` | 1050×840 / 300 | `f517d266ef53348662411e9682c59582caae14458becc99ffa099a1ae42b3185` | `eebe33a76ef7d00c981b1535fa4798294d6890640c3bccc55928d00aadbbfe5d` |
| `Table_I_camera_ready_ablation.png` | manifest、`tables/table_i.csv` | 2148×675 / 300 | `8d81cc4404d93ea2da988223cb71d47a90168832eb2e03099da718bc27b4dc34` | `e9f086a4be1efd63a8b4b4293c409f56464b3febf33944bd26e9c1a1a9deb4c5` |

全部文件为 RGBA PNG，DPI 元数据为约 `(299.9994, 299.9994)`；字体声明为 Times New Roman，运行环境实际解析至 `C:\Windows\Fonts\times.ttf`，生成日志没有字体 fallback。核心离散调色板为色盲友好的蓝 `#0072B2`、橙 `#D55E00`、绿 `#009E73`、浅蓝 `#56B4E9`、灰 `#666666` 和黑 `#111111`；Fig. 3 热图另使用 Matplotlib `Blues`/`Oranges` 连续色图并逐格标注数值。逐图人工检查了标题、图例、坐标轴、单位、颜色、裁切和灰度可读性；Fig. 1 的七个框内文字、Fig. 3 的图例/IRN 注释还由像素 bbox 测试覆盖。

来源检查确认：Fig. 2 contamination 200 行中 74 flagged/126 clean；轨迹 9 行来自 A0 seed 0 development 的 68X outbound `19→20→21`；Fig. 3 为完整 3×3 网格且中心点等于 Rule C 生产审计行；Fig. 4 的 758 行只含 A4 和种子 0–3；Fig. 5 只读取 400 行真实 L1 CSV，从可行点重建 cumulative-best，不读取汇总表伪造曲线。每个 sidecar 的输入路径、字节数、SHA-256、脚本哈希、输出哈希和 manifest 哈希均与实物一致。

第二次完整绘图写入 `data/camera_ready_revision_20260716/verification/figure_reproduction`。用户布局修正后，Fig. 2 和 Fig. 3 又以同一生产 CSV 独立写入 `verification/figure_reproduction_user_layout_fix`；两张新 PNG 与当前生产图逐字节相同，四张未受影响的生产图仍与原完整复现目录相同。本轮复现日志 SHA-256 为 `30ae911f1c957386cd66621fdae000f3b5fd4628211baeaa7ff49da51cc1bea0`，两图规格、生产/复现字节一致性、输入哈希和脚本哈希检查日志 SHA-256 为 `90a9fdee15bf17b1e568a7f4924ca5ea349d74fa8186d687527195a3b06e6e92`。

## 验证结果

| 命令或检查 | 结果 | 结论 |
|---|---|---|
| 用户布局修正前 `python -m pytest -p no:cacheprovider -ra tests/test_l1_objective.py tests/test_paper_*.py` | `120 passed, 2 warnings`，退出 0；日志 SHA-256 `34b60fcc741ba523083871694c6ed0a6392ca66ef0739a42402f8016b572cc0f` | 原完整回归通过；两条均为 sklearn GPR 核参数边界 `ConvergenceWarning` |
| 用户布局修正后 `python -m pytest -q -p no:cacheprovider tests/test_paper_figures.py` | `17 passed`；日志 SHA-256 `8098a8134d23e912617c465c78ab17d5c16c76dd9825e76755fbc4f2adb28d9b` | 包含 Fig. 2 说明/散点防遮挡、Fig. 3 柱顶留白和既有 IRN 注释/图例防重叠测试 |
| 用户布局修正后完整相关回归 | `123 passed, 2 warnings`，退出 0；日志 SHA-256 `341d95e16cbc9ee570aad5a4e2495fceb36d90219463ee1e562c67f35aeffa3f` | PowerShell 枚举 `test_l1_objective.py` 及全部 `test_paper_*.py`；两条仍为 sklearn GPR 核参数边界 `ConvergenceWarning` |
| 完整回归首次 PowerShell 通配符调用 | 未收集测试，退出 4；日志 SHA-256 `299d10855d651b47a0dafdca32ce9a0e9d458b4084ecd2c65ce3650a5aebd76c` | `tests\test_paper_*.py` 被 PowerShell 原样传给 pytest；保留失败日志后改为显式枚举同一测试集，未降低标准 |
| `python -m compileall ...` | 18 个文件，0 错误、0 警告；日志 SHA-256 `542261291e41e01dfa3c133c642b536f98e7076f51c4bdad8b4f66533655c0b0` | 源码和入口脚本语法检查通过 |
| `python scripts/smoke/p14_smoke.py` | 退出 0；fixture raw K-S `0.5199`、clean K-S `0.4444`，两张 smoke 图写出 | 既有 P14 workflow 通过 |
| 用户布局修正后 `git diff --check` | 退出 0；只有 Windows 行尾提示；日志 SHA-256 `08f8f041c746680edac06292523d63ac8f70218dbfcbcd998aa8b5803b19db7a` | 无空白错误 |
| `python -m pip check` | 退出 1；日志 SHA-256 `3c9afdac4a32fe76fcfa87e0202e4c168c9c1182c1215ff6e76baa7fcc103beb` | 共享环境存在既有冲突；未安装、升级或降级依赖 |
| 根目录 `python -m pytest -q` | 收集 `DAPPER-master` 时 `TypeError: unsupported operand type(s) for +: 'set' and 'list'` | 第三方本地目录既有收集错误；项目测试显式限定 `tests` |
| manifest 实际哈希核验 | 10 个数据项、10 个 simulator/mapping 项全部匹配 | 20 项输入未漂移 |
| pilot 状态核验 | `run-status/v1`、attempt=1、exit=0、输出哈希一致 | 真实执行链通过 |
| L1 冻结输出独立审计 | 325 个物理 run 全成功；400/10/5 个聚合/汇总/选择记录；297/103 个可行/不可行报告行；seed、shared 候选、状态和输出哈希不一致数均为 0 | L1 真实补跑及其来源链通过 |
| L2/final 独立审计 | 421 L2 成员=420 成功+1 失败；14 outcomes；40 final 成功+10 blocked；461 manifests/460 stopinfo 链全部核验 | 四共同种子降级符合设计；唯一失败和 29 未调度项均已披露 |
| evaluate 独立重算 | 160 指标、5 行 Table I、audit manifest/metrics 及 Fig. 2–4 输入全部零差异 | 生产指标和表格来源链通过 |
| 图片规格与视觉检查 | 6 PNG 格式/尺寸/DPI/字体/输入/脚本/sidecar 全通过；人工检查图例、坐标轴、颜色和裁切 | 图片满足设计与生产规格 |
| 图片重复生成 | 4 张未受影响图与原全量复现目录一致；修正后 Fig. 2/3 与新独立复现目录逐字节一致；QA 日志 SHA-256 `90a9fdee15bf17b1e568a7f4924ca5ea349d74fa8186d687527195a3b06e6e92` | 6/6 最终图片均可由本次冻结结果重复生成 |
| IRN 证据读取核验 | 45 个 link、460 个唯一 segment；开发 31/排除 1、4351 median；跨日 24/排除 2、4383 median | 已写入 `audit-manifest/v2` 并独立重建一致 |
| 源码/图片阶段 Word 边界检查 | 当时 Word 文件/跟踪差异均为 0；README 既有 +2 行和未跟踪文件保留；pycache 状态数量 0；日志 SHA-256 `f0e21da0d86cbaa64e110e2afc031ab99df407a0620e6cda03a9178de03eeabf` | 证明 Word 阶段开始前未触碰主稿，且未处理用户无关改动 |
| IEEE Word 结构与分页 | Word 原生 `ComputeStatistics`=6、3419 词；Letter 8.5×11 in；四边 0.75 in；双栏间距 288 dxa；12 个 section；9 个 inline shapes；1 个 6×7 原生表格 | 版心、分栏、分页、跨栏图和原生表格布局符合指定模板；第 6 页未新增人工分页或缩小字号 |
| Word 修订审计 | 09:58 用户最新版的 2 项既有修订先接受为独立比较基线；当前审阅稿含 `32 ins + 32 del` | OOXML 接受全部修订后与 clean 候选逐段、逐表一致；保留 12 sections、5 drawings、7 math objects、1 个 6×7 原生表格和全部媒体哈希 |
| Word 嵌入图与表格来源 | Figure 1 与原稿字节一致；Figures 2–5 匹配生产 PNG；Table I 从冻结 `table_i.csv` 写成可编辑 Word 表格，未嵌入 Table I PNG | 没有手工调图、错用旧实验图片或以图片冒充表格 |
| Word 最终视觉检查 | 上一版已完成 Microsoft Word 六页栅格检查；当前 09:58 语言修订版尚未重新渲染 | LibreOffice 不存在；Microsoft Word 调用被当前应用执行额度限制拒绝。未把上一版 PDF 冒充为当前版验证结果 |

### 2026-07-17 用户最新版扩写与重新分页

用户手动调整后的审阅稿 SHA-256 为 `e800f1364f3ef6401bc96f928f8e1d63f3d600abbdffaa96f553031f130bb5fba`，已原样备份为 `Operator-Aware Robust Calibration for Bus-Corridor Digital Twins via Bayesian Optimization and Iterative Ensemble Smoothing.camera-ready-tracked.user-edited-20260717-023623.docx`。Word 接受其 203 项既有修订后形成只读扩写基线 `data/camera_ready_revision_20260716/manuscript/user-latest-accepted-base.docx`，SHA-256 为 `7725de156c1a85524212fd6a5d218a7544baf38465bd8084ae068e8d4ad1f5dc`。该处理保留了用户的最新格式调整，不把既有修订再次归到本轮扩写名下。

本轮新增 3 段、539 词，集中在 Fig. 5 之后的局限性和结论区域。新增内容只解释现有合同和结果：成功评估与确定性失败记录、共享初始设计和 L2 随机输入、冻结审计键、retention 与 K-S 的不同分母含义、A1–A4 的混合结果、BO 四胜一负及未来跨日验证边界。没有新增实验数字、外部来源或未验证结论。按 `humanizer`、`humanizer-zh` 和 `intj-global-guardrails` 复查，新增段落未发现高频 AI 套话词、Unicode 破折号、夸大归因或“not ... but ...”置换句；保留 IEEE 技术论文所需的中性语气。

曾生成的 v5、v6 候选均为 7 页：原因是图前扩写把 Fig. 4–5 推至第 6 页，继而把参考文献推至第 7 页。v7 改为只在图后扩写后恢复 6 页；v8、v9 继续利用末页空间，最终 v9 为 6 页、3419 词。正式干净稿 SHA-256 为 `0e78e189435095476ab5af2b760876a71cc6aa1f6211d88d7005580484178b6b`，审阅稿为 `0d186a806b2a58b261966a0ec908e15ea5ae84b8a7d75a59919ad77b358a1e86`，PDF 为 `a0aeac036071f6848a65922b121a647deb3932ef0241b27f30d637c70c22ae94`。扩写脚本 SHA-256 为 `57a5db353593a34a300e85149042bdb75fcbefb344dfe424713759d2c94b6528`，可从上述用户最新版基线重复生成干净稿内容。

### 2026-07-17 09:58 用户语言修订

用户最新版审阅稿 SHA-256 为 `d8ed5d93fcbfef01b6184ca9794db5d82a6b6150a0f77eb1b6bb2406d9d58556`，已原样备份为 `Operator-Aware Robust Calibration for Bus-Corridor Digital Twins via Bayesian Optimization and Iterative Ensemble Smoothing.camera-ready-tracked.user-edited-20260717-095818.docx`。接受该稿 2 项既有修订后形成基线 `data/camera_ready_revision_20260716/manuscript/user-latest-20260717-095818-accepted-base.docx`。当前正式 tracked 文件 SHA-256 为 `a72ce6691180612238c5bee3ae36c6c1559038b978121b80c5d1d12d013a12a4`。

本轮参照原始主稿中 IV 节及其前文的叙述方式，扩充 I. INTRODUCTION 和 II. RELATED WORK 的问题背景、观测语义、数据同化、公交运营控制和 BO 文献联系。用户已经删除的 `Relevant work falls into five streams.` 未恢复。审稿意见要求的第五类文献保留为 `E. Simulation Calibration and Bayesian Optimization`。IV 节删除了与式 (5) 重复的行内 `γ = ...`，K-S 说明改为直接描述经验分布函数最大间距。结果与局限部分同时压缩了此前为填页加入的长段说明。

`humanizer` 与 `intj-global-guardrails` 专项扫描覆盖正文和所有本轮插入文本。正文中 `not`、`but`、`whereas`、`although`、`however`、`while`、`without`、`rather than`、`instead of` 等否定转折模式均为 0；高频 AI 套话词为 0；重复行内 γ 公式为 0；禁用的 K-S 否定转折原句为 0。正文中的 Unicode 长破折号只保留 IEEE `Abstract—` 标签，参考文献题名和页码范围保持原文。接受全部 32 组段落修订后，clean 候选含 120 个段落和 3197 个空格分词，与审阅稿接受结果逐段、逐表一致。

`pip check` 报告的环境级问题包括缺少 `pywin32`（jupyter-core/docker）、basemap 与 Matplotlib/packaging 版本冲突，以及 gradio-client/inference 的既有依赖冲突。这些包不在本项目相机就绪实验调用链中；最新 123 项相关测试、真实 SUMO 阶段、evaluate 与绘图均在已记录环境成功执行。首次最终测试收集因执行器缺少 `WINDIR` 退出 2，日志 SHA-256 `4129bf550fa462665f68792759d99434325e55dc8d32d95d649cc8f474266c54`；恢复系统环境变量后原样重跑通过，没有降低测试标准。

## 未完成项及原因

源码、实验、评估、Table I 和图片没有新增未完成项。09:58 语言修订版仍缺 Microsoft Word 视觉分页、逐页检查、正式 clean DOCX 和匹配 PDF。LibreOffice 渲染器未安装；Microsoft Word 调用因当前应用执行额度限制被拒绝。当前根目录 PDF 对应上一版文字，未作为本轮审阅稿的验证证据。

以下不是未说明的缺口，而是必须保留的范围/结果说明：

- L2 的 `A3/seed4/iteration1/member0` 三次真实尝试均因缺少 `68X inbound 12→13` 被 post-output validator 拒绝；其余 29 个该 outcome 成员未调度，seed 4 的 10 个 final 计划项 blocked。设计允许以四个共同种子完成 A0–A4，并要求如实报告四个，因此没有补种、硬编码或隐藏失败。
- `pip check` 的共享环境冲突和根目录 pytest 收集本地 `DAPPER-master` 的既有错误仍存在；它们不属于本项目相关测试/运行链，且最新相关 123 项测试、P14、compileall、真实 SUMO、evaluate 和绘图均已通过。没有修改第三方目录或依赖环境。
- 末页双栏平衡的首次 Word GUI 测试会产生空参考文献编号 `[16]`，该测试副本已拒绝并清理。最终实现改用独立、无编号的连续分节段落；参考文献页双栏平衡且不存在 `[16]`。
- Figure 1 的流程图内容没有修改。审稿文件没有提出该项内容变更，用户明确要求无要求则不改；仅通过周围正文排版处理其上方两栏对齐。

最终交付位置：源码与测试位于本报告所列项目路径；冻结实验/评估证据位于 `data/camera_ready_revision_20260716`；生产图片位于 `plots/camera_ready_revision_20260716`；Word 审阅稿、干净 DOCX 与 PDF 位于 `D:\Documents\Bus Project\SMC`；审稿意见映射为 `smc-camera-ready-reviewer-change-log.md`；本完成报告即 `smc-camera-ready-revision-completion-report.md`。
