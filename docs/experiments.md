# 实验记录模板
> 按日期记录，每次实验/验证补充一条，保持可复现。示例可复制后填写。

## 2025-12-16（示例）
- **线路/方向/时间窗**：`线路名称/方向`，时间窗 `HH:MM–HH:MM`
- **数据来源与采样**：高德接口（接口名/字段）；采样频率 `XX s`; 缺失/异常说明
- **SUMO 场景版本**：`sumo/config/<cfg>`；路网 `sumo/net/<net.xml>`；附加文件 `additional/`；路线 `routes/`
- **关键参数默认值**：`tau=?`, `sigma=?`, `minGap=?`, `t_board=?`, `a=?`, `b=?`, `capacityFactor=?`（未设可写 N/A）
- **运行命令**：`<python/sumo 命令>`（含工作目录、环境变量，尽量可复制粘贴）
- **输出/日志路径**：`<output 文件路径>`，日志 `<log 路径>`，截图 `<img 路径>`（如有）
- **观察/结论**：指标、现象、问题清单、下一步动作

---

## 2025-12-18 — Week 1 L1/L2 Metric Validation
- **线路/方向**: KMB 68X Inbound (元朗 -> 旺角)
- **数据集**:
    - Real: `enriched_link_stats.csv` (N=Cleaned Link Samples)
    - Sim: `stopinfo.xml` (Baseline Traffic)
- **指标结果 (Baseline)**:
    - **L1 (Macro)**:
        - Real End-to-End Time: **8962.4s**
        - Sim End-to-End Time: **2255.8s**
        - Error: **-6706.6s (-74.8%)** -> 仿真严重偏快（由轨迹图确认）。
    - **L2 (Micro)**:
        - Top Bottleneck: Link 5-6 (Real 542s vs Sim 42s), 缺失大量延误。
        - 速度分布：EMD 距离极大，KS 检验 p-value ~ 0.0。
- **结论**:
    - 当前 OSM 路网 + 自由流配置无法复现真实公交运行。
    - **必须**进行高精度路网重建（引入信号灯/路口延误）及拥堵注入。Week 2 重点明确。

## 待填写记录
- 日期：
- 线路/方向/时间窗：
- 数据来源与采样：
- SUMO 场景版本：
- 关键参数默认值：SUMO 版本 1.20.0；Python 3.11.4（venv，system-site-packages）
- 运行命令：
- 输出/日志路径：
- 观察/结论：

## 2025-12-16（记录）
- 线路/方向/时间窗：预定 KMB 68X（基线稳态）+ 960（过海复杂工况），双向，待采集窗口 1h
- 数据来源与采样：KMB ETA + data.gov.hk（IRN 分段速度、检测器/分段 CSV、JTI、STN、HKO）
- SUMO 场景版本：冒烟测试，内置网 `D:\SUMO\doc\examples\duarouter\flows2routes\input_net.net.xml`
- 关键参数默认值：SUMO 版本 1.20.0；Python 3.11.4（venv，system-site-packages）
- 运行命令：`python scripts/traci_smoke.py`
- 输出/日志路径：标准输出（无额外日志）
- 观察/结论：traci 冒烟通过，SUMO_HOME/PYTHONPATH 生效；下一步采集 68X/960 的 ETA+路况，并准备场景文件

## 2025-12-16（高德数据采集与验证）
- **线路/方向/时间窗**：西安 229 路双向，晚高峰约 18:20 起 1h 轮询（部分轮询被中断）。
- **数据来源与采样**：高德 `v3/bus/linename`+`v3/bus/lineid`（静态）；`v4/etd/driving` 尝试失败（10012 权限不足）；`v3/traffic/status/road` 成功（adcode=610100，长安南路，extensions=all），可 60s 轮询。
- **运行命令**：`python scripts/gaode_downloader.py --key <key> --city 610100 --keyword 229 --line 229 --etd --interval 60 --duration 3600 --count 48 --strategy 1 --firsttime-offset 300 --traffic-road --road-name 长安南路 --road-level 5`
- **输出/日志路径**：原始 `data/raw/229-*{linename|lineid|traffic}.json`；日志 `data/raw/229-fetch.log`；清洗 `data/processed/229_stops.csv`, `229_line_meta.csv`, `traffic_road.csv`。
- **观察/结论**：静态线路/站点获取正常；etd 无权限（10012）无法拿时间分布/路况；traffic/status/road 可用，返回 speed/status/polyline；需用路况 map-match 替代站间速度，ETA 需换数据源或升级权限。

## 2025-12-17 — HK ETA 全量站轮询器调整与运行

- 线路/方向/时间窗：KMB 68X + 960 双向，目标轮询 1h，起始约 17:35（每轮完整跑完全站 stop/stop-eta）。
- 数据来源与采样：KMB `/route` `/route-stop` `/route-eta` `/stop` `/stop-eta`；data.gov.hk `/traffic-speed-map/notification.csv`、IRN 下载版 `irnAvgSpeed-all.xml`、检测器信息 `traffic_speed_volume_occ_info.csv`、STN/JTI/HKO；过滤掉已知失败接口（irn_avg_speed direct、road_network_segments、tsm_raw/raw_dl、TDAS、JSON_BUS）。
- 运行命令：`python scripts/hk_collect.py --routes 68X,960 --service-type 1 --duration 3600 --interval 60`
- 输出/日志路径：`data/raw/kmb-route-*.json`、`kmb-route-stop-*-*.json`、`kmb-route-eta-*.json`、全量站点 `kmb-stop-*.json` / `kmb-stop-eta-*.json`；`tsm-notification-*.csv`、`irnAvgSpeed-all-*.xml`、`traffic_speed_volume_occ_info-*.csv`、`trafficnews-*.xml`、`Journeytimev2-*.xml`、`hko-rhrread-*.json`；终端打印每轮耗时与对齐信息。
- 观察/结论：轮询间隔按照设定间隔（默认 60s）对齐，每轮耗时单独记录；为保证单轮数据完整，启动后不在轮内截断，截止检查发生在轮与轮之间，总时长约 1h（最后一轮可能略跨过截止）；若个别接口出现 SSL/HTTP 报错仅打印 warn 不中断。

## 2025-12-17 — HK 数据组织与里程计算

- 任务：整理 HK 原始数据、生成站点序列与里程表。
- 操作：
  - 运行 `python scripts/organize_raw_files.py` 完成 5950 个 raw 文件归类（KMB ETA/route/stop，IRN，JTI，STN，HKO 等）。
  - 新增 `scripts/clean_kmb_shapes.py`，使用 IRN `CENTERLINE`（EPSG:2326）+ KMB route-stop/stop 计算站点累积里程，输出 `data/processed/kmb_route_stop_dist.csv`（68X/960 双向，110 行）。
- 结果：
  - 归档：`data/raw` 按类别整理（kmb/、irn_download/、detector_locations/ 等）。
  - 里程：基于路网最短路径，`cum_dist_m`、`link_dist_m` 均按中心线长度估算。
  - 运行日志：加载 CENTERLINE 约 36k 行，构图 ~7.86M 节点/10.69M 边，生成 CSV 完成；Shapely 距离计算出现若干 RuntimeWarning（空/无效几何），未影响输出。

## 2025-12-18 — Week 1 自动化基线构建（Data/Sim/Analyst Agents）

- **线路/方向/时间窗**：KMB 68X（元朗公园 -> 旺角/佐敦），方向 `Inbound`（对应 API bound `I`），时间窗 `17:35 – 18:35` (UTC/HKT 自动对齐)。
- **数据来源与采样**：
  - KMB Raw ETA (`data/raw/kmb/route-eta/*.json`)：清洗出 `station_eta.csv` (8773 条记录)。
  - `clean_kmb_eta.py`：自动识别 `eta_seq=1` 的到站/离站事件，已处理 UTC/HKT 时区差异（Pandas UTC 模式）。
- **SUMO 场景版本**：
  - 路网：`sumo/net/hk_baseline.net.xml` (OSM 基础)。
  - 站点：`sumo/additional/bus_stops.add.xml` (自动映射，过滤掉距离路网 >100m 的 15 个孤立站点，修正停靠长度 10m->15m 以适配双层巴士)。
  - 线路：`sumo/routes/baseline.rou.xml` (自动填充路段空白 `edge_X` -> `edge_X+1` 以修复连通性，生成 `flow_68X`，间隔 600s/班)。
  - 配置：`sumo/config/baseline.sumocfg` (步长 1s，时长 3600s)。
- **关键参数默认值**：`accel=2.5`, `decel=4.5`, `sigma=0.5`, `length=12`, `minGap=2.5`, `maxSpeed=20.0` (m/s)。
- **运行命令**：
  - 数据清洗：`python scripts/clean_kmb_eta.py`
  - 场景生成：`python scripts/generate_sumo_stops.py` && `python scripts/generate_sumo_routes.py`
  - 仿真运行：`sumo -c sumo/config/baseline.sumocfg`
  - 基线对比：`python scripts/compare_week1.py`
- **输出/日志路径**：
  - 仿真输出：`sumo/output/stopinfo.xml` (含 `started`, `ended`, `delay` 等)。
  - 对比结果：`data/processed/week1_comparison.csv` (站点 ID 对齐，L1 初步误差)。
  - 问题清单：`docs/week1_problem_list.md`。
- **观察/结论**：
  - **流程跑通**：能够从原始 JSON 直出对比 CSV，闭环验证成功。
  - **拓扑修复**：OSM 路网简陋导致大量站点必须被 Skip，且路线连通性靠“猜名字”修补，不可持续。Week 2 必须引入 `RdNet_IRNP.gdb`。
  - **指标对齐**：SUMO `started` 属性对应实际到站时间；CSV 中已合并展示，待计算 RMSE。

## 2025-12-18 — Week 1 路段层（Layer 2）指标实现与特征合并

- **线路/方向/时间窗**：同上（68X Inbound, 17:35-18:35）。
- **任务目标**：(1) 生成路段层运行指标（Link Travel Time/Speed）；(2) 合并外部环境特征（Weather, Incident, Traffic）。
- **实现方案**：
  - **数据清洗**：`clean_kmb_links.py` 从连续的 `station_eta` 记录中推导站间 `(seq_i -> seq_i+1)` 的出发/到达时间，结合静态站间距离 `kmb_route_stop_dist.csv` 计算路段均速。异常剔除策略：`dt > 10s` 且 `2 < v < 100 km/h`。
  - **特征工程**：`merge_features.py` 使用 `pd.merge_asof` 将 HKO（气温/降雨）、STN（交通事件数）、JTI（主干道平均行车时间）按时间戳对齐合并到路段记录。
    - *Update*: 修复了 JTI XML 命名空间问题（`JOURNEY_DATA` vs `JOURNEY_TIME`），成功提取出宏观拥堵指数。
  - **指标计算**：`compare_link_metrics.py` 计算 Real vs Sim 的 L2 级核心指标：
    - **分布统计**：P10/P50/P90 分位速度、均值、标准差。
    - **距离度量**：Wasserstein Distance (EMD) 和 KS 统计量。
- **运行命令**：
  - `python scripts/clean_kmb_links.py`
  - `python scripts/merge_features.py`
  - `python scripts/compare_link_metrics.py`
- **输出结果**：
  - 原始记录：`data/processed/link_times.csv` (每车每路段), `link_speeds.csv` (同上)。
  - 增强数据：`data/processed/enriched_link_stats.csv` (含 Temp, Rain, Incidents, JTI Avg Time)。
  - 评价指标：`data/processed/link_metrics_overall.csv` (EMD, KS, Real/Sim Stats)。
- **观察/结论（Initial Insights）**：
  - **特征完整度**：天气（Temp: ~23.6C）、降雨、STN 事件（常值 1）、JTI（~13.1 min）均已成功合并。
  - **仿真未校准**：Link 1-2 真实均速 3.44 km/h，仿真均速 78.64 km/h (EMD=75.19)，表明仿真处于完全自由流状态，极度缺乏拥堵机制。
  - **长直路段准确**：Link 15-16 (高速段) 真实 57.4 km/h vs 仿真 60.6 km/h，EMD 仅 3.2，证明车辆动力学参数基本正确，误差源于交通流缺失。
  - **数据覆盖**：因真实数据清洗过滤，部分拥堵路段无有效速度记录，已修复脚本以静态距离表为基准，确保仿真侧指标计算完整。

## 2025-12-19 — Week 2 Network Reconstruction
- **线路/方向**: KMB 68X / 960 (Inbound/Outbound)
- **目标**: 替换 OSM 路网，解决拓扑断裂、站点无法映射、车道属性错误问题。
- **数据源**:
    - `RdNet_IRNP.gdb` (High Precision Road Network): Centerline, Speed Limit, Turn Table.
    - `JSON_BUS.json` (Bus Shapes).
    - `kmb_route_stop_dist.csv` (Mapped Stops).
- **实现方案**:
    1. **解析**: `inspect_irn_gdb.py` 确认 IRN 结构。
    2. **转换**: `convert_irn_to_sumo_xml.py` 将 GDB 转为 `.nod.xml`, `.edg.xml`，并解析 `TURN` 表生成 `.con.xml` (显式连接)。
       - **关键修复 1 (双向路)**: 解析 `TRAVEL_DIRECTION` (3=Bi, 1=Forward)，自动生成 `_rev` 反向边。
       - **关键修复 2 (显式连接)**: 自动生成 25,400+ 条基于 TURN 表的连接，解决立交桥/隧道断路。
    3. **生成**: `netconvert` 配置 `irn_xml.netccfg`。
       - **关键修复 3 (坐标系)**: 启用 `offset.disable-normalization` 和 `proj.plain-geo`，保持 HK 1980 Grid 原始坐标。
    4. **映射**: `generate_sumo_stops_irn.py` 使用 `pyproj` (WGS84->HK1980) 将 106 个站点成功吸附 (100% 成功率)。
- **验证结果**:
    - **连通性**: WCC 占比 99.2% (物理连通)；Route 68X/960 连通性检查仅剩 ~9 个断点 (主要在隧道口/私家路汇入)，仿真可容错。
    - **仿真**: `hk_irn.sumocfg` 成功运行，车辆按真实站点停靠，无严重死锁。
- **结论**:
    - 路网重建完成，质量远超 Week 1 OSM 版本。
    - 此版本路网 (`hk_irn.net.xml`) 已准备好用于 L1 微观校准。

## 2025-12-19 — Week 2 Experiment 1: Network Reconstruction with Signals
- **线路/方向**: KMB 68X / 960 (Inbound/Outbound)
- **目标**: 在重建的 IRN 路网上启用信号灯，验证其对仿真行程时间的影响。
- **配置变更**:
    - **Base**: `hk_irn.net.xml` (Based on IRN High Precision Network).
    - **Variation**: 
        - V1 (No Signals): `netconvert` config without TLS guessing.
        - V2 (With Signals): Included `<tls.guess value="true"/>` and `<tls.join value="true"/>`.
- **验证结果 (End-to-End Travel Time Error)**:
    - **Real World**: ~8962s (149 min).
    - **Week 1 (OSM)**: 2256s (Error: **-74.8%**).
    - **Week 2 (IRN No TLS)**: 2857s (Error: **-68.1%**).
    - **Week 2 (IRN With TLS)**: 3394s (Error: **-62.1%**).
- **观察**:
    - 引入信号灯增加了约 9 分钟 (540s) 的行程延误，误差收敛了 6 个百分点。
    - 依然存在严重偏差 (-62%)，说明单纯依靠静态路网和信号灯无法复现真实路况。
    - **新问题**: 启用信号灯后，Teleport (Wrong Lane) 警告增加，表明自动猜测的红绿灯连接逻辑在某些复杂路口（如转向限制）可能存在冲突。
- **下一步**:
    - 必须引入 **Micro-Calibration**：调整 `speedFactor`，增加 `dwell time` (停靠时间)，并注入背景交通流 (Background Traffic)。

## 2025-12-19 — Week 2 Data Collection (Timer Triggered)
- **线路/方向/时间窗**: KMB 68X + 960 (Inbound/Outbound), 17:00-18:00 (Timer Triggered).
- **操作**:
    - 修改 `hk_collect.py` 增加定时启动逻辑 (Target: 17:00 UTC+8)。
    - 执行采集指令，程序成功在 16:5x 启动并等待，于 17:00 自动开始。
    - 采集完成后使用 `organize_raw_files.py` 整理了约 6734 个文件。
- **运行命令**: `python scripts/hk_collect.py --routes 68X,960 --service-type 1 --duration 3600 --interval 60`
- **输出**:
    - `data/raw/` 下各分类文件夹 (kmb, irn, etc.) 中的 17:00-18:00 时段数据。
- **结论**: 定时采集功能验证成功，数据已归档，可用于 Week 2 Micro-Calibration。

## 2025-12-19 — Week 2 Experiment 2: L1 Validation with Calibrated Parameters
- **线路/方向**: KMB 68X / 960 (Inbound/Outbound)
- **目标**: 验证引入背景交通流 (Background Traffic) 和 真实动态停站时间 (Dwell Time) 后，能否解决“仿真过快”的问题。
- **实验设置**:
    - **Network**: `hk_irn.net.xml` (Week 2 Reconstructed).
    - **Bus Route**: `baseline_irn_calibrated.rou.xml`
        - **Dwell Times**: LogNormal 分布 (Mean 117.7s, Std 67.2s)，基于 17:00 ETA 数据推导并动态注入。
    - **Background Traffic**: `background.rou.xml`
        - **Data Source**: Smart Lamppost Detectors (17:00 晚高峰真实流量)。
        - **Injection**: 约 138 万辆/小时 (折合当量) 注入到有效路网 Edge。
- **验证结果 (Speed & L1 Error)**:
    - **Real World Speed**: 12.42 km/h
    - **Week 1 (Baseline)**: ~50 km/h
    - **Week 2 (Exp 2)**: **4.85 km/h**
    - **L1 Error (Stop 1->2)**: Real 339.8s vs Sim 457.0s (+34%)。
- **结论**:
    - **过度修正 (Over-Correction)**: 仿真车速从极快 (~50km/h) 骤降至极慢 (4.85km/h)。证明背景流和停站模型是调节这一指标的强力手段。
    - **方向正确**: 成功逆转了“仿真跑太快”的根本性难题。现在的挑战从“由于缺乏阻力而过快”转变为“由于阻力过大或路网瓶颈而过慢/拥堵”。
    - **新发现 (Bottleneck)**: 仿真中出现大量 `No connection` 和 `Teleport` 警告，导致车辆被卡住或传送。这表明 **Week 3 Network Reconstruction (Connectivity Repair)** 将是下一步的关键，只有修复路网连通性，才能让这巨大的车流顺畅运行，从而获得更准确的微观速度数据。

## 2025-12-20 — Week 2 Experiment 2.1: Network Connectivity Repair
- **线路/方向**: KMB 68X / 960 (Inbound/Outbound)
- **目标**: 彻底修复 `duarouter` 无法生成有效路径的问题，消除仿真中的 "Teleport" 警告，确保所有公交线路在 IRN 路网上物理连通。
- **问题分析**:
    - 初始 `duarouter` 生成的路由文件几乎为空 (<1KB)，大量报错 `No connection between edge...`。
    - 原因：IRN 路网在立交桥、隧道口、复杂路口处存在物理断裂（GDB 拓扑与 SUMO 生成逻辑不完全匹配），导致长距离公交线路无法寻路。
- **修复方案 (Iterative Repair Strategy)**:
    - **工具链开发**:
        - `debug_connection.py`: 自动分析断点坐标与前后端节点 ID。
        - `hk_irn_patch.edg.xml`: 专门定义的 "Bridge Edges"（桥接边）文件，用于填补物理空缺。
        - `append_bridge_connections_v2.py`: 动态注入显式连接（Connections），确保新旧路网融合。
        - `check_bridge_connections.py`: 自动化回归测试脚本，验证所有手动修复点的连通性。
    - **执行过程 (Phases 1-8)**:
        - 采用“报错-定位-修复-验证”的循环模式，共执行 8 轮修复。
        - 累计修复了 **33 个关键断点**（涉及 68X/960 双向主干道及高速路段）。
        - 解决了节点 ID 变更（Cluster ID vs Raw ID）导致的 `netconvert` 插入失败问题。
- **验证结果**:
    - **连通性检查**: `check_bridge_connections.py` 显示 **33/33 [OK]**，所有补丁均成功注入。
    - **DuARouter**:
        - **Exit Code**: 0 (Success).
        - **Output File**: `sumo/routes/baseline_irn_duarouter.rou.xml` 大小从 <1KB 跃升至 **~99KB**。
        - **Route Coverage**: 成功生成了 4 条流（68X In/Out, 960 In/Out）的完整边缘序列。
- **结论**:
    - 路网连通性修复工作宣告完成。
    - 当前路网 `hk_irn_v2.net.xml` 配合修正后的路由文件已具备进行无 Teleport 仿真的基础。
    - 下一步可直接进行 Experiment 2.2（L1 参数校准与轨迹对比）。

## 2025-12-20 — Experiment 2.2: Simulation Calibration & L1 Validation
- **线路/方向**: KMB 68X / 960 (Inbound)
- **目标**: 解决所有仿真警告（Bus stop too short, Not downstream），生成有效的轨迹对比图，并量化仿真与现实的 L1 差距。
- **修复方案**:
    - **站台修复**: 运行 `fix_bus_stops_v2.py`，将所有站台强制延伸至 20m，并针对“车道过短”的站点实施智能重定位（Relocation）或硬编码车道分配。
    - **连通性修复**: 针对最后 3 个顽固的 "Not downstream" 错误，在 `sumo/net/fix_connections.con.xml` 中手动定义了显式连接，并生成 `hk_irn_v3.net.xml`。
    - **路由清理**: 使用 `cleanup_routes.py` 移除了已废弃站点的路由引用。
- **数据恢复**:
    - 在 `stopinfo_exp2.xml` 意外覆盖后，执行了恢复性仿真（`sumo_run_restore_data.log`），成功获取完整仿真数据。
- **验证结果 (L1 Metric)**:
    - **68X Inbound (元朗 -> 旺角)**:
        - 截断位置：长沙湾 (Cheung Sha Wan) [Stop ~13]。
        - 耗时对比：仿真 1963.7s vs 现实 4828.8s (Error: -59.3%)。
        - 结论：仿真在市区段畅通无阻，远快于现实。
    - **960 Inbound (湾仔 -> 屯门)**:
        - 截断位置：西营盘/皇后街 (Queen St) [Stop ~8]。
        - 耗时对比：仿真 998.0s vs 现实 2192.4s (Error: -54.5%)。
        - 结论：仿真在港岛北岸畅通，且未进入西隧。
- **关键发现**:
    1. **仿真时长瓶颈**: 3600s 的总时长不足以让长途线路（如 68X, 960）跑完全程，导致轨迹在中途截断。
    2. **速度差异**: 缺少背景交通流及路口拥堵，导致仿真车速约为现实的 2 倍。
    3. **地理验证**: 确认车辆确实在正确的物理路径上行驶（如 960 从湾仔沿海边行驶），路网几何正确。
- **下一步计划 (Strategy Pivot)**:
    - **路网裁剪 (Network Cropping)**: 放弃延长仿真时间跑完全程的计划。依据本次仿真确定的终点坐标（即 1小时行车范围），定义新的 Bounding Box：
        - **68X 边界**: 长沙湾 (Lat ~22.337, Long ~114.148)。
        - **960 边界**: 西营盘 (Lat ~22.288, Long ~114.147)。
    - **行动**: 使用上述坐标确定极值矩形框，进一步大幅减小路网区域。这将显著降低计算负载，专注于市区高密度路段的参数校准。
    - **背景流**: 在缩小后的路网中引入背景交通流。

## 2025-12-20 — Week 2 Experiment 2.3: Network Cropping & Robust Trajectory Verification
- **线路/方向**: KMB 68X / 960 (Inbound)
- **目标**: 解决仿真运行缓慢问题，通过裁剪路网至市区核心（长沙湾-西营盘），建立快速迭代的实验环境，并确立第三周校准的基线。
- **操作与变更**:
    - **路网裁剪**: 使用 `netconvert --keep-edges.in-boundary` 将路网文件从 ~500MB 大幅缩减至 **38MB**。
    - **路由清洗**: 开发 `crop_network_v2.py`，替代不兼容的 `duarouter`，自动过滤无效站点（保留 42 个）、修剪 24 条巴士路线及背景流。
    - **仿真运行**: `experiment2_cropped.sumocfg`，时长延长至 **3600s** (1小时)。
    - **轨迹分析**: 开发 `compare_week2_robust.py`，解决轨迹图中间断裂问题，并针对真实数据进行了截断处理（只对比仿真覆盖的里程）。
- **验证结果 (L1 Metric)**:
    - **仿真性能**: 实时因子达到 **~12x** (1小时仿真耗时约 5分钟)，吞吐量正常。
    - **68X Inbound (核心拥堵区: 旺角->美孚, 5.1km)**:
        - 仿真耗时 3285s vs 现实 5103s。
        - 误差 **-35.6%** (仿真比现实快，缺失拥堵)。
    - **960 Inbound (过海段: 湾仔->西隧, 4.0km)**:
        - 仿真耗时 2378s vs 现实 2225s。
        - 误差 **+6.9%** (仿真略慢，与现实吻合度高)。
- **结论**:
    1. **策略成功**: 裁剪路网既保留了关键实验路段，又极大提升了运行效率。
    2. **差异化基线**: 发现拥堵模型具有区域性差异（九龙拥堵严重被低估，港岛/过海段较准）。
    3. **下一步 (Week 3 Calibration)**: 针对 68X 路段，通过增加背景流密度或调整驾驶模型参数（`minGap`, `accel`, `sigma`）来“制造”拥堵，目标是将误差从 -35% 收敛至 ±10% 以内；对于 960，保持现状或微调。

## 2025-12-20 — Week 2 Experiment 2.4: Evaluation System Refinement & Universal 5KM Alignment
- **线路/方向**: KMB 68X / 960 (Inbound)
- **目标**: 解决可视化层面的数据不对齐、颜色过淡及逻辑错误（瞬移现象），建立标准化的 5KM 研究区域评估体系。
- **核心逻辑改进 (Universal 5KM Alignment)**:
    - **逻辑源头**: 借鉴并深化了 `compare_week2_robust.py` 的序列对齐思想，将其工程化到全套评估脚本中。
    - **共同起点 ($D_{start}$)**: 自动探测仿真车出现的第一个物理站点序列，强制 Real World 数据在同一位置切除冗余起点。
    - **精确归零 (Temporal Re-zeroing)**: 
        - 对于仿真：每班次在 $D_{start}$ 站点的到站时间为 $T=0$。
        - 对于现实：通过线性插值计算车辆到达 $D_{start}$ 位置的时刻 $T_{real\_offset}$，并以此作为基准归零。
        - 消除“瞬移”：彻底放弃了“物理起点 0m 到仿真起点 190m”的补全线，图表 X 轴从实际对齐点开始展示。
    - **同步终点**: 自动识别仿真数据的最大里程 $D_{sim\_max}$，并同步裁剪真实数据，解决“蓝线（实测）太长”导致的比例失调。
- **可视化升级**:
    - **Dark Colormaps**: 为 Space-Time Diagram 引入带阈值的 `DarkBlues` 和 `DarkOranges` 色阶（起始亮度 0.4），彻底解决低速点不可见的问题。
    - **全场景应用**: 该对齐与裁剪逻辑已同步应用至轨迹图、时空图、箱线图、车头时距图。
- **运行命令**:
    - 一键评估：`python scripts/evaluate_baseline.py`
    - 该脚本会自动调用 `plot_trajectory.py`, `plot_spacetime.py`, `plot_link_boxplot.py`, `plot_headway.py`。
- **产出文件**:
    - 评估报告：`docs/baseline_evaluation_report.md`
    - 对比图表：`plots/trajectory_{ROUTE}_baseline.png`, `plots/spacetime_{ROUTE}_baseline.png` 等。
- **结论**:
    - **数据闭环**: 实现了仿真与现实在微观时序上的“同起同终”，斜率对比（速度）变得极度直观。
    - **基准统一**: 所有的后续校准（Week 3）将严格基于这套评估脚本。只要曲线在图中拟合，即代表校准成功。
    - **稳定性**: 修复了 `mcolors`, `numpy`, `real_links` 等多处 `NameError`，确保自动化脚本可稳健运行。

---

## Week 3 Experiment 3.0: L1 微观参数定义与空间构建 (2025-12-20)
- **目标**: 筛选 L1 层核心校准参数，定义物理搜索空间，并生成初始样本集以启动贝叶斯优化。
- **参数筛选 ($\theta_{L1}$)**:
    - `t_board`: 单乘客上车耗时 [0.5, 5.0] s (核心停站参数)。
    - `tau`: 驾驶员反应时间 [0.1, 2.0] s (Krauss 模型)。
    - `sigma`: 驾驶不完美度 [0.1, 0.8] (随机减速强度)。
    - `minGap`: 最小跟车间距 [0.1, 5.0] m (排队密度)。
    - `accel`: 最大加速度 [0.5, 3.0] m/s² (动力性能)。
    - `decel`: 最大减速度 [1.0, 5.0] m/s² (制动性能)。
- **操作与产出**:
    - **配置文件**: [l1_parameter_config.json](file:///d:/Documents/Bus%20Project/Sorce%20code/config/calibration/l1_parameter_config.json)
    - **采样脚本**: [generate_l1_samples.py](file:///d:/Documents/Bus%20Project/Sorce%20code/scripts/calibration/generate_l1_samples.py)
    - **初始样本**: 使用拉丁超立方采样 (LHS) 生成 20 组初始参数，保存至 [l1_initial_samples.csv](file:///d:/Documents/Bus%20Project/Sorce%20code/data/calibration/l1_initial_samples.csv)。
- **基线快照**: 记录了 Experiment 2.3 的全部默认参数至 [baseline_parameters.json](file:///d:/Documents/Bus%20Project/Sorce%20code/config/calibration/baseline_parameters.json)，作为后续优化的原始对照。
- **结论**: 确立了 6 维校准空间，初始样本分布均匀，为后续 MVC 代理模型训练准备好了冷启动数据。

---

## Week 3 Experiment 3.1: L1 微观参数校准闭环验证 (2025-12-20)

### 实验目标
实现并验证 L1 层（微观驾驶行为与停站）的校准闭环。确保从“参数采样 -> XML 注入 -> SUMO 仿真 -> RMSE 计算 -> 代理模型更新”的全流程自动化，并在真实背景流量环境下运行。

### 关键修复点 (Lessons Learned)
1. **仿真环境真实性修复**：
    - **问题**：初始运行仅需 1 秒，仿真处于“真空”状态。
    - **修复**：在 `run_simulation` 命令中重新加入了 `background_cropped.rou.xml`。
    - **结论**：虽然仿真时长从 <1s 增加到约 230s (RTF ~15)，但车辆总数从 24 辆恢复到 13 万辆，RMSE 计算才具有物理意义。
2. **参数冲突处理**：
    - **问题**：`vType` 定义在路由文件和追加文件中冲突。
    - **修复**：取消独立的 `vType` 追加文件，改为直接解析并修改 `.rou.xml` 模板中的 `vType` 属性。
3. **路由模板对齐**：
    - **修复**：将基础模板从 `baseline_irn_duarouter.rou.xml` 切换为专门为裁剪路网修复过的 `fixed_routes_cropped.rou.xml`。

### 实验数据记录
- **仿真配置**: `hk_cropped.net.xml`, `background_cropped.rou.xml`
- **采样策略**: 20 次初始 LHS + N 次 BO (本次验证运行了 2 次 BO)
- **初步指标**:
    - **Iteration 1**: RMSE = 174.2s (RTF 15.6)
    - **Iteration 2**: RMSE = 149.0s (RTF 18.1)

### 阶段性结论
L1 最小闭环已完全跑通。加入背景流后的 RMSE (150s 左右) 真实反映了 68X 路线在晚高峰期间的运行偏差。框架已准备好进行大规模参数优化搜索。

### 基础设施更新 (Infrastructure Updates)
- **日志系统增强**: 为了支持并行实验和历史回溯，我们将 `run_calibration_l1_loop.py` 的日志输出改为带时间戳格式 (`l1_calibration_log_YYYYMMDD_HHMMSS.csv`)。每次运行均生成独立文件，不再覆盖旧数据。同时更新了绘图脚本以自动识别最新日志。