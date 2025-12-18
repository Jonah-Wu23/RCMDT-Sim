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
