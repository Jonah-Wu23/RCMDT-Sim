"""
平峰数据收集脚本 (2025/12/30 15:00-16:00)

复用 hk_collect.py 中的核心函数，数据保存到 data2/raw/ 目录。

功能：
- 定时等待到目标时间 15:00
- 收集1小时数据（默认60秒间隔轮询）
- 计时开始、中间计时、结束汇总
- 异常处理防止数据残缺

用法:
  python scripts/hk_collect_offpeak.py --routes 68X,960 --interval 60 --duration 3600
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path
from typing import Iterable, Tuple

import requests

# ========== 配置 ==========
RAW_DIR = Path("data2/raw")
TARGET_TIME = datetime.datetime(2025, 12, 30, 15, 0, 0)
DEFAULT_DURATION = 3600  # 1小时
DEFAULT_INTERVAL = 60    # 60秒轮询间隔

# ========== API URLs（从 hk_collect.py 复制）==========
BASE_KMB = "https://data.etabus.gov.hk/v1/transport/kmb"
URL_IRN_SPEED_DL = (
    "https://res.data.gov.hk/api/get-download-file?"
    "name=https%3A%2F%2Fresource.data.one.gov.hk%2Ftd%2Ftraffic-detectors%2FirnAvgSpeed-all.xml"
)
URL_TSM_RAW_DL = (
    "https://res.data.gov.hk/api/get-download-file?"
    "name=https%3A%2F%2Fresource.data.one.gov.hk%2Ftd%2Ftraffic-detectors%2FrawSpeedVol-all.xml"
)
URL_TSM_LOC = "https://static.data.gov.hk/td/traffic-data-strategic-major-roads/info/traffic_speed_volume_occ_info.csv"
URL_TSM_NOTIFICATION = "https://static.data.gov.hk/td/traffic-speed-map/notification.csv"
URL_STN = "https://www.td.gov.hk/en/special_news/trafficnews.xml"
URL_JTI = "https://resource.data.one.gov.hk/td/jss/Journeytimev2.xml"
URL_HKO = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"


# ========== 工具函数 ==========
def ts() -> str:
    """返回当前时间戳字符串"""
    return time.strftime("%Y%m%d-%H%M%S")


def save_text(name: str, content: str) -> Path:
    """保存文本文件"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def save_bytes(name: str, content: bytes) -> Path:
    """保存二进制文件"""
    path = RAW_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def fetch(url: str, params=None, binary: bool = False) -> Tuple[int, bytes | str]:
    """HTTP GET 请求"""
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.status_code, resp.content if binary else resp.text


def _fetch_route_stop(route: str, bound: str, svc: str, timestamp: str, stop_ids: set):
    """抓 route-stop，同时收集 stop_id 到集合里"""
    status, txt = fetch(f"{BASE_KMB}/route-stop/{route}/{bound}/{svc}")
    save_text(f"kmb/route-stop/kmb-route-stop-{route}-{bound}-{timestamp}.json", txt)
    try:
        data = json.loads(txt)
        if isinstance(data, dict):
            for item in data.get("data", []):
                sid = item.get("stop")
                if sid:
                    stop_ids.add(sid)
    except Exception as e:
        print(f"[warn] parse route-stop {route}-{bound} failed: {e}")


# ========== 核心收集函数 ==========
def poll_once(routes: Iterable[str], svc: str, poll_count: int, start_time: float, end_time: float) -> int:
    """
    按 hk_collect 轮询一次，收集 KMB + 路况数据。
    增加计时信息输出。
    返回成功数和失败数。
    """
    timestamp = ts()
    stop_ids = set()
    success_count = 0
    fail_count = 0
    
    elapsed = time.time() - start_time
    remaining = end_time - time.time()
    
    print(f"\n{'='*60}")
    print(f"[轮询 #{poll_count}] {timestamp}")
    print(f"  已用时间: {elapsed/60:.1f} 分钟")
    print(f"  剩余时间: {remaining/60:.1f} 分钟")
    print(f"  收集路线: {','.join(routes)}")
    print('='*60)

    def safe_call(name: str, fn):
        nonlocal success_count, fail_count
        try:
            fn()
            print(f"  [OK]   {name}")
            success_count += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            fail_count += 1

    # KMB 路线信息
    safe_call(
        "kmb-route",
        lambda: save_text(f"kmb/route/kmb-route-{timestamp}.json", fetch(f"{BASE_KMB}/route")[1]),
    )
    
    # 各路线的 route-stop 和 route-eta
    for route in routes:
        for bound in ("inbound", "outbound"):
            safe_call(
                f"kmb-route-stop-{route}-{bound}",
                lambda r=route, b=bound: _fetch_route_stop(r, b, svc, timestamp, stop_ids),
            )
        safe_call(
            f"kmb-route-eta-{route}",
            lambda r=route: save_text(
                f"kmb/route-eta/kmb-route-eta-{r}-{timestamp}.json", 
                fetch(f"{BASE_KMB}/route-eta/{r}/{svc}")[1]
            ),
        )

    print(f"  [INFO] 收集到站点数: {len(stop_ids)}")

    # 全量站点 stop/stop-eta
    for sid in sorted(stop_ids):
        safe_call(
            f"kmb-stop-{sid}",
            lambda s=sid: save_text(f"kmb/stop/kmb-stop-{s}-{timestamp}.json", fetch(f"{BASE_KMB}/stop/{s}")[1]),
        )
        safe_call(
            f"kmb-stop-eta-{sid}",
            lambda s=sid: save_text(f"kmb/stop-eta/kmb-stop-eta-{s}-{timestamp}.json", fetch(f"{BASE_KMB}/stop-eta/{s}")[1]),
        )

    # TSM 通知
    safe_call(
        "tsm-notification",
        lambda: save_text(f"tsm_notification/tsm-notification-{timestamp}.csv", fetch(URL_TSM_NOTIFICATION)[1]),
    )

    # IRN 速度数据
    safe_call(
        "irnAvgSpeed-all",
        lambda: save_bytes(f"irn_download/irnAvgSpeed-all-{timestamp}.xml", fetch(URL_IRN_SPEED_DL, binary=True)[1]),
    )

    # 原始探测器数据
    safe_call(
        "rawSpeedVol-all",
        lambda: save_bytes(f"detector_locations/rawSpeedVol-all-{timestamp}.xml", fetch(URL_TSM_RAW_DL, binary=True)[1]),
    )

    # 探测器位置信息
    safe_call(
        "tsm-detector-loc",
        lambda: save_text(f"detector_locations/traffic_speed_volume_occ_info-{timestamp}.csv", fetch(URL_TSM_LOC)[1]),
    )

    # 交通新闻
    safe_call("stn", lambda: save_text(f"stn/trafficnews-{timestamp}.xml", fetch(URL_STN)[1]))
    
    # 行程时间指标
    safe_call("jti", lambda: save_text(f"jti/Journeytimev2-{timestamp}.xml", fetch(URL_JTI)[1]))
    
    # 天气数据
    safe_call(
        "hko-rhrread",
        lambda: save_text(
            f"hko/hko-rhrread-{timestamp}.json", 
            fetch(URL_HKO, params={"dataType": "rhrread", "lang": "en"})[1]
        ),
    )

    print(f"\n[轮询 #{poll_count} 完成] 成功: {success_count}, 失败: {fail_count}")
    return success_count


def wait_for_target_time(target: datetime.datetime):
    """等待到目标时间，每10秒打印倒计时"""
    print(f"\n{'#'*60}")
    print(f"# 计时器已启动")
    print(f"# 目标时间: {target}")
    print(f"# 当前时间: {datetime.datetime.now()}")
    print(f"{'#'*60}\n")
    
    while True:
        now = datetime.datetime.now()
        if now >= target:
            print(f"\n[{now}] 目标时间已到达！开始收集数据...\n")
            break
        
        remaining = (target - now).total_seconds()
        if remaining > 60:
            print(f"[倒计时] 距离开始还有 {remaining/60:.1f} 分钟...")
            time.sleep(30)  # 每30秒打印一次
        else:
            print(f"[倒计时] 距离开始还有 {remaining:.0f} 秒...")
            time.sleep(10)  # 最后一分钟每10秒打印


def main():
    ap = argparse.ArgumentParser(description="平峰数据收集器 (2025/12/30 15:00-16:00)")
    ap.add_argument("--routes", default="68X,960", help="逗号分隔的 KMB 路线，默认 68X,960")
    ap.add_argument("--service-type", default="1", help="KMB service_type，默认 1")
    ap.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help=f"轮询间隔（秒），默认 {DEFAULT_INTERVAL}")
    ap.add_argument("--duration", type=int, default=DEFAULT_DURATION, help=f"轮询总时长（秒），默认 {DEFAULT_DURATION}")
    ap.add_argument("--no-wait", action="store_true", help="跳过等待，立即开始收集")
    args = ap.parse_args()

    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    
    print(f"\n{'='*60}")
    print(f"平峰数据收集脚本")
    print(f"  目标时间: {TARGET_TIME}")
    print(f"  收集时长: {args.duration} 秒 ({args.duration/60:.1f} 分钟)")
    print(f"  轮询间隔: {args.interval} 秒")
    print(f"  收集路线: {routes}")
    print(f"  数据目录: {RAW_DIR.absolute()}")
    print(f"{'='*60}\n")

    # 等待到目标时间
    if not args.no_wait:
        wait_for_target_time(TARGET_TIME)
    else:
        print("[跳过等待] 立即开始收集...")

    # 开始轮询收集
    start_time = time.time()
    end_time = start_time + args.duration
    next_tick = start_time
    poll_count = 0
    total_success = 0

    print(f"\n[收集开始] {datetime.datetime.now()}")
    print(f"  预计结束: {datetime.datetime.now() + datetime.timedelta(seconds=args.duration)}")

    while True:
        now = time.time()
        if now >= end_time:
            break
        
        # 对齐到预期起点
        if now < next_tick:
            sleep_head = min(next_tick - now, end_time - now)
            if sleep_head > 0:
                time.sleep(sleep_head)
        
        poll_count += 1
        loop_start = time.time()
        success = poll_once(routes, args.service_type, poll_count, start_time, end_time)
        total_success += success
        elapsed = time.time() - loop_start
        
        # 保持轮询起点间隔约为 interval 秒
        next_tick += args.interval
        remaining = end_time - time.time()
        sleep_sec = min(max(next_tick - time.time(), 0), remaining)
        
        if sleep_sec <= 0:
            print(f"[INFO] 本轮耗时 {elapsed:.1f}s，间隔已到，立即继续下一轮")
            continue
        print(f"[INFO] 本轮耗时 {elapsed:.1f}s，休眠 {sleep_sec:.1f}s 对齐间隔")
        time.sleep(sleep_sec)

    # 收集结束汇总
    total_elapsed = time.time() - start_time
    print(f"\n{'#'*60}")
    print(f"# 数据收集完成！")
    print(f"# 结束时间: {datetime.datetime.now()}")
    print(f"# 总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"# 总轮询次数: {poll_count}")
    print(f"# 总成功请求数: {total_success}")
    print(f"# 数据保存位置: {RAW_DIR.absolute()}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
