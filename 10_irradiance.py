#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 10: 辐照度计算 (5m WGS84, Ineichen 晴空模型)
=====================================================
计算到达地面的太阳辐射强度 (W/m²)

基于 Ineichen 晴空模型:
  E_dir = DNI × cos(zenith) × beam_factor
  E_dif = DHI × SVF

beam_factor 考虑建筑遮挡和树冠透射 (TAU_DIR=0.3)

输入:
  - data/svf/{city}/svf_5m_wgs84.tif
  - data/shadows/building/{city}/{date}/build_shadow_{time}.tif
  - data/shadows/canopy/{city}/{date}/canopy_shadow_{time}.tif
  - data/meteorology/extreme_dates.json

输出:
  - data/irradiance/{city}/{date}/irradiance_{time}.tif (2波段: 直射, 散射)
"""

import os
os.environ['CPL_LOG_ERRORS'] = 'OFF'

import re
import json
import time
import warnings
import numpy as np
import pandas as pd
import rasterio
import requests
import pytz
from pathlib import Path
from datetime import datetime
from timezonefinder import TimezoneFinder
from pvlib.solarposition import get_solarposition
from pvlib.atmosphere import kasten96_lt, alt2pres, get_relative_airmass, get_absolute_airmass
from pvlib.clearsky import ineichen

warnings.filterwarnings("ignore")

# ============ 路径配置 ============
FULL_DIR = Path(__file__).parent
SVF_DIR = FULL_DIR / "data" / "svf"
BUILD_SHADOW_DIR = FULL_DIR / "data" / "shadows" / "building"
CANOPY_SHADOW_DIR = FULL_DIR / "data" / "shadows" / "canopy"
DSM_DIR = FULL_DIR / "data" / "dsm"
# 默认外置盘; 可用环境变量 IRRADIANCE_OUT 覆盖(如外置盘NTFS损坏时改到本地)
OUTPUT_DIR = Path(os.environ.get("IRRADIANCE_OUT", "/media/ps/Elements SE/irradiance"))
EXTREME_JSON = FULL_DIR / "data" / "meteorology" / "extreme_dates.json"
# 浑浊度缓存始终用外置盘那份(复用已积累的3996条, 避免重查NASA POWER)
CACHE_FILE = Path("/media/ps/Elements SE/irradiance/turbidity_cache.json")

TAU_DIR = 0.30
START_HOUR, END_HOUR = 8, 18
TILE_SIZE = 4000  # tiles for very large cities
NASA_POWER = "https://power.larc.nasa.gov/api/temporal/daily/point"

_tf = TimezoneFinder()


def normalize(s):
    s = str(s).lower().strip()
    s = re.sub(r'[^a-z0-9]', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s


def load_extreme_dates():
    with open(EXTREME_JSON) as f:
        data = json.load(f)
    city_dates = {}
    for entry in data:
        top5 = entry.get('top5_combined_extreme', [])
        if not top5:
            continue
        city_dates[entry['city']] = {
            'dates': [d['date'] for d in top5],
            'center': (entry['center']['lat'], entry['center']['lon']),
            'folder': normalize(entry['city']),
        }
    return city_dates


def _load_turbidity_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def _save_turbidity_cache(cache):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)


def _get_timezone(lat, lon):
    tz_name = _tf.timezone_at(lat=lat, lng=lon)
    if tz_name:
        return pytz.timezone(tz_name)
    utc_offset = int(round(lon / 15))
    return pytz.FixedOffset(utc_offset * 60)


def fetch_aod_pwv(lat, lon, date_str):
    params = {
        'latitude': lat, 'longitude': lon,
        'start': date_str, 'end': date_str,
        'parameters': 'AOD_55,TQV',
        'format': 'JSON', 'community': 'RE',
    }
    try:
        # 强制直连, 绕过本地代理(127.0.0.1:7890)瞬时抖动导致的 ProxyError
        r = requests.get(NASA_POWER, params=params, timeout=30, proxies={'http': None, 'https': None})
        data = r.json()['properties']['parameter']
        return {
            'aod550': data['AOD_55'].get(date_str, 0.15) or 0.15,
            'pwvap': data['TQV'].get(date_str, 30.0) or 30.0,
        }
    except Exception as e:
        print(f"    WARN: NASA POWER fallback for ({lat:.1f},{lon:.1f}) {date_str}: {e}")
        return {'aod550': 0.15, 'pwvap': 30.0}


def compute_linke_turbidity(lat, lon, altitude, date_compact, cache):
    key = f"{lat:.3f}_{lon:.3f}_{date_compact}"
    if key in cache:
        return cache[key]

    atm = fetch_aod_pwv(lat, lon, date_compact)
    pressure = alt2pres(altitude)

    # 太阳正午：用 pvlib 找当天最小天顶角时刻
    tz = _get_timezone(lat, lon)
    local_midnight = tz.localize(datetime.strptime(date_compact, "%Y%m%d"), is_dst=False)
    utc_midnight = local_midnight.astimezone(pytz.UTC)
    # 在 UTC 00:00-24:00 范围内搜索太阳正午
    ts_scan = pd.date_range(utc_midnight, utc_midnight + pd.Timedelta(hours=24), freq='15min')
    sol_scan = get_solarposition(ts_scan, lat, lon)
    solar_noon_idx = sol_scan['apparent_zenith'].idxmin()
    ts = pd.DatetimeIndex([solar_noon_idx])
    sol = get_solarposition(ts, lat, lon)
    zen = sol['apparent_zenith'].iloc[0]

    if zen >= 85:
        cache[key] = 3.0
        return 3.0

    am_rel = get_relative_airmass(zen)
    am_abs = get_absolute_airmass(am_rel, pressure)

    if not np.isfinite(am_abs):
        cache[key] = 3.0
        return 3.0

    tl = kasten96_lt(am_abs, atm['pwvap'], atm['aod550'])
    try:
        tl_val = float(tl.iloc[0]) if hasattr(tl, 'iloc') else float(tl)
    except Exception:
        tl_val = 3.0

    tl_val = max(1.0, min(tl_val, 10.0))
    cache[key] = tl_val
    return tl_val


def compute_clearsky(lat, lon, utc_timestamps, turbidity, altitude):
    ts = pd.DatetimeIndex(utc_timestamps)
    sol = get_solarposition(ts, lat, lon)
    zen = sol['apparent_zenith']

    pressure = alt2pres(altitude)
    am_rel = get_relative_airmass(zen)
    am_abs = get_absolute_airmass(am_rel, pressure)

    cs = ineichen(apparent_zenith=zen, airmass_absolute=am_abs,
                  linke_turbidity=turbidity, altitude=altitude)

    dni = cs['dni'].values.copy()
    dhi = cs['dhi'].values.copy()
    dni[zen.values >= 90] = 0
    dhi[zen.values >= 90] = 0

    return dni, dhi, zen.values


def process_city_date(folder, date_str, lat, lon, svf_gpu, svf_np, profile, altitude, cache):
    """处理单个城市单个日期"""
    import cupy as cp

    date_compact = date_str.replace("-", "")

    bld_dir = BUILD_SHADOW_DIR / folder / date_compact
    can_dir = CANOPY_SHADOW_DIR / folder / date_compact

    if not bld_dir.exists() and not can_dir.exists():
        return {'computed': 0, 'cached': 0, 'skip': True}

    out_dir = OUTPUT_DIR / folder / date_compact
    out_dir.mkdir(parents=True, exist_ok=True)

    # Linke turbidity (带缓存)
    turbidity = compute_linke_turbidity(lat, lon, altitude, date_compact, cache)

    # 本地 8-18 时 → UTC timestamps
    tz = _get_timezone(lat, lon)
    utc_timestamps = []
    hours = list(range(START_HOUR, END_HOUR + 1))
    for h in hours:
        dt_local = tz.localize(datetime.strptime(f"{date_str} {h:02d}:00", "%Y-%m-%d %H:%M"), is_dst=False)
        utc_timestamps.append(dt_local.astimezone(pytz.UTC))

    # Clearsky
    dni_arr, dhi_arr, zen_arr = compute_clearsky(lat, lon, utc_timestamps, turbidity, altitude)

    computed, cached = 0, 0
    if svf_gpu is not None:
        rows, cols = svf_gpu.shape
    else:
        rows, cols = svf_np.shape

    for idx, h in enumerate(hours):
        time_str = f"{date_compact}T{h:02d}00"
        out_file = out_dir / f"irradiance_{time_str}.tif"

        if out_file.exists():
            cached += 1
            continue

        zen = zen_arr[idx]
        if zen >= 90:
            continue

        # 读取阴影
        bld_file = bld_dir / f"build_shadow_{time_str}.tif"
        can_file = can_dir / f"canopy_shadow_{time_str}.tif"

        bld_shadow = np.zeros((rows, cols), dtype=bool)
        can_shadow = np.zeros((rows, cols), dtype=bool)

        has_data = False
        if bld_file.exists():
            with rasterio.open(bld_file) as src:
                bld_shadow = src.read(1) > 0
                has_data = True
        if can_file.exists():
            with rasterio.open(can_file) as src:
                can_shadow = src.read(1) > 0
                has_data = True

        if not has_data:
            continue

        cosz = float(np.cos(np.deg2rad(zen)))

        if svf_gpu is not None:
            # GPU path
            bld_gpu = cp.asarray(bld_shadow)
            can_gpu = cp.asarray(can_shadow)
            beam = (~bld_gpu).astype(cp.float32) * (1 - (1 - TAU_DIR) * can_gpu.astype(cp.float32))
            del bld_gpu, can_gpu
            E_dir = float(dni_arr[idx]) * cosz * beam
            E_dif = float(dhi_arr[idx]) * svf_gpu
            del beam
            out_np = cp.asnumpy(cp.stack([E_dir, E_dif])).astype(np.float32)
            del E_dir, E_dif
            cp.get_default_memory_pool().free_all_blocks()
        else:
            # CPU path for large cities
            beam = (~bld_shadow).astype(np.float32) * (1 - (1 - TAU_DIR) * can_shadow.astype(np.float32))
            E_dir = float(dni_arr[idx]) * cosz * beam
            E_dif = float(dhi_arr[idx]) * svf_np
            out_np = np.stack([E_dir, E_dif]).astype(np.float32)
            del beam, E_dir, E_dif

        # 保存 (float16 + DEFLATE: 6.6x smaller than float32 + LZW)
        prof = profile.copy()
        prof.update(count=2, dtype='float16', compress='deflate', zlevel=6)
        with rasterio.open(out_file, 'w', **prof) as dst:
            dst.write(out_np.astype(np.float16))

        del out_np
        computed += 1

    return {'computed': computed, 'cached': cached, 'skip': False}


def process_city_date_tiled(folder, date_str, lat, lon, svf_file, profile, altitude, cache):
    """Tile-based processing for very large cities (CPU only, constant memory)"""
    date_compact = date_str.replace("-", "")

    bld_dir = BUILD_SHADOW_DIR / folder / date_compact
    can_dir = CANOPY_SHADOW_DIR / folder / date_compact

    if not bld_dir.exists() and not can_dir.exists():
        return {'computed': 0, 'cached': 0, 'skip': True}

    out_dir = OUTPUT_DIR / folder / date_compact
    out_dir.mkdir(parents=True, exist_ok=True)

    turbidity = compute_linke_turbidity(lat, lon, altitude, date_compact, cache)

    tz = _get_timezone(lat, lon)
    utc_timestamps = []
    hours = list(range(START_HOUR, END_HOUR + 1))
    for h in hours:
        dt_local = tz.localize(datetime.strptime(f"{date_str} {h:02d}:00", "%Y-%m-%d %H:%M"), is_dst=False)
        utc_timestamps.append(dt_local.astimezone(pytz.UTC))

    dni_arr, dhi_arr, zen_arr = compute_clearsky(lat, lon, utc_timestamps, turbidity, altitude)

    with rasterio.open(svf_file) as svf_src:
        rows, cols = svf_src.height, svf_src.width

    computed, cached = 0, 0

    for idx, h in enumerate(hours):
        time_str = f"{date_compact}T{h:02d}00"
        out_file = out_dir / f"irradiance_{time_str}.tif"

        if out_file.exists():
            cached += 1
            continue

        zen = zen_arr[idx]
        if zen >= 90:
            continue

        bld_file = bld_dir / f"build_shadow_{time_str}.tif"
        can_file = can_dir / f"canopy_shadow_{time_str}.tif"

        bld_exists = bld_file.exists()
        can_exists = can_file.exists()
        if not bld_exists and not can_exists:
            continue

        cosz = float(np.cos(np.deg2rad(zen)))
        dni_val = float(dni_arr[idx])
        dhi_val = float(dhi_arr[idx])

        # Create output file
        prof = profile.copy()
        prof.update(count=2, dtype='float16', compress='deflate', zlevel=6,
                    width=cols, height=rows)

        with rasterio.open(out_file, 'w', **prof) as dst, \
             rasterio.open(svf_file) as svf_src:

            # Optionally open shadow sources
            bld_src = rasterio.open(bld_file) if bld_exists else None
            can_src = rasterio.open(can_file) if can_exists else None

            for row_off in range(0, rows, TILE_SIZE):
                for col_off in range(0, cols, TILE_SIZE):
                    win_rows = min(TILE_SIZE, rows - row_off)
                    win_cols = min(TILE_SIZE, cols - col_off)
                    window = rasterio.windows.Window(col_off, row_off, win_cols, win_rows)

                    # Read SVF tile
                    svf_tile = svf_src.read(1, window=window).astype(np.float32)

                    # Read shadow tiles
                    bld_tile = np.zeros((win_rows, win_cols), dtype=bool)
                    can_tile = np.zeros((win_rows, win_cols), dtype=bool)
                    if bld_src:
                        bld_tile = bld_src.read(1, window=window) > 0
                    if can_src:
                        can_tile = can_src.read(1, window=window) > 0

                    beam = (~bld_tile).astype(np.float32) * (1 - (1 - TAU_DIR) * can_tile.astype(np.float32))
                    E_dir = dni_val * cosz * beam
                    E_dif = dhi_val * svf_tile

                    out_tile = np.stack([E_dir, E_dif]).astype(np.float16)
                    dst.write(out_tile, window=window)
                    del svf_tile, bld_tile, can_tile, beam, E_dir, E_dif, out_tile

            if bld_src:
                bld_src.close()
            if can_src:
                can_src.close()

        computed += 1

    return {'computed': computed, 'cached': cached, 'skip': False}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Step 10: 辐照度计算")
    parser.add_argument("--cities", "-c", nargs='+', help="指定城市")
    parser.add_argument("--limit", "-l", type=int, default=0)
    args = parser.parse_args()

    import cupy as cp
    _test = cp.array([1.0])
    _ = _test.sum()

    print("=" * 60)
    print("Step 10: 辐照度计算 (Ineichen 晴空模型)")
    print("=" * 60)

    city_dates = load_extreme_dates()
    print(f"极端日期: {len(city_dates)} 城市")

    if args.cities:
        city_dates = {k: v for k, v in city_dates.items() if k in args.cities}
    if args.limit > 0:
        city_dates = dict(list(city_dates.items())[:args.limit])

    print(f"实际处理: {len(city_dates)} 城市")
    print(f"每城市: 5 日期 × {END_HOUR - START_HOUR + 1} 时刻\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    turb_cache = _load_turbidity_cache()
    print(f"浑浊度缓存: {len(turb_cache)} 条")

    total_computed, total_cached, total_fail = 0, 0, 0
    city_count = 0
    t0 = time.time()

    for city_name, info in city_dates.items():
        folder = info['folder']
        lat, lon = info['center']

        # 检查 SVF
        svf_file = SVF_DIR / folder / "svf_5m_wgs84.tif"
        try:
            _svf_ok = svf_file.exists()
        except (OSError, IOError) as e:
            print(f"  [{city_name}] [SKIP-IO] SVF exists(): {e}")
            total_fail += len(info['dates'])
            continue
        if not _svf_ok:
            total_fail += len(info['dates'])
            continue

        # 读取 profile 和网格尺寸
        with rasterio.open(svf_file) as src:
            profile = src.profile
            svf_rows, svf_cols = src.height, src.width
            svf_pixels = svf_rows * svf_cols

        # 全部走 GPU
        try:
            with rasterio.open(svf_file) as src:
                svf_np = src.read(1).astype(np.float32)
        except Exception as e:
            print(f"  [{city_name}] ERROR reading SVF: {e}")
            total_fail += len(info['dates'])
            continue
        svf_gpu = cp.asarray(svf_np)
        svf_np = None
        use_tiled = False
        print(f"  [{city_name}] grid ({svf_rows}x{svf_cols}), using GPU path")

        # 海拔
        dsm_file = DSM_DIR / folder / "dsm_5m_wgs84.tif"
        try:
            _dsm_exists = dsm_file.exists()
        except (OSError, IOError) as e:
            print(f"  [{city_name}] [SKIP-IO] DSM exists(): {e} — 用默认海拔")
            _dsm_exists = False
        if _dsm_exists:
            try:
                with rasterio.open(dsm_file) as src:
                    dsm = src.read(1)
                    valid = dsm[~np.isnan(dsm)]
                    altitude = float(valid.mean()) if len(valid) > 0 else 0
                del dsm
            except (OSError, IOError) as e:
                print(f"  [{city_name}] [SKIP-IO] DSM read: {e} — 用默认海拔")
                altitude = 50.0
        else:
            altitude = 50.0

        for date_str in info['dates']:
            try:
                if use_tiled:
                    result = process_city_date_tiled(folder, date_str, lat, lon, svf_file, profile, altitude, turb_cache)
                else:
                    result = process_city_date(folder, date_str, lat, lon, svf_gpu, svf_np, profile, altitude, turb_cache)
            except (OSError, IOError) as e:
                # 容错: 外置盘NTFS偶发I/O error(损坏文件/目录), 跳过该日期不中断整城
                print(f"    [SKIP-IO] {city_name} {date_str}: {e}")
                total_fail += 1
                continue
            if result.get('skip'):
                total_fail += 1
            else:
                total_computed += result['computed']
                total_cached += result['cached']

        # 每 50 城市保存缓存
        city_count += 1
        if city_count % 50 == 0:
            _save_turbidity_cache(turb_cache)

        if svf_gpu is not None:
            del svf_gpu
        if svf_np is not None:
            del svf_np
        cp.get_default_memory_pool().free_all_blocks()
        print(f"  [{city_name}] done")

    _save_turbidity_cache(turb_cache)
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"完成: computed={total_computed}, cached={total_cached}, fail={total_fail}")
    print(f"耗时: {elapsed:.0f}s | 浑浊度缓存: {len(turb_cache)} 条")
    print(f"输出: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
