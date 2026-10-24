#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 13: 平均辐射温度 (MRT) 计算 — 5m分辨率 (修正版)
=====================================================
基于 ISO 7726 / Höppe (1992) / SOLWEIG 模型修正

修正内容:
  1. 加入人体短波吸收率/长波发射率比 A_SW/EPS_P (0.70/0.97)
  2. 直射辐射: E_dir是水平面直射, 先恢复DNI再用动态F_cyl (SOLWEIG)
  3. 地面长波使用 SOLWEIG 正弦地表温度模型 (替代 T_air)
  4. 视角因子分解: F_sky=0.5*svf, F_ground=0.5, F_wall=0.5*(1-svf)
  5. 反射短波方案C: 地面反射(恒定) + 墙壁反射(利用SVF)

输入:
  - 辐照度:  /media/ps/Elements SE/irradiance/{city}/{date}/irradiance_*.tif (2波段)
  - SVF:     data/svf/{city}/svf_5m_wgs84.tif
  - 反照率:  data/albedo/{city}/albedo_5m.tif
  - 气温:    data/meteorology/{city}/{date}/meteo_8-18.csv
  - 树冠:    data/canopy/{city}/canopy_5m.tif

输出:
  - data/mrt/{city}/{date}/mrt_{timestamp}.tif (°C, int16)
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============ 路径配置 ============
FULL_DIR = Path(__file__).parent
# 敏感性分析可用环境变量覆盖: IRR_DIR / MRT_OUT / TAU_SVF / IRR_NO_LOCAL
IRR_DIR = Path(os.environ.get("IRR_DIR", "/media/ps/Elements SE/irradiance"))
IRR_DIR_LOCAL = FULL_DIR / "data" / "irradiance_local"  # 16城辐照度在本地(外置盘NTFS损坏时备用)
SVF_DIR = FULL_DIR / "data" / "svf"
ALBEDO_DIR = FULL_DIR / "data" / "albedo"
METEO_DIR = FULL_DIR / "data" / "meteorology"
CANOPY_DIR = FULL_DIR / "data" / "canopy"
OUTPUT_DIR = Path(os.environ.get("MRT_OUT", str(FULL_DIR / "data" / "mrt")))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 物理常数
SIGMA = 5.67e-8

# 人体辐射特性 (ISO 7726 / VDI 3787)
A_SW = 0.70       # 人体短波吸收率
EPS_P = 0.97      # 人体长波发射率
# F_cyl 动态计算: 0.28*cos(beta) + 0.06*sin(beta)

# 表面发射率
EPS_SKY = 0.90    # 天空等效发射率
EPS_GROUND = 0.95  # 地面发射率
EPS_WALL = 0.95   # 墙面发射率
ALBEDO_WALL = 0.20  # 墙面反照率 (SOLWEIG默认, UMEP Manual)

# 树冠透过率
TAU_DIF = float(os.environ.get("TAU_SVF", "0.40"))

# SOLWEIG 地表温度参数 (Lindberg et al. 2016)
TGK_IMPERVIOUS = 0.37   # 不透水面温度增益系数 (°C/°太阳高度角)
TSTART_IMPERVIOUS = -3.41  # 基线偏移

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
_log = logging.getLogger("step13_mrt")


def get_available_cities() -> List[str]:
    """返回同时有辐照度 + SVF + 反照率的城市"""
    def _safe_dirs(base):
        out = set()
        if not base.exists():
            return out
        try:
            for d in base.iterdir():
                try:
                    if d.is_dir() and d.name != 'turbidity_cache.json':
                        out.add(d.name)
                except (OSError, IOError):
                    continue
        except (OSError, IOError):
            pass
        return out
    irr_cities = _safe_dirs(IRR_DIR) | _safe_dirs(IRR_DIR_LOCAL)
    svf_cities = _safe_dirs(SVF_DIR)
    albedo_cities = _safe_dirs(ALBEDO_DIR)
    return sorted(irr_cities & svf_cities & albedo_cities)


def read_raster(path: Path, template: Dict, resampling=Resampling.bilinear) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    h, w = template['height'], template['width']
    with rasterio.open(path) as src:
        same_grid = (
            src.crs == template['crs']
            and src.width == w
            and src.height == h
            and src.transform == template['transform']
        )
        if same_grid:
            arr = src.read(1).astype(np.float32)
        else:
            arr = np.full((h, w), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1), destination=arr,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=template['transform'], dst_crs=template['crs'],
                resampling=resampling,
            )
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr


def read_irradiance(path: Path, template: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """读取辐照度 (2波段) 并对齐到模板"""
    if not path.exists():
        return None
    h, w = template['height'], template['width']
    with rasterio.open(path) as src:
        if src.count < 2:
            return None
        same_grid = (
            src.crs == template['crs']
            and src.width == w
            and src.height == h
            and src.transform == template['transform']
        )
        if same_grid:
            e_dir = src.read(1).astype(np.float32)
            e_dif = src.read(2).astype(np.float32)
        else:
            e_dir = np.full((h, w), np.nan, dtype=np.float32)
            e_dif = np.full((h, w), np.nan, dtype=np.float32)
            for band_idx, dst in [(1, e_dir), (2, e_dif)]:
                reproject(
                    source=rasterio.band(src, band_idx), destination=dst,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=template['transform'], dst_crs=template['crs'],
                    resampling=Resampling.bilinear,
                )
        if src.nodata is not None:
            e_dir[e_dir == src.nodata] = np.nan
            e_dif[e_dif == src.nodata] = np.nan
        return e_dir, e_dif


def get_irradiance_files(city: str) -> Dict[str, List[Path]]:
    """获取辐照度文件，按日期分组(默认先外置盘; IRR_LOCAL_FIRST=1时本地优先)"""
    result = {}
    # 外置盘NTFS坏块时, 用IRR_LOCAL_FIRST=1让本地副本优先(避免读到坏tif)
    bases = (IRR_DIR_LOCAL, IRR_DIR) if os.environ.get("IRR_LOCAL_FIRST") == "1" else (IRR_DIR, IRR_DIR_LOCAL)
    if os.environ.get("IRR_NO_LOCAL") == "1":  # 敏感性分析: 只用IRR_DIR, 不回退本地副本
        bases = (IRR_DIR,)
    for base in bases:
        city_dir = base / city
        try:
            if not city_dir.exists():
                continue
            date_dirs = sorted(city_dir.iterdir())
        except (OSError, IOError):
            continue
        for date_dir in date_dirs:
            try:
                if not date_dir.is_dir():
                    continue
                files = sorted(date_dir.glob("irradiance_*.tif"))
            except (OSError, IOError):
                continue
            if files:
                date_str = f"{date_dir.name[:4]}-{date_dir.name[4:6]}-{date_dir.name[6:8]}"
                result.setdefault(date_str, files)
    return result


def parse_timestamp(filename: str) -> Optional[tuple]:
    """从 irradiance_YYYYMMDDTHHMM.tif 解析日期和小时"""
    try:
        stem = Path(filename).stem
        ts = stem.split('_')[1]
        date_part = ts[:8]
        hour = int(ts[9:11])
        date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
        return date_str, hour, ts
    except Exception:
        return None


def load_meteo_day(city: str, date_str: str) -> Dict[int, float]:
    """一次性加载整天气温"""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    for offset in [0, 1, -1]:
        d = dt + timedelta(days=offset)
        d_str = d.strftime('%Y-%m-%d')
        path = METEO_DIR / city / d_str / "meteo_8-18.csv"
        if path.exists():
            try:
                df = pd.read_csv(path)
                result = {}
                for _, row in df.iterrows():
                    h = int(str(row['local_time'])[11:13])
                    result[h] = float(row['temperature_c'])
                return result
            except Exception:
                pass
    return {}


def estimate_solar_elevation(lat: float, date_str: str, hour: int) -> float:
    """估算太阳高度角 (度), 用于动态F_cyl和地表温度"""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    doy = dt.timetuple().tm_yday
    decl = 23.45 * np.sin(np.radians(360 * (284 + doy) / 365))
    lat_rad = np.radians(lat)
    dec_rad = np.radians(decl)
    hour_angle = np.radians(15 * (hour - 12))
    sin_alt = (np.sin(lat_rad) * np.sin(dec_rad)
               + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hour_angle))
    return max(float(np.degrees(np.arcsin(np.clip(sin_alt, -1, 1)))), 0.0)


def compute_F_cyl(solar_elev_deg: float) -> float:
    """动态圆柱投影系数 (SOLWEIG)"""
    beta = np.deg2rad(solar_elev_deg)
    return 0.28 * np.cos(beta) + 0.06 * np.sin(beta)


def estimate_max_solar_elevation(lat: float, date_str: str) -> float:
    """估算当天最大太阳高度角"""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    doy = dt.timetuple().tm_yday
    decl = 23.45 * np.sin(np.radians(360 * (284 + doy) / 365))
    return max(90.0 - abs(lat) + decl, 0.0)


def compute_surface_temp_offset(solar_alt_max: float, hour: int) -> float:
    """SOLWEIG 正弦地表温度模型: T_surf = T_air + offset"""
    if solar_alt_max <= 0:
        return 0.0
    tg_amplitude = TGK_IMPERVIOUS * solar_alt_max + TSTART_IMPERVIOUS
    # 正弦日内变化, 峰值在14:00 (TMAX_LST)
    phase = np.pi * (hour - 6) / 16.0  # 6:00-22:00 映射到 0-pi
    tg_offset = tg_amplitude * max(np.sin(phase), 0.0)
    return max(tg_offset, 0.0)


def get_city_lat(city: str) -> Optional[float]:
    """从SVF栅格获取城市纬度"""
    svf_path = SVF_DIR / city / "svf_5m_wgs84.tif"
    if not svf_path.exists():
        return None
    with rasterio.open(svf_path) as src:
        bounds = src.bounds
        return (bounds.top + bounds.bottom) / 2.0


def compute_mrt_city(city: str, force: bool = False) -> Optional[Dict]:
    """计算单个城市的 MRT"""
    svf_path = SVF_DIR / city / "svf_5m_wgs84.tif"
    if not svf_path.exists():
        _log.warning(f"{city}: SVF 不存在")
        return None

    albedo_path = ALBEDO_DIR / city / "albedo_5m.tif"
    if not albedo_path.exists():
        _log.warning(f"{city}: albedo 不存在")
        return None

    irr_dates = get_irradiance_files(city)
    if not irr_dates:
        _log.warning(f"{city}: 辐照度数据不存在")
        return None

    _log.info(f"{city}: {len(irr_dates)} 个日期, {sum(len(v) for v in irr_dates.values())} 个时次")

    # 城市纬度
    city_lat = get_city_lat(city)
    if city_lat is None:
        _log.warning(f"{city}: 无法获取纬度")
        return None

    # 加载参考模板
    with rasterio.open(svf_path) as src:
        template = {
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height,
        }

    # 加载 SVF
    svf = read_raster(svf_path, template, Resampling.bilinear)
    svf = np.clip(svf, 0.05, 1.0)

    # 加载反照率
    albedo = read_raster(albedo_path, template, Resampling.bilinear)
    if albedo is None:
        _log.warning(f"{city}: 反照率加载失败")
        return None
    albedo = np.clip(albedo, 0.05, 0.5)

    # 加载树冠掩膜
    canopy_path = CANOPY_DIR / city / "canopy_5m.tif"
    canopy = read_raster(canopy_path, template, Resampling.nearest)
    canopy_mask = (canopy > 0).astype(np.float32) if canopy is not None else None

    # 有效 SVF (树冠修正)
    if canopy_mask is not None:
        svf_eff = svf * (1.0 - (1.0 - TAU_DIF) * canopy_mask)
    else:
        svf_eff = svf

    # 视角因子分解 (站立人, 上下半球各0.5)
    F_sky = 0.5 * svf_eff
    F_ground = np.full_like(svf_eff, 0.5)
    F_wall = 0.5 * (1.0 - svf_eff)

    # 输出元数据
    out_meta = {
        'driver': 'GTiff', 'dtype': 'int16', 'width': template['width'],
        'height': template['height'], 'count': 1, 'crs': template['crs'],
        'transform': template['transform'], 'compress': 'lzw', 'nodata': -999,
        'tiled': True, 'blockxsize': 256, 'blockysize': 256,
    }

    total_files = 0
    skip_temp = 0
    meteo_cache = {}

    for date_str, irr_files in irr_dates.items():
        # 每个日期预加载气象数据
        if date_str not in meteo_cache:
            meteo_cache[date_str] = load_meteo_day(city, date_str)

        # 当天最大太阳高度角
        alt_max = estimate_max_solar_elevation(city_lat, date_str)

        for irr_fp in irr_files:
            parsed = parse_timestamp(irr_fp.name)
            if parsed is None:
                continue
            date_str_parsed, hour, ts = parsed

            # 输出路径
            out_dir = OUTPUT_DIR / city / date_str.replace("-", "")
            out_path = out_dir / f"mrt_{ts}.tif"
            if not force and out_path.exists():
                total_files += 1
                continue

            # 获取气温
            Ta_C = meteo_cache.get(date_str_parsed, {}).get(hour)
            if Ta_C is None:
                skip_temp += 1
                Ta_C = 30.0
            T_air_K = Ta_C + 273.15

            # 太阳高度角 (动态计算)
            solar_elev = estimate_solar_elevation(city_lat, date_str_parsed, hour)
            sin_beta = max(np.sin(np.deg2rad(solar_elev)), 0.05)

            # 地表温度偏移 (SOLWEIG 正弦模型)
            tg_offset = compute_surface_temp_offset(alt_max, hour)
            T_surf_K = T_air_K + tg_offset
            T_wall_K = 0.5 * T_air_K + 0.5 * T_surf_K  # 墙面=气温与地表均值

            # 预计算长波标量因子
            lw_sky_factor = EPS_SKY * SIGMA * T_air_K**4
            lw_ground_factor = EPS_GROUND * SIGMA * T_surf_K**4
            lw_wall_factor = EPS_WALL * SIGMA * T_wall_K**4

            # 读取辐照度 (2波段, 对齐到模板)
            # 容错: 坏tif(IFD损坏/IO error)会抛异常, 跳过该时次而不中断整城
            try:
                irr = read_irradiance(irr_fp, template)
            except (rasterio.errors.RasterioIOError, OSError, Exception) as e:
                _log.error(f"{city}: {irr_fp.name} 读取失败,跳过: {e}")
                continue
            if irr is None:
                continue
            E_dir, E_dif = irr

            # 有效像元掩膜
            valid = np.isfinite(E_dir) & np.isfinite(E_dif) & np.isfinite(svf_eff)

            # ---- 短波 ----
            # 直射: E_dir 是水平面直射(DNI×cos(z)×beam), 需先恢复DNI
            DNI = E_dir / sin_beta  # 阳光区: DNI≈E_dir/cos(z); 阴影区: E_dir≈0, 不影响
            F_cyl = compute_F_cyl(solar_elev)
            E_dir_body = F_cyl * DNI
            # 天空散射
            E_dif_sky = F_sky * E_dif
            # 地面反射短波 (恒定, 站立人下半球始终看到地面)
            E_ref_ground = F_ground * albedo * (E_dir + E_dif)
            # 墙壁反射短波 (随SVF变化, 使用墙面反照率0.20)
            E_ref_wall = F_wall * ALBEDO_WALL * E_dif
            # 总短波
            E_sw = E_dir_body + E_dif_sky + E_ref_ground + E_ref_wall

            # ---- 长波 ----
            E_lw = F_sky * lw_sky_factor + F_ground * lw_ground_factor + F_wall * lw_wall_factor

            # ---- MRT (ISO 7726) ----
            E_abs = (A_SW / EPS_P) * E_sw + E_lw
            T_mrt_K = np.sqrt(np.sqrt(np.maximum(E_abs, 0) / SIGMA))
            T_mrt_C = np.clip(np.round(T_mrt_K - 273.15), -40, 200).astype(np.int16)

            # 无效像元设为 nodata
            T_mrt_C[~valid] = -999

            # 保存
            out_dir.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, 'w', **out_meta) as dst:
                dst.write(T_mrt_C, 1)

            total_files += 1

    return {
        'city': city,
        'dates': len(irr_dates),
        'files': total_files,
        'skip_temp': skip_temp,
    }


def _mrt_worker(city_name):
    """多进程worker: 处理单城, 返回结果或None(异常时)。"""
    try:
        r = compute_mrt_city(city_name, force=False)
        if r:
            return (city_name, r)
    except Exception as e:
        return (city_name, {'error': str(e)})
    return None


def main():
    parser = argparse.ArgumentParser(description="Step 13: MRT 计算 (5m, 修正版)")
    parser.add_argument("--cities", "-c", nargs='+', help="指定城市")
    parser.add_argument("--limit", "-l", type=int, default=0)
    parser.add_argument("--force", "-f", action="store_true", help="覆盖已有文件")
    parser.add_argument("--workers", "-w", type=int, default=0, help="并行进程数(0=自动,用一半核)")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 13: MRT 计算 (修正版) — 5m")
    print("=" * 60)
    print(f"公式修正: A_SW={A_SW}, EPS_P={EPS_P}, F_cyl=动态(太阳高度角)")
    print(f"地表温度: SOLWEIG正弦模型")
    print(f"视角因子: F_sky/F_ground/F_wall 分解")
    print()

    cities = get_available_cities()
    print(f"有辐照度+SVF+反照率的城市: {len(cities)}")

    if args.cities:
        cities = [c for c in args.cities if c in cities]
    if args.limit > 0:
        cities = cities[:args.limit]

    print(f"待处理: {len(cities)} 城市")
    # 多进程并行数(可用 --workers 覆盖); 默认用一半核
    n_workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 4) // 2)
    print(f"并行进程数: {n_workers}")
    print()

    results = []
    if n_workers <= 1:
        for city_name in tqdm(cities, desc="计算MRT"):
            try:
                r = compute_mrt_city(city_name, force=args.force)
                if r:
                    results.append(r)
                    _log.info(f"{city_name}: {r['dates']} dates, {r['files']} files, skip_temp={r['skip_temp']}")
            except Exception as e:
                _log.error(f"{city_name}: {e}")
    else:
        from multiprocessing import Pool
        with Pool(n_workers) as pool:
            for item in tqdm(pool.imap_unordered(_mrt_worker, cities), total=len(cities), desc="计算MRT"):
                if item is None:
                    continue
                city_name, r = item
                if 'error' in r:
                    _log.error(f"{city_name}: {r['error']}")
                else:
                    results.append(r)
                    _log.info(f"{city_name}: {r['dates']} dates, {r['files']} files, skip_temp={r['skip_temp']}")

    # 保存摘要
    summary = OUTPUT_DIR / "mrt_summary.json"
    with open(summary, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total_skip_temp = sum(r.get('skip_temp', 0) for r in results)
    print("\n" + "=" * 60)
    print("MRT 计算完成")
    print("=" * 60)
    print(f"  成功: {len(results)} 城市")
    print(f"  气温缺失(使用30°C): {total_skip_temp} 时次")
    print(f"  输出: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
