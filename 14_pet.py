#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 14: PET (生理等效温度) 计算 — 5m WGS84 CUDA 加速
=====================================================
从 MRT + 气象数据 (CSV) 计算 PET

PET 公式:
  1. 计算实际环境 PMV(Ta, Tmrt, v, RH)
  2. 二分搜索: 在参考环境 (Tmrt=Ta, v=0.1m/s, pa=12hPa) 下
     达到相同 PMV 的气温即为 PET

输入:
  - MRT:    data/mrt/{city}/{YYYYMMDD}/mrt_{timestamp}.tif (int16, °C)
  - 气象:   data/meteorology/{city}/{date}/meteo_8-18.csv

输出:
  - data/pet/{city}/{YYYYMMDD}/pet_{timestamp}.tif (int8, °C)
"""

import os
import math
import json
import logging
import argparse
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from rasterio.windows import Window
import warnings
warnings.filterwarnings('ignore')

from numba import cuda
if not cuda.is_available():
    raise RuntimeError("CUDA GPU 不可用，本脚本仅支持 GPU 运行")

# ============ 路径配置 ============
# 敏感性分析可用环境变量覆盖: MRT_DIR / PET_OUT
FULL_DIR = Path(__file__).parent
MRT_DIR = Path(os.environ.get("MRT_DIR", str(FULL_DIR / "data" / "mrt")))
METEO_DIR = FULL_DIR / "data" / "meteorology"
OUTPUT_DIR = Path(os.environ.get("PET_OUT", str(FULL_DIR / "data" / "3_5_pet")))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ 物理参数 ============
MET = 3.5
CLO = 0.5
REF_V = 0.1
REF_PA = 1200.0       # 12 hPa
TMIN, TMAX = -20.0, 60.0
ITERS = 7
NODATA_PET = -99
WIND_FLOOR = 0.3

# GPU 参数
CHUNK_SIZE = 8192
N_STREAMS = 4

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
_log = logging.getLogger("step14_pet")


# ======================== CUDA kernels ========================

@cuda.jit(device=True, fastmath=True)
def pmv(tdb, tr, vr, rh, pa_fixed=-1.0):
    """Fanger PMV (ISO 7730)"""
    M = MET * 58.15
    W = 0.0
    Icl = CLO
    Icl_r = 0.155 * Icl

    if Icl_r > 0.078:
        f_cl = 1.05 + 0.645 * Icl_r
    else:
        f_cl = 1.00 + 1.290 * Icl_r

    if pa_fixed > 0:
        pa = pa_fixed
    else:
        pa = (rh * 0.01) * 610.94 * math.exp((17.625 * tdb) / (tdb + 243.04))

    hcf = 12.1 * math.sqrt(vr)
    tcl = tdb + (35.5 - tdb) / (3.5 * Icl + 0.1)

    for _ in range(8):
        hr = 4.0 * 5.67e-8 * f_cl * ((tcl + tr + 546.3) * 0.5) ** 3
        hcn = 2.38 * abs(tcl - tdb) ** 0.25
        hc = hcf if hcf > hcn else hcn
        tcl = (35.7 - 0.028 * M + Icl_r * f_cl * (hr * tr + hc * tdb)) / \
              (1.0 + Icl_r * f_cl * (hr + hc))

    hl1 = 3.05e-3 * (5733.0 - 6.99 * (M - W) - pa)
    hl2 = 0.42 * max(0.0, M - 58.15)
    hl3 = 1.7e-5 * M * (5867.0 - pa)
    hl4 = 0.0014 * M * (34.0 - tdb)
    hl5 = 3.96e-8 * f_cl * ((tcl + 273.15) ** 4 - (tr + 273.15) ** 4)
    hl6 = f_cl * hc * (tcl - tdb)

    ts = 0.303 * math.exp(-0.036 * M) + 0.028
    return ts * (M - (hl1 + hl2 + hl3 + hl4 + hl5 + hl6))


@cuda.jit(fastmath=True)
def pet_kernel(Tr, ta, u, rh, PET, n):
    """PET 二分搜索: ta/u/rh 为标量, Tr 为数组"""
    i = cuda.grid(1)
    if i >= n:
        return

    tr = Tr[i]

    if tr <= -50.0:
        PET[i] = NODATA_PET
        return

    pmv_target = pmv(ta, tr, u, rh, -1.0)

    lo, hi = TMIN, TMAX
    for _ in range(ITERS):
        mid = 0.5 * (lo + hi)
        if pmv(mid, mid, REF_V, 0.0, REF_PA) < pmv_target:
            lo = mid
        else:
            hi = mid

    PET[i] = 0.5 * (lo + hi)


# ======================== 辅助函数 ========================

def get_cities_with_mrt() -> List[str]:
    """获取有 MRT 数据的城市列表"""
    cities = set()
    if not MRT_DIR.exists():
        return []
    for d in MRT_DIR.iterdir():
        if d.is_dir() and any(d.glob('**/*.tif')):
            cities.add(d.name)
    return sorted(cities)


def parse_mrt_timestamp(filename: str) -> Optional[Tuple[str, int, str]]:
    """从 mrt_YYYYMMDDTHHMM.tif 解析日期、小时、时间戳"""
    try:
        stem = Path(filename).stem
        ts = stem.split('_')[1]       # 20230804T0800
        date_part = ts[:8]            # 20230804
        hour = int(ts[9:11])          # 08
        date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
        return date_str, hour, ts
    except Exception:
        return None


def _read_csv_hour(csv_path: Path, hour: int) -> Optional[Tuple[float, float, float]]:
    """从 CSV 文件中查找指定小时的气象数据"""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            ts = str(row['local_time'])
            h_match = int(ts[11:13])
            if h_match == hour:
                ta = float(row['temperature_c'])
                rh = float(row['humidity_pct'])
                ws = float(row['wind_speed_ms'])
                return ta, max(rh, 5.0), max(ws, WIND_FLOOR)
    except Exception:
        pass
    return None


def get_meteo_from_csv(city: str, date_str: str, hour: int) -> Tuple[float, float, float]:
    """从 CSV 读取标量气象值 (Ta, RH, WS)，搜索前后一天"""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    for offset in [0, 1, -1]:
        d = dt + timedelta(days=offset)
        d_str = d.strftime('%Y-%m-%d')
        result = _read_csv_hour(METEO_DIR / city / d_str / "meteo_8-18.csv", hour)
        if result is not None:
            return result
    return 30.0, 50.0, WIND_FLOOR


def get_mrt_files(city: str) -> List[Path]:
    """获取某城市的所有 MRT 文件"""
    mrt_base = MRT_DIR / city
    if not mrt_base.exists():
        return []
    return sorted(mrt_base.glob("**/mrt_*.tif"))


# ======================== GPU 处理 ========================

def compute_pet(city: str, force: bool = False) -> Optional[Dict]:
    """GPU PET 计算 (单城市)"""
    mrt_files = get_mrt_files(city)
    if not mrt_files:
        return None

    with rasterio.open(mrt_files[0]) as src:
        H, W = src.height, src.width
        ref_crs = src.crs
        ref_transform = src.transform

    out_meta = {
        'driver': 'GTiff', 'dtype': 'int8', 'width': W, 'height': H,
        'count': 1, 'crs': ref_crs, 'transform': ref_transform,
        'compress': 'deflate', 'nodata': NODATA_PET,
        'tiled': True, 'blockxsize': 256, 'blockysize': 256,
    }

    windows = [
        Window(c, r, min(CHUNK_SIZE, W - c), min(CHUNK_SIZE, H - r))
        for r in range(0, H, CHUNK_SIZE)
        for c in range(0, W, CHUNK_SIZE)
    ]

    max_n = CHUNK_SIZE * CHUNK_SIZE
    streams = [cuda.stream() for _ in range(N_STREAMS)]

    Tr_p = [cuda.pinned_array(max_n, np.float32) for _ in range(N_STREAMS)]
    PET_p = [cuda.pinned_array(max_n, np.float32) for _ in range(N_STREAMS)]

    d_Tr = [cuda.device_array(max_n, np.float32) for _ in range(N_STREAMS)]
    d_PET = [cuda.device_array(max_n, np.float32) for _ in range(N_STREAMS)]

    threads = 256
    files_written = 0

    skip_count = 0
    for mrt_fp in mrt_files:
        try:
            parsed = parse_mrt_timestamp(mrt_fp.name)
            if parsed is None:
                continue
            date_str, hour, ts = parsed

            date_compact = date_str.replace("-", "")
            out_dir = OUTPUT_DIR / city / date_compact
            out_path = out_dir / f"pet_{ts}.tif"
            if out_path.exists():
                if not force:
                    files_written += 1
                    continue
                out_path.unlink()

            ta_val, rh_val, ws_val = get_meteo_from_csv(city, date_str, hour)
            ta_s = np.float32(ta_val)
            rh_s = np.float32(rh_val)
            ws_s = np.float32(ws_val)

            out_dir.mkdir(parents=True, exist_ok=True)

            with rasterio.open(mrt_fp) as mrt_src, \
                 rasterio.open(out_path, 'w', **out_meta) as dst:

                # pending: sid -> (n_pixels, nodata_mask, shape, window)
                pending = {}

                def _flush_slot(sid):
                    """同步该 slot 并写盘"""
                    streams[sid].synchronize()
                    n_px, nodata_mask, shape, prev_win = pending.pop(sid)
                    raw = PET_p[sid][:n_px].reshape(shape)
                    out = np.clip(np.round(raw), -50, 60).astype(np.int8)
                    out[nodata_mask] = np.int8(NODATA_PET)
                    dst.write(out, 1, window=prev_win)

                for idx, win in enumerate(windows):
                    sid = idx % N_STREAMS
                    s = streams[sid]

                    # 复用 slot 前先刷掉上一轮结果
                    if sid in pending:
                        _flush_slot(sid)

                    Tmrt = mrt_src.read(1, window=win).astype(np.float32)
                    n = Tmrt.size

                    nodata_mrt = (Tmrt == -999) | (Tmrt < -50.0)
                    Tmrt[nodata_mrt] = -99.0

                    Tr_p[sid][:n] = Tmrt.ravel()
                    cuda.to_device(Tr_p[sid][:n], to=d_Tr[sid][:n], stream=s)

                    blocks = (n + threads - 1) // threads
                    pet_kernel[blocks, threads, s](
                        d_Tr[sid], ta_s, ws_s, rh_s, d_PET[sid], n
                    )

                    d_PET[sid][:n].copy_to_host(PET_p[sid][:n], stream=s)
                    pending[sid] = (n, nodata_mrt, Tmrt.shape, win)

                # 刷完剩余 slot
                for sid in list(pending.keys()):
                    _flush_slot(sid)

            files_written += 1

        except Exception as e:
            skip_count += 1
            _log.warning(f"  {city}: 跳过 {mrt_fp.name}: {e}")
            # 清理可能写入一半的文件
            if 'out_path' in locals() and out_path.exists():
                try:
                    out_path.unlink()
                except Exception:
                    pass
            continue

    if skip_count > 0:
        _log.info(f"  {city}: 跳过 {skip_count} 个损坏文件")

    return {'city': city, 'files': files_written, 'skipped': skip_count}


# ======================== 主流程 ========================

def main():
    parser = argparse.ArgumentParser(description="Step 14: PET 计算 (5m WGS84, CUDA)")
    parser.add_argument("--cities", "-c", nargs='+', help="指定城市")
    parser.add_argument("--limit", "-l", type=int, default=0)
    parser.add_argument("--force", "-f", action="store_true", help="覆盖已有文件")
    args = parser.parse_args()

    dev = cuda.get_current_device()
    dev_name = dev.name.decode() if isinstance(dev.name, bytes) else dev.name

    print("=" * 60)
    print("Step 14: PET (生理等效温度) 计算 — 5m WGS84 (GPU)")
    print("=" * 60)
    print(f"  GPU: {dev_name}")
    print(f"  CHUNK_SIZE={CHUNK_SIZE}  N_STREAMS={N_STREAMS}  ITERS={ITERS}")
    print(f"  参数: MET={MET}, CLO={CLO}, REF_V={REF_V}, REF_PA={REF_PA}")
    print()

    cities = get_cities_with_mrt()
    print(f"有 MRT 的城市: {len(cities)}")

    if args.cities:
        cities = [c for c in args.cities if c in cities]
    if args.limit > 0:
        cities = cities[:args.limit]

    print(f"待处理: {len(cities)} 城市")
    print()

    results = []
    total_files = 0
    skip = 0

    for city_name in tqdm(cities, desc="计算PET"):
        try:
            r = compute_pet(city_name, force=args.force)
            if r and r['files'] > 0:
                results.append(r)
                total_files += r['files']
                _log.info(f"{city_name}: {r['files']} files")
            else:
                skip += 1
        except Exception as e:
            _log.error(f"{city_name}: {e}")
            skip += 1

    summary = OUTPUT_DIR / "pet_summary.json"
    with open(summary, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("PET 计算完成")
    print("=" * 60)
    print(f"  成功: {len(results)} 城市")
    print(f"  总文件数: {total_files}")
    print(f"  跳过: {skip}")
    print(f"  输出: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
