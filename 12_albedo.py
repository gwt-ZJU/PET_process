#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 12: 地表反照率计算 (5m, 512城市)
======================================
基于 NDVI + 建筑 + 树冠 计算地表反照率

地表类型与反照率:
  - 建筑 (building):   0.22
  - 水体 (water):      0.05
  - 植被 (NDVI>0.3 或树冠): 0.25
  - 不透水面 (impervious):  0.15
  - 其他/裸土:         0.20

输入:
  - NDVI:   data/ndvi/{city}/ndvi_5m.tif
  - 建筑:   data/build/{city}/buildings_5m.tif
  - 树冠:   data/canopy/{city}/canopy_5m.tif

输出:
  - data/albedo/{city}/albedo_5m.tif
"""

import os
os.environ['CPL_LOG_ERRORS'] = 'OFF'

import re
import json
import logging
import argparse
import numpy as np
import rasterio
from pathlib import Path
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============ 路径配置 ============
FULL_DIR = Path(__file__).parent
NDVI_DIR = FULL_DIR / "data" / "ndvi"
BUILD_DIR = FULL_DIR / "data" / "build"
CANOPY_DIR = FULL_DIR / "data" / "canopy"
OUTPUT_DIR = FULL_DIR / "data" / "albedo"

RESOLUTION = 5

# 反照率默认值
ALBEDO = {
    'vegetation': 0.25,
    'building': 0.22,
    'impervious': 0.15,
    'water': 0.05,
    'bare_soil': 0.20,
    'default': 0.18,
}

NDVI_VEG = 0.3       # NDVI > 0.3 → 植被
NDVI_WATER = -0.1    # NDVI < -0.1 → 水体 (收紧阈值避免硬质地面误判)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("albedo")


def normalize(name):
    s = str(name).lower().strip()
    s = re.sub(r'[^a-z0-9]', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s


def read_and_resample(path, template_profile, template_shape):
    """读取栅格并重采样到模板网格"""
    with rasterio.open(path) as src:
        if (CRS.from_user_input(src.crs) == CRS.from_user_input(template_profile['crs']) and
                src.width == template_shape[1] and
                src.height == template_shape[0]):
            return src.read(1)

        arr = np.zeros(template_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_profile['transform'],
            dst_crs=template_profile['crs'],
            resampling=Resampling.nearest,
        )
        return arr


def compute_albedo(city_folder):
    """计算单个城市的反照率"""
    res = RESOLUTION
    res_str = f"{res}m"

    build_path = BUILD_DIR / city_folder / f"buildings_{res_str}.tif"
    ndvi_path = NDVI_DIR / city_folder / f"ndvi_{res_str}.tif"
    canopy_path = CANOPY_DIR / city_folder / f"canopy_{res_str}.tif"

    out_dir = OUTPUT_DIR / city_folder
    out_path = out_dir / f"albedo_{res_str}.tif"

    # 需要至少有建筑数据
    if not build_path.exists():
        return {'city': city_folder, 'status': 'skip', 'reason': 'no build data'}

    if out_path.exists():
        return {'city': city_folder, 'status': 'cached'}

    # 用 buildings 作为模板 (UTM投影, 5m分辨率)
    with rasterio.open(build_path) as src:
        template = {
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height,
        }
        profile = src.profile.copy()
        h, w = src.height, src.width

    # 读取建筑
    build = None
    with rasterio.open(build_path) as src:
        build = src.read(1)

    # 读取NDVI (可能需要重采样)
    ndvi = None
    if ndvi_path.exists():
        ndvi = read_and_resample(ndvi_path, template, (h, w))
    else:
        log.warning(f"{city_folder}: 无 NDVI 数据")
        return {'city': city_folder, 'status': 'skip', 'reason': 'no ndvi'}

    # 读取树冠 (可选)
    canopy = None
    if canopy_path.exists():
        canopy = read_and_resample(canopy_path, template, (h, w))

    # 构建掩膜
    bld_mask = (build > 0).astype(np.uint8)
    can_mask = (canopy > 0).astype(np.uint8) if canopy is not None else np.zeros((h, w), np.uint8)

    ndvi_clean = np.where(np.isnan(ndvi), 0, ndvi)

    veg_mask = (ndvi_clean > NDVI_VEG).astype(np.uint8)
    water_mask = (ndvi_clean < NDVI_WATER).astype(np.uint8)

    # 裸土: NDVI在[-0.1, 0.3]且非建筑非树冠
    bare_mask = ((ndvi_clean >= NDVI_WATER) & (ndvi_clean <= NDVI_VEG) &
                 (bld_mask == 0) & (can_mask == 0)).astype(np.uint8)

    # 不透水面: NDVI < -0.1 但非水体(不会)且非建筑非树冠 → 用NDVI [0附近浅负值]
    # 实际: 非veg/bare/water/build/canopy 的剩余像素
    imp_mask = ((bare_mask == 0) & (veg_mask == 0) & (water_mask == 0) &
                (bld_mask == 0) & (can_mask == 0)).astype(np.uint8)

    # 赋值 (优先级从低到高: 不透水面 → 裸土 → 植被 → 水体 → 建筑)
    albedo = np.full((h, w), ALBEDO['impervious'], dtype=np.float32)
    albedo[bare_mask == 1] = ALBEDO['bare_soil']
    albedo[(veg_mask == 1) | (can_mask == 1)] = ALBEDO['vegetation']
    albedo[water_mask == 1] = ALBEDO['water']
    albedo[bld_mask == 1] = ALBEDO['building']

    # 保存
    out_dir.mkdir(parents=True, exist_ok=True)
    profile.update({
        'dtype': 'float32',
        'count': 1,
        'compress': 'lzw',
        'nodata': -9999,
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
    })

    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(albedo, 1)

    # 统计
    total_px = h * w
    stats = {
        'city': city_folder,
        'status': 'success',
        'mean_albedo': float(np.mean(albedo)),
        'ndvi_veg_pct': float(100 * np.sum(veg_mask) / total_px),
        'canopy_pct': float(100 * np.sum(can_mask) / total_px),
        'bld_pct': float(100 * np.sum(bld_mask) / total_px),
        'imp_pct': float(100 * np.sum(imp_mask) / total_px),
        'bare_pct': float(100 * np.sum(bare_mask) / total_px),
        'water_pct': float(100 * np.sum(water_mask) / total_px),
        'size': f"{w}x{h}",
    }

    del build, ndvi, canopy, bld_mask, can_mask, veg_mask, water_mask, imp_mask, bare_mask, albedo
    return stats


def main():
    parser = argparse.ArgumentParser(description="Step 12: 地表反照率计算 (5m)")
    parser.add_argument("--cities", "-c", nargs='+', help="指定城市名称")
    parser.add_argument("--limit", "-l", type=int, default=0,
                        help="限制处理城市数量")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 12: 地表反照率计算 (5m)")
    print("=" * 60)
    print(f"反照率: building={ALBEDO['building']}, veg={ALBEDO['vegetation']}, "
          f"water={ALBEDO['water']}, impervious={ALBEDO['impervious']}, "
          f"bare_soil={ALBEDO['bare_soil']}")

    # 收集所有有建筑数据的城市
    build_cities = set()
    for d in BUILD_DIR.iterdir():
        if d.is_dir() and (d / f"buildings_{RESOLUTION}m.tif").exists():
            build_cities.add(d.name)

    ndvi_cities = set()
    for d in NDVI_DIR.iterdir():
        if d.is_dir() and (d / f"ndvi_{RESOLUTION}m.tif").exists():
            ndvi_cities.add(d.name)

    cities = sorted(build_cities & ndvi_cities)

    if args.cities:
        folders = [normalize(c) for c in args.cities]
        cities = [c for c in cities if c in folders]

    if args.limit > 0:
        cities = cities[:args.limit]

    print(f"有建筑数据: {len(build_cities)}, 有NDVI: {len(ndvi_cities)}")
    print(f"交集 (可处理): {len(cities)} 城市")
    print()

    results = []
    stats = {'success': 0, 'cached': 0, 'skip': 0}

    for city_folder in tqdm(cities, desc="计算反照率"):
        result = compute_albedo(city_folder)
        results.append(result)
        status = result['status']
        if status in stats:
            stats[status] += 1

        if status == 'success':
            log.info(f"{city_folder}: 均值={result['mean_albedo']:.3f}, "
                     f"建筑={result['bld_pct']:.1f}%, NDVI植被={result['ndvi_veg_pct']:.1f}%, "
                     f"树冠={result['canopy_pct']:.1f}%, 裸土={result['bare_pct']:.1f}%, "
                     f"不透水={result['imp_pct']:.1f}%, 水体={result['water_pct']:.1f}%")

    # Save summary
    summary_file = OUTPUT_DIR / "albedo_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"  成功: {stats['success']}")
    print(f"  缓存: {stats['cached']}")
    print(f"  跳过: {stats['skip']}")
    print(f"  输出: {OUTPUT_DIR}")
    print(f"  摘要: {summary_file}")


if __name__ == "__main__":
    main()
