# PET_process

Pedestrian-level PET (Physiological Equivalent Temperature) computation pipeline, from clear-sky irradiance to PET rasters at 5 m resolution.

## Pipeline

| Script | Step | Output |
|---|---|---|
| `10_irradiance.py` | Clear-sky (Ineichen) irradiance with building/canopy shadow beam attenuation (τ_tree) | 2-band rasters (direct, diffuse), W/m² |
| `12_albedo.py` | Surface albedo classification from NDVI + buildings + canopy | `albedo_5m.tif` |
| `13_mrt.py` | Mean Radiant Temperature (ISO 7726 / Höppe 1992 / SOLWEIG corrections) | `mrt_*.tif`, °C |
| `14_pet.py` | PET via PMV bisection search, CUDA-accelerated | `pet_*.tif`, °C |

## Key parameters

- Tree transmissivity: τ_dir = 0.30 (beam), τ_dif = 0.40 (canopy SVF correction)
- Body: A_SW = 0.70, ε_p = 0.97; MET = 3.5, CLO = 0.5
- Surface emissivity: ground/wall 0.95, sky 0.90
- Wall albedo 0.20 (SOLWEIG default)

Paths inside the scripts are configured for the original project layout and can be adapted via constants / environment variables at the top of each file.

## Requirements

Python 3.12 environment with: numpy, pandas, rasterio, pvlib, timezonefinder, pytz, geopandas, numba (CUDA), tqdm.
