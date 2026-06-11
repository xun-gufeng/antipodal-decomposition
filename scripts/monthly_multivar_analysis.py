"""
Monthly & Multi-Variable Complementarity Analysis
==================================================
Reproduces Tables S3 and S4 in the SI Appendix.

Uses NCEP/NCAR Reanalysis data (OPeNDAP or local files).
Reference period: 1961-1990.

Output:
  - monthly_complementarity.csv  (Table S3)
  - multivar_complementarity.csv (Table S4)

Requirements: numpy, pandas, xarray, scipy, netcdf4
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os

# ── Configuration ──
OUTPUT_DIR = './results/'
DATA_DIR = './data/ncep/'
USE_OPENDAP = True  # Set False to use local files

LUOSHU_PAIRS = [(1, 9), (2, 8), (3, 7), (4, 6)]
CENTER = (34.62, 112.45)
CENTRAL_RADIUS = 2.0
REF_START, REF_END = 1961, 1990

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Palace Assignment ──
def assign_palace(lats, lons, center=CENTER, radius=CENTRAL_RADIUS):
    clat, clon = center
    dlat = lats - clat
    dlon = (lons - clon + 180) % 360 - 180
    dlon_corr = dlon * np.cos(np.radians(lats))
    dist = np.sqrt(dlat**2 + dlon_corr**2)
    az = np.degrees(np.arctan2(dlon_corr, dlat)) % 360
    sector = (az / 45).astype(int) % 8
    palace_map = np.array([1, 8, 3, 4, 9, 2, 7, 6])
    palaces = palace_map[sector]
    palaces[dist < radius] = 5
    return palaces

# ── Pairing Enumeration ──
def generate_all_pairings():
    items = [i for i in range(1, 10) if i != 5]
    def _pairings(remaining):
        if not remaining:
            yield []
            return
        first = remaining[0]
        for i in range(1, len(remaining)):
            rest = remaining[1:i] + remaining[i+1:]
            for p in _pairings(rest):
                yield [(first, remaining[i])] + p
    seen = set()
    for p in _pairings(items):
        norm = tuple(sorted(tuple(sorted(pair)) for pair in p))
        if norm not in seen:
            seen.add(norm)
            yield list(norm)

def compute_paired_cv(pair_means):
    mu = np.mean(pair_means)
    if mu == 0:
        return np.inf
    return np.std(pair_means, ddof=0) / mu

def luoshu_rank(palace_amplitudes, all_pairings):
    pm = [(palace_amplitudes.get(a, 0) + palace_amplitudes.get(b, 0)) / 2
          for a, b in LUOSHU_PAIRS]
    cv_l = compute_paired_cv(pm)
    all_cvs = []
    for pairing in all_pairings:
        pm_i = [(palace_amplitudes.get(a, 0) + palace_amplitudes.get(b, 0)) / 2
                for a, b in pairing]
        all_cvs.append(compute_paired_cv(pm_i))
    rank = sum(1 for c in all_cvs if c < cv_l) + 1
    return cv_l, rank

print("Generating all 105 pairings...")
ALL_PAIRINGS = list(generate_all_pairings())
print(f"Total: {len(ALL_PAIRINGS)}")

# ── NCEP Data Sources ──
# NOTE on NCEP grid resolution:
#   surface/ variables (air, wspd) use 2.5°×2.5° lat-lon grid (144×73)
#   surface_gauss/ and other_gauss/ variables (tmax, tmin, tcdc) use T62 Gaussian grid (~1.9°)
NCEP_OPENDAP = {
    'air':  'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/surface/air.mon.mean.nc',
    'tmax': 'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/surface_gauss/tmax.2m.mon.mean.nc',
    'tmin': 'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/surface_gauss/tmin.2m.mon.mean.nc',
    'tcdc': 'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/other_gauss/tcdc.eatm.mon.mean.nc',
    'wspd': 'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/surface/wspd.sig995.mon.mean.nc',
}

NCEP_LOCAL = {
    'air':  f'{DATA_DIR}air.mon.mean.nc',
    'tmax': f'{DATA_DIR}tmax.2m.mon.mean.nc',
    'tmin': f'{DATA_DIR}tmin.2m.mon.mean.nc',
    'tcdc': f'{DATA_DIR}tcdc.eatm.mon.mean.nc',
    'wspd': f'{DATA_DIR}wspd.sig995.mon.mean.nc',
}

NCEP_VARMAP = {
    'air': 'air', 'tmax': 'tmax', 'tmin': 'tmin',
    'tcdc': 'tcdc', 'wspd': 'wspd',
}

# ── Helper: compute seasonal amplitude per palace ──
def compute_palace_amplitudes(data_array, lats_2d, lons_2d):
    palace_grid = assign_palace(lats_2d.ravel(), lons_2d.ravel()).reshape(lats_2d.shape)
    palace_ampls = {}
    palace_lats = {}
    for p in range(1, 10):
        mask = (palace_grid == p)
        if mask.sum() == 0:
            palace_ampls[p] = np.nan
            palace_lats[p] = np.nan
            continue
        palace_lats[p] = float(lats_2d[mask].mean())
        monthly_means = []
        for month in range(1, 13):
            m_data = data_array.sel(time=data_array.time.dt.month == month)
            vals = m_data.values
            if vals.ndim == 3:
                vals = vals[:, mask].mean()
            elif vals.ndim == 2:
                vals = vals[mask].mean()
            monthly_means.append(float(vals))
        palace_ampls[p] = max(monthly_means) - min(monthly_means)
    return palace_ampls, palace_lats, palace_grid

# ── Load dataset ──
def load_ncep(key):
    if USE_OPENDAP:
        url = NCEP_OPENDAP[key]
        print(f"  Loading {key} from OPeNDAP...")
        return xr.open_dataset(url, engine='netcdf4')
    else:
        path = NCEP_LOCAL[key]
        print(f"  Loading {key} from local file...")
        return xr.open_dataset(path)

# ══════════════════════════════════════════
# PART 1: Monthly Complementarity (Table S3)
# ══════════════════════════════════════════
# NOTE: Monthly analysis uses palace-mean absolute temperature (not seasonal
# amplitude). When the mean of pair means approaches 0°C (winter months),
# CV = sigma/mu becomes unstable. However, since the sum of all 8 outer-palace
# temperatures is constant across all 105 pairings, mu is identical for every
# scheme in a given month; thus CV ranking = sigma ranking, and the rank is
# valid even when the absolute CV value is not interpretable.
print("\n" + "=" * 60)
print("PART 1: Monthly Complementarity")
print("=" * 60)

ds_air = load_ncep('air')
time_mask = (ds_air.time.dt.year >= REF_START) & (ds_air.time.dt.year <= REF_END)
lat_vals = ds_air.lat.values
lon_vals = ds_air.lon.values
lat_mask = (lat_vals >= 15) & (lat_vals <= 55)
lon_mask = (lon_vals >= 70) & (lon_vals <= 145)

air_china = ds_air['air'].sel(time=time_mask).isel(lat=lat_mask, lon=lon_mask)
lats_2d, lons_2d = np.meshgrid(air_china.lat.values, air_china.lon.values, indexing='ij')
palace_grid = assign_palace(lats_2d.ravel(), lons_2d.ravel()).reshape(lats_2d.shape)

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

monthly_results = []
for month in range(1, 13):
    palace_means = {}
    for p in range(1, 10):
        mask = (palace_grid == p)
        if mask.sum() == 0:
            continue
        m_data = air_china.sel(time=air_china.time.dt.month == month)
        palace_means[p] = float(m_data.values[:, mask].mean())

    pair_means = [(palace_means.get(a, 0) + palace_means.get(b, 0)) / 2
                  for a, b in LUOSHU_PAIRS]
    cv_m = compute_paired_cv(pair_means)

    all_cvs = []
    for pairing in ALL_PAIRINGS:
        pm_i = [(palace_means.get(a, 0) + palace_means.get(b, 0)) / 2
                for a, b in pairing]
        all_cvs.append(compute_paired_cv(pm_i))
    rank_m = sum(1 for c in all_cvs if c < cv_m) + 1

    monthly_results.append({
        'month': month, 'month_name': month_names[month-1],
        'luoshu_cv': round(cv_m, 4), 'rank_105': rank_m,
    })
    print(f"  {month_names[month-1]:3s}: CV={cv_m:.4f}, Rank={rank_m}/105")

monthly_df = pd.DataFrame(monthly_results)
monthly_df.to_csv(os.path.join(OUTPUT_DIR, 'monthly_complementarity.csv'), index=False)
print(f"\nAll 12 months rank <= {monthly_df['rank_105'].max()}/105")

# ══════════════════════════════════════════
# PART 2: Multi-Variable Complementarity (Table S4)
# ══════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 2: Multi-Variable Complementarity")
print("=" * 60)

VARIABLES = {
    'TMEAN': {'key': 'air',  'unit': '°C', 'desc': 'Surface air temperature'},
    'TMAX':  {'key': 'tmax', 'unit': '°C', 'desc': 'Maximum temperature'},
    'TMIN':  {'key': 'tmin', 'unit': '°C', 'desc': 'Minimum temperature'},
    'WIND':  {'key': 'wspd', 'unit': 'm/s', 'desc': 'Surface wind speed'},
    'CLD':   {'key': 'tcdc', 'unit': '%',  'desc': 'Total cloud cover'},
}

multivar_results = []
for vname, vinfo in VARIABLES.items():
    print(f"\nProcessing {vname} ({vinfo['desc']})...")
    try:
        ds = load_ncep(vinfo['key'])
        var_data = ds[NCEP_VARMAP[vinfo['key']]]

        lat_v = ds.lat.values
        lon_v = ds.lon.values
        lat_m = (lat_v >= 15) & (lat_v <= 55)
        lon_m = (lon_v >= 70) & (lon_v <= 145)
        t_mask = (ds.time.dt.year >= REF_START) & (ds.time.dt.year <= REF_END)
        data_china = var_data.sel(time=t_mask).isel(lat=lat_m, lon=lon_m)

        lats_v, lons_v = np.meshgrid(data_china.lat.values, data_china.lon.values, indexing='ij')
        palace_ampls, palace_lats, _ = compute_palace_amplitudes(data_china, lats_v, lons_v)

        cv, rank = luoshu_rank(palace_ampls, ALL_PAIRINGS)

        lat_list = [palace_lats.get(p, np.nan) for p in range(1, 10) if p != 5]
        ampl_list = [palace_ampls.get(p, np.nan) for p in range(1, 10) if p != 5]
        valid = ~np.isnan(lat_list) & ~np.isnan(ampl_list)
        r_lat, p_lat = stats.pearsonr(np.array(lat_list)[valid], np.array(ampl_list)[valid])

        multivar_results.append({
            'variable': vname,
            'desc': vinfo['desc'],
            'luoshu_cv': round(cv, 4),
            'rank_105': rank,
            'r_lat': round(r_lat, 3),
            'p_lat': f'{p_lat:.2e}',
        })
        print(f"  CV={cv:.4f}, Rank={rank}/105, r_lat={r_lat:.3f}")
        ds.close()
    except Exception as e:
        print(f"  Error: {e}")

multivar_df = pd.DataFrame(multivar_results)
multivar_df.to_csv(os.path.join(OUTPUT_DIR, 'multivar_complementarity.csv'), index=False)
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(multivar_df[['variable', 'luoshu_cv', 'rank_105', 'r_lat']].to_string(index=False))
print(f"\nResults saved to {OUTPUT_DIR}")
