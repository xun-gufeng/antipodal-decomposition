"""
洛书时空耦合实验 v2（高效版）
核心假说：洛书约束不体现为宫间空间差异，而体现为宫内时间结构与轴间耦合

检验1：轴对宫异常相关 — 去季节后，对宫异常是否轴特异耦合
检验2：六气共振 — 宫-气五行匹配时方差是否放大
检验3：季节循环轴间相位对称性 — 对宫季节循环相位差是否符合物理预期
"""

import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
from scipy import stats, signal
import datetime
import warnings
warnings.filterwarnings('ignore')

QI_NAMES = ['初之气', '二之气', '三之气', '四之气', '五之气', '终之气']
MONTH_TO_QI = {1:0, 2:0, 3:1, 4:1, 5:2, 6:2, 7:3, 8:3, 9:4, 10:4, 11:5, 12:5}
QI_WUXING = {0: '木', 1: '火', 2: '火', 3: '土', 4: '金', 5: '水'}
PALACE_WUXING = {1:'水', 2:'土', 3:'木', 4:'木', 5:'土', 6:'金', 7:'金', 8:'土', 9:'火'}
PALACE_NAMES = {1:'坎(北)', 2:'坤(西南)', 3:'震(东)', 4:'巽(东南)',
                5:'中', 6:'乾(西北)', 7:'兑(西)', 8:'艮(东北)', 9:'离(南)'}

PALACE_RESONANT_QI = {
    1: [5], 2: [3], 3: [0], 4: [0], 5: [3], 6: [4], 7: [4], 8: [3], 9: [1, 2],
}

AXES = {
    '水火': {'pair': (1, 9), 'var': 'bowen', 'expect': 'anti_corr',
             'desc': '1↔9坎离: Bowen ratio异常应反号'},
    '木金': {'pair': (3, 7), 'var': 'rhum', 'expect': 'anti_corr',
             'desc': '3↔7震兑: 湿度异常应反号'},
    '坤艮': {'pair': (2, 8), 'var': 'dtr', 'expect': 'pos_corr',
             'desc': '2↔8坤艮: DTR异常应同号'},
    '巽乾': {'pair': (4, 6), 'var': 'gflux', 'expect': 'anti_corr',
             'desc': '4↔6巽乾: 地热通量异常应反号'},
}

PALACE_MAP = {
    ('上','左'): 6, ('上','中'): 1, ('上','右'): 8,
    ('中','左'): 7, ('中','中'): 5, ('中','右'): 3,
    ('下','左'): 2, ('下','中'): 9, ('下','右'): 4,
}

def lat_to_row(lat):
    if lat > 35: return '上'
    elif lat < 25: return '下'
    else: return '中'

def lon_to_col(lon):
    if lon < 107.5: return '左'
    elif lon > 112.5: return '右'
    else: return '中'

DATA_DIR = './data/ncep/'

# ============================================================
# 1. 数据提取
# ============================================================
print("=" * 70)
print("洛书时空耦合实验 v2")
print("=" * 70)

def extract_local(varname, filepath):
    ds = nc.Dataset(filepath)
    time_var = np.array(ds.variables['time'][:])
    origin = datetime.datetime(1800, 1, 1)
    dates = [origin + datetime.timedelta(hours=float(t)) for t in time_var]
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]
    lat_mask = (lat >= 22.5) & (lat <= 42.5)
    lon_mask = (lon >= 97.5) & (lon <= 122.5)
    lat_sel = lat[lat_mask]
    lon_sel = lon[lon_mask]
    data = ds.variables[varname][:, lat_mask, lon_mask]
    ds.close()
    records = []
    for li, la in enumerate(lat_sel):
        for lj, lo in enumerate(lon_sel):
            row = lat_to_row(la)
            col = lon_to_col(lo)
            palace = PALACE_MAP.get((row, col), None)
            if palace is None: continue
            for t, dt in enumerate(dates):
                if dt.year < 1948 or dt.year > 2025: continue
                val = float(data[t, li, lj])
                if val < -9e30 or np.isnan(val): continue
                records.append((dt.year, dt.month, palace, val))
    df = pd.DataFrame(records, columns=['year','month','palace',varname])
    return df.groupby(['year','month','palace'])[varname].mean().reset_index()

local_vars = {
    'shtfl': f'{DATA_DIR}shtfl.sfc.mon.mean.nc',
    'lhtfl': f'{DATA_DIR}lhtfl.sfc.mon.mean.nc',
    'wspd': f'{DATA_DIR}wspd.mon.mean.nc',
    'rhum': f'{DATA_DIR}rhum.mon.mean.nc',
    'air': f'{DATA_DIR}air.mon.mean.nc',
}

dfs = {}
for varname, filepath in local_vars.items():
    print(f"  {varname}...", end=" ", flush=True)
    dfs[varname] = extract_local(varname, filepath)
    print(f"OK")

df = dfs['shtfl']
for varname in ['lhtfl', 'wspd', 'rhum', 'air']:
    df = df.merge(dfs[varname], on=['year','month','palace'], how='outer')
del dfs

opendap_vars = {
    'tmax': 'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/surface_gauss/tmax.2m.mon.mean.nc',
    'tmin': 'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/surface_gauss/tmin.2m.mon.mean.nc',
    'tcdc': 'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/other_gauss/tcdc.eatm.mon.mean.nc',
    'gflux': 'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/surface_gauss/gflux.sfc.mon.mean.nc',
}

for varname, url in opendap_vars.items():
    print(f"  {varname} (OPeNDAP)...", end=" ", flush=True)
    try:
        ds = xr.open_dataset(url, engine='netcdf4')
        lat = ds.lat.values; lon = ds.lon.values
        lat_mask = (lat >= 22.5) & (lat <= 42.5)
        lon_mask = (lon >= 97.5) & (lon <= 122.5)
        lat_idx = np.where(lat_mask)[0]; lon_idx = np.where(lon_mask)[0]
        data = ds[varname].isel(lat=lat_idx, lon=lon_idx).values
        time_var = ds.time.values; ds.close()
        records = []
        for li, la in enumerate(lat[lat_mask]):
            for lj, lo in enumerate(lon[lon_mask]):
                row = lat_to_row(float(la)); col = lon_to_col(float(lo))
                palace = PALACE_MAP.get((row, col), None)
                if palace is None: continue
                for t in range(len(time_var)):
                    dt = pd.Timestamp(time_var[t])
                    if dt.year < 1948 or dt.year > 2025: continue
                    val = float(data[t, li, lj])
                    if val < -9e30 or np.isnan(val): continue
                    records.append((dt.year, dt.month, palace, val))
        df_v = pd.DataFrame(records, columns=['year','month','palace',varname])
        df_v = df_v.groupby(['year','month','palace'])[varname].mean().reset_index()
        df = df.merge(df_v, on=['year','month','palace'], how='outer')
        del df_v, data
        print(f"OK")
    except Exception as e:
        print(f"FAIL: {e}")

df['qi'] = df['month'].map(MONTH_TO_QI)
df['bowen'] = np.where(np.abs(df['lhtfl']) > 1, df['shtfl'] / df['lhtfl'], np.nan)
df['dtr'] = df['tmax'] - df['tmin']
df['wind_humid'] = df['wspd'] * df['rhum']
print(f"数据: {len(df)} rows")

# ============================================================
# 2. 去季节异常
# ============================================================
print("\n构建去季节+去趋势异常...")
anomaly_cols = ['bowen', 'rhum', 'dtr', 'gflux', 'shtfl', 'lhtfl', 'wspd', 'air', 'tcdc', 'wind_humid']
for col in anomaly_cols:
    anom_col = f'{col}_anom'
    clim = df.groupby(['month', 'palace'])[col].transform('mean')
    df[anom_col] = df[col] - clim
    for p in range(1, 10):
        mask = df['palace'] == p
        if mask.sum() == 0: continue
        years = df.loc[mask, 'year'].values
        vals = df.loc[mask, anom_col].values
        valid = ~np.isnan(vals)
        if valid.sum() > 20:
            slope, intercept = np.polyfit(years[valid], vals[valid], 1)
            df.loc[mask, anom_col] = vals - (slope * years + intercept)
print("完成")

# ============================================================
# 检验1：轴对宫异常相关
# ============================================================
print("\n" + "=" * 70)
print("检验1：轴对宫异常相关")
print("=" * 70)

N_PERM = 5000
np.random.seed(42)

# 预构建异常矩阵：palace_data[palace] = DataFrame(index=year_month, columns=anom_vars)
palace_anom = {}
for p in range(1, 10):
    sub = df[df['palace'] == p][['year', 'month'] + [f'{c}_anom' for c in anomaly_cols]].copy()
    sub = sub.set_index(['year', 'month'])
    palace_anom[p] = sub

# 预计算所有C(9,2)=36对宫的异常相关
print("预计算36对宫相关矩阵...")
corr_matrix = {}
for p1 in range(1, 10):
    for p2 in range(p1+1, 10):
        common_idx = palace_anom[p1].index.intersection(palace_anom[p2].index)
        if len(common_idx) < 50:
            continue
        for col in anomaly_cols:
            anom_col = f'{col}_anom'
            v1 = palace_anom[p1].loc[common_idx, anom_col].values
            v2 = palace_anom[p2].loc[common_idx, anom_col].values
            valid = ~(np.isnan(v1) | np.isnan(v2))
            if valid.sum() > 50:
                r = np.corrcoef(v1[valid], v2[valid])[0, 1]
                corr_matrix[(p1, p2, col)] = r

print(f"预计算完成: {len(corr_matrix)} 个对宫×变量组合")

# 对每个轴检验
print(f"\n{'轴':>6} {'对宫':>6} {'变量':>6} {'观测r':>8} {'全对宫μ':>8} {'全对宫σ':>8} {'p值':>8} {'方向':>4}")
print("-" * 65)

test1_results = {}
for axis_name, axis_info in AXES.items():
    pa, pb = axis_info['pair']
    var = axis_info['var']
    key = (min(pa,pb), max(pa,pb), var)
    obs_r = corr_matrix.get(key, None)
    if obs_r is None:
        print(f"{axis_name:>6} {pa}↔{pb} {var:>6}  无数据")
        continue
    
    # Null: 所有其他对宫在该变量上的相关
    all_corrs = [v for (p1,p2,c), v in corr_matrix.items() if c == var and (p1,p2) != (min(pa,pb), max(pa,pb))]
    all_corrs = np.array(all_corrs)
    perm_mu = np.mean(all_corrs)
    perm_sd = np.std(all_corrs)
    
    # 双侧: 观测相关是否比随机对宫更极端
    p_val = np.mean(np.abs(all_corrs - perm_mu) >= np.abs(obs_r - perm_mu))
    p_val = max(min(p_val, 1.0), 1.0/len(all_corrs))
    
    expect = axis_info['expect']
    direction = '✓' if (expect == 'anti_corr' and obs_r < perm_mu) or (expect == 'pos_corr' and obs_r > perm_mu) else '✗'
    
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
    print(f"{axis_name:>6} {pa}↔{pb} {var:>6} {obs_r:+.4f} {perm_mu:+.4f} {perm_sd:.4f} {p_val:.4f}{sig} {direction}")
    test1_results[axis_name] = {'r': obs_r, 'p': p_val, 'expect': expect, 'direction': direction}

# 同纬度对照
print("\n同纬度非轴对宫异常相关:")
control_pairs = [
    ('1↔8(北)', 1, 8, 'bowen'), ('9↔2(南)', 9, 2, 'bowen'),
    ('3↔4(东)', 3, 4, 'rhum'), ('7↔6(西)', 7, 6, 'rhum'),
]
for name, pa, pb, var in control_pairs:
    key = (min(pa,pb), max(pa,pb), var)
    obs_r = corr_matrix.get(key, None)
    if obs_r is None:
        print(f"  {name}: 无数据")
        continue
    all_corrs = np.array([v for (p1,p2,c), v in corr_matrix.items() if c == var and (p1,p2) != (min(pa,pb), max(pa,pb))])
    perm_mu = np.mean(all_corrs)
    p_val = np.mean(np.abs(all_corrs - perm_mu) >= np.abs(obs_r - perm_mu))
    p_val = max(min(p_val, 1.0), 1.0/max(len(all_corrs),1))
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
    print(f"  {name}: r={obs_r:+.4f}, perm_μ={perm_mu:+.4f}, p={p_val:.4f}{sig}")

# 打印全部对宫相关矩阵（bowen和rhum）
print("\n对宫异常相关全景 (关键变量):")
for var in ['bowen', 'rhum', 'dtr', 'gflux']:
    print(f"\n  {var}:")
    for p1 in range(1, 10):
        row_vals = []
        for p2 in range(1, 10):
            if p1 == p2:
                row_vals.append('  ---')
            else:
                key = (min(p1,p2), max(p1,p2), var)
                r = corr_matrix.get(key, None)
                row_vals.append(f'{r:+.3f}' if r is not None else '  N/A')
        print(f"    宫{p1}: {' '.join(row_vals)}")

# ============================================================
# 检验2：六气共振（高效版）
# ============================================================
print("\n" + "=" * 70)
print("检验2：六气共振（宫-气五行匹配方差放大）")
print("=" * 70)

# 预聚合到 (year, qi, palace) → variance of anomalies within (year, qi, palace)
# 注意：每个(year, qi, palace)有2个月，方差=这2个月的离散度
# 更好的做法：用每个(year, qi, palace)的均值，然后计算共振气vs非共振气的年际方差

df_yqp = df.groupby(['year', 'qi', 'palace']).agg({
    f'{c}_anom': 'mean' for c in anomaly_cols
}).reset_index()

print(f"聚合到(year,qi,palace): {len(df_yqp)} rows")

N_PERM2 = 5000

test2_all = {}
for axis_name, axis_info in AXES.items():
    var_anom = f"{axis_info['var']}_anom"
    pa, pb = axis_info['pair']
    print(f"\n{axis_name}轴 ({axis_info['var']}):")
    print(f"  {'宫':>3} {'名':>8} {'五行':>4} {'共振气':>10} {'共振σ':>7} {'非共振σ':>7} {'σ比':>6} {'p值':>8}")
    print("  " + "-" * 60)
    
    for palace in [pa, pb]:
        sub = df_yqp[df_yqp['palace'] == palace][['year', 'qi', var_anom]].dropna()
        if len(sub) < 30:
            continue
        
        resonant_qi = PALACE_RESONANT_QI[palace]
        
        # 每年：共振气均值 vs 非共振气均值
        # 然后：共振气的年际方差 vs 非共振气的年际方差
        yearly = sub.pivot_table(index='year', columns='qi', values=var_anom)
        res_cols = [c for c in resonant_qi if c in yearly.columns]
        nonres_cols = [c for c in range(6) if c not in resonant_qi and c in yearly.columns]
        
        if not res_cols or not nonres_cols:
            continue
        
        res_mean = yearly[res_cols].mean(axis=1).dropna()
        nonres_mean = yearly[nonres_cols].mean(axis=1).dropna()
        
        obs_res_var = res_mean.var()
        obs_nonres_var = nonres_mean.var()
        obs_ratio = obs_res_var / obs_nonres_var if obs_nonres_var > 0 else np.nan
        
        if np.isnan(obs_ratio):
            continue
        
        # Permutation: 随机选同数量qi作为"共振"
        perm_ratios = np.zeros(N_PERM2)
        n_res = len(res_cols)
        for k in range(N_PERM2):
            rand_qi = np.random.choice(range(6), n_res, replace=False)
            rand_nonres = [q for q in range(6) if q not in rand_qi]
            rq = [c for c in rand_qi if c in yearly.columns]
            nrq = [c for c in rand_nonres if c in yearly.columns]
            if not rq or not nrq:
                perm_ratios[k] = 1.0
                continue
            rm = yearly[rq].mean(axis=1).dropna()
            nrm = yearly[nrq].mean(axis=1).dropna()
            if nrm.var() > 0:
                perm_ratios[k] = rm.var() / nrm.var()
            else:
                perm_ratios[k] = 1.0
        
        # 单侧检验：共振气方差 > 非共振气 → ratio > 1
        p_val = np.mean(perm_ratios >= obs_ratio)
        p_val = max(min(p_val, 1.0), 1.0/N_PERM2)
        
        qi_str = ','.join([QI_NAMES[q] for q in resonant_qi])
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        print(f"  {palace:>3} {PALACE_NAMES[palace]:>8} {PALACE_WUXING[palace]:>4} {qi_str:>10} "
              f"{np.sqrt(obs_res_var):7.3f} {np.sqrt(obs_nonres_var):7.3f} {obs_ratio:6.3f} {p_val:8.4f}{sig}")
        test2_all[(axis_name, palace)] = {'p': p_val, 'ratio': obs_ratio, 'res_var': obs_res_var, 'nonres_var': obs_nonres_var}

# ============================================================
# 检验3：季节循环轴间相位对称性
# ============================================================
print("\n" + "=" * 70)
print("检验3：季节循环轴间相位对称性")
print("=" * 70)

def compute_seasonal_phase(palace, varname):
    sub = df[(df['palace'] == palace)][['month', varname]].dropna()
    if len(sub) < 36:
        return None, None, None, None
    clim = sub.groupby('month')[varname].mean()
    if len(clim) < 12:
        return None, None, None, None
    vals = clim.reindex(range(1, 13), fill_value=clim.mean()).values
    vals_detrend = vals - vals.mean()
    fft = np.fft.fft(vals_detrend)
    amp = np.abs(fft[1]) / 12 * 2
    phase = np.angle(fft[1])
    phase_month = (-phase / (2 * np.pi) * 12) % 12 + 1
    # 半年循环
    amp2 = np.abs(fft[2]) / 12 * 2
    phase2 = np.angle(fft[2])
    return amp, phase_month, amp2, phase2

print(f"\n{'轴':>6} {'对宫':>6} {'变量':>6} {'A峰值月':>8} {'B峰值月':>8} {'相位差':>6} {'预期':>12} {'匹配':>4}")
print("-" * 65)

test3_results = {}
for axis_name, axis_info in AXES.items():
    pa, pb = axis_info['pair']
    var = axis_info['var']
    amp_a, pm_a, amp2_a, ph2_a = compute_seasonal_phase(pa, var)
    amp_b, pm_b, amp2_b, ph2_b = compute_seasonal_phase(pb, var)
    
    if pm_a is None or pm_b is None:
        print(f"{axis_name:>6} {pa}↔{pb} {var:>6}  数据不足")
        continue
    
    diff = abs(pm_a - pm_b)
    if diff > 6: diff = 12 - diff
    
    if axis_info['expect'] == 'anti_corr':
        expect_desc = '≈6月(反相)'
        match = '✓' if 4 <= diff <= 8 else '✗'
    else:
        expect_desc = '≈0月(同相)'
        match = '✓' if diff <= 2 or diff >= 10 else '✗'
    
    print(f"{axis_name:>6} {pa}↔{pb} {var:>6} {pm_a:8.1f} {pm_b:8.1f} {diff:6.1f} {expect_desc:>12} {match}")
    
    # Permutation检验
    all_pm = []
    for p in range(1, 10):
        _, pm, _, _ = compute_seasonal_phase(p, var)
        if pm is not None:
            all_pm.append((p, pm))
    
    if len(all_pm) >= 4:
        pm_values = np.array([x[1] for x in all_pm])
        perm_diffs = []
        for _ in range(N_PERM2):
            i, j = np.random.choice(len(pm_values), 2, replace=False)
            d = abs(pm_values[i] - pm_values[j])
            if d > 6: d = 12 - d
            perm_diffs.append(d)
        perm_diffs = np.array(perm_diffs)
        
        if axis_info['expect'] == 'anti_corr':
            p_val = np.mean(np.abs(perm_diffs - 6) <= np.abs(diff - 6))
        else:
            p_val = np.mean(perm_diffs <= diff)
        p_val = max(min(p_val, 1.0), 1.0/N_PERM2)
    else:
        p_val = 1.0
    
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
    print(f"  → perm p={p_val:.4f}{sig}, perm_μ={np.mean(perm_diffs):.1f}月")
    test3_results[axis_name] = {'diff': diff, 'p': p_val, 'match': match}

# ============================================================
# 综合评估
# ============================================================
print("\n" + "=" * 70)
print("综合评估")
print("=" * 70)

all_pvals = {}

print("\n检验1（轴对宫异常相关）:")
for axis, res in test1_results.items():
    all_pvals[('异常相关', axis)] = res['p']
    print(f"  {axis}: r={res['r']:+.4f}, p={res['p']:.4f}, 预期方向{res['direction']}")

print("\n检验2（六气共振）:")
for (axis, palace), info in test2_all.items():
    all_pvals[('六气共振', f'{axis}_宫{palace}')] = info['p']
    sig = '***' if info['p'] < 0.001 else '**' if info['p'] < 0.01 else '*' if info['p'] < 0.05 else ''
    print(f"  {axis} 宫{palace}: σ比={info['ratio']:.3f}, p={info['p']:.4f}{sig}")

print("\n检验3（相位对称性）:")
for axis, res in test3_results.items():
    all_pvals[('相位对称', axis)] = res['p']
    print(f"  {axis}: 相位差={res['diff']:.1f}月, p={res['p']:.4f}, 匹配{res['match']}")

n_tests = len(all_pvals)
best_p = min(all_pvals.values()) if all_pvals else 1.0
fwer_p = min(best_p * n_tests, 1.0)

sig_raw = [(k, v) for k, v in all_pvals.items() if v < 0.05]
print(f"\n{'='*50}")
print(f"总检验数: {n_tests}")
print(f"最佳原始p值: {best_p:.4f}")
print(f"FWER校正后: {fwer_p:.4f}")
print(f"原始p<0.05: {len(sig_raw)}/{n_tests}")
for (test, axis), p in sorted(sig_raw, key=lambda x: x[1]):
    print(f"  {test} {axis}: p={p:.4f}")

# ============================================================
# 物理解读
# ============================================================
print("\n" + "=" * 70)
print("物理解读")
print("=" * 70)

print("""
三轮检验的方法论逻辑：

1. 空间均值差异 → 被纬度/海陆稀释 → 全阴性
2. 轴特异空间差异 → FWER不显著
3. 时空耦合（本实验）→ 从"空间差异"转向"时间结构耦合"

检验1（异常相关）：
- 去除季节循环和趋势后，对宫异常是否轴特异耦合
- 关键诊断：如果对宫异常相关不比随机对宫更极端，
  说明洛书轴不驱动年际耦合

检验2（六气共振）：
- 宫的五行对应之气是否有放大的年际变异
- 最直接检验"宫-气-五行"时空对应

检验3（相位对称性）：
- 季节循环峰值月是否在对宫之间符合物理预期
- 反相关轴→峰值差≈6月；同相关轴→峰值同月
""")

if fwer_p < 0.05:
    print(f"结论：时空耦合发现FWER显著信号 → 洛书约束有时间维度物理基础")
elif best_p < 0.05:
    print(f"结论：有边际信号(最佳p={best_p:.4f})但FWER不显著(p={fwer_p:.4f})")
else:
    print(f"结论：时空耦合仍未发现显著信号(最佳p={best_p:.4f}, FWER p={fwer_p:.4f})")
    print("  四轮实验全阴性闭环：空间均值→轴特异→时空耦合→均不显著")
    print("  可能结论：洛书约束在月均气候层面无可检测物理效应")

print("\n实验完成")
