"""
洛书轴特异实验 v3：高效版
先聚合到(year, qi, palace)小表，再置换检验
"""

import numpy as np
import pandas as pd
import netCDF4 as nc
from scipy import stats
import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 编码体系
# ============================================================
QI_NAMES = ['初之气', '二之气', '三之气', '四之气', '五之气', '终之气']
MONTH_TO_QI = {1:0, 2:0, 3:1, 4:1, 5:2, 6:2, 7:3, 8:3, 9:4, 10:4, 11:5, 12:5}
PALACE_MAP = {
    ('上','左'): 6, ('上','中'): 1, ('上','右'): 8,
    ('中','左'): 7, ('中','中'): 5, ('中','右'): 3,
    ('下','左'): 2, ('下','中'): 9, ('下','右'): 4,
}
PALACE_NAMES = {1:'坎(北)', 2:'坤(西南)', 3:'震(东)', 4:'巽(东南)',
                5:'中', 6:'乾(西北)', 7:'兑(西)', 8:'艮(东北)', 9:'离(南)'}

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
# 1. 提取各变量并聚合到 (year, month, palace) 级别
# ============================================================
print("=" * 70)
print("洛书轴特异实验 v3 (高效版)")
print("=" * 70)

def extract_local(varname, filepath):
    """提取本地nc并聚合到宫级别"""
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

# 提取所有本地变量
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
    print(f"OK ({len(dfs[varname])})")

# 逐个merge
df = dfs['shtfl']
for varname in ['lhtfl', 'wspd', 'rhum', 'air']:
    df = df.merge(dfs[varname], on=['year','month','palace'], how='outer')
del dfs

# OPeNDAP变量：用xarray提取中国区域子集
import xarray as xr

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
        lat = ds.lat.values
        lon = ds.lon.values
        lat_mask = (lat >= 22.5) & (lat <= 42.5)
        lon_mask = (lon >= 97.5) & (lon <= 122.5)
        lat_sel = lat[lat_mask]
        lon_sel = lon[lon_mask]
        lat_idx = np.where(lat_mask)[0]
        lon_idx = np.where(lon_mask)[0]
        data = ds[varname].isel(lat=lat_idx, lon=lon_idx).values
        time_var = ds.time.values
        ds.close()
        
        records = []
        for li, la in enumerate(lat_sel):
            for lj, lo in enumerate(lon_sel):
                row = lat_to_row(float(la))
                col = lon_to_col(float(lo))
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

print(f"\n合并后: {len(df)} rows, 变量: {[c for c in df.columns if c not in ['year','month','palace']]}")

# 添加六气和派生变量
df['qi'] = df['month'].map(MONTH_TO_QI)
df['bowen'] = np.where(np.abs(df['lhtfl']) > 1, df['shtfl'] / df['lhtfl'], np.nan)
df['dtr'] = df['tmax'] - df['tmin']
df['wind_humid'] = df['wspd'] * df['rhum']

# ============================================================
# 2. 聚合到 (year, qi, palace) 级别
# ============================================================
df_yq = df.groupby(['year', 'qi', 'palace']).agg({
    'shtfl': 'mean', 'lhtfl': 'mean', 'bowen': 'mean',
    'dtr': 'mean', 'tcdc': 'mean', 'gflux': 'mean',
    'rhum': 'mean', 'wspd': 'mean', 'wind_humid': 'mean',
    'air': 'mean'
}).reset_index()

print(f"聚合后: {len(df_yq)} rows (expect ~78yr×6qi×9palace={78*6*9})")

# ============================================================
# 3. 构建快查矩阵：变量 × (year, qi) × palace
# ============================================================
# 为了permutation高效，构建字典：
# metric_data[metric][(year, qi)] = {palace: value}

metric_data = {}
for metric in ['bowen', 'dtr', 'tcdc', 'gflux', 'rhum', 'wind_humid', 'shtfl', 'lhtfl', 'wspd', 'air']:
    sub = df_yq[['year','qi','palace',metric]].dropna()
    metric_data[metric] = {}
    for (y, q), g in sub.groupby(['year','qi']):
        metric_data[metric][(y, q)] = dict(zip(g['palace'], g[metric]))

# ============================================================
# 4. 高效Permutation Test
# ============================================================
N_PERM = 10000
np.random.seed(42)

def fast_axis_test(metric, palace_a, palace_b, n_perm=N_PERM):
    """
    快速permutation test：基于预构建的metric_data
    统计量：mean(B-A) across (year, qi) pairs
    """
    data = metric_data.get(metric, {})
    if not data:
        return {}, {}
    
    # 观测值
    obs_diffs = []
    for key, pdict in data.items():
        if palace_a in pdict and palace_b in pdict:
            obs_diffs.append(pdict[palace_b] - pdict[palace_a])
    
    if len(obs_diffs) < 20:
        return {}, {}
    
    obs_mean = np.mean(obs_diffs)
    
    # Permutation: shuffle palace labels
    all_palaces = list(range(1, 10))
    
    # 预构建值数组
    keys = []
    vals_a = []
    vals_b = []
    for key, pdict in data.items():
        if palace_a in pdict and palace_b in pdict:
            keys.append(key)
            vals_a.append(pdict[palace_a])
            vals_b.append(pdict[palace_b])
    
    vals_a = np.array(vals_a)
    vals_b = np.array(vals_b)
    obs_stat = np.mean(vals_b - vals_a)
    
    # 构建所有9宫值矩阵: shape (n_pairs, 9)
    palace_vals = np.zeros((len(keys), 9))
    for i, key in enumerate(keys):
        pdict = data[key]
        for p in range(1, 10):
            if p in pdict:
                palace_vals[i, p-1] = pdict[p]
            else:
                palace_vals[i, p-1] = np.nan
    
    # Permutation: 随机选两个宫做差
    perm_stats = np.zeros(n_perm)
    for k in range(n_perm):
        idx_a = np.random.randint(0, 9)
        idx_b = np.random.randint(0, 9)
        while idx_b == idx_a:
            idx_b = np.random.randint(0, 9)
        diff = palace_vals[:, idx_b] - palace_vals[:, idx_a]
        valid = ~np.isnan(diff)
        if valid.sum() > 20:
            perm_stats[k] = np.mean(diff[valid])
        else:
            perm_stats[k] = 0
    
    # 双侧p值
    p_val = np.mean(np.abs(perm_stats) >= np.abs(obs_stat))
    p_val = max(min(p_val, 1.0), 1.0/n_perm)
    
    return obs_stat, p_val, len(obs_diffs)

def fast_axis_test_by_qi(metric, palace_a, palace_b, n_perm=N_PERM):
    """按六气分组做permutation test"""
    data = metric_data.get(metric, {})
    if not data:
        return {}, {}
    
    results = {}
    for qi in range(6):
        # 筛选该气
        obs_diffs = []
        palace_vals_list = []
        keys_qi = []
        
        for (y, q), pdict in data.items():
            if q != qi: continue
            if palace_a in pdict and palace_b in pdict:
                obs_diffs.append(pdict[palace_b] - pdict[palace_a])
                keys_qi.append((y, q))
                pvals = []
                for p in range(1, 10):
                    pvals.append(pdict.get(p, np.nan))
                palace_vals_list.append(pvals)
        
        if len(obs_diffs) < 10:
            continue
        
        obs_stat = np.mean(obs_diffs)
        palace_vals = np.array(palace_vals_list)  # (n_years, 9)
        
        perm_stats = np.zeros(n_perm)
        for k in range(n_perm):
            idx_a = np.random.randint(0, 9)
            idx_b = np.random.randint(0, 9)
            while idx_b == idx_a:
                idx_b = np.random.randint(0, 9)
            diff = palace_vals[:, idx_b] - palace_vals[:, idx_a]
            valid = ~np.isnan(diff)
            if valid.sum() > 5:
                perm_stats[k] = np.mean(diff[valid])
            else:
                perm_stats[k] = 0
        
        p_val = np.mean(np.abs(perm_stats) >= np.abs(obs_stat))
        p_val = max(min(p_val, 1.0), 1.0/n_perm)
        results[qi] = (obs_stat, p_val, len(obs_diffs))
    
    return results

# ============================================================
# 5. 运行检验
# ============================================================
print("\n" + "=" * 70)
print("轴特异检验")
print("=" * 70)

all_results = {}

# --- 全景：各宫物理量 ---
print("\n各宫物理量:")
print(f"{'宫':>3} {'名':>8} {'β':>7} {'shtfl':>8} {'lhtfl':>8} {'DTR':>6} {'tcdc%':>6} {'rhum%':>6} {'gflux':>7}")
print("-" * 70)
for p in range(1, 10):
    sub = df_yq[df_yq['palace'] == p]
    if len(sub) == 0: continue
    print(f" {p}  {PALACE_NAMES[p]:>8} {sub['bowen'].mean():7.3f} "
          f"{sub['shtfl'].mean():8.1f} {sub['lhtfl'].mean():8.1f} "
          f"{sub['dtr'].mean():6.2f} {sub['tcdc'].mean():6.1f} "
          f"{sub['rhum'].mean():6.1f} {sub['gflux'].mean():7.2f}")

# --- 水火轴 (1↔9): Bowen ratio ---
print("\n" + "-" * 50)
print("水火轴 (1↔9坎离): Bowen ratio 方向性")
print("预期：离9(南/火)β>坎1(北/水)β → Δβ>0 (但可能是纬度效应)")
print("-" * 50)

obs, pval, n = fast_axis_test('bowen', 1, 9)
print(f"  全局: Δβ={obs:+.4f}, p={pval:.4f}, n={n}")
all_results[('水火_bowen_19', 'all')] = pval

print("\n  按六气:")
res_wf = fast_axis_test_by_qi('bowen', 1, 9)
for qi in sorted(res_wf.keys()):
    obs_qi, p_qi, n_qi = res_wf[qi]
    sig = '***' if p_qi < 0.001 else '**' if p_qi < 0.01 else '*' if p_qi < 0.05 else ''
    print(f"    {QI_NAMES[qi]}: Δβ={obs_qi:+.4f}, p={p_qi:.4f} {sig}")
    all_results[('水火_bowen_19', qi)] = p_qi

# 同纬度3↔7 Bowen ratio (消除纬度混淆)
print("\n  同纬度对宫3(震/东)↔7(兑/西) Bowen ratio:")
obs37, p37, n37 = fast_axis_test('bowen', 3, 7)
print(f"  全局: Δβ={obs37:+.4f}, p={p37:.4f}, n={n37}")
all_results[('水火_bowen_37', 'all')] = p37

# --- 木金轴 (3↔7): rhum + wind_humid ---
print("\n" + "-" * 50)
print("木金轴 (3↔7震兑): 湿度+风湿耦合")
print("预期：震3(东/沿海)湿, 兑7(西/内陆)干")
print("-" * 50)

obs_rh, p_rh, n_rh = fast_axis_test('rhum', 3, 7)
print(f"  rhum 7-3: {obs_rh:+.3f}%, p={p_rh:.4f}, n={n_rh}")
all_results[('木金_rhum_37', 'all')] = p_rh

obs_wh, p_wh, n_wh = fast_axis_test('wind_humid', 3, 7)
print(f"  wind_humid 7-3: {obs_wh:+.4f}, p={p_wh:.4f}, n={n_wh}")
all_results[('木金_wh_37', 'all')] = p_wh

print("\n  rhum 3↔7 按六气:")
res_rh = fast_axis_test_by_qi('rhum', 3, 7)
for qi in sorted(res_rh.keys()):
    obs_qi, p_qi, n_qi = res_rh[qi]
    sig = '***' if p_qi < 0.001 else '**' if p_qi < 0.01 else '*' if p_qi < 0.05 else ''
    print(f"    {QI_NAMES[qi]}: Δrhum={obs_qi:+.3f}, p={p_qi:.4f} {sig}")
    all_results[('木金_rhum_37', qi)] = p_qi

# --- 坤艮轴 (2↔8): DTR + tcdc ---
print("\n" + "-" * 50)
print("坤艮轴 (2↔8): DTR缓冲 + 云量")
print("预期：土行'匀和'→ DTR小, tcdc大")
print("-" * 50)

obs_dtr, p_dtr, n_dtr = fast_axis_test('dtr', 2, 8)
print(f"  DTR 8-2: {obs_dtr:+.3f}°C, p={p_dtr:.4f}, n={n_dtr}")
all_results[('坤艮_dtr_28', 'all')] = p_dtr

obs_tc, p_tc, n_tc = fast_axis_test('tcdc', 2, 8)
print(f"  tcdc 8-2: {obs_tc:+.2f}%, p={p_tc:.4f}, n={n_tc}")
all_results[('坤艮_tcdc_28', 'all')] = p_tc

# 土行vs非土行
earth = [2, 5, 8]
non_earth = [1, 3, 4, 6, 7, 9]
print("\n  土行(2,5,8) vs 非土行 DTR (配对t检验):")
for qi in range(6):
    sub = df_yq[df_yq['qi'] == qi]
    ye = sub[sub['palace'].isin(earth)].groupby('year')['dtr'].mean()
    yn = sub[sub['palace'].isin(non_earth)].groupby('year')['dtr'].mean()
    common = ye.index.intersection(yn.index)
    if len(common) < 10: continue
    t, p = stats.ttest_rel(ye[common], yn[common])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"    {QI_NAMES[qi]}: Δ={((ye[common]-yn[common]).mean()):+.3f}, t={t:.3f}, p={p:.4f} {sig}")

# --- 巽乾轴 (4↔6): gflux ---
print("\n" + "-" * 50)
print("巽乾轴 (4↔6): 地热通量")
print("-" * 50)

obs_gf, p_gf, n_gf = fast_axis_test('gflux', 4, 6)
print(f"  gflux 6-4: {obs_gf:+.3f}, p={p_gf:.4f}, n={n_gf}")
all_results[('巽乾_gflux_46', 'all')] = p_gf

res_gf = fast_axis_test_by_qi('gflux', 4, 6)
for qi in sorted(res_gf.keys()):
    obs_qi, p_qi, n_qi = res_gf[qi]
    sig = '***' if p_qi < 0.001 else '**' if p_qi < 0.01 else '*' if p_qi < 0.05 else ''
    print(f"    {QI_NAMES[qi]}: Δgflux={obs_qi:+.3f}, p={p_qi:.4f} {sig}")
    all_results[('巽乾_gflux_46', qi)] = p_qi

# ============================================================
# 6. 综合评估
# ============================================================
print("\n" + "=" * 70)
print("综合评估")
print("=" * 70)

n_tests = len(all_results)
best_p = min(all_results.values()) if all_results else 1.0
fwer_p = min(best_p * n_tests, 1.0)

print(f"  总检验数: {n_tests}")
print(f"  最佳原始p值: {best_p:.4f}")
print(f"  FWER校正后: {fwer_p:.4f}")

sig_raw = [(k, v) for k, v in all_results.items() if v < 0.05]
print(f"\n  原始p<0.05 ({len(sig_raw)}/{n_tests}):")
for (test, qi), p in sorted(sig_raw, key=lambda x: x[1]):
    qi_name = QI_NAMES[qi] if isinstance(qi, int) else '全局'
    print(f"    {test} {qi_name}: p={p:.4f}")

# ============================================================
# 7. 物理解读
# ============================================================
print("\n" + "=" * 70)
print("物理解读")
print("=" * 70)

print("""
核心方法论转变的验证：

旧方法：所有轴用同一套变量(T,rhum,prate,dswrf)测偏相关符号反号
→ 全阴性（p>0.2）

新方法：每轴用物理匹配的变量
- 水火轴：Bowen ratio (shtfl/lhtfl) 量度垂直热力结构
- 木金轴：湿度+风 量度水平水汽-风场耦合
- 坤艮轴：DTR+云量 量度缓冲/转化能力
- 巽乾轴：地热通量 量度收敛方向能量

关键诊断：
1. Bowen ratio 1↔9 的信号是否纯粹是纬度效应？
   → 同纬度3↔7 Bowen ratio 差异是真正的经度(海陆)效应
   → 如果3↔7也有显著信号，说明不是纯纬度混淆

2. 湿度3↔7 是木金轴最直接的物理体现：
   → 震3(中国东部沿海)确实湿度更高
   → 兑7(中国西部内陆)确实更干燥
   → 这是海陆分布的物理事实，不是洛书独有的
   → 但如果permutation不显著，说明9宫内其他对也有类似差异
""")

# 最终判定
if fwer_p < 0.05:
    print("结论：轴特异方法发现FWER显著信号 → 洛书约束有物理基础")
elif best_p < 0.05:
    print(f"结论：有边际信号(最佳p={best_p:.4f})但FWER校正后不显著")
else:
    print(f"结论：轴特异方法仍未发现显著信号(最佳p={best_p:.4f})")
    print("  可能原因：1)洛书约束在月均气候层面确实没有信号")
    print("  2)9宫空间分辨率太粗，信号被稀释")
    print("  3)需要更高维度(垂直层次/日变化)的数据")

print("\n实验完成")
