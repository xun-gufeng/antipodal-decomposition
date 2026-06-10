"""
Z₅季节响应量子化验证实验
========================

核心假设：Z₅分组对九宫气候变量的季节响应（振幅/相位/波形）施加约束，
使响应空间被切割为5个离散允许模式，而非连续分布。

与旧框架的关键区别：
- 不去季节！季节循环是信号，不是噪声
- 不去纬度！纬度是洛书九宫的内嵌结构，不是混淆变量
- 测量对象：季节响应参数的Z₅分组约束，而非"涌现时间量子τ"

实验设计：
Test 1: Z₅组内季节波形相似性（置换检验）
Test 2: Z₅对季节振幅/相位的约束（置换检验）
Test 3: 跨纬度Z₅约束检验——火(2,7)和木(3,8)是关键
Test 4: 量子化间隙检验——响应参数是否聚类为5个离散模式
Test 5: 季节维度Z₅反相关方向性——Z₅约束是否随季节变化
"""

import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
from scipy import stats, optimize
from itertools import permutations
import datetime
import warnings
import json
warnings.filterwarnings('ignore')

# ============================================================
# 0. 常量与映射（使用修正后映射）
# ============================================================

# 修正后洛书映射：位置(row,col) → 宫号
# 上=北(高纬), 下=南(低纬), 左=西, 右=东（面南而立）
PALACE_MAP = {
    ('上','左'): 6, ('上','中'): 1, ('上','右'): 8,
    ('中','左'): 7, ('中','中'): 5, ('中','右'): 3,
    ('下','左'): 2, ('下','中'): 9, ('下','右'): 4,
}

# 修正后五行归属
PALACE_WUXING = {1:'水', 2:'火', 3:'木', 4:'金', 5:'土', 6:'水', 7:'火', 8:'木', 9:'金'}
PALACE_NAMES = {1:'坎(北)', 2:'坤(西南)', 3:'震(东)', 4:'巽(东南)',
                5:'中', 6:'乾(西北)', 7:'兑(西)', 8:'艮(东北)', 9:'离(南)'}

# Z₅五行分组
Z5_GROUPS = {'水':[1, 6], '火':[2, 7], '木':[3, 8], '金':[4, 9], '土':[5]}

# 各宫纬度
PALACE_LAT = {1:40.0, 2:22.5, 3:30.0, 4:22.5, 5:30.0, 6:40.0, 7:30.0, 8:40.0, 9:22.5}

# 洛书对宫
LUOSHU_AXIS_PAIRS = [(1,9), (2,8), (3,7), (4,6)]

DATA_DIR = './data/ncep/'

MONTH_TO_QI = {1:0, 2:0, 3:1, 4:1, 5:2, 6:2, 7:3, 8:3, 9:4, 10:4, 11:5, 12:5}
QI_NAMES = ['初之气', '二之气', '三之气', '四之气', '五之气', '终之气']

# 本地稳定变量
LOCAL_VARS = {
    'shtfl': f'{DATA_DIR}shtfl.sfc.mon.mean.nc',
    'lhtfl': f'{DATA_DIR}lhtfl.sfc.mon.mean.nc',
    'wspd': f'{DATA_DIR}wspd.mon.mean.nc',
    'rhum': f'{DATA_DIR}rhum.mon.mean.nc',
    'air': f'{DATA_DIR}air.mon.mean.nc',
}

N_PERM = 10000  # 置换检验次数

# ============================================================
# 1. 数据加载
# ============================================================

def lat_to_row(lat):
    if lat > 35: return '上'
    elif lat < 25: return '下'
    else: return '中'

def lon_to_col(lon):
    if lon < 107.5: return '左'
    elif lon > 112.5: return '右'
    else: return '中'

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

print("=" * 70)
print("Z₅季节响应量子化验证实验")
print("=" * 70)
print("\n加载数据...")

dfs = {}
for varname, filepath in LOCAL_VARS.items():
    print(f"  {varname}...", end=" ", flush=True)
    dfs[varname] = extract_local(varname, filepath)
    print("OK")

df = dfs['shtfl']
for varname in ['lhtfl', 'wspd', 'rhum', 'air']:
    df = df.merge(dfs[varname], on=['year','month','palace'], how='outer')
del dfs

# 派生变量
df['bowen'] = np.where(np.abs(df['lhtfl']) > 1, df['shtfl'] / df['lhtfl'], np.nan)

# 气变量
variables = ['shtfl', 'bowen', 'wspd', 'rhum', 'air']
print(f"数据: {len(df)} rows, 变量: {variables}")

# ============================================================
# 2. 季节循环提取
# ============================================================

print("\n" + "=" * 70)
print("季节循环提取：月气候态 + 正弦拟合")
print("=" * 70)

# 每宫×每变量：提取12月气候态
climatology = {}  # climatology[var][palace] = array(12)

for var in variables:
    climatology[var] = {}
    for p in range(1, 10):
        sub = df[(df['palace'] == p)][['month', var]].dropna()
        if len(sub) < 36: continue
        monthly = sub.groupby('month')[var].mean()
        if len(monthly) < 12: continue
        vals = monthly.reindex(range(1, 13)).values
        climatology[var][p] = vals

# 正弦拟合：y(t) = A * sin(2π*t/12 + φ) + C
# 参数：振幅A, 相位φ, 均值C
# 相位φ表示峰值出现的时间（月）

def fit_sin(monthly_vals):
    """拟合年周期正弦函数，返回(A, φ, C, R²)"""
    t = np.arange(12)
    y = monthly_vals.copy()
    valid = ~np.isnan(y)
    if valid.sum() < 8:
        return np.nan, np.nan, np.nan, np.nan
    
    C0 = np.nanmean(y)
    A0 = (np.nanmax(y) - np.nanmin(y)) / 2
    
    # 粗搜相位
    best_r2 = -np.inf
    best_phi = 0
    for phi_trial in np.linspace(0, 2*np.pi, 360):
        y_pred = A0 * np.sin(2*np.pi*t/12 + phi_trial) + C0
        ss_res = np.nansum((y - y_pred)**2)
        ss_tot = np.nansum((y - C0)**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        if r2 > best_r2:
            best_r2 = r2
            best_phi = phi_trial
    
    # 精细拟合
    try:
        def sin_model(t_arr, A, phi, C):
            return A * np.sin(2*np.pi*t_arr/12 + phi) + C
        
        popt, _ = optimize.curve_fit(sin_model, t[valid], y[valid],
                                      p0=[A0, best_phi, C0],
                                      maxfev=10000)
        A, phi, C = popt
        A = abs(A)  # 振幅取绝对值
        y_pred = sin_model(t, *popt)
        ss_res = np.sum((y[valid] - y_pred[valid])**2)
        ss_tot = np.sum((y[valid] - C)**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        return A, phi, C, r2
    except:
        return A0, best_phi, C0, best_r2

# 提取季节参数
seasonal_params = {}  # seasonal_params[var][palace] = {'A':, 'phi':, 'C':, 'r2':}

for var in variables:
    seasonal_params[var] = {}
    print(f"\n  {var} 季节拟合:")
    for p in range(1, 10):
        if p not in climatology[var]:
            continue
        A, phi, C, r2 = fit_sin(climatology[var][p])
        # 将相位转为峰值月（0-11 → 1-12月）
        peak_month = ((3 - phi/(2*np.pi)*12) % 12)
        if peak_month <= 0: peak_month += 12
        seasonal_params[var][p] = {'A': A, 'phi': phi, 'C': C, 'r2': r2, 'peak_month': peak_month}
        print(f"    宫{p} {PALACE_NAMES[p]:>8} ({PALACE_LAT[p]}°N, {PALACE_WUXING[p]}): "
              f"A={A:.3f}, 峰值月={peak_month:.1f}, R²={r2:.3f}")

# ============================================================
# Test 1: Z₅组内季节波形相似性（置换检验）
# ============================================================

print("\n" + "=" * 70)
print("Test 1: Z₅组内季节波形相似性")
print("=" * 70)
print("零假设：Z₅分组对季节波形无约束，组内相似度不高于随机分组")
print("检验：Z₅组内波形相关均值 vs 10000次随机分组分布\n")

def within_group_waveform_corr(clim_dict, grouping):
    """计算分组内所有配对的波形相关均值"""
    corrs = []
    for group_name, members in grouping.items():
        if len(members) < 2:
            continue
        for i, p1 in enumerate(members):
            for p2 in members[i+1:]:
                if p1 in clim_dict and p2 in clim_dict:
                    v1 = clim_dict[p1]
                    v2 = clim_dict[p2]
                    valid = ~(np.isnan(v1) | np.isnan(v2))
                    if valid.sum() >= 6:
                        r = np.corrcoef(v1[valid], v2[valid])[0, 1]
                        corrs.append(r)
    return np.mean(corrs) if corrs else np.nan

def random_grouping_structure():
    """生成与Z₅结构相同的随机分组（2,2,2,2,1）"""
    palaces = list(range(1, 10))
    np.random.shuffle(palaces)
    groups = {}
    group_names = ['G1', 'G2', 'G3', 'G4', 'G5']
    idx = 0
    for i, size in enumerate([2, 2, 2, 2, 1]):
        groups[group_names[i]] = palaces[idx:idx+size]
        idx += size
    return groups

t1_results = {}

for var in variables:
    # Z₅实际得分
    z5_score = within_group_waveform_corr(climatology[var], Z5_GROUPS)
    
    # 置换分布
    null_scores = []
    for _ in range(N_PERM):
        rg = random_grouping_structure()
        s = within_group_waveform_corr(climatology[var], rg)
        if not np.isnan(s):
            null_scores.append(s)
    
    null_scores = np.array(null_scores)
    p_value = np.mean(null_scores >= z5_score) if not np.isnan(z5_score) else np.nan
    
    t1_results[var] = {
        'z5_score': z5_score,
        'null_mean': np.mean(null_scores),
        'null_std': np.std(null_scores),
        'p_value': p_value,
        'z_score': (z5_score - np.mean(null_scores)) / np.std(null_scores) if np.std(null_scores) > 0 else 0
    }
    
    sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    print(f"  {var}: Z₅组内相关={z5_score:.4f}, 随机均值={np.mean(null_scores):.4f}±{np.std(null_scores):.4f}, "
          f"p={p_value:.4f} {sig}")

# ============================================================
# Test 2: Z₅对季节振幅/相位的约束（置换检验）
# ============================================================

print("\n" + "=" * 70)
print("Test 2: Z₅对季节振幅/相位的约束")
print("=" * 70)
print("检验：Z₅组内振幅/相位方差是否小于随机分组\n")

def within_group_param_variance(params_dict, param_key, grouping):
    """计算分组内参数的组内方差均值"""
    variances = []
    for group_name, members in grouping.items():
        if len(members) < 2:
            continue
        vals = [params_dict[p][param_key] for p in members if p in params_dict]
        if len(vals) >= 2:
            variances.append(np.var(vals))
    return np.mean(variances) if variances else np.nan

def within_group_circular_variance(params_dict, grouping):
    """计算分组内相位的圆方差均值"""
    circ_vars = []
    for group_name, members in grouping.items():
        if len(members) < 2:
            continue
        phis = [params_dict[p]['phi'] for p in members if p in params_dict]
        if len(phis) >= 2:
            # 圆方差 = 1 - R, R = |mean resultant|
            R = abs(np.mean(np.exp(1j * np.array(phis))))
            circ_vars.append(1 - R)
    return np.mean(circ_vars) if circ_vars else np.nan

t2_results = {}

for var in variables:
    params = seasonal_params[var]
    
    # 振幅约束
    z5_amp_var = within_group_param_variance(params, 'A', Z5_GROUPS)
    null_amp_vars = []
    for _ in range(N_PERM):
        rg = random_grouping_structure()
        v = within_group_param_variance(params, 'A', rg)
        if not np.isnan(v):
            null_amp_vars.append(v)
    null_amp_vars = np.array(null_amp_vars)
    amp_p = np.mean(null_amp_vars <= z5_amp_var) if not np.isnan(z5_amp_var) else np.nan
    
    # 相位约束（圆方差）
    z5_circ_var = within_group_circular_variance(params, Z5_GROUPS)
    null_circ_vars = []
    for _ in range(N_PERM):
        rg = random_grouping_structure()
        v = within_group_circular_variance(params, rg)
        if not np.isnan(v):
            null_circ_vars.append(v)
    null_circ_vars = np.array(null_circ_vars)
    phi_p = np.mean(null_circ_vars <= z5_circ_var) if not np.isnan(z5_circ_var) else np.nan
    
    t2_results[var] = {
        'amp_z5_var': z5_amp_var, 'amp_null_mean': np.mean(null_amp_vars),
        'amp_p': amp_p,
        'phi_z5_circvar': z5_circ_var, 'phi_null_mean': np.mean(null_circ_vars),
        'phi_p': phi_p
    }
    
    amp_sig = '***' if amp_p < 0.001 else '**' if amp_p < 0.01 else '*' if amp_p < 0.05 else 'ns'
    phi_sig = '***' if phi_p < 0.001 else '**' if phi_p < 0.01 else '*' if phi_p < 0.05 else 'ns'
    print(f"  {var}:")
    print(f"    振幅: Z₅组内方差={z5_amp_var:.4f}, 随机均值={np.mean(null_amp_vars):.4f}, "
          f"p={amp_p:.4f} {amp_sig} (越小越好)")
    print(f"    相位: Z₅组内圆方差={z5_circ_var:.4f}, 随机均值={np.mean(null_circ_vars):.4f}, "
          f"p={phi_p:.4f} {phi_sig} (越小越好)")

# ============================================================
# Test 3: 跨纬度Z₅约束检验（关键测试）
# ============================================================

print("\n" + "=" * 70)
print("Test 3: 跨纬度Z₅约束检验")
print("=" * 70)
print("关键：火(2,7)跨越22.5°N↔30°N, 木(3,8)跨越30°N↔40°N")
print("如果Z₅约束超越纬度，这些跨纬度对应该比纬度预测更相似")
print("零假设：跨纬度对的相似度=同纬度随机配对的相似度\n")

# 定义所有跨纬度配对（Z₅预测应相似）
z5_cross_lat = {
    '火(2,7)': (2, 7),  # 22.5°N ↔ 30°N
    '木(3,8)': (3, 8),  # 30°N ↔ 40°N
}

# 同纬度配对（对照）
same_lat_pairs = {
    '1,6(40°N)': (1, 6),  # Z₅同组(水)
    '4,9(22.5°N)': (4, 9),  # Z₅同组(金)
    '1,8(40°N)': (1, 8),  # Z₅不同组(水/木)
    '6,8(40°N)': (6, 8),  # Z₅不同组(水/木)
    '2,4(22.5°N)': (2, 4),  # Z₅不同组(火/金)
    '4,9同纬': (4, 9),  # Z₅同组
    '3,7(30°N)': (3, 7),  # Z₅不同组(木/火)
    '5,7(30°N)': (5, 7),  # Z₅不同组(土/火)
    '3,5(30°N)': (3, 5),  # Z₅不同组(木/土)
}

# 更精确的对照：Z₅不同组但纬度差相同的配对
# 火(2,7): 22.5°N ↔ 30°N, 纬度差=7.5° → 对照：(2,3), (2,5), (4,3), (4,5), (9,3), (9,5), (9,7)
# 木(3,8): 30°N ↔ 40°N, 纬度差=10° → 对照：(3,1), (3,6), (5,1), (5,6), (7,1), (7,6), (7,8)

z5_cross_lat_controls = {
    '火(2,7)': {
        'pair': (2, 7),
        'lat_diff': abs(PALACE_LAT[2] - PALACE_LAT[7]),
        'z5_same': True,
        # 对照：纬度差≈7.5°, Z₅不同组的配对
        'controls': [(2,3), (2,5), (4,3), (4,5), (9,3), (9,5), (9,7)]
    },
    '木(3,8)': {
        'pair': (3, 8),
        'lat_diff': abs(PALACE_LAT[3] - PALACE_LAT[8]),
        'z5_same': True,
        # 对照：纬度差=10°, Z₅不同组的配对
        'controls': [(3,1), (3,6), (5,1), (5,6), (7,1), (7,6), (7,8)]
    }
}

t3_results = {}

for var in variables:
    print(f"\n  {var}:")
    t3_results[var] = {}
    
    for name, info in z5_cross_lat_controls.items():
        p1, p2 = info['pair']
        
        # Z₅跨纬度对的波形相关
        if p1 in climatology[var] and p2 in climatology[var]:
            v1, v2 = climatology[var][p1], climatology[var][p2]
            valid = ~(np.isnan(v1) | np.isnan(v2))
            z5_r = np.corrcoef(v1[valid], v2[valid])[0, 1] if valid.sum() >= 6 else np.nan
        else:
            z5_r = np.nan
        
        # 对照组波形相关
        ctrl_rs = []
        for cp1, cp2 in info['controls']:
            if cp1 in climatology[var] and cp2 in climatology[var]:
                v1, v2 = climatology[var][cp1], climatology[var][cp2]
                valid = ~(np.isnan(v1) | np.isnan(v2))
                if valid.sum() >= 6:
                    r = np.corrcoef(v1[valid], v2[valid])[0, 1]
                    ctrl_rs.append(r)
        
        ctrl_rs = np.array(ctrl_rs)
        ctrl_mean = np.mean(ctrl_rs) if len(ctrl_rs) > 0 else np.nan
        
        # Z₅对是否比对照组更相似（|r|更大）
        if not np.isnan(z5_r) and len(ctrl_rs) > 0:
            # 单边检验：Z₅对|r| > 对照|r|
            p_val = np.mean(np.abs(ctrl_rs) >= abs(z5_r))
        else:
            p_val = np.nan
        
        t3_results[var][name] = {
            'z5_r': z5_r,
            'ctrl_mean_r': ctrl_mean,
            'ctrl_rs': ctrl_rs.tolist(),
            'p_value': p_val
        }
        
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f"    {name}: Z₅对r={z5_r:+.4f}, 对照|r|均值={np.mean(np.abs(ctrl_rs)):.4f}, "
              f"p={p_val:.4f} {sig}")

# ============================================================
# Test 4: 量子化间隙检验
# ============================================================

print("\n" + "=" * 70)
print("Test 4: 量子化间隙检验")
print("=" * 70)
print("检验：季节响应参数是否聚类为5个离散模式（Z₅量子化）")
print("方法：Z₅分组silhouette分数 vs 纬度分组 vs 随机分组\n")

from sklearn.metrics import silhouette_score

def compute_silhouette(feature_matrix, labels):
    """计算silhouette分数"""
    if len(set(labels)) < 2:
        return np.nan
    try:
        return silhouette_score(feature_matrix, labels)
    except:
        return np.nan

# 为每宫构建特征向量：[A, peak_month, C, 波形PC1, 波形PC2]
t4_results = {}

for var in variables:
    print(f"\n  {var}:")
    
    # 构建特征矩阵
    palaces_available = [p for p in range(1, 10) if p in seasonal_params[var] and p in climatology[var]]
    if len(palaces_available) < 5:
        print("    数据不足，跳过")
        continue
    
    # 特征1：季节参数 (A, peak_month, C)
    param_features = []
    for p in palaces_available:
        sp = seasonal_params[var][p]
        param_features.append([sp['A'], sp['peak_month'], sp['C']])
    param_features = np.array(param_features)
    
    # 标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    param_features_scaled = scaler.fit_transform(param_features)
    
    # 特征2：波形PC
    waveforms = np.array([climatology[var][p] for p in palaces_available])
    # 标准化每个波形
    waveforms_norm = np.array([(w - np.nanmean(w))/np.nanstd(w) if np.nanstd(w) > 0 else w - np.nanmean(w) 
                                for w in waveforms])
    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(3, waveforms_norm.shape[1]))
    wave_pcs = pca.fit_transform(np.nan_to_num(waveforms_norm))
    
    # 合并特征
    features = np.hstack([param_features_scaled, wave_pcs])
    
    # Z₅分组标签
    z5_labels = np.array([list(Z5_GROUPS.keys()).index(PALACE_WUXING[p]) for p in palaces_available])
    
    # 纬度分组标签（按纬度聚类）
    lat_labels = np.array([0 if PALACE_LAT[p] < 25 else 1 if PALACE_LAT[p] > 35 else 2 
                           for p in palaces_available])
    
    # Z₅ silhouette
    z5_sil = compute_silhouette(features, z5_labels)
    lat_sil = compute_silhouette(features, lat_labels)
    
    # 随机分组silhouette分布
    rand_sils = []
    for _ in range(N_PERM):
        rand_labels = z5_labels.copy()
        np.random.shuffle(rand_labels)
        s = compute_silhouette(features, rand_labels)
        if not np.isnan(s):
            rand_sils.append(s)
    rand_sils = np.array(rand_sils)
    
    z5_p = np.mean(rand_sils >= z5_sil) if not np.isnan(z5_sil) else np.nan
    
    t4_results[var] = {
        'z5_silhouette': z5_sil,
        'lat_silhouette': lat_sil,
        'rand_mean': np.mean(rand_sils),
        'rand_std': np.std(rand_sils),
        'z5_p': z5_p
    }
    
    z5_sig = '***' if z5_p < 0.001 else '**' if z5_p < 0.01 else '*' if z5_p < 0.05 else 'ns'
    print(f"    Z₅ silhouette={z5_sil:.4f} (p={z5_p:.4f} {z5_sig})")
    print(f"    纬度 silhouette={lat_sil:.4f}")
    print(f"    随机均值={np.mean(rand_sils):.4f}±{np.std(rand_sils):.4f}")
    
    # 量子化判断
    if z5_sil > 0 and z5_p < 0.05:
        if lat_sil > z5_sil:
            print(f"    → Z₅有量子化信号，但纬度解释力更强（纬度>Z₅）")
        else:
            print(f"    → Z₅量子化信号显著，且超越纬度预测！")
    else:
        print(f"    → Z₅量子化信号不显著")

# ============================================================
# Test 5: 季节维度Z₅反相关方向性
# ============================================================

print("\n" + "=" * 70)
print("Test 5: Z₅反相关方向性的季节变化")
print("=" * 70)
print("检验：Z₅分组内的反相关方向性是否随季节(六气)变化")
print("如果Z₅量子化了响应模式，不同季节的约束强度应有差异\n")

t5_results = {}

for var in variables:
    print(f"\n  {var}:")
    t5_results[var] = {}
    
    for qi_idx in range(6):
        qi_months = [m for m, q in MONTH_TO_QI.items() if q == qi_idx]
        
        # 提取该气的气候值
        qi_clim = {}
        for p in range(1, 10):
            if p not in climatology[var]:
                continue
            qi_vals = climatology[var][p][[m-1 for m in qi_months]]
            qi_clim[p] = qi_vals
        
        if len(qi_clim) < 4:
            continue
        
        # Z₅组内相关（该气）
        z5_corrs = []
        for wx, members in Z5_GROUPS.items():
            if len(members) < 2:
                continue
            for i, p1 in enumerate(members):
                for p2 in members[i+1:]:
                    if p1 in qi_clim and p2 in qi_clim:
                        v1, v2 = qi_clim[p1], qi_clim[p2]
                        valid = ~(np.isnan(v1) | np.isnan(v2))
                        if valid.sum() >= 2:
                            r = np.corrcoef(v1[valid], v2[valid])[0, 1]
                            z5_corrs.append(r)
        
        mean_r = np.mean(z5_corrs) if z5_corrs else np.nan
        
        # 置换检验
        null_rs = []
        for _ in range(5000):  # 减少计算量
            rg = random_grouping_structure()
            r_corrs = []
            for gn, members in rg.items():
                if len(members) < 2: continue
                for i, p1 in enumerate(members):
                    for p2 in members[i+1:]:
                        if p1 in qi_clim and p2 in qi_clim:
                            v1, v2 = qi_clim[p1], qi_clim[p2]
                            valid = ~(np.isnan(v1) | np.isnan(v2))
                            if valid.sum() >= 2:
                                r = np.corrcoef(v1[valid], v2[valid])[0, 1]
                                r_corrs.append(r)
            if r_corrs:
                null_rs.append(np.mean(r_corrs))
        
        null_rs = np.array(null_rs)
        if not np.isnan(mean_r) and len(null_rs) > 0:
            p_val = np.mean(null_rs >= mean_r)
        else:
            p_val = np.nan
        
        t5_results[var][qi_idx] = {'mean_r': mean_r, 'p': p_val, 'null_mean': np.mean(null_rs)}
        
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f"    {QI_NAMES[qi_idx]}: Z₅组内r={mean_r:+.4f}, 随机均值={np.mean(null_rs):.4f}, "
              f"p={p_val:.4f} {sig}")

# ============================================================
# 综合评估
# ============================================================

print("\n" + "=" * 70)
print("综合评估：Z₅季节响应量子化")
print("=" * 70)

# 统计各测试的显著性
print("\n各变量在各测试中的表现：")
print(f"{'变量':>8} | {'T1波形相似':>10} | {'T2振幅约束':>10} | {'T2相位约束':>10} | {'T3跨纬度':>10} | {'T4量子化':>10}")
print("-" * 70)

for var in variables:
    t1_sig = '***' if t1_results[var]['p_value'] < 0.001 else '**' if t1_results[var]['p_value'] < 0.01 else '*' if t1_results[var]['p_value'] < 0.05 else 'ns'
    t2_amp_sig = '***' if t2_results[var]['amp_p'] < 0.001 else '**' if t2_results[var]['amp_p'] < 0.01 else '*' if t2_results[var]['amp_p'] < 0.05 else 'ns'
    t2_phi_sig = '***' if t2_results[var]['phi_p'] < 0.001 else '**' if t2_results[var]['phi_p'] < 0.01 else '*' if t2_results[var]['phi_p'] < 0.05 else 'ns'
    
    # T3: 取最显著的跨纬度对
    t3_best_p = min([v['p_value'] for v in t3_results[var].values()] + [1.0])
    t3_sig = '***' if t3_best_p < 0.001 else '**' if t3_best_p < 0.01 else '*' if t3_best_p < 0.05 else 'ns'
    
    # T4
    t4_sig = '***' if t4_results.get(var, {}).get('z5_p', 1.0) < 0.001 else '**' if t4_results.get(var, {}).get('z5_p', 1.0) < 0.01 else '*' if t4_results.get(var, {}).get('z5_p', 1.0) < 0.05 else 'ns'
    
    print(f"{var:>8} | {t1_sig:>10} | {t2_amp_sig:>10} | {t2_phi_sig:>10} | {t3_sig:>10} | {t4_sig:>10}")

print("\n" + "=" * 70)
print("物理解读")
print("=" * 70)

# 检查跨纬度Z₅对的详细结果
print("\n关键：跨纬度Z₅对(火2↔7, 木3↔8)的季节响应相似性")
print("如果这些跨纬度对比同纬度差但Z₅不同的对照对更相似，")
print("说明Z₅约束超越了纬度预测——这是量子化的最强证据\n")

for var in variables:
    print(f"  {var}:")
    for name, info in z5_cross_lat_controls.items():
        r = t3_results[var][name]
        ctrl_abs_mean = np.mean(np.abs(r['ctrl_rs'])) if r['ctrl_rs'] else np.nan
        print(f"    {name}: Z₅对|r|={abs(r['z5_r']):.4f}, 对照|r|均值={ctrl_abs_mean:.4f}, "
              f"差值={abs(r['z5_r'])-ctrl_abs_mean:+.4f}")

# 检查T5：季节约束变化
print("\nZ₅约束的季节变化（Test 5显著变量）：")
seasonal_variation_found = False
for var in variables:
    qi_ps = [t5_results[var].get(qi, {}).get('p', 1.0) for qi in range(6)]
    sig_qis = [QI_NAMES[qi] for qi, p in enumerate(qi_ps) if p < 0.05]
    if sig_qis:
        seasonal_variation_found = True
        print(f"  {var}: Z₅约束在{', '.join(sig_qis)}显著")

if not seasonal_variation_found:
    print("  无变量在特定季节显示显著Z₅约束")

print("\n" + "=" * 70)
print("实验完成")
print("=" * 70)

# 保存结果
results_summary = {
    'test1_waveform_similarity': {var: {'z5_score': float(r['z5_score']) if not np.isnan(r['z5_score']) else None,
                                         'p_value': float(r['p_value']) if not np.isnan(r['p_value']) else None}
                                  for var, r in t1_results.items()},
    'test2_amplitude_constraint': {var: {'p_value': float(r['amp_p']) if not np.isnan(r['amp_p']) else None}
                                   for var, r in t2_results.items()},
    'test2_phase_constraint': {var: {'p_value': float(r['phi_p']) if not np.isnan(r['phi_p']) else None}
                               for var, r in t2_results.items()},
    'test3_cross_latitude': {var: {name: {'z5_r': float(r['z5_r']) if not np.isnan(r['z5_r']) else None,
                                          'p_value': float(r['p_value']) if not np.isnan(r['p_value']) else None}
                                   for name, r in var_results.items()}
                             for var, var_results in t3_results.items()},
    'test4_quantization': {var: {'z5_silhouette': float(r['z5_silhouette']) if not np.isnan(r.get('z5_silhouette', np.nan)) else None,
                                  'z5_p': float(r['z5_p']) if not np.isnan(r.get('z5_p', np.nan)) else None,
                                  'lat_silhouette': float(r['lat_silhouette']) if not np.isnan(r.get('lat_silhouette', np.nan)) else None}
                           for var, r in t4_results.items()},
    'test5_seasonal_variation': {var: {str(qi): {'mean_r': float(r['mean_r']) if not np.isnan(r.get('mean_r', np.nan)) else None,
                                                  'p': float(r['p']) if not np.isnan(r.get('p', np.nan)) else None}
                                       for qi, r in var_results.items()}
                                 for var, var_results in t5_results.items()},
}

with open('./seasonal_quantization_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)

print("结果已保存: seasonal_quantization_results.json")
