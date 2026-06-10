#!/usr/bin/env python3
"""
实验E2: 下临宫旋转Permutation检验

核心思路: 下临宫每年轮转到不同宫，信号不在单个宫的时序里，
而在"正确的旋转映射是否比随机映射更准"。

方法:
1. 用实际司天→下临宫映射，计算方向一致率 (observed)
2. 随机打乱司天标签(保持每年气候数据不变)，重算方向一致率 (null)
3. 重复10000次，得null分布
4. observed落在null分布的位置 = p值

这样直接检验: "天元玉册的司天→下临宫映射是否有超出随机的预测力"

同时检验宫2 wspd信号是否在permutation下仍然接近显著
"""

import numpy as np
import xarray as xr
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# 1. 参数定义(与expE一致)
# ============================================================
lat_lo, lat_hi = 29.62, 39.62
lon_lo, lon_hi = 107.45, 117.45
ext_lat_lo, ext_lat_hi = 17.5, 42.5
ext_lon_lo, ext_lon_hi = 97.5, 122.5

PALACE_WUXING = {
    1: '水', 2: '土', 3: '木', 4: '木', 5: '土',
    6: '金', 7: '金', 8: '土', 9: '火'
}

# 新映射(主预测)
XIALIN_PRED = {
    2: {'var': 'rhum', 'direction': 'neg'},
    7: {'var': 'air', 'direction': 'pos'},
    1: {'var': 'air', 'direction': 'pos'},
    3: {'var': 'wspd', 'direction': 'neg'},
    9: {'var': 'air', 'direction': 'neg'},
}

SITIAN_XIALIN = {
    '厥阴风木': 2, '少阴君火': 7, '少阳相火': 7,
    '太阴湿土': 1, '阳明燥金': 3, '太阳寒水': 9,
}

SITIAN_WUXING = {
    '厥阴风木': '木', '少阴君火': '火', '少阳相火': '火',
    '太阴湿土': '土', '阳明燥金': '金', '太阳寒水': '水',
}

KE = {'木': '土', '土': '水', '水': '火', '火': '金', '金': '木'}

QI_MONTHS = {
    1: [1, 2], 2: [3, 4], 3: [5, 6],
    4: [7, 8], 5: [9, 10], 6: [11, 12],
}

TIANGAN = ['甲','乙','丙','丁','戊','己','庚','辛','壬','癸']
DIZHI = ['子','丑','寅','卯','辰','巳','午','未','申','酉','戌','亥']

DIZHI_SITIAN = {
    '子': '少阴君火', '午': '少阴君火',
    '丑': '太阴湿土', '未': '太阴湿土',
    '寅': '少阳相火', '申': '少阳相火',
    '卯': '阳明燥金', '酉': '阳明燥金',
    '辰': '太阳寒水', '戌': '太阳寒水',
    '巳': '厥阴风木', '亥': '厥阴风木',
}

def get_year_info(year):
    tigan_idx = (year - 4) % 10
    dizhi_idx = (year - 4) % 12
    TIANGAN_YUN = {0:'土',1:'金',2:'水',3:'木',4:'火',5:'土',6:'金',7:'水',8:'木',9:'火'}
    yun_wuxing = TIANGAN_YUN[tigan_idx]
    is_too_much = tigan_idx % 2 == 0
    sitian = DIZHI_SITIAN[DIZHI[dizhi_idx]]
    sitian_wuxing = SITIAN_WUXING[sitian]
    xialin_palace = SITIAN_XIALIN[sitian]
    zhongjian_ke_sitian = (KE.get(yun_wuxing) == sitian_wuxing)
    return {
        'year': year, 'yun_wuxing': yun_wuxing, 'is_too_much': is_too_much,
        'sitian': sitian, 'sitian_wuxing': sitian_wuxing,
        'xialin_palace': xialin_palace,
        'zhongjian_ke_sitian': zhongjian_ke_sitian,
    }

def assign_palace(lat, lon):
    if lat < lat_lo: row = 0
    elif lat > lat_hi: row = 2
    else: row = 1
    if lon < lon_lo: col = 0
    elif lon > lon_hi: col = 2
    else: col = 1
    palace_map = {(0,0):6,(0,1):1,(0,2):8,(1,0):7,(1,1):5,(1,2):3,(2,0):2,(2,1):9,(2,2):4}
    return palace_map.get((row,col), None)

# ============================================================
# 2. 加载数据
# ============================================================
print("=" * 70)
print("实验E2: 下临宫旋转Permutation检验")
print("核心: 正确映射 vs 随机映射的方向一致率比较")
print("=" * 70)

DATA_DIR = './data/ncep_raw'
variables = {}
for varname in ['air', 'rhum', 'wspd']:
    ds = xr.open_dataset(f'{DATA_DIR}/{varname}.mon.mean.nc')
    variables[varname] = ds[varname]

lat = variables['air'].lat.values
lon = variables['air'].lon.values

import pandas as pd
dates = pd.to_datetime(variables['air'].time.values)
years_arr = dates.year.values
months_arr = dates.month.values

valid_years = list(range(1948, 2026))

# 九宫格点
palace_grids = {p: [] for p in range(1, 10)}
for i, la in enumerate(lat):
    for j, lo in enumerate(lon):
        if ext_lat_lo <= la <= ext_lat_hi and ext_lon_lo <= lo <= ext_lon_hi:
            p = assign_palace(la, lo)
            if p is not None:
                palace_grids[p].append((i, j))

# ============================================================
# 3. 预计算: 所有变量×所有宫的年均异常矩阵
# ============================================================
print("\n预计算异常矩阵...")

anom_matrix = {}  # anom_matrix[varname][palace] = {year: anomaly}

for varname in ['air', 'rhum', 'wspd']:
    print(f"  {varname}...", end=' ')
    data = variables[varname].values
    anom_matrix[varname] = {}
    
    for p in range(1, 10):
        grids = palace_grids[p]
        if len(grids) == 0:
            continue
        
        palace_data = np.array([data[:, i, j] for i, j in grids]).mean(axis=0)
        
        # qi标准化
        qi_groups = {qi: [] for qi in range(1, 7)}
        qi_indices = {qi: [] for qi in range(1, 7)}
        for t in range(len(palace_data)):
            for qi, qm in QI_MONTHS.items():
                if months_arr[t] in qm:
                    qi_groups[qi].append(palace_data[t])
                    qi_indices[qi].append(t)
        
        standardized = np.full(len(palace_data), np.nan)
        for qi in range(1, 7):
            vals = np.array(qi_groups[qi])
            idxs = np.array(qi_indices[qi])
            if len(vals) > 0:
                mu = vals.mean()
                sigma = vals.std()
                if sigma > 0:
                    standardized[idxs] = (vals - mu) / sigma
        
        year_dict = {}
        for year in valid_years:
            mask = (years_arr == year) & (~np.isnan(standardized))
            if mask.sum() >= 6:
                year_dict[year] = standardized[mask].mean()
        anom_matrix[varname][p] = year_dict
    print("done")

# ============================================================
# 4. 构建年份数据表
# ============================================================
print("\n构建年份数据表...")

year_data = []
for year in valid_years:
    info = get_year_info(year)
    row = {
        'year': year,
        'sitian': info['sitian'],
        'xialin_palace': info['xialin_palace'],
        'zhongjian_ke_sitian': info['zhongjian_ke_sitian'],
        'is_too_much': info['is_too_much'],
    }
    # 所有变量×所有宫的异常
    for varname in ['air', 'rhum', 'wspd']:
        for p in range(1, 10):
            if p in anom_matrix[varname]:
                row[f'{varname}_{p}'] = anom_matrix[varname][p].get(year, np.nan)
            else:
                row[f'{varname}_{p}'] = np.nan
    year_data.append(row)

n_years = len(year_data)
sitian_list = [row['sitian'] for row in year_data]
unique_sitians = list(set(sitian_list))
print(f"总年数: {n_years}")

# ============================================================
# 5. 定义方向一致率计算函数
# ============================================================

def compute_consistency(sitian_assignment, year_data, pred_map):
    """
    给定司天分配，计算方向一致率
    sitian_assignment: list of 司天 (length = n_years)
    pred_map: {下临宫: {'var':..., 'direction':...}}
    """
    n_consistent = 0
    n_total = 0
    
    for i, row in enumerate(year_data):
        sitian = sitian_assignment[i]
        xialin_p = SITIAN_XIALIN[sitian]
        pred = pred_map[xialin_p]
        varname = pred['var']
        direction = pred['direction']
        
        anom = row.get(f'{varname}_{xialin_p}', np.nan)
        if np.isnan(anom):
            continue
        
        n_total += 1
        if direction == 'pos' and anom > 0:
            n_consistent += 1
        elif direction == 'neg' and anom < 0:
            n_consistent += 1
    
    return n_consistent / n_total if n_total > 0 else np.nan

def compute_mean_anomaly(sitian_assignment, year_data, pred_map):
    """计算方向性均值异常(正值=与预测同向)"""
    signed_anoms = []
    
    for i, row in enumerate(year_data):
        sitian = sitian_assignment[i]
        xialin_p = SITIAN_XIALIN[sitian]
        pred = pred_map[xialin_p]
        varname = pred['var']
        direction = pred['direction']
        
        anom = row.get(f'{varname}_{xialin_p}', np.nan)
        if np.isnan(anom):
            continue
        
        # 统一为正值=与预测同向
        if direction == 'neg':
            anom = -anom
        
        signed_anoms.append(anom)
    
    return np.mean(signed_anoms) if signed_anoms else np.nan

# ============================================================
# 6. 检验1: 全局方向一致率Permutation
# ============================================================
print("\n" + "=" * 70)
print("检验1: 全局方向一致率Permutation (N=10000)")
print("=" * 70)

# 观测值
obs_rate = compute_consistency(sitian_list, year_data, XIALIN_PRED)
obs_mean = compute_mean_anomaly(sitian_list, year_data, XIALIN_PRED)

print(f"观测方向一致率: {obs_rate:.1%}")
print(f"观测方向性均值异常: {obs_mean:+.4f}")

# Permutation: 保持司天频次(6类各13年)，随机打乱
n_perm = 10000
null_rates = np.zeros(n_perm)
null_means = np.zeros(n_perm)

# 司天频次: 6类×13年=78
sitian_pool = []
for s in unique_sitians:
    sitian_pool.extend([s] * 13)  # 每类恰好13年

print(f"\nRunning {n_perm} permutations...")
for b in range(n_perm):
    # 随机打乱(保持每年气候数据不变，只改变司天标签)
    shuffled = np.random.permutation(sitian_pool).tolist()
    null_rates[b] = compute_consistency(shuffled, year_data, XIALIN_PRED)
    null_means[b] = compute_mean_anomaly(shuffled, year_data, XIALIN_PRED)

# p值
p_rate = np.mean(null_rates >= obs_rate)
p_mean = np.mean(null_means >= obs_mean)

print(f"\nNull分布: 均值={np.mean(null_rates):.1%}, 标准差={np.std(null_rates):.1%}")
print(f"95%分位: {np.percentile(null_rates, 95):.1%}")
print(f"99%分位: {np.percentile(null_rates, 99):.1%}")
print(f"\nPermutation p(一致率≥观测): {p_rate:.4f}")
print(f"Permutation p(均值异常≥观测): {p_mean:.4f}")

# ============================================================
# 7. 检验2: 逐宫方向一致率Permutation
# ============================================================
print("\n" + "=" * 70)
print("检验2: 逐宫方向一致率Permutation")
print("=" * 70)

for palace in [2, 7, 1, 3, 9]:
    pred = XIALIN_PRED[palace]
    varname = pred['var']
    direction = pred['direction']
    
    # 该宫下临的司天类型
    target_sitians = [s for s, p in SITIAN_XIALIN.items() if p == palace]
    
    # 观测: 该宫下临年的方向一致率
    xialin_years_idx = [i for i, row in enumerate(year_data) if row['sitian'] in target_sitians]
    xialin_anoms = [year_data[i].get(f'{varname}_{palace}', np.nan) for i in xialin_years_idx]
    xialin_anoms = [a for a in xialin_anoms if not np.isnan(a)]
    
    if direction == 'pos':
        obs_consistent = sum(1 for a in xialin_anoms if a > 0)
    else:
        obs_consistent = sum(1 for a in xialin_anoms if a < 0)
    obs_n = len(xialin_anoms)
    obs_r = obs_consistent / obs_n if obs_n > 0 else np.nan
    
    # Permutation: 随机选同样多的年份，看该宫该变量方向一致率
    all_anoms_for_palace = [year_data[i].get(f'{varname}_{palace}', np.nan) for i in range(n_years)]
    all_anoms_valid = [(i, a) for i, a in enumerate(all_anoms_for_palace) if not np.isnan(a)]
    
    null_consistent_rates = np.zeros(n_perm)
    for b in range(n_perm):
        # 随机选obs_n个年份
        rand_idx = np.random.choice(len(all_anoms_valid), obs_n, replace=False)
        rand_anoms = [all_anoms_valid[j][1] for j in rand_idx]
        if direction == 'pos':
            null_consistent_rates[b] = sum(1 for a in rand_anoms if a > 0) / obs_n
        else:
            null_consistent_rates[b] = sum(1 for a in rand_anoms if a < 0) / obs_n
    
    p_palace = np.mean(null_consistent_rates >= obs_r)
    
    print(f"\n宫{palace}({PALACE_WUXING[palace]}) | {varname} {direction}")
    print(f"  观测: {obs_consistent}/{obs_n} = {obs_r:.1%}")
    print(f"  Null: 均值={np.mean(null_consistent_rates):.1%}, 95%={np.percentile(null_consistent_rates, 95):.1%}")
    print(f"  Permutation p: {p_palace:.4f}")

# ============================================================
# 8. 检验3: 宫2 wspd专项Permutation
# ============================================================
print("\n" + "=" * 70)
print("检验3: 宫2 wspd↓专项(前次p=0.08的permutation验证)")
print("=" * 70)

# 观测: 厥阴司天13年, 宫2 wspd均值
xialin2_idx = [i for i, row in enumerate(year_data) if row['sitian'] == '厥阴风木']
xialin2_wspd = [year_data[i].get('wspd_2', np.nan) for i in xialin2_idx]
xialin2_wspd = [a for a in xialin2_wspd if not np.isnan(a)]

obs_mean_2 = np.mean(xialin2_wspd)
obs_t, obs_p_ttest = stats.ttest_1samp(xialin2_wspd, 0) if len(xialin2_wspd) > 1 else (np.nan, np.nan)

print(f"厥阴→2宫 wspd: 均值={obs_mean_2:+.4f}, t={obs_t:+.3f}, parametric p={obs_p_ttest:.4f}")

# Permutation: 从所有年份随机选13个，看宫2 wspd均值
all_wspd_2 = [year_data[i].get('wspd_2', np.nan) for i in range(n_years)]
all_wspd_2_valid = [a for a in all_wspd_2 if not np.isnan(a)]

null_means_2 = np.zeros(n_perm)
for b in range(n_perm):
    rand_anoms = np.random.choice(all_wspd_2_valid, 13, replace=False)
    null_means_2[b] = np.mean(rand_anoms)

# 单侧检验(预测wspd↓)
p_wspd_neg = np.mean(null_means_2 <= obs_mean_2)
p_wspd_2side = np.mean(np.abs(null_means_2) >= np.abs(obs_mean_2))

print(f"Permutation p(均值≤观测): {p_wspd_neg:.4f}")
print(f"Permutation p(|均值|≥|观测|): {p_wspd_2side:.4f}")
print(f"Null分布: 均值={np.mean(null_means_2):+.4f}, 5%分位={np.percentile(null_means_2, 5):+.4f}")

# ============================================================
# 9. 检验4: 宫2 全变量扫描Permutation
# ============================================================
print("\n" + "=" * 70)
print("检验4: 宫2 全变量扫描Permutation(修正多重比较)")
print("=" * 70)

for varname in ['air', 'rhum', 'wspd']:
    xialin2_vals = [year_data[i].get(f'{varname}_2', np.nan) for i in xialin2_idx]
    xialin2_vals = [a for a in xialin2_vals if not np.isnan(a)]
    
    all_vals = [year_data[i].get(f'{varname}_2', np.nan) for i in range(n_years)]
    all_vals_valid = [a for a in all_vals if not np.isnan(a)]
    
    obs_m = np.mean(xialin2_vals)
    obs_t_val, _ = stats.ttest_1samp(xialin2_vals, 0) if len(xialin2_vals) > 1 else (np.nan, np.nan)
    
    # Permutation
    null_t_vals = np.zeros(n_perm)
    null_means_v = np.zeros(n_perm)
    for b in range(n_perm):
        rand_vals = np.random.choice(all_vals_valid, 13, replace=False)
        null_means_v[b] = np.mean(rand_vals)
        t_v, _ = stats.ttest_1samp(rand_vals, 0) if len(rand_vals) > 1 else (0, 1)
        null_t_vals[b] = t_v
    
    # 双侧
    p_two = np.mean(np.abs(null_t_vals) >= np.abs(obs_t_val))
    p_neg = np.mean(null_t_vals <= obs_t_val)  # 左侧
    p_pos = np.mean(null_t_vals >= obs_t_val)  # 右侧
    
    print(f"\n宫2 {varname}: 观测均值={obs_m:+.4f}, t={obs_t_val:+.3f}")
    print(f"  Permutation p(双侧): {p_two:.4f}, p(左): {p_neg:.4f}, p(右): {p_pos:.4f}")

# ============================================================
# 10. 检验5: 太阴→1宫上半年air↑ Permutation
# ============================================================
print("\n" + "=" * 70)
print("检验5: 太阴→1宫上半年air↑ Permutation")
print("=" * 70)

# 需要分qi数据
qi_anom_matrix = {}
for varname in ['air']:
    print(f"  计算 {varname} 分qi异常...")
    data = variables[varname].values
    qi_anom_matrix[varname] = {}
    
    for p in [1]:
        grids = palace_grids[p]
        palace_data = np.array([data[:, i, j] for i, j in grids]).mean(axis=0)
        
        qi_groups = {qi: [] for qi in range(1, 7)}
        qi_indices = {qi: [] for qi in range(1, 7)}
        for t in range(len(palace_data)):
            for qi, qm in QI_MONTHS.items():
                if months_arr[t] in qm:
                    qi_groups[qi].append(palace_data[t])
                    qi_indices[qi].append(t)
        
        standardized = np.full(len(palace_data), np.nan)
        for qi in range(1, 7):
            vals = np.array(qi_groups[qi])
            idxs = np.array(qi_indices[qi])
            if len(vals) > 0:
                mu = vals.mean()
                sigma = vals.std()
                if sigma > 0:
                    standardized[idxs] = (vals - mu) / sigma
        
        qi_anom_matrix[varname][p] = {}
        for year in valid_years:
            qi_vals = {}
            for qi in range(1, 7):
                qi_months = QI_MONTHS[qi]
                mask = (years_arr == year) & np.isin(months_arr, qi_months) & (~np.isnan(standardized))
                if mask.sum() >= len(qi_months):
                    qi_vals[qi] = standardized[mask].mean()
            qi_anom_matrix[varname][p][year] = qi_vals

# 太阴司天年上半年1宫air异常
taiyin_idx = [i for i, row in enumerate(year_data) if row['sitian'] == '太阴湿土']
taiyin_upper_air1 = []
for i in taiyin_idx:
    y = year_data[i]['year']
    qi_dict = qi_anom_matrix['air'][1].get(y, {})
    vals = [qi_dict.get(qi, np.nan) for qi in [1, 2, 3]]
    vals = [v for v in vals if not np.isnan(v)]
    if vals:
        taiyin_upper_air1.append(np.mean(vals))

obs_upper_mean = np.mean(taiyin_upper_air1) if taiyin_upper_air1 else np.nan
print(f"太阴→1宫上半年air: 均值={obs_upper_mean:+.4f}, n={len(taiyin_upper_air1)}")

# Permutation: 随机选13年
all_upper_air1 = []
for y in valid_years:
    qi_dict = qi_anom_matrix['air'][1].get(y, {})
    vals = [qi_dict.get(qi, np.nan) for qi in [1, 2, 3]]
    vals = [v for v in vals if not np.isnan(v)]
    if vals:
        all_upper_air1.append(np.mean(vals))

null_upper = np.zeros(n_perm)
for b in range(n_perm):
    rand_vals = np.random.choice(all_upper_air1, 13, replace=False)
    null_upper[b] = np.mean(rand_vals)

p_upper = np.mean(null_upper >= obs_upper_mean)
print(f"Permutation p(均值≥观测): {p_upper:.4f}")
print(f"Null: 均值={np.mean(null_upper):+.4f}, 95%={np.percentile(null_upper, 95):+.4f}")

# ============================================================
# 11. 检验6: 多重比较校正
# ============================================================
print("\n" + "=" * 70)
print("检验6: 多重比较校正 — min-p permutation")
print("=" * 70)
print("思路: 如果下临宫系统有整体预测力，至少一个(宫,变量)组合")
print("应该比随机强。用min-p方法控制FWER。")

# 对5个下临宫×3个变量=15个假设做多重比较
# 每个假设: 下临年该宫该变量均值 vs 零
# 观测: 15个中最小的p值
# Permutation: 对每个permutation计算15个p值，取最小值

print("\n计算15个(宫,变量)观测统计量...")

combos = [(p, v) for p in [2, 7, 1, 3, 9] for v in ['air', 'rhum', 'wspd']]
obs_stats = {}

for palace, varname in combos:
    target_sitians = [s for s, p in SITIAN_XIALIN.items() if p == palace]
    xialin_idx = [i for i, row in enumerate(year_data) if row['sitian'] in target_sitians]
    vals = [year_data[i].get(f'{varname}_{palace}', np.nan) for i in xialin_idx]
    vals = [a for a in vals if not np.isnan(a)]
    
    if len(vals) > 1:
        t_val, _ = stats.ttest_1samp(vals, 0)
        obs_stats[(palace, varname)] = t_val
    else:
        obs_stats[(palace, varname)] = 0.0

obs_min_abs_t = min(abs(v) for v in obs_stats.values())
obs_max_abs_t = max(abs(v) for v in obs_stats.values())

# 找最大|t|对应的组合
best_combo = max(obs_stats.keys(), key=lambda k: abs(obs_stats[k]))
print(f"最大|t|: 宫{best_combo[0]} {best_combo[1]}, t={obs_stats[best_combo]:+.3f}")

# Permutation
print(f"\nRunning {n_perm} permutations for FWER...")
null_max_t = np.zeros(n_perm)

for b in range(n_perm):
    # 随机打乱司天
    shuffled = np.random.permutation(sitian_pool).tolist()
    
    max_t_this = 0
    for palace, varname in combos:
        target_sitians = [s for s, p in SITIAN_XIALIN.items() if p == palace]
        # 打乱后指向该宫的年份
        xialin_idx = [i for i in range(n_years) if shuffled[i] in target_sitians]
        vals = [year_data[i].get(f'{varname}_{palace}', np.nan) for i in xialin_idx]
        vals = [a for a in vals if not np.isnan(a)]
        
        if len(vals) > 1:
            t_val, _ = stats.ttest_1samp(vals, 0)
            if abs(t_val) > max_t_this:
                max_t_this = abs(t_val)
    
    null_max_t[b] = max_t_this

# FWER校正后的p值
p_fwer = np.mean(null_max_t >= abs(obs_stats[best_combo]))
print(f"\nFWER校正后p值(宫{best_combo[0]} {best_combo[1]}): {p_fwer:.4f}")
print(f"Null最大|t|分布: 均值={np.mean(null_max_t):.3f}, 95%={np.percentile(null_max_t, 95):.3f}")

# 每个组合的FWER校正p值
print(f"\n所有组合FWER校正结果:")
for palace, varname in combos:
    obs_t = abs(obs_stats[(palace, varname)])
    p_fwer_combo = np.mean(null_max_t >= obs_t)
    sig = '***' if p_fwer_combo < 0.001 else '**' if p_fwer_combo < 0.01 else '*' if p_fwer_combo < 0.05 else ''
    print(f"  宫{palace} {varname}: |t|={obs_t:.3f}, FWER-p={p_fwer_combo:.4f}{sig}")

# ============================================================
# 12. 总结
# ============================================================
print("\n" + "=" * 70)
print("实验E2总结")
print("=" * 70)
print(f"1. 全局方向一致率 permutation p = {p_rate:.4f}")
print(f"2. 宫2 wspd↓ permutation p(左) = {p_wspd_neg:.4f}")
print(f"3. 太阴→1宫上半年air↑ permutation p = {p_upper:.4f}")
print(f"4. 最佳组合 FWER-p = {p_fwer:.4f}")
